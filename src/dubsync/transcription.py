from __future__ import annotations

from math import isfinite
from pathlib import Path

from .audio import AudioNormalizationLimits, normalize_audio
from .asr_timing import clamp_asr_word_durations
from .cache import JsonDiskCache, write_json_atomic, write_text_atomic
from .config import load_style_profile, load_yaml
from .cost import CostMeter, asr_dollars_per_hour, audio_seconds, record_llm_usage
from .cue_segmentation import group_word_indices_for_cues
from .llm_providers import drain_usage_events, llm_config_for_pass, punctuation_adapter_from_config
from .models import AlignmentResult, Cue, QCFlag, Word
from .output_order import finalize_cues_for_output
from .providers import (
    CachedASRAdapter,
    adapter_from_config,
    apply_asr_language,
    apply_local_asr_config,
    apply_transcription_provider_config,
)
from .profanity import apply_german_profanity_censorship, censor_german_profanity_flags
from .punctuation import apply_punctuation_pass
from .reports import write_qc_report
from .srt_io import write_srt
from .style_profile import GenerationConstraints, StyleProfile
from .text_metrics import wrap_visual_width
from .timing_refinement import boundary_refinement_config_from_config, refine_cues_to_speech_activity
from .vad import (
    min_coverage_from_config,
    speech_activity_adapter_from_config,
    speech_activity_flags_for_cues,
    trailing_silence_flags_for_cues,
)
from .verify import cps_sanity_flags, lint_cues, score_cues


class TranscriptionResult:
    def __init__(self, output_srt: Path, episode_workdir: Path, cost_meter: CostMeter, report: dict[str, object]):
        self.output_srt = output_srt
        self.episode_workdir = episode_workdir
        self.cost_meter = cost_meter
        self.report = report


def build_cues_from_words(
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float = 0.8,
    max_cue_duration_seconds: float = 5.0,
    preserve_timing: bool = False,
) -> list[Cue]:
    cues, _ = _build_cues_with_word_ownership(
        words,
        profile,
        max_gap_seconds=max_gap_seconds,
        max_cue_duration_seconds=max_cue_duration_seconds,
        preserve_timing=preserve_timing,
    )
    return cues


def _build_cues_with_word_ownership(
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
    preserve_timing: bool,
) -> tuple[list[Cue], AlignmentResult]:
    """Keep lexical ownership from segmentation; display padding is not evidence."""
    groups = group_word_indices_for_cues(
        words,
        list(range(len(words))),
        profile,
        max_gap_seconds=max_gap_seconds,
        max_cue_duration_seconds=max_cue_duration_seconds,
    )
    cues = [
        _cue_from_group(
            index, [words[word_index] for word_index in group], profile, preserve_timing=preserve_timing
        )
        for index, group in enumerate(groups, start=1)
    ]
    alignment = AlignmentResult(
        cue_word_indices={index: list(group) for index, group in enumerate(groups, start=1)}
    )
    return (cues if preserve_timing else _cap_generated_overlaps(cues, profile)), alignment


def generate_srt_from_audio(
    audio_path: Path,
    output_path: Path,
    workdir: Path,
    style_path: Path | None = None,
    providers_path: Path | None = None,
    no_llm: bool = False,
    fps: float | None = None,
    local: bool = False,
    language: str | None = None,
    transcription_provider: str = "default",
    allow_gemini_transcribe_web: bool = False,
    style_profile: StyleProfile | None = None,
    generation_constraints: GenerationConstraints | None = None,
    audio_limits: AudioNormalizationLimits | None = None,
) -> TranscriptionResult:
    episode_workdir = workdir / audio_path.stem
    episode_workdir.mkdir(parents=True, exist_ok=True)
    profile = style_profile.model_copy(deep=True) if style_profile is not None else load_style_profile(style_path) or StyleProfile()
    if fps is not None:
        profile = profile.model_copy(update={"fps": fps})

    provider_config = apply_transcription_provider_config(
        _provider_config(load_yaml(providers_path), local=local),
        transcription_provider,
    )
    provider_config = apply_asr_language(provider_config, language)
    if local:
        no_llm = True
    asr_config = provider_config.get("asr", {})
    if not isinstance(asr_config, dict):
        raise ValueError("providers.yaml asr section must be a mapping")

    audio_for_asr = audio_path
    if not asr_config.get("fixture_path"):
        audio_for_asr = normalize_audio(
            audio_path,
            episode_workdir / "audio.16k.wav",
            limits=audio_limits,
        )

    provider = str(asr_config.get("provider", "fixture"))
    model = str(asr_config.get("model_id", asr_config.get("model", provider)))
    cost_meter = CostMeter()
    adapter = CachedASRAdapter(
        adapter_from_config(
            provider_config,
            local_mode=local,
            allow_gemini_transcribe_web=allow_gemini_transcribe_web,
        ),
        JsonDiskCache(episode_workdir / "asr-cache"),
        model,
        asr_config,
        cost_meter=cost_meter,
        cost_provider=model,
        dollars_per_hour=asr_dollars_per_hour(provider, asr_config),
    )
    words = adapter.transcribe(audio_for_asr)
    flags: list[QCFlag] = list(adapter.last_repair_flags)
    asr_metadata = {
        "provider": provider,
        "model": model,
        "repair_flags": [flag.model_dump() for flag in adapter.last_repair_flags],
    }
    _write_json(
        episode_workdir / "asr.json",
        {
            "words": [word.model_dump() for word in words],
            "metadata": asr_metadata,
        },
    )

    boundary_refinement = boundary_refinement_config_from_config(provider_config)
    speech_regions = []
    speech_activity_adapter = speech_activity_adapter_from_config(provider_config)
    if speech_activity_adapter is not None:
        speech_regions = speech_activity_adapter.detect(audio_for_asr)
        if getattr(speech_activity_adapter, "fallback_used", False):
            flags.append(
                QCFlag(
                    kind="vad_provider_fallback",
                    cue_ids=[],
                    message="Configured VAD provider fell back to energy-based speech activity detection.",
                    severity="warning",
                )
            )
        _write_json(episode_workdir / "vad.json", {"regions": [region.model_dump() for region in speech_regions]})
    timing_config = provider_config.get("timing", {})
    if not isinstance(timing_config, dict):
        raise ValueError("providers.yaml timing section must be a mapping")
    words, word_clamp_flags = clamp_asr_word_durations(
        words,
        speech_regions,
        max_word_duration=_positive_float(timing_config, "max_word_duration", 2.0),
        max_region_overrun=boundary_refinement.max_trailing_silence_ms / 1000.0,
    )
    flags.extend(word_clamp_flags)

    generation_config = provider_config.get("generation", {})
    if not isinstance(generation_config, dict):
        raise ValueError("providers.yaml generation section must be a mapping")
    constraints = (
        generation_constraints.model_copy(deep=True)
        if generation_constraints is not None
        else _generation_constraints(provider_config, generation_config)
    )
    cues, alignment = _build_cues_with_word_ownership(
        words,
        profile,
        max_gap_seconds=constraints.max_gap_seconds,
        max_cue_duration_seconds=constraints.max_cue_duration_seconds,
        preserve_timing=True,
    )
    if speech_regions:
        cues, timing_flags = refine_cues_to_speech_activity(
            cues,
            speech_regions,
            profile,
            boundary_refinement,
            words=words,
            alignment=alignment,
        )
        flags.extend(timing_flags)

    if not no_llm:
        punctuation_adapter = punctuation_adapter_from_config(provider_config)
        if punctuation_adapter is not None:
            cues, punctuation_flags = apply_punctuation_pass(
                cues,
                punctuation_adapter,
                source_cues=[],
                scene_gap_seconds=_punctuation_scene_gap(provider_config),
                max_chars_per_line=profile.max_chars_per_line,
                max_lines_per_cue=profile.max_lines_per_cue,
            )
            flags.extend(punctuation_flags)
            _record_punctuation_cost(cost_meter, punctuation_adapter, provider_config)

    output_config = provider_config.get("output", {})
    if not isinstance(output_config, dict):
        raise ValueError("providers.yaml output section must be a mapping")
    duration_seconds = audio_seconds(audio_for_asr)
    cues, output_flags = finalize_cues_for_output(
        cues,
        profile,
        no_overlaps=bool(output_config.get("no_overlaps", True)),
        max_cps=constraints.max_cps,
        max_cue_duration_seconds=constraints.max_cue_duration_seconds,
        preserve_timing=True,
        media_duration_ms=round(duration_seconds * 1000) if duration_seconds > 0 else None,
        merge_duplicates=False,
    )
    flags.extend(output_flags)
    if speech_activity_adapter is not None:
        flags.extend(speech_activity_flags_for_cues(cues, speech_regions, min_coverage_from_config(provider_config)))
        flags.extend(
            trailing_silence_flags_for_cues(
                cues, speech_regions, max_trailing_silence_ms=boundary_refinement.max_trailing_silence_ms
            )
        )
    cues, profanity_flags = apply_german_profanity_censorship(cues)
    flags.extend(profanity_flags)
    flags.extend(cps_sanity_flags(cues, max_cps=constraints.max_cps, min_cps=constraints.min_cps))
    flags = censor_german_profanity_flags(flags)

    style_issues = lint_cues(cues, profile)
    cue_scores = score_cues(cues, words, alignment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_path, write_srt(cues, renumber=True))
    _write_json(
        episode_workdir / "generate.json",
        {
            "mode": "generate",
            "cues": [cue.model_dump() for cue in cues],
            "profile": profile.model_dump(),
            "constraints": constraints.model_dump(),
            "asr": asr_metadata,
            "cue_word_indices": alignment.cue_word_indices,
        },
    )
    report = write_qc_report(
        episode_workdir / "qc_report.json",
        episode_workdir / "qc_report.html",
        cues,
        flags,
        style_issues,
        cue_scores=cue_scores,
        summary_metadata={
            "fps": profile.fps,
            "fps_source": "explicit" if fps is not None else "fallback",
            "fps_detection_confident": fps is not None,
            "asr_provider": provider,
            "asr_model": model,
            "asr_repair_count": len(adapter.last_repair_flags),
        },
    )
    write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())
    return TranscriptionResult(output_path, episode_workdir, cost_meter, report)


def _cue_from_group(
    index: int, group: list[Word], profile: StyleProfile, *, preserve_timing: bool = False
) -> Cue:
    text = " ".join(word.text.strip() for word in group)
    lines = wrap_visual_width(text, profile.max_chars_per_line) or [text]
    start_ms = profile.snap_floor(max(0, group[0].start * 1000 - profile.lead_in_ms))
    spoken_end_ms = profile.snap_ceil(max(word.end for word in group) * 1000 + profile.tail_ms)
    minimum_end_ms = spoken_end_ms if preserve_timing else profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
    speaker_ids = [word.speaker_id for word in group if word.speaker_id]
    speaker_id = max(set(speaker_ids), key=speaker_ids.count) if speaker_ids else None
    return Cue(
        index=index,
        start_ms=start_ms,
        end_ms=max(spoken_end_ms, minimum_end_ms, start_ms + 1),
        lines=lines,
        speaker_id=speaker_id,
    )


def _cap_generated_overlaps(cues: list[Cue], profile: StyleProfile) -> list[Cue]:
    result: list[Cue] = []
    for index, cue in enumerate(cues):
        next_start = cues[index + 1].start_ms if index + 1 < len(cues) else None
        end_ms = cue.end_ms
        if next_start is not None and cue.start_ms < next_start < end_ms:
            end_ms = max(cue.start_ms + 1, next_start)
        snapped_end_ms = profile.snap_floor(end_ms)
        result.append(cue.with_timing(cue.start_ms, snapped_end_ms if snapped_end_ms > cue.start_ms else end_ms))
    return result


def _provider_config(config: dict[str, object], *, local: bool) -> dict[str, object]:
    return apply_local_asr_config(config, local)


def _punctuation_scene_gap(config: dict[str, object]) -> float:
    llm = config.get("llm", {})
    if not isinstance(llm, dict):
        return 4.0
    punctuation = llm.get("punctuation", {})
    if not isinstance(punctuation, dict):
        return 4.0
    return _positive_float(punctuation, "scene_gap_seconds", 4.0)


def _record_punctuation_cost(meter: CostMeter, adapter: object, config: dict[str, object]) -> None:
    llm_config = llm_config_for_pass(config, "punctuation")
    provider = str(llm_config.get("provider", "gemini"))
    model = str(llm_config.get("model", provider))
    for event in drain_usage_events(adapter):
        record_llm_usage(meter, provider, model, llm_config, event)


def _positive_float(source: dict[str, object], key: str, default: float) -> float:
    value = float(source.get(key, default))
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and greater than zero")
    return value


def _generation_constraints(
    provider_config: dict[str, object],
    generation_config: dict[str, object],
) -> GenerationConstraints:
    timing_config = provider_config.get("timing", {})
    if not isinstance(timing_config, dict):
        raise ValueError("providers.yaml timing section must be a mapping")
    return GenerationConstraints(
        max_gap_seconds=_positive_float(generation_config, "max_gap_seconds", 0.8),
        max_cue_duration_seconds=_positive_float(generation_config, "max_cue_duration_seconds", 5.0),
        min_cps=_nonnegative_float(timing_config, "min_cps", 2.0),
        max_cps=_positive_float(timing_config, "max_cps", 30.0),
    )


def _nonnegative_float(source: dict[str, object], key: str, default: float) -> float:
    value = float(source.get(key, default))
    if not isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and zero or greater")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    write_json_atomic(path, payload)
