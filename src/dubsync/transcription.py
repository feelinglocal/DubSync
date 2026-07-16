from __future__ import annotations

import json
from pathlib import Path

from .audio import AudioNormalizationLimits, normalize_audio
from .cache import JsonDiskCache
from .config import load_style_profile, load_yaml
from .cost import CostMeter, asr_dollars_per_hour, record_llm_usage
from .llm_providers import drain_usage_events, llm_config_for_pass, punctuation_adapter_from_config
from .models import AlignmentResult, Cue, QCFlag, Word
from .output_order import finalize_cues_for_output
from .providers import CachedASRAdapter, adapter_from_config, apply_asr_language
from .punctuation import apply_punctuation_pass
from .reports import write_qc_report
from .srt_io import write_srt
from .style_profile import GenerationConstraints, StyleProfile
from .text_metrics import wrap_visual_width
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
) -> list[Cue]:
    ordered = sorted(
        (word for word in words if word.text.strip() and word.end >= word.start and word.start >= 0),
        key=lambda word: (word.start, word.end),
    )
    groups = _word_groups(
        ordered,
        profile,
        max_gap_seconds=max_gap_seconds,
        max_cue_duration_seconds=max_cue_duration_seconds,
    )
    cues = [_cue_from_group(index, group, profile) for index, group in enumerate(groups, start=1)]
    return _cap_generated_overlaps(cues, profile)


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
    style_profile: StyleProfile | None = None,
    generation_constraints: GenerationConstraints | None = None,
    audio_limits: AudioNormalizationLimits | None = None,
) -> TranscriptionResult:
    episode_workdir = workdir / audio_path.stem
    episode_workdir.mkdir(parents=True, exist_ok=True)
    profile = style_profile.model_copy(deep=True) if style_profile is not None else load_style_profile(style_path) or StyleProfile()
    if fps is not None:
        profile = profile.model_copy(update={"fps": fps})

    provider_config = apply_asr_language(_provider_config(load_yaml(providers_path), local=local), language)
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
        adapter_from_config(provider_config),
        JsonDiskCache(episode_workdir / "asr-cache"),
        model,
        asr_config,
        cost_meter=cost_meter,
        cost_provider=model,
        dollars_per_hour=asr_dollars_per_hour(provider, asr_config),
    )
    words = adapter.transcribe(audio_for_asr)
    _write_json(episode_workdir / "asr.json", {"words": [word.model_dump() for word in words]})

    generation_config = provider_config.get("generation", {})
    if not isinstance(generation_config, dict):
        raise ValueError("providers.yaml generation section must be a mapping")
    constraints = (
        generation_constraints.model_copy(deep=True)
        if generation_constraints is not None
        else _generation_constraints(provider_config, generation_config)
    )
    cues = build_cues_from_words(
        words,
        profile,
        max_gap_seconds=constraints.max_gap_seconds,
        max_cue_duration_seconds=constraints.max_cue_duration_seconds,
    )

    flags: list[QCFlag] = []
    if not no_llm:
        punctuation_adapter = punctuation_adapter_from_config(provider_config)
        if punctuation_adapter is not None:
            cues, punctuation_flags = apply_punctuation_pass(
                cues,
                punctuation_adapter,
                scene_gap_seconds=_punctuation_scene_gap(provider_config),
                max_chars_per_line=profile.max_chars_per_line,
                max_lines_per_cue=profile.max_lines_per_cue,
            )
            flags.extend(punctuation_flags)
            _record_punctuation_cost(cost_meter, punctuation_adapter, provider_config)

    output_config = provider_config.get("output", {})
    if not isinstance(output_config, dict):
        raise ValueError("providers.yaml output section must be a mapping")
    cues, output_flags = finalize_cues_for_output(
        cues,
        profile,
        no_overlaps=bool(output_config.get("no_overlaps", True)),
        max_cps=constraints.max_cps,
        max_cue_duration_seconds=constraints.max_cue_duration_seconds,
    )
    flags.extend(output_flags)
    flags.extend(cps_sanity_flags(cues, max_cps=constraints.max_cps, min_cps=constraints.min_cps))

    alignment = AlignmentResult(cue_word_indices=_cue_word_indices(cues, words))
    style_issues = lint_cues(cues, profile)
    cue_scores = score_cues(cues, words, alignment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(write_srt(cues, renumber=True), encoding="utf-8")
    _write_json(
        episode_workdir / "generate.json",
        {
            "mode": "generate",
            "cues": [cue.model_dump() for cue in cues],
            "profile": profile.model_dump(),
            "constraints": constraints.model_dump(),
        },
    )
    report = write_qc_report(
        episode_workdir / "qc_report.json",
        episode_workdir / "qc_report.html",
        cues,
        flags,
        style_issues,
        cue_scores=cue_scores,
    )
    (episode_workdir / "cost.json").write_text(cost_meter.to_json(), encoding="utf-8")
    return TranscriptionResult(output_path, episode_workdir, cost_meter, report)


def _word_groups(
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and _starts_new_cue(
            current,
            word,
            profile,
            max_gap_seconds=max_gap_seconds,
            max_cue_duration_seconds=max_cue_duration_seconds,
        ):
            groups.append(current)
            current = []
        current.append(word)
        if _ends_sentence(word.text) and word.end - current[0].start >= profile.min_cue_dur:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _starts_new_cue(
    current: list[Word],
    word: Word,
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> bool:
    previous = current[-1]
    if word.start - previous.end > max_gap_seconds:
        return True
    if previous.speaker_id and word.speaker_id and previous.speaker_id != word.speaker_id:
        return True
    if word.end - current[0].start > max_cue_duration_seconds:
        return True
    candidate = " ".join(item.text.strip() for item in [*current, word])
    return len(wrap_visual_width(candidate, profile.max_chars_per_line)) > profile.max_lines_per_cue


def _cue_from_group(index: int, group: list[Word], profile: StyleProfile) -> Cue:
    text = " ".join(word.text.strip() for word in group)
    lines = wrap_visual_width(text, profile.max_chars_per_line) or [text]
    start_ms = profile.snap_floor(max(0, group[0].start * 1000 - profile.lead_in_ms))
    spoken_end_ms = profile.snap_ceil(group[-1].end * 1000 + profile.tail_ms)
    minimum_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
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
        if next_start is not None and end_ms > next_start:
            end_ms = max(cue.start_ms + 1, next_start)
        result.append(cue.with_timing(cue.start_ms, profile.snap_floor(end_ms) if end_ms > cue.start_ms else end_ms))
    return result


def _cue_word_indices(cues: list[Cue], words: list[Word]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for cue in cues:
        mapping[cue.index] = [
            index
            for index, word in enumerate(words)
            if word.end * 1000 >= cue.start_ms and word.start * 1000 <= cue.end_ms
        ]
    return mapping


def _provider_config(config: dict[str, object], *, local: bool) -> dict[str, object]:
    if not local:
        return dict(config)
    next_config = dict(config)
    existing = next_config.get("asr", {})
    asr_config = dict(existing) if isinstance(existing, dict) else {}
    asr_config["provider"] = "whisperx"
    asr_config.pop("fixture_path", None)
    next_config["asr"] = asr_config
    return next_config


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
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero")
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
    if value < 0:
        raise ValueError(f"{key} must be zero or greater")
    return value


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", "...", "…"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
