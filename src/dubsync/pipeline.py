from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .adjudication import AdjudicationEngine, KeepSRTAdapter
from .aligner import align_cues_to_words
from .asr_timing import clamp_asr_word_durations
from .audio import normalize_audio
from .audio_snippets import extract_audio_snippets
from .cache import CacheKey, JsonDiskCache
from .changes import apply_adjudication_decisions
from .config import load_style_profile, load_yaml
from .cost import CostMeter, asr_dollars_per_hour, record_llm_usage
from .forced_alignment import apply_forced_alignment, forced_alignment_adapter_from_config
from .llm_providers import (
    _ADJUDICATION_PROMPT_VERSION,
    drain_usage_events,
    llm_adapter_from_config,
    llm_config_for_pass,
    punctuation_adapter_from_config,
)
from .models import AdjudicationDecision, AlignmentResult, AudioSnippet, Cue, CueContext, DivergenceSpan, ForcedAlignmentCue, QCFlag, Word
from .output_order import finalize_cues_for_output
from .overlap import apply_overlap_policy
from .overlap_detection import overlap_detection_adapter_from_config, overlap_flags_for_regions
from .providers import CachedASRAdapter, adapter_from_config, apply_asr_language
from .punctuation import apply_punctuation_pass
from .recue import rebuild_cues
from .reports import write_changes_diff, write_qc_report
from .srt_io import parse_srt_text, write_srt
from .silence import silence_flags_for_cues
from .source_quality import detect_source_errors
from .source_order import sort_cues_chronologically
from .speaker_mapping import speaker_mapping_adapter_from_config, speaker_mapping_flags
from .style_profile import StyleProfile, derive_style_profile
from .timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity
from .tokenize import alphanumeric_signature
from .vad import (
    dropped_line_flags_for_unmatched_cues,
    min_coverage_from_config,
    speech_activity_adapter_from_config,
    speech_activity_flags_for_cues,
)
from .verify import cps_sanity_flags, lint_cues, score_cues

VERIFY_STAGE_FLAG_KINDS = frozenset(
    {
        "asr_word_clamped",
        "cps_cue_merged",
        "cps_duration_extended",
        "cue_on_silence",
        "cue_without_speech_activity",
        "duplicate_cue_merged",
        "forced_alignment_refined",
        "impossible_cps_fast",
        "impossible_cps_slow",
        "output_overlap_resolved",
        "overlap_detected",
        "speaker_transition_gap_inserted",
        "timing_refined",
    }
)


class PipelineResult:
    def __init__(self, output_srt: Path, episode_workdir: Path, cost_meter: CostMeter, report: dict[str, object]):
        self.output_srt = output_srt
        self.episode_workdir = episode_workdir
        self.cost_meter = cost_meter
        self.report = report


def sync_episode(
    srt_path: Path,
    audio_path: Path,
    output_path: Path,
    workdir: Path,
    style_path: Path | None = None,
    providers_path: Path | None = None,
    no_llm: bool = False,
    fps: float | None = None,
    resume: str | None = None,
    local: bool = False,
    language: str | None = None,
) -> PipelineResult:
    resume_stage = _normalize_resume_stage(resume)
    episode_workdir = workdir / srt_path.stem
    episode_workdir.mkdir(parents=True, exist_ok=True)
    cost_meter = CostMeter()
    cues = _source_cues_for_run(srt_path, episode_workdir, resume_stage)
    cues, source_order_flags = sort_cues_chronologically(cues)
    style_artifact_path = episode_workdir / "style_profile.json"
    profile = load_style_profile(style_path) or _load_style_profile_for_resume(style_artifact_path, resume_stage) or derive_style_profile(cues)
    if fps is not None:
        profile = profile.model_copy(update={"fps": fps})
    if _should_write_ingest_artifacts(resume_stage, episode_workdir, style_path, fps):
        _write_json(episode_workdir / "ingest.json", {"cues": [cue.model_dump() for cue in cues]})
        _write_json(style_artifact_path, profile.model_dump())

    provider_config = apply_asr_language(_apply_local_mode(load_yaml(providers_path), local), language)
    if local:
        no_llm = True
    audio_for_asr = audio_path
    if _should_load_asr_artifact(resume_stage):
        words = _load_asr_artifact(episode_workdir / "asr.json")
        audio_for_asr = _resume_audio_for_verify(audio_path, episode_workdir)
    else:
        asr_config = provider_config.get("asr", {}) if isinstance(provider_config, dict) else {}
        if isinstance(asr_config, dict) and not asr_config.get("fixture_path"):
            audio_for_asr = normalize_audio(audio_path, episode_workdir / "audio.16k.wav")
        raw_adapter = adapter_from_config(provider_config)
        if isinstance(asr_config, dict):
            asr_provider = str(asr_config.get("provider", "fixture"))
            model_name = str(asr_config.get("model_id", asr_config.get("model", asr_provider)))
            dollars_per_hour = asr_dollars_per_hour(asr_provider, asr_config)
        else:
            asr_provider = "fixture"
            model_name = "fixture"
            dollars_per_hour = None
        adapter = CachedASRAdapter(
            raw_adapter,
            JsonDiskCache(episode_workdir / "asr-cache"),
            model_name,
            asr_config if isinstance(asr_config, dict) else {},
            cost_meter=cost_meter,
            cost_provider=model_name,
            dollars_per_hour=dollars_per_hour,
        )
        words = adapter.transcribe(audio_for_asr)
        _write_json(episode_workdir / "asr.json", {"words": [word.model_dump() for word in words]})

    if resume_stage == "verify":
        rebuilt = _load_rebuild_artifact(episode_workdir / "rebuild.json")
        return _run_verify_stage(
            episode_workdir=episode_workdir,
            output_path=output_path,
            audio_path=audio_path,
            audio_for_asr=_resume_audio_for_verify(audio_path, episode_workdir),
            provider_config=provider_config,
            profile=profile,
            source_cues=cues,
            rebuilt=rebuilt,
            words=words,
            alignment=_load_alignment_artifact(episode_workdir / "align.json"),
            flags=_load_report_flags(episode_workdir / "qc_report.json"),
            cost_meter=cost_meter,
            include_dropped_line_flags=False,
        )

    if resume_stage in {"adjudicate", "rebuild"}:
        alignment = _load_alignment_artifact(episode_workdir / "align.json")
    else:
        alignment = align_cues_to_words(cues, words)
        alignment = _alignment_with_adjudication_context(alignment, cues)
        _write_json(episode_workdir / "align.json", alignment.model_dump())

    flags: list[QCFlag] = [*source_order_flags, *detect_source_errors(cues)]
    decisions: list[AdjudicationDecision] = []
    if resume_stage == "rebuild":
        decisions, adjudication_flags = _load_adjudication_artifact(episode_workdir / "adjudicate.json")
        flags.extend(adjudication_flags)
    elif alignment.divergence_spans:
        adjudication_flags: list[QCFlag]
        audio_snippets = (
            {}
            if no_llm
            else _adjudication_audio_snippets(
                audio_for_asr,
                episode_workdir,
                alignment.divergence_spans,
                provider_config,
            )
        )
        cached_adjudication = (
            None
            if no_llm
            else _load_cached_adjudication(
                episode_workdir,
                alignment.divergence_spans,
                provider_config,
                audio_snippets=audio_snippets,
            )
        )
        if cached_adjudication is None:
            llm_adapter = KeepSRTAdapter() if no_llm else llm_adapter_from_config(provider_config, pass_name="adjudication")
            engine = AdjudicationEngine(
                llm_adapter,
                confidence_gate=_adjudication_confidence_gate(provider_config),
                scene_gap_seconds=_adjudication_scene_gap_seconds(provider_config),
                audio_snippets=audio_snippets,
            )
            decisions, adjudication_flags = engine.adjudicate(alignment.divergence_spans)
            if not no_llm:
                _write_cached_adjudication(
                    episode_workdir,
                    alignment.divergence_spans,
                    provider_config,
                    decisions,
                    adjudication_flags,
                    audio_snippets=audio_snippets,
                )
            _record_llm_usage_events(cost_meter, llm_adapter, provider_config, pass_name="adjudication")
        else:
            decisions, adjudication_flags = cached_adjudication
        if no_llm:
            for span in alignment.divergence_spans:
                adjudication_flags.append(
                    QCFlag(
                        kind="divergence_unresolved",
                        cue_ids=span.cue_ids,
                        message="Text divergence found while LLM adjudication is disabled.",
                        old_text=span.srt_text,
                        new_text=span.asr_text,
                        start=span.start,
                        end=span.end,
                    )
                )
        flags.extend(adjudication_flags)
        _write_adjudication_artifact(episode_workdir / "adjudicate.json", decisions, adjudication_flags)
    else:
        _write_adjudication_artifact(episode_workdir / "adjudicate.json", [], [])

    adlib_cue_ids_by_case, adlib_reconciliation_flags = _adlib_cue_ids_by_case(
        cues,
        alignment.divergence_spans,
        decisions,
        alignment.unmatched_cue_ids,
    )
    flags.extend(adlib_reconciliation_flags)
    alignment = _alignment_with_decision_words(
        alignment,
        decisions,
        alignment.divergence_spans,
        adlib_cue_ids_by_case,
    )
    adjudicated_cues, change_flags = apply_adjudication_decisions(
        cues,
        alignment.divergence_spans,
        decisions,
        profile,
        adlib_cue_ids_by_case=adlib_cue_ids_by_case,
    )
    flags.extend(change_flags)
    rebuilt, recue_flags = rebuild_cues(
        adjudicated_cues,
        words,
        alignment,
        profile,
        max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
        max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
    )
    flags.extend(recue_flags)
    rebuilt, overlap_flags = apply_overlap_policy(rebuilt, profile.overlap_policy)
    flags.extend(overlap_flags)
    speaker_mapping_uses_llm = _speaker_mapping_uses_llm(provider_config)
    speaker_mapping_adapter = None if no_llm and speaker_mapping_uses_llm else speaker_mapping_adapter_from_config(provider_config)
    if speaker_mapping_adapter is not None:
        cached_speaker_map = (
            _load_cached_speaker_mapping(episode_workdir, rebuilt, provider_config) if speaker_mapping_uses_llm else None
        )
        if cached_speaker_map is None:
            speaker_map = speaker_mapping_adapter.map_speakers(rebuilt)
            if speaker_mapping_uses_llm:
                _write_cached_speaker_mapping(episode_workdir, rebuilt, provider_config, speaker_map)
            _record_llm_usage_events(cost_meter, speaker_mapping_adapter, provider_config, pass_name="speaker_mapping")
        else:
            speaker_map = cached_speaker_map
        _write_json(episode_workdir / "speaker_map.json", speaker_map)
        rebuilt = _cues_with_speaker_characters(rebuilt, speaker_map)
        flags.extend(speaker_mapping_flags(speaker_map))
    if not no_llm:
        punctuation_adapter = punctuation_adapter_from_config(provider_config)
        if punctuation_adapter is not None:
            punctuation_input = rebuilt
            cached_punctuation = _load_cached_punctuation(
                episode_workdir,
                punctuation_input,
                provider_config,
                max_chars_per_line=profile.max_chars_per_line,
                max_lines_per_cue=profile.max_lines_per_cue,
            )
            if cached_punctuation is None:
                rebuilt, punctuation_flags = apply_punctuation_pass(
                    punctuation_input,
                    punctuation_adapter,
                    scene_gap_seconds=_punctuation_scene_gap_seconds(provider_config),
                    max_chars_per_line=profile.max_chars_per_line,
                    max_lines_per_cue=profile.max_lines_per_cue,
                )
                _write_cached_punctuation(
                    episode_workdir,
                    punctuation_input,
                    provider_config,
                    rebuilt,
                    punctuation_flags,
                    max_chars_per_line=profile.max_chars_per_line,
                    max_lines_per_cue=profile.max_lines_per_cue,
                )
                _record_llm_usage_events(cost_meter, punctuation_adapter, provider_config, pass_name="punctuation")
            else:
                rebuilt, punctuation_flags = cached_punctuation
            flags.extend(punctuation_flags)
    return _run_verify_stage(
        episode_workdir=episode_workdir,
        output_path=output_path,
        audio_path=audio_path,
        audio_for_asr=audio_for_asr,
        provider_config=provider_config,
        profile=profile,
        source_cues=cues,
        rebuilt=rebuilt,
        words=words,
        alignment=alignment,
        flags=flags,
        cost_meter=cost_meter,
        include_dropped_line_flags=True,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_resume_stage(resume: str | None) -> str | None:
    if resume is None:
        return None
    stage = resume.strip().lower()
    allowed = {"ingest", "asr", "align", "adjudicate", "rebuild", "verify"}
    if stage not in allowed:
        raise ValueError(f"Unsupported resume stage: {resume}")
    return stage


def _apply_local_mode(config: dict[str, object], local: bool) -> dict[str, object]:
    if not local:
        return config
    next_config = dict(config)
    existing_asr = next_config.get("asr", {})
    asr_config = dict(existing_asr) if isinstance(existing_asr, dict) else {}
    asr_config["provider"] = "whisperx"
    asr_config.pop("fixture_path", None)
    next_config["asr"] = asr_config
    return next_config


def _should_load_asr_artifact(resume_stage: str | None) -> bool:
    return resume_stage in {"align", "adjudicate", "rebuild", "verify"}


def _should_load_ingest_artifact(resume_stage: str | None) -> bool:
    return resume_stage in {"asr", "align", "adjudicate", "rebuild", "verify"}


def _source_cues_for_run(srt_path: Path, episode_workdir: Path, resume_stage: str | None) -> list[Cue]:
    ingest_path = episode_workdir / "ingest.json"
    if _should_load_ingest_artifact(resume_stage) and ingest_path.exists():
        return _load_ingest_artifact(ingest_path)
    return parse_srt_text(srt_path.read_text(encoding="utf-8-sig"))


def _should_write_ingest_artifacts(
    resume_stage: str | None,
    episode_workdir: Path,
    style_path: Path | None,
    fps: float | None,
) -> bool:
    if not _should_load_ingest_artifact(resume_stage):
        return True
    if style_path is not None or fps is not None:
        return True
    return not (episode_workdir / "ingest.json").exists()


def _load_style_profile_for_resume(path: Path, resume_stage: str | None) -> StyleProfile | None:
    if not _should_load_ingest_artifact(resume_stage):
        return None
    return _load_style_profile_artifact(path)


def _load_ingest_artifact(path: Path) -> list[Cue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Cue.model_validate(item) for item in payload.get("cues", [])]


def _load_asr_artifact(path: Path) -> list[Word]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume without ASR artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Word.model_validate(item) for item in payload.get("words", [])]


def _load_alignment_artifact(path: Path) -> AlignmentResult:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume verify without alignment artifact: {path}")
    return AlignmentResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _load_rebuild_artifact(path: Path) -> list[Cue]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume verify without rebuild artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Cue.model_validate(item) for item in payload.get("cues", [])]


def _load_adjudication_artifact(path: Path) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume rebuild without adjudication artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_flags = payload.get("flags", [])
    return (
        [AdjudicationDecision.model_validate(item) for item in payload.get("decisions", [])],
        [QCFlag.model_validate(item) for item in raw_flags] if isinstance(raw_flags, list) else [],
    )


def _write_adjudication_artifact(path: Path, decisions: list[AdjudicationDecision], flags: list[QCFlag]) -> None:
    _write_json(
        path,
        {
            "decisions": [decision.model_dump() for decision in decisions],
            "flags": [flag.model_dump() for flag in flags],
        },
    )


def _load_cached_adjudication(
    episode_workdir: Path,
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    audio_snippets: dict[str, AudioSnippet] | None = None,
) -> tuple[list[AdjudicationDecision], list[QCFlag]] | None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    payload = cache.read(_adjudication_cache_key(spans, provider_config, audio_snippets=audio_snippets))
    if payload is None:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list) or not isinstance(payload.get("flags"), list):
        raise ValueError("invalid LLM adjudication cache artifact")
    return (
        [AdjudicationDecision.model_validate(item) for item in payload["decisions"]],
        [QCFlag.model_validate(item) for item in payload["flags"]],
    )


def _write_cached_adjudication(
    episode_workdir: Path,
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    decisions: list[AdjudicationDecision],
    flags: list[QCFlag],
    audio_snippets: dict[str, AudioSnippet] | None = None,
) -> None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    cache.write(
        _adjudication_cache_key(spans, provider_config, audio_snippets=audio_snippets),
        {
            "decisions": [decision.model_dump() for decision in decisions],
            "flags": [flag.model_dump() for flag in flags],
        },
    )


def _load_cached_punctuation(
    episode_workdir: Path,
    cues: list[Cue],
    provider_config: dict[str, object],
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
) -> tuple[list[Cue], list[QCFlag]] | None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    payload = cache.read(
        _punctuation_cache_key(
            cues,
            provider_config,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        )
    )
    if payload is None:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("cues"), list) or not isinstance(payload.get("flags"), list):
        raise ValueError("invalid LLM punctuation cache artifact")
    return (
        [Cue.model_validate(item) for item in payload["cues"]],
        [QCFlag.model_validate(item) for item in payload["flags"]],
    )


def _write_cached_punctuation(
    episode_workdir: Path,
    input_cues: list[Cue],
    provider_config: dict[str, object],
    output_cues: list[Cue],
    flags: list[QCFlag],
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
) -> None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    cache.write(
        _punctuation_cache_key(
            input_cues,
            provider_config,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        ),
        {
            "cues": [cue.model_dump() for cue in output_cues],
            "flags": [flag.model_dump() for flag in flags],
        },
    )


def _load_cached_speaker_mapping(
    episode_workdir: Path,
    cues: list[Cue],
    provider_config: dict[str, object],
) -> dict[str, str] | None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    payload = cache.read(_speaker_mapping_cache_key(cues, provider_config))
    if payload is None:
        return None
    mapping = payload.get("mapping") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("invalid LLM speaker mapping cache artifact")
    return {str(speaker_id): str(character) for speaker_id, character in mapping.items()}


def _write_cached_speaker_mapping(
    episode_workdir: Path,
    cues: list[Cue],
    provider_config: dict[str, object],
    mapping: dict[str, str],
) -> None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    cache.write(_speaker_mapping_cache_key(cues, provider_config), {"mapping": dict(mapping)})


def _adjudication_cache_key(
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    audio_snippets: dict[str, AudioSnippet] | None = None,
) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "adjudication")
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    payload = {
        "pass": "adjudication",
        "prompt_version": _ADJUDICATION_PROMPT_VERSION,
        "confidence_gate": _adjudication_confidence_gate(provider_config),
        "scene_gap_seconds": _adjudication_scene_gap_seconds(provider_config),
        "spans": [span.model_dump(mode="json") for span in spans],
        "audio_snippets": _audio_snippet_cache_payload(audio_snippets or {}),
    }
    return CacheKey.from_payload(payload, model=model, params=_llm_cache_params(llm_config))


def _punctuation_cache_key(
    cues: list[Cue],
    provider_config: dict[str, object],
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "punctuation")
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    payload = {
        "pass": "punctuation",
        "scene_gap_seconds": _punctuation_scene_gap_seconds(provider_config),
        "line_constraints": {
            "max_chars_per_line": max_chars_per_line,
            "max_lines_per_cue": max_lines_per_cue,
        },
        "cues": [cue.model_dump(mode="json") for cue in cues],
    }
    return CacheKey.from_payload(payload, model=model, params=_llm_cache_params(llm_config))


def _speaker_mapping_cache_key(cues: list[Cue], provider_config: dict[str, object]) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "speaker_mapping")
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    mapping_config = provider_config.get("speaker_mapping", {}) if isinstance(provider_config, dict) else {}
    payload = {
        "pass": "speaker_mapping",
        "cues": [cue.model_dump(mode="json") for cue in cues],
    }
    params = {**_llm_cache_params(llm_config), "speaker_mapping": mapping_config}
    return CacheKey.from_payload(payload, model=model, params=params)


def _llm_cache_params(llm_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in llm_config.items() if key != "responses"}


def _adjudication_audio_snippets(
    audio_path: Path,
    episode_workdir: Path,
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
) -> dict[str, AudioSnippet]:
    enabled, pad_seconds, max_duration_seconds = _adjudication_audio_snippet_options(provider_config)
    if not enabled:
        return {}
    snippets = extract_audio_snippets(
        audio_path,
        spans,
        episode_workdir / "audio-snippets",
        pad_seconds=pad_seconds,
        max_duration_seconds=max_duration_seconds,
    )
    _write_json(episode_workdir / "audio_snippets.json", {"snippets": [snippet.model_dump() for snippet in snippets]})
    return {snippet.case_id: snippet for snippet in snippets}


def _adjudication_audio_snippet_options(provider_config: dict[str, object]) -> tuple[bool, float, float]:
    llm_config = llm_config_for_pass(provider_config, "adjudication")
    value = llm_config.get("audio_snippet_double_check", False)
    if value in (False, None):
        return (False, 2.0, 20.0)
    if value is True:
        return (True, 2.0, 20.0)
    if not isinstance(value, dict):
        raise ValueError("llm.adjudication.audio_snippet_double_check must be a mapping or boolean")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("llm.adjudication.audio_snippet_double_check.enabled must be boolean")
    pad_seconds = _float_config(
        value,
        "pad_seconds",
        2.0,
        "llm.adjudication.audio_snippet_double_check.pad_seconds",
    )
    max_duration_seconds = _float_config(
        value,
        "max_duration_seconds",
        20.0,
        "llm.adjudication.audio_snippet_double_check.max_duration_seconds",
    )
    if pad_seconds < 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.pad_seconds must be non-negative")
    if max_duration_seconds <= 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.max_duration_seconds must be positive")
    return (enabled, pad_seconds, max_duration_seconds)


def _float_config(source: dict[str, object], key: str, default: float, label: str) -> float:
    value = source.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc


def _timing_float_config(provider_config: dict[str, object], key: str, default: float) -> float:
    timing_config = provider_config.get("timing", {}) if isinstance(provider_config, dict) else {}
    if not isinstance(timing_config, dict):
        return default
    value = _float_config(timing_config, key, default, f"timing.{key}")
    if value <= 0:
        raise ValueError(f"timing.{key} must be positive")
    return value


def _output_no_overlaps(provider_config: dict[str, object]) -> bool:
    output_config = provider_config.get("output", {}) if isinstance(provider_config, dict) else {}
    if not isinstance(output_config, dict):
        return True
    value = output_config.get("no_overlaps", True)
    if not isinstance(value, bool):
        raise ValueError("output.no_overlaps must be boolean")
    return value


def _boundary_refinement_config(provider_config: dict[str, object]) -> BoundaryRefinementConfig:
    vad_config = provider_config.get("vad", {}) if isinstance(provider_config, dict) else {}
    if not isinstance(vad_config, dict):
        return BoundaryRefinementConfig(enabled=False)
    value = vad_config.get("boundary_refinement", False)
    if value in (False, None):
        return BoundaryRefinementConfig(enabled=False)
    if value is True:
        return BoundaryRefinementConfig(
            max_word_duration_ms=int(_timing_float_config(provider_config, "max_word_duration", 2.0) * 1000)
        )
    if not isinstance(value, dict):
        raise ValueError("vad.boundary_refinement must be a mapping or boolean")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("vad.boundary_refinement.enabled must be boolean")
    return BoundaryRefinementConfig(
        enabled=enabled,
        start_pad_ms=_int_config(value, "start_pad_ms", 40, "vad.boundary_refinement.start_pad_ms"),
        end_pad_ms=_int_config(value, "end_pad_ms", 40, "vad.boundary_refinement.end_pad_ms"),
        max_end_extension_ms=_int_config(
            value,
            "max_end_extension_ms",
            300,
            "vad.boundary_refinement.max_end_extension_ms",
        ),
        max_leading_silence_ms=_int_config(
            value,
            "max_leading_silence_ms",
            150,
            "vad.boundary_refinement.max_leading_silence_ms",
        ),
        max_trailing_silence_ms=_int_config(
            value,
            "max_trailing_silence_ms",
            300,
            "vad.boundary_refinement.max_trailing_silence_ms",
        ),
        max_word_duration_ms=int(
            _timing_float_config(provider_config, "max_word_duration", 2.0) * 1000
        ),
    )


def _int_config(source: dict[str, object], key: str, default: int, label: str) -> int:
    value = source.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _audio_snippet_cache_payload(audio_snippets: dict[str, AudioSnippet]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for case_id, snippet in sorted(audio_snippets.items()):
        path = Path(snippet.path)
        payload.append(
            {
                "case_id": case_id,
                "mime_type": snippet.mime_type,
                "start": snippet.start,
                "end": snippet.end,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            }
        )
    return payload


def _load_style_profile_artifact(path: Path) -> StyleProfile | None:
    if not path.exists():
        return None
    return StyleProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _load_report_flags(path: Path) -> list[QCFlag]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_flags = payload.get("flags", [])
    if not isinstance(raw_flags, list):
        return []
    return [QCFlag.model_validate(item) for item in raw_flags]


def _resume_audio_for_verify(audio_path: Path, episode_workdir: Path) -> Path:
    normalized_audio = episode_workdir / "audio.16k.wav"
    return normalized_audio if normalized_audio.exists() else audio_path


def _run_verify_stage(
    *,
    episode_workdir: Path,
    output_path: Path,
    audio_path: Path,
    audio_for_asr: Path,
    provider_config: dict[str, object],
    profile: StyleProfile,
    source_cues: list[Cue],
    rebuilt: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    flags: list[QCFlag],
    cost_meter: CostMeter,
    include_dropped_line_flags: bool,
) -> PipelineResult:
    flags = _without_stale_verify_flags(flags)
    forced_alignments: list[ForcedAlignmentCue] = []
    effective_words = words
    forced_alignment_adapter = forced_alignment_adapter_from_config(provider_config)
    if forced_alignment_adapter is not None:
        forced_alignments = forced_alignment_adapter.align(audio_for_asr, rebuilt)
        _write_json(episode_workdir / "forced_align.json", {"cues": [alignment.model_dump() for alignment in forced_alignments]})
        rebuilt, forced_alignment_flags = apply_forced_alignment(rebuilt, forced_alignments, profile)
        flags.extend(forced_alignment_flags)
    overlap_detection_adapter = overlap_detection_adapter_from_config(provider_config)
    if overlap_detection_adapter is not None:
        overlap_regions = overlap_detection_adapter.detect(audio_for_asr)
        _write_json(episode_workdir / "overlap.json", {"regions": [region.model_dump() for region in overlap_regions]})
        flags.extend(overlap_flags_for_regions(rebuilt, overlap_regions))
    speech_activity_adapter = speech_activity_adapter_from_config(provider_config)
    if speech_activity_adapter is not None:
        speech_regions = speech_activity_adapter.detect(audio_for_asr)
        _write_json(episode_workdir / "vad.json", {"regions": [region.model_dump() for region in speech_regions]})
        effective_words, word_clamp_flags = clamp_asr_word_durations(
            words,
            speech_regions,
            max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
        )
        flags.extend(word_clamp_flags)
        min_coverage = min_coverage_from_config(provider_config)
        rebuilt, timing_flags = refine_cues_to_speech_activity(
            rebuilt,
            speech_regions,
            profile,
            _boundary_refinement_config(provider_config),
            words=effective_words,
            alignment=alignment,
        )
        flags.extend(timing_flags)
        if include_dropped_line_flags:
            flags.extend(
                dropped_line_flags_for_unmatched_cues(
                    source_cues,
                    alignment.unmatched_cue_ids,
                    speech_regions,
                    min_coverage,
                )
            )
        flags.extend(speech_activity_flags_for_cues(rebuilt, speech_regions, min_coverage))
    rebuilt, final_order_flags = finalize_cues_for_output(
        rebuilt,
        profile,
        no_overlaps=_output_no_overlaps(provider_config),
        max_cps=_timing_float_config(provider_config, "max_cps", 30.0),
    )
    flags.extend(final_order_flags)
    style_issues = lint_cues(rebuilt, profile)
    flags.extend(
        cps_sanity_flags(
            rebuilt,
            max_cps=_timing_float_config(provider_config, "max_cps", 30.0),
            min_cps=_timing_float_config(provider_config, "min_cps", 2.0),
        )
    )
    cue_scores = score_cues(rebuilt, effective_words, alignment, forced_alignments)
    if audio_for_asr != audio_path:
        flags.extend(silence_flags_for_cues(audio_for_asr, rebuilt))

    flags = _unique_flags(flags)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(write_srt(rebuilt, renumber=True), encoding="utf-8")
    _write_json(episode_workdir / "rebuild.json", {"cues": [cue.model_dump() for cue in rebuilt]})

    report = write_qc_report(
        episode_workdir / "qc_report.json",
        episode_workdir / "qc_report.html",
        rebuilt,
        flags,
        style_issues,
        cue_scores,
    )
    _write_json(
        episode_workdir / "verify.json",
        {
            "stage": "verify",
            "summary": report["summary"],
            "cue_scores": report["cue_scores"],
            "flags": report["flags"],
            "style_issues": report["style_issues"],
        },
    )
    write_changes_diff(episode_workdir / "changes.diff.srt", flags)
    _write_json(episode_workdir / "cost.json", cost_meter.as_dict())

    return PipelineResult(output_path, episode_workdir, cost_meter, report)


def _without_stale_verify_flags(flags: list[QCFlag]) -> list[QCFlag]:
    return [flag for flag in flags if flag.kind not in VERIFY_STAGE_FLAG_KINDS]


def _unique_flags(flags: list[QCFlag]) -> list[QCFlag]:
    seen: set[str] = set()
    unique: list[QCFlag] = []
    for flag in flags:
        key = json.dumps(flag.model_dump(), sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)
    return unique


def _speaker_mapping_uses_llm(provider_config: dict[str, object]) -> bool:
    mapping_config = provider_config.get("speaker_mapping", {}) if isinstance(provider_config, dict) else {}
    return isinstance(mapping_config, dict) and str(mapping_config.get("provider", "")).lower() == "llm"


def _record_llm_usage_events(
    cost_meter: CostMeter,
    adapter: object,
    provider_config: dict[str, object],
    pass_name: str | None = None,
) -> None:
    llm_config = llm_config_for_pass(provider_config, pass_name)
    if not isinstance(llm_config, dict):
        return
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    for event in drain_usage_events(adapter):
        record_llm_usage(cost_meter, provider, model, llm_config, event)


def _adjudication_scene_gap_seconds(provider_config: dict[str, object]) -> float:
    return _llm_scene_gap_seconds(provider_config, "adjudication")


def _adjudication_confidence_gate(provider_config: dict[str, object]) -> float:
    llm_config = llm_config_for_pass(provider_config, "adjudication")
    value = llm_config.get("confidence_gate", 0.7)
    try:
        confidence_gate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("llm.adjudication.confidence_gate must be numeric") from exc
    if not 0 <= confidence_gate <= 1:
        raise ValueError("llm.adjudication.confidence_gate must be between 0 and 1")
    return confidence_gate


def _punctuation_scene_gap_seconds(provider_config: dict[str, object]) -> float:
    return _llm_scene_gap_seconds(provider_config, "punctuation")


def _llm_scene_gap_seconds(provider_config: dict[str, object], pass_name: str) -> float:
    llm_config = llm_config_for_pass(provider_config, pass_name)
    value = llm_config.get("scene_gap_seconds", 4.0)
    try:
        scene_gap_seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm.{pass_name}.scene_gap_seconds must be numeric") from exc
    if scene_gap_seconds < 0:
        raise ValueError(f"llm.{pass_name}.scene_gap_seconds must be non-negative")
    return scene_gap_seconds


def _default_llm_model(provider: str) -> str:
    if provider == "gemini":
        return "gemini-3.5-flash"
    if provider == "openai":
        return "gpt-5.5"
    if provider == "anthropic":
        return "claude-sonnet-5"
    return provider


def _alignment_with_adjudication_context(
    alignment: AlignmentResult,
    cues: list[Cue],
    radius: int = 2,
) -> AlignmentResult:
    if not alignment.divergence_spans:
        return alignment
    return alignment.model_copy(
        update={
            "divergence_spans": [
                _span_with_adjudication_context(span, cues, radius)
                for span in alignment.divergence_spans
            ]
        }
    )


def _span_with_adjudication_context(span: DivergenceSpan, cues: list[Cue], radius: int) -> DivergenceSpan:
    positions = {cue.index: position for position, cue in enumerate(cues)}
    span_positions = [positions[cue_id] for cue_id in span.cue_ids if cue_id in positions]
    if not span_positions:
        return span

    first = min(span_positions)
    last = max(span_positions)
    before = cues[max(0, first - radius) : first]
    after = cues[last + 1 : last + 1 + radius]
    return span.model_copy(
        update={
            "context_before": [_cue_context(cue) for cue in before],
            "context_after": [_cue_context(cue) for cue in after],
        }
    )


def _cue_context(cue: Cue) -> CueContext:
    return CueContext(
        cue_id=cue.index,
        text=cue.plain_text,
        start=cue.start_ms / 1000.0,
        end=cue.end_ms / 1000.0,
    )


def _cues_with_speaker_characters(cues: list[Cue], speaker_map: dict[str, str]) -> list[Cue]:
    return [
        cue.model_copy(update={"character": speaker_map.get(cue.speaker_id)})
        if cue.speaker_id in speaker_map
        else cue
        for cue in cues
    ]


def _adlib_cue_ids_by_case(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    unmatched_cue_ids: list[int],
) -> tuple[dict[str, int], list[QCFlag]]:
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    next_index = max((cue.index for cue in cues), default=0) + 1
    cue_ids: dict[str, int] = {}
    flags: list[QCFlag] = []
    unmatched = {cue_id for cue_id in unmatched_cue_ids}
    used_reconciled: set[int] = set()
    for span in spans:
        decision = decisions_by_case.get(span.case_id)
        if decision is None or decision.verdict == "keep_srt":
            continue
        if span.cue_ids or not decision.final_text.strip():
            continue
        reconciled = _reconciled_adlib_source_cue(
            cues,
            unmatched - used_reconciled,
            span,
            decision.final_text,
        )
        if reconciled is not None:
            cue_ids[span.case_id] = reconciled.index
            used_reconciled.add(reconciled.index)
            flags.append(
                QCFlag(
                    kind="adlib_reconciled",
                    cue_ids=[reconciled.index],
                    message="Ad-lib insertion matched a nearby unmatched source cue and reused that cue instead of creating a duplicate.",
                    old_text=reconciled.text,
                    new_text=decision.final_text,
                    start=span.start,
                    end=span.end,
                )
            )
            continue
        cue_ids[span.case_id] = next_index
        next_index += 1
    return cue_ids, flags


def _reconciled_adlib_source_cue(
    cues: list[Cue],
    candidate_ids: set[int],
    span: DivergenceSpan,
    final_text: str,
) -> Cue | None:
    candidates = [cue for cue in cues if cue.index in candidate_ids]
    if not candidates:
        return None
    final_signature = " ".join(alphanumeric_signature(final_text))
    scored: list[tuple[float, Cue]] = []
    for cue in candidates:
        timing_match = _span_overlaps_cue_with_pad(span, cue, pad_seconds=3.0)
        cue_signature = " ".join(alphanumeric_signature(cue.plain_text))
        similarity = fuzz.ratio(final_signature, cue_signature) / 100.0 if final_signature and cue_signature else 0.0
        if not timing_match or similarity < 0.8:
            continue
        scored.append((similarity, cue))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], -item[1].start_ms))[1]


def _span_overlaps_cue_with_pad(span: DivergenceSpan, cue: Cue, pad_seconds: float) -> bool:
    if span.start is None and span.end is None:
        return False
    span_start = (span.start if span.start is not None else span.end or 0.0) - pad_seconds
    span_end = (span.end if span.end is not None else span.start or 0.0) + pad_seconds
    return span_end >= cue.start_ms / 1000.0 and span_start <= cue.end_ms / 1000.0


def _alignment_with_decision_words(alignment, decisions, spans, adlib_cue_ids_by_case=None):
    timed_decisions = {
        decision.case_id: decision
        for decision in decisions
        if decision.verdict in {"keep_srt", "use_audio", "hybrid"}
    }
    if not timed_decisions:
        return alignment

    adlib_cue_ids_by_case = adlib_cue_ids_by_case or {}
    cue_word_indices = {cue_id: list(indices) for cue_id, indices in alignment.cue_word_indices.items()}
    for span in spans:
        if span.case_id not in timed_decisions:
            continue
        adlib_cue_id = adlib_cue_ids_by_case.get(span.case_id)
        if adlib_cue_id is not None:
            cue_word_indices[adlib_cue_id] = sorted(set(span.asr_word_indices))
            continue
        for cue_id, spoken_indices in _span_word_indices_by_cue(span).items():
            combined = sorted(set(cue_word_indices.get(cue_id, []) + spoken_indices))
            if combined:
                cue_word_indices[cue_id] = combined
    return alignment.model_copy(update={"cue_word_indices": cue_word_indices})


def _span_word_indices_by_cue(span: DivergenceSpan) -> dict[int, list[int]]:
    cue_ids = list(dict.fromkeys(span.cue_ids))
    if not cue_ids:
        return {}
    partitions = _partition_contiguous(span.asr_word_indices, len(cue_ids))
    return {
        cue_id: partition
        for cue_id, partition in zip(cue_ids, partitions, strict=False)
    }


def _partition_contiguous(indices: list[int], parts: int) -> list[list[int]]:
    if parts <= 0:
        return []
    base_size, remainder = divmod(len(indices), parts)
    partitions: list[list[int]] = []
    offset = 0
    for part_index in range(parts):
        size = base_size + (1 if part_index < remainder else 0)
        partitions.append(list(indices[offset : offset + size]))
        offset += size
    return partitions
