from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import re
from math import ceil, isfinite
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .adjudication import AdjudicationEngine, KeepSRTAdapter, confidence_gated_decision
from .adjudication_regions import (
    PROTECTED_SOURCE_PREFIX, SPEECH_REPEAT_PREFIX, is_joint_region,
    is_song_caption_cue, protect_song_captions,
    validated_protected_source_regions,
)
from .adjudication_snippets import BoundedAudioSnippetBatchSource
from .aligner import MISSING_AUDIO_GUARD_VERSION, _words_touch, align_cues_to_words
from .audio import AudioNormalizationLimits, normalize_audio
from .audio_snippets import DEFAULT_MAX_COVERING_SNIPPET_SECONDS, extract_audio_snippets
from .cache import CacheKey, JsonDiskCache, _sha256_file, write_json_atomic, write_text_atomic
from .changes import (
    ReplacementOwnershipError, anchor_confidence_is_acceptable, apply_adjudication_decisions,
    indexed_multi_cue_replacements, lexical_edit_costs, whole_cue_replacement_plan,
    protected_replacement_targets, single_token_prefix_replacement_targets,
)
from .config import load_style_profile, load_yaml
from .cost import CostMeter, asr_dollars_per_hour, audio_seconds, record_gemini_context_cost, record_llm_usage
from .cue_segmentation import (
    segment_generated_adlib_cues, settle_collapsed_generated_adlibs,
    split_overlong_existing_cues, split_speaker_turn_cues,
)
from .detached_speech import separate_detached_speech
from .edit_consistency import (
    held_decisions as decisions_with_held_cases, hold_fragmenting_replacements,
    settle_edits_with_held_timing, settle_one_letter_residues,
)
from .editorial_guard import episode_editorial_addition_flags
from .forced_alignment import apply_forced_alignment, forced_alignment_adapter_from_config, usable_forced_alignments_by_cue
from .gemini_audio_context import validate_audio_context_config
from .hybrid_adjudication import HYBRID_POLICY_VERSION
from .llm_providers import (
    _ADJUDICATION_PROMPT_VERSION,
    _ADJUDICATION_REVIEW_PROMPT_VERSION,
    _PUNCTUATION_PROMPT_VERSION,
    _SPEAKER_MAPPING_PROMPT_VERSION,
    drain_usage_events,
    adjudication_fallback_config,
    llm_adapter_from_config,
    llm_config_for_pass,
    punctuation_adapter_from_config,
)
from .models import AdjudicationDecision, AlignmentResult, AudioSnippet, Cue, CueContext, DivergenceSpan, ForcedAlignmentCue, QCFlag, Word
from .observability import name_spelling_inconsistency_flags, span_coverage_flags
from .output_order import finalize_cues_for_output, source_order_inversion_flags
from .overlap import apply_overlap_policy, reconcile_overlap_flags
from .overlap_detection import overlap_detection_adapter_from_config, overlap_flags_for_regions
from .providers import (
    CachedASRAdapter,
    adapter_from_config,
    apply_asr_language,
    asr_language_code,
    apply_local_asr_config,
    apply_transcription_provider_config,
    repair_word_stream,
)
from .profanity import apply_german_profanity_censorship, censor_german_profanity_flags
from .punctuation import apply_punctuation_pass
from .recue import cue_spoken_spans, preserve_source_timings, rebuild_cues, shared_word_cue_ids, shared_word_timing_flags
from .reports import write_change_log, write_qc_report
from .srt_io import parse_srt_text, write_srt
from .silence import silence_flags_for_cues
from .source_quality import detect_source_errors
from .source_order import sort_cues_chronologically
from .speaker_evidence import has_known_different_speakers, speakers_known_different
from .speaker_mapping import speaker_mapping_adapter_from_config, speaker_mapping_flags
from .style_profile import FPSDetection, StyleProfile, derive_style_profile, detect_fps_with_confidence
from .subtitle_annotations import cue_has_bracketed_screen_text, cue_has_spoken_text, speech_text_for_alignment
from .timing_refinement import (
    BoundaryRefinementConfig, SpeechEvidence, boundary_refinement_config_from_config,
    min_duration_policy_from_config, refine_cues_to_speech_activity, speech_evidence_for_words,
)
from .tokenize import alphanumeric_signature, normalize_token
from .vad import (
    cue_ids_with_audible_words,
    dropped_line_flags_for_unmatched_cues,
    min_coverage_from_config,
    speech_activity_adapter_from_config,
    speech_activity_flags_for_cues,
    trailing_silence_flags_for_cues,
)
from .verify import cps_sanity_flags, lint_cues, score_cues

VERIFY_STAGE_FLAG_KINDS = frozenset(
    {
        "asr_word_clamped",
        "cps_cue_merged",
        "cps_duration_extended",
        "cue_on_silence",
        "cue_without_speech_activity",
        "cue_with_excessive_trailing_silence",
        "adlib_removed_without_speech_activity",
        "duplicate_cue_merged",
        "forced_alignment_refined",
        "forced_alignment_unresolved",
        "forced_alignment_unavailable",
        "shared_word_timing_preserved",
        "output_overlap_preserved",
        "impossible_cps_fast",
        "impossible_cps_slow",
        "invalid_cue_duration",
        "output_overlap_resolved",
        "output_overlap_unresolved",
        "media_boundary_clamped",
        "cue_outside_media",
        "min_duration_unattainable",
        "overlap_detected",
        "speaker_transition_gap_inserted",
        "timing_refined",
        "timing_refinement_held",
        "vad_provider_fallback",
    }
)

_TRANSIENT_ADJUDICATION_FLAG_KINDS = frozenset(
    {
        "audio_snippet_unavailable",
        "adjudication_audio_unavailable",
        "adjudication_review_unavailable",
        "invalid_llm_response",
        "llm_provider_unavailable",
    }
)

_TRANSIENT_PUNCTUATION_FLAG_KINDS = frozenset({"punctuation_provider_unavailable"})

_REBUILD_POLICY_VERSION = 10
_ADJUDICATION_POLICY_VERSION = 3
_PUNCTUATION_POLICY_VERSION = 1


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
    transcription_provider: str = "default",
    allow_gemini_transcribe_web: bool = False,
    audio_limits: AudioNormalizationLimits | None = None,
    style_profile: StyleProfile | None = None,
) -> PipelineResult:
    resume_stage = _normalize_resume_stage(resume)
    episode_workdir = workdir / srt_path.stem
    episode_workdir.mkdir(parents=True, exist_ok=True)
    provenance_flags = (
        _validate_resume_audio_provenance(audio_path, episode_workdir)
        if _should_load_asr_artifact(resume_stage)
        else []
    )
    if resume_stage == "verify":
        _validate_rebuild_policy(episode_workdir / "rebuild.json")
    cost_meter = CostMeter()
    cues, ingest_metadata = _source_cues_for_run(srt_path, episode_workdir, resume_stage)
    cues, source_order_flags = sort_cues_chronologically(cues)
    source_order_flags = [
        *(QCFlag.model_validate(item) for item in ingest_metadata.get("ingest_flags", [])),
        *source_order_flags,
    ]
    fps_detection = detect_fps_with_confidence(cues)
    fps_override_flags = _fps_override_mismatch_flags(cues, fps, fps_detection)
    explicit_style_override = style_profile is not None or style_path is not None
    fps_detection_flags = (
        []
        if style_path is not None
        else _fps_detection_flags(cues, fps, fps_detection)
    )
    style_artifact_path = episode_workdir / "style_profile.json"
    source_profile = derive_style_profile(cues)
    profile = (
        style_profile.model_copy(deep=True)
        if style_profile is not None
        else load_style_profile(style_path)
        or _load_style_profile_for_resume(style_artifact_path, resume_stage)
        or source_profile
    )
    if fps is not None:
        profile = profile.model_copy(update={"fps": fps})
    fps_summary_metadata = _fps_summary_metadata(
        profile,
        fps_detection,
        explicitly_configured=fps is not None or style_path is not None,
    )
    if _should_write_ingest_artifacts(resume_stage, episode_workdir, explicit_style_override, fps):
        _write_json(episode_workdir / "ingest.json", {"cues": [cue.model_dump() for cue in cues], **ingest_metadata})
        _write_json(style_artifact_path, profile.model_dump())

    provider_config = apply_transcription_provider_config(
        _apply_local_mode(load_yaml(providers_path), local),
        transcription_provider,
    )
    provider_config = apply_asr_language(provider_config, language)
    if local:
        no_llm = True
    audio_for_asr = audio_path
    if _should_load_asr_artifact(resume_stage):
        asr_artifact_path = episode_workdir / "asr.json"
        words, asr_repair_flags = _load_asr_artifact_with_repair(asr_artifact_path)
        asr_repair_flags.extend(provenance_flags)
        audio_for_asr = _resume_audio_for_verify(audio_path, episode_workdir)
    else:
        asr_config = provider_config.get("asr", {}) if isinstance(provider_config, dict) else {}
        source_audio_sha256 = None
        if isinstance(asr_config, dict) and not asr_config.get("fixture_path"):
            source_audio_sha256 = _sha256_file(audio_path)
            audio_for_asr = normalize_audio(
                audio_path,
                episode_workdir / "audio.16k.wav",
                limits=audio_limits,
            )
        raw_adapter = adapter_from_config(
            provider_config,
            local_mode=local,
            allow_gemini_transcribe_web=allow_gemini_transcribe_web,
        )
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
        try:
            words = adapter.transcribe(audio_for_asr)
        except Exception:
            _write_json(episode_workdir / "asr_failure.json", {
                "provider": asr_provider, "model": model_name,
                "usage": adapter.last_usage, "cost": cost_meter.as_dict(),
            })
            if not (episode_workdir / "cost.json").exists():
                write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())
            raise
        asr_repair_flags = list(adapter.last_repair_flags)
        asr_metadata: dict[str, object] = {
            "provider": asr_provider,
            "model": model_name,
            "usage": adapter.last_usage,
            "cache_hit": adapter.last_cache_hit,
            "audio_provenance": {
                "source_sha256": source_audio_sha256 or adapter.last_cache_key.audio_sha256,
                "asr_input_sha256": adapter.last_cache_key.audio_sha256,
                "normalized": audio_for_asr != audio_path,
            },
            "repair_flags": [flag.model_dump() for flag in asr_repair_flags],
        }
        # Evidence that is not part of a word (Scribe logprob and audio events,
        # MAI chunk languages and speaker links) stays with the transcript.
        provider_evidence = getattr(adapter, "last_evidence", None)
        if provider_evidence is not None:
            asr_metadata["provider_evidence"] = provider_evidence
        _write_json(
            episode_workdir / "asr.json",
            {
                "words": [word.model_dump() for word in words],
                "repair_flags": [flag.model_dump() for flag in asr_repair_flags],
                "metadata": asr_metadata,
            },
        )
    long_audio_llm_flag = None if no_llm else _long_audio_llm_skip_flag(audio_for_asr, provider_config)
    llm_disabled_for_episode = no_llm or long_audio_llm_flag is not None

    if resume_stage == "verify":
        resume_alignment = _load_alignment_artifact(episode_workdir / "align.json")
        _validate_alignment_screen_text_provenance(resume_alignment, cues)
        protected_source_regions = _protected_regions_for_alignment(resume_alignment, cues, words)
        resume_decisions = _load_adjudication_artifact(
            episode_workdir / "adjudicate.json"
        )[0]
        selected, confidence_flags = _confidence_gate_decisions(
            resume_alignment.divergence_spans, resume_decisions, provider_config,
            _load_report_flags(episode_workdir / "qc_report.json"),
        )
        if selected != resume_decisions or confidence_flags:
            raise ValueError(
                "Cannot resume verify with decisions below the current confidence gate; "
                "resume from rebuild to preserve uncertain source text and timing."
            )
        rebuilt = _load_rebuild_artifact(episode_workdir / "rebuild.json")
        unsafe_cases = _unsafe_incomplete_source_resume_case_ids(
            resume_alignment.divergence_spans,
            provider_config,
            resume_decisions,
            rebuilt,
            cues,
            alignment_unresolved=resume_alignment.diagnostics.unresolved,
            missing_audio_cue_ids=set(resume_alignment.diagnostics.missing_audio_cue_ids),
            protected_source_regions=protected_source_regions,
        )
        if unsafe_cases:
            raise RuntimeError(
                "Cannot resume verify from stale incomplete-source adjudication cases "
                f"({', '.join(unsafe_cases)}); resume from rebuild so source-authority "
                "safety checks can be applied."
            )
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
            alignment=resume_alignment,
            flags=[
                *_load_report_flags(episode_workdir / "qc_report.json"),
                *source_order_flags,
                *fps_override_flags,
                *fps_detection_flags,
                *asr_repair_flags,
                *detect_source_errors(cues),
            ],
            cost_meter=cost_meter,
            include_dropped_line_flags=False,
            decisions=resume_decisions,
            fps_summary_metadata=fps_summary_metadata,
        )

    if resume_stage in {"adjudicate", "rebuild"}:
        alignment = _load_alignment_artifact(episode_workdir / "align.json")
        _validate_alignment_screen_text_provenance(alignment, cues)
    else:
        episode_language = _episode_language_code(provider_config, language)
        # A known language decides which article-like words are numbers.
        alignment = (
            align_cues_to_words(cues, words, language=episode_language) if episode_language
            else align_cues_to_words(cues, words)
        )
        alignment = _alignment_with_adjudication_context(alignment, cues)
        alignment = _alignment_with_song_caption_guard(alignment, cues, words)
        _write_json(episode_workdir / "align.json", alignment.model_dump())

    protected_source_regions = _protected_regions_for_alignment(alignment, cues, words)
    flags: list[QCFlag] = [
        *source_order_flags,
        *fps_override_flags,
        *fps_detection_flags,
        *asr_repair_flags,
        *alignment.flags,
        *detect_source_errors(cues),
    ]
    if long_audio_llm_flag is not None:
        flags.append(long_audio_llm_flag)
    decisions: list[AdjudicationDecision] = []
    if resume_stage == "rebuild":
        decisions, adjudication_flags = _load_adjudication_artifact(episode_workdir / "adjudicate.json")
        decisions, adjudication_flags = _apply_incomplete_source_holds_to_decisions(
            alignment.divergence_spans,
            provider_config,
            decisions,
            adjudication_flags,
            source_cue_count=_spoken_source_cue_count(cues),
            alignment_unresolved=alignment.diagnostics.unresolved,
            missing_audio_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
            protected_source_regions=protected_source_regions,
            song_caption_cue_ids=_song_caption_cue_ids(cues),
        )
        flags.extend(adjudication_flags)
        _write_adjudication_artifact(episode_workdir / "adjudicate.json", decisions, adjudication_flags)
    elif alignment.divergence_spans:
        provider_spans, held_decisions, incomplete_source_flags = (
            _hold_incomplete_source_insertions(
                alignment.divergence_spans,
                provider_config,
                source_cue_count=_spoken_source_cue_count(cues),
                alignment_unresolved=alignment.diagnostics.unresolved,
                missing_audio_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
                protected_source_regions=protected_source_regions,
                song_caption_cue_ids=_song_caption_cue_ids(cues),
            )
        )
        provider_decisions: list[AdjudicationDecision] = []
        provider_flags: list[QCFlag] = []
        if provider_spans:
            audio_snippet_source = (
                None
                if llm_disabled_for_episode
                else _adjudication_audio_snippet_source(
                    audio_for_asr,
                    episode_workdir,
                    provider_config,
                )
            )
            audio_snippet_context = (
                audio_snippet_source.cache_context()
                if audio_snippet_source is not None
                else None
            )
            if not llm_disabled_for_episode:
                audio_snippet_context = _adjudication_audio_cache_context(
                    audio_path, audio_for_asr, provider_config, audio_snippet_context,
                )
            cached_adjudication = (
                None
                if llm_disabled_for_episode
                else _load_cached_adjudication(
                    episode_workdir,
                    provider_spans,
                    provider_config,
                    audio_snippet_context=audio_snippet_context,
                    source_cues=cues,
                    source_words=words,
                )
            )
            if cached_adjudication is None:
                llm_adapter = (
                    KeepSRTAdapter()
                    if llm_disabled_for_episode
                    else llm_adapter_from_config(provider_config, pass_name="adjudication")
                )
                _set_adapter_episode_context(llm_adapter, cues, words=words)
                engine = AdjudicationEngine(
                    llm_adapter,
                    confidence_gate=_adjudication_confidence_gate(provider_config),
                    scene_gap_seconds=_adjudication_scene_gap_seconds(provider_config),
                    audio_snippet_batches=(
                        audio_snippet_source.load
                        if audio_snippet_source is not None
                        else None
                    ),
                    require_audio_snippets=(
                        audio_snippet_source is not None
                        and _episode_audio_options(provider_config) is None
                    ),
                    required_audio_case_ids={
                        SPEECH_REPEAT_PREFIX + case_id.removeprefix(PROTECTED_SOURCE_PREFIX)
                        for case_id in protected_source_regions
                    },
                    max_batch_spans=llm_config_for_pass(provider_config, "adjudication").get("max_batch_spans", 25),
                    max_concurrent_batches=llm_config_for_pass(provider_config, "adjudication").get("max_concurrent_batches", 1),
                    retry_timed_out_batches=llm_config_for_pass(provider_config, "adjudication").get("retry_timed_out_batches", False),
                )
                with _adjudication_audio_session(
                    llm_adapter, audio_path, audio_for_asr,
                    {} if llm_disabled_for_episode else provider_config,
                    episode_workdir, cost_meter, provider_flags,
                ):
                    provider_decisions, engine_flags = engine.adjudicate(provider_spans)
                snippet_flags = (
                    audio_snippet_source.flags()
                    if audio_snippet_source is not None
                    else []
                )
                provider_flags.extend([*snippet_flags, *engine_flags])
                if audio_snippet_source is not None:
                    _write_json(
                        episode_workdir / "audio_snippets.json",
                        audio_snippet_source.manifest(),
                    )
                if not llm_disabled_for_episode:
                    _write_cached_adjudication(
                        episode_workdir,
                        provider_spans,
                        provider_config,
                        provider_decisions,
                        provider_flags,
                        audio_snippet_context=audio_snippet_context,
                        source_cues=cues,
                        source_words=words,
                    )
                provider_flags.extend(
                    _record_llm_usage_events(
                        cost_meter,
                        llm_adapter,
                        provider_config,
                        pass_name="adjudication",
                    )
                )
            else:
                provider_decisions, provider_flags = cached_adjudication
        else:
            _write_json(episode_workdir / "audio_snippets.json", {"snippets": []})
        decisions_by_case = {
            decision.case_id: decision
            for decision in [*held_decisions, *provider_decisions]
        }
        decisions = [
            decisions_by_case[span.case_id]
            for span in alignment.divergence_spans
            if span.case_id in decisions_by_case
        ]
        adjudication_flags = [*incomplete_source_flags, *provider_flags]
        if llm_disabled_for_episode:
            for span in provider_spans:
                if _is_punctuation_only_span(span):
                    # Identical words in another tokenisation are not a divergence.
                    continue
                adjudication_flags.append(
                    QCFlag(
                        kind="divergence_unresolved",
                        cue_ids=span.cue_ids,
                        message=(
                            "Text divergence found while LLM adjudication is disabled "
                            "for this episode."
                        ),
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

    selected_decisions, confidence_flags = _confidence_gate_decisions(
        alignment.divergence_spans, decisions, provider_config, flags,
    )
    if selected_decisions != decisions or confidence_flags:
        decisions = selected_decisions
        flags.extend(confidence_flags)
        _write_adjudication_artifact(
            episode_workdir / "adjudicate.json", decisions,
            _unique_flags([*_load_adjudication_artifact(episode_workdir / "adjudicate.json")[1], *confidence_flags]),
        )
    confidence_held_cue_ids = _confidence_held_source_cue_ids(flags)
    source_timing_held_cue_ids = _source_timing_held_cue_ids(flags)

    alignment, decisions = _absorb_redecoded_insertions(alignment, decisions, words)

    # ASR word edges are repaired against the audio once, before any cue is
    # timed; rebuild and verification share the same repaired words.
    speech_evidence = speech_evidence_for_words(
        speech_activity_adapter_from_config(provider_config), words, audio_for_asr, provider_config,
        max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
        asr_artifact_path=episode_workdir / "asr.json",
    )
    timing_words = speech_evidence.words

    # One case can hold words that are seconds apart. Only the group spoken at
    # a cue's own time may edit that cue; the others are placed on their own.
    alignment, decisions, detached_speech_flags = separate_detached_speech(
        cues, alignment, decisions, words,
        max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
        protected_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
        speech_regions=speech_evidence.regions if speech_evidence.detected else None,
    )
    flags.extend(detached_speech_flags)

    adlib_cue_ids_by_case, adlib_reconciliation_flags = _adlib_cue_ids_by_case(
        cues,
        alignment.divergence_spans,
        decisions,
        alignment.unmatched_cue_ids,
    )
    flags.extend(adlib_reconciliation_flags)
    alignment, flags = _release_reconciled_cues(alignment, flags)
    adlib_cue_ids_by_case, inline_speaker_flags = _validate_inline_adlib_ownership(
        cues, words, alignment, decisions, adlib_cue_ids_by_case, profile,
        protected_cue_ids=confidence_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids),
        max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
    )
    flags.extend(inline_speaker_flags)
    source_cue_ids = {cue.index for cue in cues}
    generated_adlib_cue_ids = {
        cue_id
        for cue_id in adlib_cue_ids_by_case.values()
        if cue_id not in source_cue_ids
    }
    def apply_text(text_decisions: list[AdjudicationDecision]) -> tuple[list[Cue], list[QCFlag]]:
        return apply_adjudication_decisions(
            cues,
            alignment.divergence_spans,
            text_decisions,
            profile,
            adlib_cue_ids_by_case=adlib_cue_ids_by_case,
            protected_cue_ids=confidence_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids),
            words=words,
            max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
            token_matches=alignment.token_matches,
        )

    # Text and word ownership are applied by two passes that must agree. A
    # replacement that only one of them can apply is held as a whole.
    decisions, fragment_hold_flags = hold_fragmenting_replacements(
        cues, alignment.divergence_spans, decisions, apply_text,
    )
    flags.extend(fragment_hold_flags)
    mapping_held_case_ids: set[str] = set()
    alignment = _alignment_with_decision_words(
        alignment,
        decisions,
        alignment.divergence_spans,
        adlib_cue_ids_by_case,
        source_cues=cues,
        words=words,
        protected_cue_ids=confidence_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids),
        max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
        held_case_ids=mapping_held_case_ids,
    )
    flags.extend(flag for flag in alignment.flags if flag.kind == "adjudication_word_mapping_held")
    decisions = decisions_with_held_cases(
        decisions, alignment.divergence_spans, mapping_held_case_ids,
        "The replacement has no unique word ownership; source text was kept for review.",
    )
    timing_held_cue_ids = _timing_evidence_held_cue_ids(flags)
    adjudicated_cues, change_flags = apply_text(decisions)
    adjudicated_cues, alignment, segmentation_flags, cue_id_expansions = segment_generated_adlib_cues(
        adjudicated_cues,
        words,
        alignment,
        generated_adlib_cue_ids,
        profile,
        max_gap_seconds=_generation_float_config(provider_config, "max_gap_seconds", 0.8),
        max_cue_duration_seconds=_generation_float_config(
            provider_config,
            "max_cue_duration_seconds",
            5.0,
        ),
    )
    change_flags = _expanded_adlib_cue_flags(change_flags, cue_id_expansions)
    adjudicated_cues, alignment, speaker_split_flags, speaker_expansions = split_speaker_turn_cues(
        adjudicated_cues, words, alignment, profile,
        protected_cue_ids=confidence_held_cue_ids | timing_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids) | shared_word_cue_ids(alignment),
    )
    change_flags = _expanded_adlib_cue_flags(change_flags, speaker_expansions)
    flags.extend(speaker_split_flags)
    flags.extend(change_flags)
    flags.extend(segmentation_flags)
    timing_held_cue_ids = _timing_evidence_held_cue_ids(flags)
    enforce_existing_line_limit = explicit_style_override or (
        profile.max_lines_per_cue < source_profile.max_lines_per_cue
        or profile.max_chars_per_line < source_profile.max_chars_per_line
    )
    line_limit_cue_ids = (
        ({cue.index for cue in cues} - confidence_held_cue_ids)
        if enforce_existing_line_limit else set()
    ) | {cue_id for children in speaker_expansions.values() for cue_id in children}
    line_limit_cue_ids -= timing_held_cue_ids | shared_word_cue_ids(alignment)
    if line_limit_cue_ids:
        adjudicated_cues, alignment, sync_line_flags, _ = split_overlong_existing_cues(
            adjudicated_cues,
            words,
            alignment,
            profile,
            source_cue_ids=line_limit_cue_ids,
            max_gap_seconds=_generation_float_config(provider_config, "max_gap_seconds", 0.8),
            max_cue_duration_seconds=_generation_float_config(
                provider_config,
                "max_cue_duration_seconds",
                5.0,
            ),
        )
        flags.extend(sync_line_flags)
    def rebuild(cue_list: list[Cue]) -> tuple[list[Cue], list[QCFlag]]:
        return rebuild_cues(
            cue_list,
            timing_words,
            alignment,
            profile,
            max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
            max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
            protected_cue_ids=source_timing_held_cue_ids | timing_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids) | shared_word_cue_ids(alignment),
            min_duration_policy=min_duration_policy_from_config(provider_config),
        )

    rebuilt, recue_flags = rebuild(adjudicated_cues)
    rebuilt, recue_flags, flags = settle_edits_with_held_timing(
        adjudicated_cues, rebuilt, recue_flags, flags,
        source_cues=cues, words=timing_words, alignment=alignment,
        rebuild=rebuild,
        max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
        max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
    )
    rebuilt, alignment, recue_flags, flags = settle_one_letter_residues(
        rebuilt, recue_flags, flags,
        source_cues=cues, words=timing_words, alignment=alignment,
        spans=alignment.divergence_spans, decisions=decisions, profile=profile,
        fixed_cue_ids=confidence_held_cue_ids | source_timing_held_cue_ids | timing_held_cue_ids
        | set(alignment.diagnostics.missing_audio_cue_ids) | shared_word_cue_ids(alignment),
        split_cue_ids={cue_id for cue_id, children in speaker_expansions.items() if len(children) > 1},
    )
    recue_flags, unconfirmed_source_timed_cue_ids = _fold_unconfirmed_evidence_holds(recue_flags, flags)
    source_timing_held_cue_ids |= unconfirmed_source_timed_cue_ids
    rebuilt, alignment, recue_flags, flags = _settle_collapsed_adlibs(
        rebuilt, timing_words, alignment, profile, recue_flags, flags,
        generated_cue_ids=_generated_cue_ids(generated_adlib_cue_ids, cue_id_expansions, speaker_expansions),
        fixed_cue_ids=confidence_held_cue_ids | source_timing_held_cue_ids | timing_held_cue_ids
        | set(alignment.diagnostics.missing_audio_cue_ids) | shared_word_cue_ids(alignment),
    )
    flags.extend(recue_flags)
    timing_held_cue_ids = _timing_evidence_held_cue_ids(flags)
    flags.extend(source_order_inversion_flags(
        rebuilt,
        # A reconciled cue was spoken at another place than written; like a
        # generated insertion it has acoustic order, not a source position.
        source_cue_ids=source_cue_ids - _reconciled_cue_ids(flags),
        protected_cue_ids=source_timing_held_cue_ids | timing_held_cue_ids | set(alignment.diagnostics.missing_audio_cue_ids) | shared_word_cue_ids(alignment),
    ))
    # Accepted insertions can precede their source-list anchor acoustically.
    # Overlap and boundary policies must see temporal order, retaining cue IDs.
    rebuilt = sorted(rebuilt, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    rebuilt, overlap_flags = apply_overlap_policy(
        rebuilt,
        profile.overlap_policy,
        protected_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids) | confidence_held_cue_ids | timing_held_cue_ids | shared_word_cue_ids(alignment),
    )
    flags.extend(overlap_flags)
    speaker_mapping_uses_llm = _speaker_mapping_uses_llm(provider_config)
    speaker_mapping_adapter = (
        None
        if llm_disabled_for_episode and speaker_mapping_uses_llm
        else speaker_mapping_adapter_from_config(provider_config)
    )
    if speaker_mapping_adapter is not None:
        cached_speaker_map = (
            _load_cached_speaker_mapping(episode_workdir, rebuilt, provider_config) if speaker_mapping_uses_llm else None
        )
        if cached_speaker_map is None:
            speaker_map = speaker_mapping_adapter.map_speakers(rebuilt)
            if speaker_mapping_uses_llm:
                _write_cached_speaker_mapping(episode_workdir, rebuilt, provider_config, speaker_map)
            flags.extend(
                _record_llm_usage_events(
                    cost_meter,
                    speaker_mapping_adapter,
                    provider_config,
                    pass_name="speaker_mapping",
                )
            )
        else:
            speaker_map = cached_speaker_map
        _write_json(episode_workdir / "speaker_map.json", speaker_map)
        rebuilt = _cues_with_speaker_characters(rebuilt, speaker_map)
        flags.extend(speaker_mapping_flags(speaker_map))
    if not llm_disabled_for_episode:
        punctuation_skip_flag = _long_audio_punctuation_skip_flag(audio_for_asr, provider_config)
        if punctuation_skip_flag is not None:
            flags.append(punctuation_skip_flag)
        else:
            punctuation_adapter = punctuation_adapter_from_config(provider_config)
            if punctuation_adapter is not None:
                punctuation_input = rebuilt
                _set_adapter_episode_context(punctuation_adapter, punctuation_input)
                cached_punctuation = _load_cached_punctuation(
                    episode_workdir,
                    punctuation_input,
                    provider_config,
                    max_chars_per_line=profile.max_chars_per_line,
                    max_lines_per_cue=profile.max_lines_per_cue,
                    source_cues=cues,
                )
                if cached_punctuation is None:
                    rebuilt, punctuation_flags = apply_punctuation_pass(
                        punctuation_input,
                        punctuation_adapter,
                        scene_gap_seconds=_punctuation_scene_gap_seconds(provider_config),
                        max_chars_per_line=profile.max_chars_per_line,
                        max_lines_per_cue=profile.max_lines_per_cue,
                        source_cues=cues,
                    )
                    _write_cached_punctuation(
                        episode_workdir,
                        punctuation_input,
                        provider_config,
                        rebuilt,
                        punctuation_flags,
                        max_chars_per_line=profile.max_chars_per_line,
                        max_lines_per_cue=profile.max_lines_per_cue,
                        source_cues=cues,
                    )
                    punctuation_flags.extend(
                        _record_llm_usage_events(
                            cost_meter,
                            punctuation_adapter,
                            provider_config,
                            pass_name="punctuation",
                        )
                    )
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
        decisions=decisions,
        fps_summary_metadata=fps_summary_metadata,
        source_timing_held_cue_ids=source_timing_held_cue_ids,
        speech_evidence=speech_evidence,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    write_json_atomic(path, payload)


def _episode_language_code(provider_config: dict[str, object], language: str | None) -> str | None:
    """ISO-639-1 code of the episode language, when the job or the ASR config names one."""
    asr_config = provider_config.get("asr", {}) if isinstance(provider_config, dict) else {}
    configured: object = None
    if isinstance(asr_config, dict):
        codes = asr_config.get("language_codes")
        configured = (
            asr_config.get("language") or asr_config.get("language_code")
            or (codes[0] if isinstance(codes, list) and len(codes) == 1 else None)
        )
    return asr_language_code(language) or asr_language_code(configured if isinstance(configured, str) else None)


def _is_punctuation_only_span(span: DivergenceSpan) -> bool:
    source_signature = alphanumeric_signature(span.srt_text)
    return bool(source_signature) and source_signature == alphanumeric_signature(span.asr_text)


def _validate_rebuild_policy(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy_version") != _REBUILD_POLICY_VERSION:
        raise ValueError(
            "Cannot resume verify from an older rebuild policy; resume from rebuild "
            "to apply current spoken-text and acoustic timing safeguards."
        )


def _confidence_gate_decisions(
    spans: list[DivergenceSpan], decisions: list[AdjudicationDecision],
    provider_config: dict[str, object], existing_flags: list[QCFlag],
) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    spans_by_id = {span.case_id: span for span in spans}
    selected: list[AdjudicationDecision] = []
    flags: list[QCFlag] = []
    for decision in decisions:
        span = spans_by_id.get(decision.case_id)
        if span is None:
            selected.append(decision)
            continue
        gated, flag = confidence_gated_decision(
            span, decision, _adjudication_confidence_gate(provider_config),
        )
        selected.append(gated)
        # This pass only withholds rewrites that a changed gate no longer
        # accepts. A keep already preserves the source: the engine reported
        # any uncertain model keep itself, and deterministic pipeline holds
        # carry their own flag instead of a second "low confidence" finding.
        if flag is not None and decision.verdict != "keep_srt":
            flags.append(flag)
    return selected, flags


_SOURCE_TIMING_HOLD_FLAG_KINDS = frozenset({
    "unresolved_alignment_adjudication_held",
    "oversized_adjudication_span_held",
})
# A model was asked and its answer could not be used. Each kind is the only
# finding of its hold; none is repeated as "low confidence".
_HELD_WORDING_FLAG_KINDS = frozenset({
    "low_confidence_adjudication",
    "adjudication_audio_unavailable",
    "llm_provider_unavailable",
    "invalid_llm_response",
})


def _confidence_held_source_cue_ids(flags: list[QCFlag]) -> set[int]:
    """Cues whose source WORDING is held: no split, merge or ownership transfer."""
    return {cue_id for flag in flags
            if flag.kind in _HELD_WORDING_FLAG_KINDS or flag.kind in _SOURCE_TIMING_HOLD_FLAG_KINDS
            for cue_id in flag.cue_ids}


def _source_timing_held_cue_ids(flags: list[QCFlag]) -> set[int]:
    # An uncertain or unavailable wording decision keeps the source TEXT only:
    # its cue still owns exactly matched words and is timed from them like any
    # confident keep. Only holds that distrust the alignment itself retain the
    # complete source cue, including its timing.
    return {cue_id for flag in flags
            if flag.kind in _SOURCE_TIMING_HOLD_FLAG_KINDS
            for cue_id in flag.cue_ids}


_UNCONFIRMED_WORDING_FLAG_KINDS = _HELD_WORDING_FLAG_KINDS | {"divergence_unresolved"}


def _fold_unconfirmed_evidence_holds(
    recue_flags: list[QCFlag], flags: list[QCFlag],
) -> tuple[list[QCFlag], set[int]]:
    """Report an unconfirmed divergence once when its words cannot time the cue.

    Source wording that no model confirmed rarely matches the spoken words, so
    the rebuild keeps such a cue at source timing. The wording flag already
    sends that cue to review; a second evidence error describes the same
    unresolved divergence. The cue stays a source-timing hold.
    """
    unconfirmed = {cue_id for flag in flags
                   if flag.kind in _UNCONFIRMED_WORDING_FLAG_KINDS for cue_id in flag.cue_ids}
    # A kept song caption with no or only coincidental matches is the same
    # case: its note already says that it stays at source text and timing.
    captions = {cue_id for flag in flags if flag.kind == _SONG_CAPTION_NOTE_KIND for cue_id in flag.cue_ids}
    retained: list[QCFlag] = []
    held: set[int] = set()
    for flag in recue_flags:
        cue_ids = set(flag.cue_ids)
        if cue_ids and (
            (flag.kind == "timing_evidence_held" and cue_ids <= unconfirmed | captions)
            or (flag.kind == "unmatched_cue" and cue_ids <= captions)
        ):
            held.update(cue_ids)
            continue
        retained.append(flag)
    return retained, held


def _generated_cue_ids(generated_adlib_cue_ids: set[int], *expansions: dict[int, list[int]]) -> set[int]:
    generated = set(generated_adlib_cue_ids)
    for expansion in expansions:
        for parent, children in expansion.items():
            if parent in generated:
                generated.update(children)
    return generated


def _settle_collapsed_adlibs(
    rebuilt: list[Cue], words: list[Word], alignment: AlignmentResult, profile: StyleProfile,
    recue_flags: list[QCFlag], flags: list[QCFlag], *,
    generated_cue_ids: set[int], fixed_cue_ids: set[int],
) -> tuple[list[Cue], AlignmentResult, list[QCFlag], list[QCFlag]]:
    """Never export a generated ad-lib at the envelope of placeholder ASR words.

    A held generated cue has no source timing to fall back to, so "hold" left
    it at its 1 ms (or zero) word envelope: a one-frame flash, or a failed
    export after ASR and adjudication were paid. Its evidence hold is replaced
    by the single flag of the outcome (joined, padded or removed).
    """
    collapsed = {
        cue_id for flag in recue_flags if flag.kind == "timing_evidence_held"
        for cue_id in flag.cue_ids if cue_id in generated_cue_ids
    }
    if not collapsed:
        return rebuilt, alignment, recue_flags, flags
    rebuilt, alignment, settle_flags, gone = settle_collapsed_generated_adlibs(
        rebuilt, words, alignment, profile, collapsed_cue_ids=collapsed, fixed_cue_ids=fixed_cue_ids,
    )
    recue_flags = [
        flag for flag in recue_flags
        if not (flag.kind == "timing_evidence_held" and flag.cue_ids and set(flag.cue_ids) <= collapsed)
    ]
    retained: list[QCFlag] = []
    for flag in flags:
        if flag.kind == "adlib_inserted" and gone.intersection(flag.cue_ids):
            remaining = [cue_id for cue_id in flag.cue_ids if cue_id not in gone]
            if not remaining:
                continue
            flag = flag.model_copy(update={"cue_ids": remaining})
        retained.append(flag)
    return rebuilt, alignment, [*recue_flags, *settle_flags], retained


def _timing_evidence_held_cue_ids(flags: list[QCFlag]) -> set[int]:
    # An ad-lib padded from collapsed words keeps that estimate: refining it
    # to its placeholder word would shrink it to one frame again.
    return {cue_id for flag in flags
            if flag.kind in {
                "timing_evidence_held", "adjudication_word_mapping_held",
                "adjudication_replacement_ownership_held", "protected_source_region_held",
                "adlib_timing_estimated",
            }
            for cue_id in flag.cue_ids}


def _protected_regions_for_alignment(
    alignment: AlignmentResult, cues: list[Cue], words: list[Word],
) -> dict[str, set[int]]:
    return validated_protected_source_regions(
        alignment.divergence_spans, alignment.token_matches, cues, words,
        protected_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
    )


def _validate_resume_audio_provenance(audio_path: Path, episode_workdir: Path) -> list[QCFlag]:
    """Check audio identity before a resumed stage can overwrite any artifacts.

    Legacy checkpoints remain readable, but their missing identity is visible in QC.
    Resume intentionally uses the persisted source/ASR configuration; changing the
    current provider settings never silently changes the cached word stream.
    """
    path = episode_workdir / "asr.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume without ASR artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    provenance = metadata.get("audio_provenance") if isinstance(metadata, dict) else None
    if provenance is None:
        return [QCFlag(
            kind="asr_audio_provenance_unverified",
            message=("Legacy ASR checkpoint has no audio identity hash. Its word timings "
                     "could not be verified against this audio; resume from asr to record provenance."),
            severity="warning",
        )]
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(key), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None
        for key in ("source_sha256", "asr_input_sha256")
    ) or not isinstance(provenance.get("normalized"), bool):
        raise ValueError("Invalid ASR audio provenance; resume from asr to rebuild acoustic evidence.")
    if _sha256_file(audio_path) != provenance["source_sha256"]:
        raise ValueError("Cannot resume: source audio has changed. Resume from asr to rebuild acoustic evidence.")
    if not provenance["normalized"] and provenance["asr_input_sha256"] != provenance["source_sha256"]:
        raise ValueError("Invalid ASR audio provenance: unnormalized input differs from source audio; resume from asr.")
    if provenance["normalized"]:
        normalized = episode_workdir / "audio.16k.wav"
        if not normalized.exists() or _sha256_file(normalized) != provenance["asr_input_sha256"]:
            raise ValueError("Cannot resume: normalized audio is missing or changed. Resume from asr to rebuild acoustic evidence.")
    return []


def _normalize_resume_stage(resume: str | None) -> str | None:
    if resume is None:
        return None
    stage = resume.strip().lower()
    allowed = {"ingest", "asr", "align", "adjudicate", "rebuild", "verify"}
    if stage not in allowed:
        raise ValueError(f"Unsupported resume stage: {resume}")
    return stage


def _apply_local_mode(config: dict[str, object], local: bool) -> dict[str, object]:
    return apply_local_asr_config(config, local)


def _should_load_asr_artifact(resume_stage: str | None) -> bool:
    return resume_stage in {"align", "adjudicate", "rebuild", "verify"}


def _should_load_ingest_artifact(resume_stage: str | None) -> bool:
    return resume_stage in {"asr", "align", "adjudicate", "rebuild", "verify"}


_INGEST_METADATA_KEYS = ("source_cue_numbers", "ingest_flags")
_REPAIRED_SOURCE_CUE_MS = 500


def _source_cues_for_run(
    srt_path: Path, episode_workdir: Path, resume_stage: str | None,
) -> tuple[list[Cue], dict[str, object]]:
    ingest_path = episode_workdir / "ingest.json"
    if _should_load_ingest_artifact(resume_stage) and ingest_path.exists():
        payload = json.loads(ingest_path.read_text(encoding="utf-8"))
        return (
            [Cue.model_validate(item) for item in payload.get("cues", [])],
            {key: payload[key] for key in _INGEST_METADATA_KEYS if key in payload},
        )
    return _hardened_source_cues(parse_srt_text(srt_path.read_text(encoding="utf-8-sig")))


def _hardened_source_cues(cues: list[Cue]) -> tuple[list[Cue], dict[str, object]]:
    """Make a customer SRT safe to process before any paid stage runs.

    The cue number is the identity of a cue in every later stage. A file whose
    numbering restarts or repeats would let one cue take the words, holds and
    timing of another, so such a file gets sequential internal ids; the
    original numbers are kept in ``ingest.json`` for reporting. A cue without
    a positive duration can stay at source timing until export, where it used
    to fail the finished job; its end is repaired here instead.
    """
    metadata: dict[str, object] = {}
    flags: list[QCFlag] = []
    numbers = [cue.index for cue in cues]
    if len(set(numbers)) != len(numbers) or any(number <= 0 for number in numbers):
        cues = [cue.model_copy(update={"index": position}) for position, cue in enumerate(cues, start=1)]
        metadata["source_cue_numbers"] = {str(cue.index): number for cue, number in zip(cues, numbers)}
        flags.append(QCFlag(
            kind="source_cue_numbers_reassigned", severity="info",
            cue_ids=[cue.index for cue, number in zip(cues, numbers) if cue.index != number],
            message=(
                "Source cue numbers were repeated or invalid; cues were numbered sequentially in "
                "file order for processing. The original numbers are recorded in ingest.json."
            ),
        ))
    starts = sorted({cue.start_ms for cue in cues})
    repaired_ids: list[int] = []
    repaired: list[Cue] = []
    for cue in cues:
        if cue.end_ms > cue.start_ms:
            repaired.append(cue)
            continue
        end_ms = cue.start_ms + _REPAIRED_SOURCE_CUE_MS
        later = next((start for start in starts if start > cue.start_ms), None)
        if later is not None:
            end_ms = min(end_ms, later)
        repaired.append(cue.with_timing(cue.start_ms, max(cue.start_ms + 1, end_ms)))
        repaired_ids.append(cue.index)
    if repaired_ids:
        flags.append(QCFlag(
            kind="source_cue_timing_repaired", cue_ids=repaired_ids, severity="warning",
            message=(
                "Source cue had a zero or negative duration; its end was moved after its start "
                "(at most to the next cue) so the cue can be processed and exported."
            ),
        ))
    if flags:
        metadata["ingest_flags"] = [flag.model_dump() for flag in flags]
    return repaired, metadata


def _should_write_ingest_artifacts(
    resume_stage: str | None,
    episode_workdir: Path,
    explicit_style_override: bool,
    fps: float | None,
) -> bool:
    if not _should_load_ingest_artifact(resume_stage):
        return True
    if explicit_style_override or fps is not None:
        return True
    return not (episode_workdir / "ingest.json").exists()


def _fps_override_mismatch_flags(
    cues: list[Cue],
    fps: float | None,
    detection: FPSDetection | None = None,
) -> list[QCFlag]:
    if fps is None or not cues:
        return []
    detection = detection or detect_fps_with_confidence(cues)
    if not detection.confident:
        return []
    detected = detection.fps
    # Treat the broadcast-equivalent pairs 23.976/24 and 29.97/30 as the same grid.
    if abs(float(fps) - detected) <= 0.05:
        return []
    return [
        QCFlag(
            kind="fps_override_mismatch",
            cue_ids=[],
            message=(
                f"Explicit {float(fps):g} fps override differs from the source SRT grid "
                f"detected near {detected:g} fps; output uses the explicit override."
            ),
            severity="warning",
            start=cues[0].start_ms / 1000.0,
            end=cues[-1].end_ms / 1000.0,
        )
    ]


def _fps_detection_flags(
    cues: list[Cue],
    fps: float | None,
    detection: FPSDetection | None = None,
) -> list[QCFlag]:
    if fps is not None or not cues:
        return []
    detection = detection or detect_fps_with_confidence(cues)
    if detection.confident:
        return []
    return [
        QCFlag(
            kind="fps_detection_low_confidence",
            cue_ids=[],
            message=(
                f"Source SRT timestamps are not confidently frame-snapped; defaulting to {detection.fps:g} fps "
                f"(best mean snap error {detection.best_error_ms:g} ms)."
            ),
            severity="warning",
            start=cues[0].start_ms / 1000.0,
            end=cues[-1].end_ms / 1000.0,
        )
    ]


def _fps_summary_metadata(
    profile: StyleProfile,
    detection: FPSDetection,
    *,
    explicitly_configured: bool,
) -> dict[str, object]:
    source = "explicit" if explicitly_configured else ("detected" if detection.confident else "fallback")
    return {
        "fps": float(profile.fps),
        "fps_source": source,
        "fps_detection_confident": bool(detection.confident),
    }


def _alignment_summary_metadata(
    alignment: AlignmentResult,
    *,
    source_cue_count: int,
) -> dict[str, object]:
    unmatched_count = len(set(alignment.unmatched_cue_ids))
    unmatched_ratio = (
        min(1.0, unmatched_count / source_cue_count)
        if source_cue_count > 0
        else 0.0
    )
    diagnostics = alignment.diagnostics
    return {
        "alignment_anchor_coverage": alignment.anchor_coverage,
        "alignment_unmatched_cue_ratio": round(unmatched_ratio, 4),
        "alignment_divergence_span_count": len(alignment.divergence_spans),
        "alignment_band_limited": diagnostics.band_limited,
        "alignment_unresolved": diagnostics.unresolved,
        "alignment_unbanded_fallback": diagnostics.unbanded_fallback,
        "alignment_prior_used": diagnostics.prior_used,
        "alignment_transform_rate": diagnostics.transform_rate,
        "alignment_missing_audio_cue_count": len(diagnostics.missing_audio_cue_ids),
    }


def _spoken_source_cue_count(cues: list[Cue]) -> int:
    return sum(1 for cue in cues if cue_has_spoken_text(cue))


def _validate_alignment_screen_text_provenance(
    alignment: AlignmentResult,
    cues: list[Cue],
) -> None:
    expected = sorted(
        cue.index for cue in cues if cue_has_bracketed_screen_text(cue)
    )
    observed = sorted(alignment.diagnostics.excluded_screen_text_cue_ids)
    if observed != expected:
        raise RuntimeError(
            "Cannot resume from this alignment artifact because its bracketed "
            "screen-text provenance does not match the current source SRT; "
            "resume from align so bracketed non-spoken text can be excluded "
            "from timing synchronization."
        )
    if alignment.diagnostics.missing_audio_guard_version != MISSING_AUDIO_GUARD_VERSION:
        raise RuntimeError(
            "Cannot resume from this alignment artifact because it predates the "
            "missing-audio timing guard; resume from align so unfinished dialogue "
            "cannot borrow timing from another passage."
        )


def _alignment_health_flags(
    alignment: AlignmentResult,
    *,
    source_cue_count: int,
    source_cues: list[Cue] | None = None,
) -> list[QCFlag]:
    coverage = _spoken_anchor_coverage(alignment, source_cues)
    if source_cue_count <= 0 or coverage >= 0.8:
        return []
    severity = "error" if coverage < 0.5 else "warning"
    return [
        QCFlag(
            kind="alignment_anchor_coverage_low",
            cue_ids=[],
            message=(
                f"Only {coverage:.1%} of source tokens anchored to "
                "acoustic words; review divergence spans and unmatched cues before delivery."
            ),
            severity=severity,
        )
    ]


def _spoken_anchor_coverage(alignment: AlignmentResult, source_cues: list[Cue] | None) -> float:
    """Anchor coverage over the source tokens a voice track can contain.

    A dubbed voice stem has no music. Song captions without a single matched
    word are not missed dialogue, so they do not count against alignment
    health. Any other unmatched text still does.
    """
    if not source_cues:
        return alignment.anchor_coverage
    matched_cue_ids = {match.cue_id for match in alignment.token_matches}
    unsung_cue_ids = {
        cue.index for cue in source_cues
        if cue.index not in matched_cue_ids and is_song_caption_cue(cue)
    }
    if not unsung_cue_ids:
        return alignment.anchor_coverage
    spoken_tokens = sum(
        len(alphanumeric_signature(speech_text_for_alignment(cue)))
        for cue in source_cues if cue.index not in unsung_cue_ids
    )
    if spoken_tokens <= 0:
        return alignment.anchor_coverage
    return min(1.0, len(alignment.token_matches) / spoken_tokens)


def _load_style_profile_for_resume(path: Path, resume_stage: str | None) -> StyleProfile | None:
    if not _should_load_ingest_artifact(resume_stage):
        return None
    return _load_style_profile_artifact(path)


def _load_ingest_artifact(path: Path) -> list[Cue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Cue.model_validate(item) for item in payload.get("cues", [])]


def _load_asr_artifact(path: Path) -> list[Word]:
    words, _flags = _load_asr_artifact_with_repair(path)
    return words


def _load_asr_artifact_with_repair(path: Path) -> tuple[list[Word], list[QCFlag]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume without ASR artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_words = payload.get("words", []) if isinstance(payload, dict) else payload
    words, repair_flags = repair_word_stream(raw_words, source="ASR resume artifact")
    return words, [*_load_asr_repair_flags_from_payload(payload, path), *repair_flags]


def _load_asr_repair_flags(path: Path) -> list[QCFlag]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _load_asr_repair_flags_from_payload(payload, path)


def _load_asr_repair_flags_from_payload(payload: object, path: Path) -> list[QCFlag]:
    if not isinstance(payload, dict):
        return []
    raw_flags = payload.get("repair_flags")
    if raw_flags is None:
        metadata = payload.get("metadata", {})
        raw_flags = metadata.get("repair_flags", []) if isinstance(metadata, dict) else []
    if not isinstance(raw_flags, list):
        raise ValueError(f"Invalid ASR repair flags artifact: {path}")
    return [QCFlag.model_validate(item) for item in raw_flags]


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
    audio_snippet_context: dict[str, object] | None = None,
    source_cues: list[Cue] | None = None,
    source_words: list[Word] | None = None,
) -> tuple[list[AdjudicationDecision], list[QCFlag]] | None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    payload = cache.read(
        _adjudication_cache_key(
            spans,
            provider_config,
            audio_snippets=audio_snippets,
            audio_snippet_context=audio_snippet_context,
            source_cues=source_cues,
            source_words=source_words,
        )
    )
    if payload is None:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list) or not isinstance(payload.get("flags"), list):
        raise ValueError("invalid LLM adjudication cache artifact")
    decisions = [AdjudicationDecision.model_validate(item) for item in payload["decisions"]]
    flags = [QCFlag.model_validate(item) for item in payload["flags"]]
    if any(flag.kind in _TRANSIENT_ADJUDICATION_FLAG_KINDS for flag in flags):
        return None
    return decisions, flags


def _write_cached_adjudication(
    episode_workdir: Path,
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    decisions: list[AdjudicationDecision],
    flags: list[QCFlag],
    audio_snippets: dict[str, AudioSnippet] | None = None,
    audio_snippet_context: dict[str, object] | None = None,
    source_cues: list[Cue] | None = None,
    source_words: list[Word] | None = None,
) -> None:
    if any(flag.kind in _TRANSIENT_ADJUDICATION_FLAG_KINDS for flag in flags):
        return
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    cache.write(
        _adjudication_cache_key(
            spans,
            provider_config,
            audio_snippets=audio_snippets,
            audio_snippet_context=audio_snippet_context,
            source_cues=source_cues,
            source_words=source_words,
        ),
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
    source_cues: list[Cue] | None = None,
) -> tuple[list[Cue], list[QCFlag]] | None:
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    payload = cache.read(
        _punctuation_cache_key(
            cues,
            provider_config,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
            source_cues=source_cues,
        )
    )
    if payload is None:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("cues"), list) or not isinstance(payload.get("flags"), list):
        raise ValueError("invalid LLM punctuation cache artifact")
    output_cues = [Cue.model_validate(item) for item in payload["cues"]]
    flags = [QCFlag.model_validate(item) for item in payload["flags"]]
    if any(flag.kind in _TRANSIENT_PUNCTUATION_FLAG_KINDS for flag in flags):
        return None
    return output_cues, flags


def _write_cached_punctuation(
    episode_workdir: Path,
    input_cues: list[Cue],
    provider_config: dict[str, object],
    output_cues: list[Cue],
    flags: list[QCFlag],
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    source_cues: list[Cue] | None = None,
) -> None:
    if any(flag.kind in _TRANSIENT_PUNCTUATION_FLAG_KINDS for flag in flags):
        return
    cache = JsonDiskCache(episode_workdir / "llm-cache")
    cache.write(
        _punctuation_cache_key(
            input_cues,
            provider_config,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
            source_cues=source_cues,
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


def _episode_audio_options(provider_config: dict[str, object]) -> dict[str, object] | None:
    config = llm_config_for_pass(provider_config, "adjudication")
    options = config.get("audio_context")
    if str(config.get("provider", "gemini")).lower() != "gemini" or options is None or options is False:
        return None
    if options is True:
        options = {"enabled": True}
    if not isinstance(options, dict):
        raise ValueError("llm.adjudication.audio_context must be a mapping or boolean")
    options = validate_audio_context_config(options)
    return options if options.get("enabled", True) else None


def _adjudication_audio_cache_context(
    original_audio: Path,
    normalized_audio: Path,
    provider_config: dict[str, object],
    snippet_context: dict[str, object] | None,
) -> dict[str, object] | None:
    options = _episode_audio_options(provider_config)
    if options is None:
        return snippet_context
    return {
        "focused_snippets": snippet_context,
        "episode_audio": {
            "policy_version": 2,
            "source_sha256": _sha256_file(original_audio),
            "normalized_sha256": _sha256_file(normalized_audio),
            "duration_seconds": audio_seconds(normalized_audio),
            "options": options,
        },
    }


@contextmanager
def _adjudication_audio_session(
    adapter: object,
    original_audio: Path,
    normalized_audio: Path,
    provider_config: dict[str, object],
    episode_workdir: Path,
    cost_meter: CostMeter,
    flags: list[QCFlag],
):
    options = _episode_audio_options(provider_config)
    configure = getattr(adapter, "set_audio_context", None)
    if options is None or not callable(configure):
        try:
            yield
        finally:
            _write_hybrid_adjudication_report(adapter, episode_workdir, flags)
            # Focused hybrid calls also incur usage when a later batch fails.
            if callable(getattr(adapter, "route_report", None)):
                flags.extend(_record_llm_usage_events(cost_meter, adapter, provider_config, pass_name="adjudication"))
                write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())
        return
    config = llm_config_for_pass(provider_config, "adjudication")
    duration = audio_seconds(normalized_audio)
    try:
        configure(
            normalized_audio if duration <= 180 else original_audio,
            duration_seconds=duration, config=options,
        )
        yield
    finally:
        adapter.close()
        _write_hybrid_adjudication_report(adapter, episode_workdir, flags)
        report = adapter.audio_context_report()
        _write_json(episode_workdir / "gemini_audio_context.json", report)
        pricing_issue = record_gemini_context_cost(
            cost_meter, str(config.get("model") or _default_llm_model("gemini")), config, report,
        )
        flags.extend(_record_llm_usage_events(cost_meter, adapter, provider_config, pass_name="adjudication"))
        if (pricing_issue or report.get("warnings") or report.get("unreported_uncached_audio_tokens_reserved")
                or report.get("unreported_cached_audio_tokens_reserved")):
            flags.append(QCFlag(
                kind="gemini_audio_context_warning", cue_ids=[],
                message="Full audio context has transport, cleanup, or cost uncertainty; see gemini_audio_context.json.",
            ))
        # Preserve incurred usage if a later stage fails before final reporting.
        write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())


def _write_hybrid_adjudication_report(adapter: object, episode_workdir: Path, flags: list[QCFlag]) -> None:
    report_method = getattr(adapter, "route_report", None)
    if not callable(report_method):
        return
    report = report_method()
    _write_json(episode_workdir / "hybrid_adjudication.json", report)
    counts = report.get("counts", {})
    flags.append(QCFlag(
        kind="hybrid_adjudication_summary", severity="info", cue_ids=[],
        message=(
            f"Primary accepted {counts.get('primary', 0)} cases; focused audio review requested for "
            f"{counts.get('review_requested', 0)} cases, accepted {counts.get('fallback', 0)}, "
            f"and held {counts.get('held', 0)}. See hybrid_adjudication.json for case routes."
        ),
    ))
    # The hybrid route turns a failed review call into ordinary held
    # decisions. Without this transient marker the hold would be cached and
    # replayed by every later run although the outage is long over.
    outage_held = sum(
        1 for item in report.get("decisions", [])
        if isinstance(item, dict) and item.get("route") == "held"
        and any(str(reason).endswith("_provider_failure") for reason in item.get("reasons", []))
    )
    if outage_held:
        flags.append(QCFlag(
            kind="adjudication_review_unavailable", severity="warning", cue_ids=[],
            message=(
                f"The adjudication review provider failed for {outage_held} cases; their source text "
                "was preserved for review and the result was not cached, so a re-run asks again."
            ),
        ))


def _adjudication_cache_key(
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    audio_snippets: dict[str, AudioSnippet] | None = None,
    audio_snippet_context: dict[str, object] | None = None,
    source_cues: list[Cue] | None = None,
    source_words: list[Word] | None = None,
) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "adjudication")
    has_review = adjudication_fallback_config(llm_config) is not None
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    payload = {
        "pass": "adjudication",
        "prompt_version": _ADJUDICATION_PROMPT_VERSION,
        "policy_version": _ADJUDICATION_POLICY_VERSION,
        "review_prompt_version": (
            _ADJUDICATION_REVIEW_PROMPT_VERSION
            if has_review else None
        ),
        "hybrid_policy_version": (
            HYBRID_POLICY_VERSION
            if has_review else None
        ),
        # Review sees neighboring ASR words as read-only ownership evidence.
        # Hash the complete stream so changing a matched neighbor cannot reuse
        # a decision made with different local audio/transcript context.
        "hybrid_asr_context_sha256": (
            hashlib.sha256(json.dumps(
                [word.model_dump(mode="json") for word in source_words],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if has_review and source_words is not None else None
        ),
        "confidence_gate": _adjudication_confidence_gate(provider_config),
        "scene_gap_seconds": _adjudication_scene_gap_seconds(provider_config),
        "spans": [span.model_dump(mode="json") for span in spans],
        "asr_word_evidence": (
            [
                {"word_index": index, **source_words[index].model_dump(mode="json")}
                for index in sorted({
                    index for span in spans for index in span.asr_word_indices
                    if isinstance(index, int) and not isinstance(index, bool)
                    and 0 <= index < len(source_words)
                })
            ]
            if source_words is not None else None
        ),
        "episode_context": (
            [cue.model_dump(mode="json") for cue in source_cues]
            if source_cues is not None
            else None
        ),
        "audio_snippets": _audio_snippet_cache_payload(audio_snippets or {}),
        "audio_snippet_context": audio_snippet_context,
    }
    return CacheKey.from_payload(payload, model=model, params=_llm_cache_params(llm_config))


def _punctuation_cache_key(
    cues: list[Cue],
    provider_config: dict[str, object],
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    source_cues: list[Cue] | None = None,
) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "punctuation")
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    payload = {
        "pass": "punctuation",
        "prompt_version": _PUNCTUATION_PROMPT_VERSION,
        "policy_version": _PUNCTUATION_POLICY_VERSION,
        "scene_gap_seconds": _punctuation_scene_gap_seconds(provider_config),
        "line_constraints": {
            "max_chars_per_line": max_chars_per_line,
            "max_lines_per_cue": max_lines_per_cue,
        },
        "cues": [cue.model_dump(mode="json") for cue in cues],
        "source_break_cues": (
            [cue.model_dump(mode="json") for cue in source_cues]
            if source_cues is not None
            else None
        ),
    }
    return CacheKey.from_payload(payload, model=model, params=_llm_cache_params(llm_config))


def _speaker_mapping_cache_key(cues: list[Cue], provider_config: dict[str, object]) -> CacheKey:
    llm_config = llm_config_for_pass(provider_config, "speaker_mapping")
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    mapping_config = provider_config.get("speaker_mapping", {}) if isinstance(provider_config, dict) else {}
    payload = {
        "pass": "speaker_mapping",
        "prompt_version": _SPEAKER_MAPPING_PROMPT_VERSION,
        "cues": [cue.model_dump(mode="json") for cue in cues],
    }
    params = {**_llm_cache_params(llm_config), "speaker_mapping": mapping_config}
    return CacheKey.from_payload(payload, model=model, params=params)


def _llm_cache_params(llm_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in llm_config.items() if key != "responses"}


def _adjudication_audio_snippet_source(
    audio_path: Path,
    episode_workdir: Path,
    provider_config: dict[str, object],
) -> BoundedAudioSnippetBatchSource | None:
    (
        enabled,
        pad_seconds,
        max_duration_seconds,
        max_snippets_per_batch,
        max_audio_duration_seconds,
    ) = (
        _adjudication_audio_snippet_options(provider_config)
    )
    if not enabled:
        return None
    return BoundedAudioSnippetBatchSource(
        audio_path,
        episode_workdir / "audio-snippets",
        pad_seconds=pad_seconds,
        max_duration_seconds=max_duration_seconds,
        max_snippets_per_batch=max_snippets_per_batch,
        max_audio_duration_seconds=max_audio_duration_seconds,
        extractor=extract_audio_snippets,
        max_concurrent_batches=llm_config_for_pass(provider_config, "adjudication").get("max_concurrent_batches", 1),
        max_covering_duration_seconds=_adjudication_covering_snippet_seconds(provider_config),
    )


def _adjudication_covering_snippet_seconds(provider_config: dict[str, object]) -> float:
    value = llm_config_for_pass(provider_config, "adjudication").get("audio_snippet_double_check", False)
    label = "llm.adjudication.audio_snippet_double_check.max_long_span_duration_seconds"
    seconds = (
        _float_config(value, "max_long_span_duration_seconds", DEFAULT_MAX_COVERING_SNIPPET_SECONDS, label)
        if isinstance(value, dict) else DEFAULT_MAX_COVERING_SNIPPET_SECONDS
    )
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return seconds


def _adjudication_audio_snippet_options(
    provider_config: dict[str, object],
) -> tuple[bool, float, float, int, float | None]:
    # Clips are cut for one batch at a time, so the episode length does not
    # bound the work. The former 90-minute default disabled every case of a
    # feature-length job; a limit now applies only when explicitly configured.
    llm_config = llm_config_for_pass(provider_config, "adjudication")
    value = llm_config.get("audio_snippet_double_check", False)
    if value in (False, None):
        return (False, 2.0, 20.0, 25, None)
    if value is True:
        return (True, 2.0, 20.0, 25, None)
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
    max_snippets_per_batch = _int_config(
        value,
        "max_snippets_per_batch",
        25,
        "llm.adjudication.audio_snippet_double_check.max_snippets_per_batch",
    )
    max_audio_duration_seconds = (
        _float_config(
            value,
            "max_audio_duration_seconds",
            0.0,
            "llm.adjudication.audio_snippet_double_check.max_audio_duration_seconds",
        )
        if value.get("max_audio_duration_seconds") is not None
        else None
    )
    if pad_seconds < 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.pad_seconds must be non-negative")
    if max_duration_seconds <= 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.max_duration_seconds must be positive")
    if max_snippets_per_batch <= 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.max_snippets_per_batch must be positive")
    if max_audio_duration_seconds is not None and max_audio_duration_seconds <= 0:
        raise ValueError("llm.adjudication.audio_snippet_double_check.max_audio_duration_seconds must be positive")
    return (
        enabled,
        pad_seconds,
        max_duration_seconds,
        max_snippets_per_batch,
        max_audio_duration_seconds,
    )


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


def _generation_float_config(provider_config: dict[str, object], key: str, default: float) -> float:
    generation_config = provider_config.get("generation", {}) if isinstance(provider_config, dict) else {}
    if not isinstance(generation_config, dict):
        return default
    value = _float_config(generation_config, key, default, f"generation.{key}")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"generation.{key} must be finite and positive")
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
    return boundary_refinement_config_from_config(provider_config)


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
    asr_path = episode_workdir / "asr.json"
    if asr_path.exists():
        payload = json.loads(asr_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        provenance = metadata.get("audio_provenance") if isinstance(metadata, dict) else None
        if isinstance(provenance, dict) and provenance.get("normalized") is False:
            return audio_path
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
    decisions: list[AdjudicationDecision] | None = None,
    fps_summary_metadata: dict[str, object] | None = None,
    source_timing_held_cue_ids: set[int] | None = None,
    speech_evidence: SpeechEvidence | None = None,
) -> PipelineResult:
    decisions = list(decisions or [])
    # A shared phrase timestamp cannot establish its internal cue boundary.
    shared_cue_ids = shared_word_cue_ids(alignment) | {
        cue_id for flag in flags if flag.kind == "shared_word_timing_preserved" for cue_id in flag.cue_ids
    }
    shared_source_cue_ids = shared_cue_ids & {cue.index for cue in source_cues}
    rebuilt = preserve_source_timings(rebuilt, source_cues, shared_source_cue_ids)
    flags = _without_stale_verify_flags(flags)
    alignment, flags = _release_reconciled_cues(alignment, flags)
    forced_alignments: list[ForcedAlignmentCue] = []
    effective_words = words
    speech_regions = []
    min_coverage = min_coverage_from_config(provider_config)
    boundary_refinement = _boundary_refinement_config(provider_config)
    missing_audio_cue_ids = set(alignment.diagnostics.missing_audio_cue_ids)
    source_timing_held_cue_ids = _source_timing_held_cue_ids(flags) | set(source_timing_held_cue_ids or ())
    timing_held_cue_ids = _timing_evidence_held_cue_ids(flags)
    protected_source_regions = _protected_regions_for_alignment(alignment, source_cues, words)
    protected_source_cue_ids = {cue_id for ids in protected_source_regions.values() for cue_id in ids}
    protected_cue_ids = missing_audio_cue_ids | source_timing_held_cue_ids | timing_held_cue_ids | protected_source_cue_ids
    unresolved_shared_cue_ids = set(shared_source_cue_ids)
    forced_alignment_adapter = forced_alignment_adapter_from_config(provider_config)
    if forced_alignment_adapter is not None:
        forced_alignment_input = [
            cue for cue in rebuilt if cue.index not in protected_cue_ids
        ]
        forced_alignments = forced_alignment_adapter.align(
            audio_for_asr,
            forced_alignment_input,
        )
        forced_alignments = [
            item for item in forced_alignments
            if item.cue_id not in shared_cue_ids
            or (isfinite(item.start) and isfinite(item.end) and 0 <= item.start < item.end)
        ]
        usable = usable_forced_alignments_by_cue(
            rebuilt, forced_alignments, protected_cue_ids=protected_cue_ids,
        )
        unresolved_shared_cue_ids -= usable.keys()
        _write_json(episode_workdir / "forced_align.json", {"cues": [alignment.model_dump() for alignment in forced_alignments]})
        rebuilt, forced_alignment_flags = apply_forced_alignment(
            rebuilt,
            forced_alignments,
            profile,
            protected_cue_ids=protected_cue_ids,
        )
        flags.extend(forced_alignment_flags)
        if unresolved_shared_cue_ids and not usable:
            flags.append(QCFlag(
                kind="forced_alignment_unavailable",
                cue_ids=sorted(unresolved_shared_cue_ids),
                message="Forced alignment returned no usable cue timings; existing timings were retained for review.",
            ))
    protected_cue_ids |= unresolved_shared_cue_ids
    flags.extend(shared_word_timing_flags(rebuilt, unresolved_shared_cue_ids))
    overlap_detection_adapter = overlap_detection_adapter_from_config(provider_config)
    if overlap_detection_adapter is not None:
        overlap_regions = overlap_detection_adapter.detect(audio_for_asr)
        _write_json(episode_workdir / "overlap.json", {"regions": [region.model_dump() for region in overlap_regions]})
        flags.extend(overlap_flags_for_regions(rebuilt, overlap_regions))
    if speech_evidence is None:
        speech_evidence = speech_evidence_for_words(
            speech_activity_adapter_from_config(provider_config), words, audio_for_asr, provider_config,
            max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
            asr_artifact_path=episode_workdir / "asr.json",
        )
    if speech_evidence.detected:
        speech_regions = speech_evidence.regions
        if speech_evidence.fallback_used:
            flags.append(
                QCFlag(
                    kind="vad_provider_fallback",
                    cue_ids=[],
                    message="Configured VAD provider fell back to energy-based speech activity detection.",
                    severity="warning",
                )
            )
        _write_json(episode_workdir / "vad.json", {"regions": [region.model_dump() for region in speech_regions]})
        effective_words = speech_evidence.words
        flags.extend(speech_evidence.word_flags)
        rebuilt, timing_flags = refine_cues_to_speech_activity(
            rebuilt,
            speech_regions,
            profile,
            boundary_refinement,
            words=effective_words,
            alignment=alignment,
            protected_cue_ids=protected_cue_ids,
            fixed_cue_ids=shared_source_cue_ids - unresolved_shared_cue_ids,
        )
        flags.extend(timing_flags)
        if include_dropped_line_flags:
            flags.extend(
                dropped_line_flags_for_unmatched_cues(
                    source_cues,
                    alignment.unmatched_cue_ids,
                    speech_regions,
                    min_coverage,
                    cue_word_indices=alignment.cue_word_indices,
                )
            )
    rebuilt, missing_audio_restore_flags = _restore_missing_audio_source_cues(
        rebuilt,
        source_cues,
        missing_audio_cue_ids,
    )
    flags.extend(missing_audio_restore_flags)
    rebuilt, confidence_restore_flags = _restore_missing_audio_source_cues(
        rebuilt, source_cues, source_timing_held_cue_ids,
        reason="low_confidence",
    )
    flags.extend(confidence_restore_flags)
    rebuilt, timing_restore_flags = _restore_missing_audio_source_cues(
        rebuilt, source_cues, timing_held_cue_ids,
        reason="timing_evidence",
    )
    flags.extend(timing_restore_flags)
    rebuilt, protected_source_restore_flags = _restore_missing_audio_source_cues(
        rebuilt, source_cues, protected_source_cue_ids, reason="protected_region",
    )
    flags.extend(protected_source_restore_flags)
    rebuilt, flags = _remove_leftover_duplicate_cues(rebuilt, source_cues, effective_words, alignment, flags)
    rebuilt, final_order_flags = finalize_cues_for_output(
        rebuilt,
        profile,
        no_overlaps=_output_no_overlaps(provider_config),
        protected_cue_ids=protected_cue_ids,
        preserve_timing=bool(effective_words or forced_alignments or speech_regions),
        media_duration_ms=_known_audio_duration_ms(audio_for_asr),
        spoken_spans=cue_spoken_spans(
            rebuilt, effective_words, alignment,
            max_word_duration=_timing_float_config(provider_config, "max_word_duration", 2.0),
            max_intra_cue_gap=_timing_float_config(provider_config, "max_intra_cue_gap", 1.5),
        ),
    )
    flags = [*reconcile_overlap_flags(flags, rebuilt, final_order_flags), *final_order_flags]
    if speech_regions:
        # A cue held for the minimum display time is not an overrun and its
        # short utterance still counts as speech activity.
        readability_floor_ms = round(profile.min_cue_dur * 1000 + profile.frame_ms)
        activity_flags = speech_activity_flags_for_cues(
            rebuilt, speech_regions, min_coverage, min_cue_duration_ms=readability_floor_ms,
        )
        rebuilt, flags, activity_flags = _remove_silent_generated_adlibs(
            rebuilt,
            flags,
            activity_flags,
            audible_cue_ids=cue_ids_with_audible_words(audio_for_asr, activity_flags, effective_words, alignment),
        )
        flags.extend(activity_flags)
        flags.extend(
            trailing_silence_flags_for_cues(
                rebuilt,
                speech_regions,
                max_trailing_silence_ms=boundary_refinement.max_trailing_silence_ms,
                min_cue_duration_ms=readability_floor_ms,
            )
        )
    style_issues = lint_cues(rebuilt, profile)
    flags.extend(
        cps_sanity_flags(
            rebuilt,
            max_cps=_timing_float_config(provider_config, "max_cps", 30.0),
            min_cps=_timing_float_config(provider_config, "min_cps", 2.0),
        )
    )
    cue_scores = score_cues(
        rebuilt, effective_words, alignment, forced_alignments,
        protected_cue_ids=protected_cue_ids,
    )
    if audio_for_asr != audio_path:
        flags.extend(silence_flags_for_cues(audio_for_asr, rebuilt))

    flags.extend(episode_editorial_addition_flags(source_cues, rebuilt))
    rebuilt, profanity_flags = apply_german_profanity_censorship(rebuilt, source_cues)
    flags.extend(profanity_flags)
    flags.extend(span_coverage_flags(source_cues, rebuilt, alignment.divergence_spans, decisions))
    flags.extend(name_spelling_inconsistency_flags(source_cues, rebuilt, asr_words=words))
    flags.extend(
        _alignment_health_flags(
            alignment,
            source_cue_count=_spoken_source_cue_count(source_cues),
            source_cues=source_cues,
        )
    )
    flags = censor_german_profanity_flags(flags, source_cues)
    flags = _unique_flags(_without_song_caption_silence_duplicates(flags))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_path, write_srt(rebuilt, renumber=True))
    _write_json(episode_workdir / "rebuild.json", {
        "policy_version": _REBUILD_POLICY_VERSION,
        "cues": [cue.model_dump() for cue in rebuilt],
    })

    report = write_qc_report(
        episode_workdir / "qc_report.json",
        episode_workdir / "qc_report.html",
        rebuilt,
        flags,
        style_issues,
        cue_scores,
        summary_metadata={
            **(fps_summary_metadata or {}),
            **_alignment_summary_metadata(
                alignment,
                source_cue_count=_spoken_source_cue_count(source_cues),
            ),
        },
        source_cues=source_cues,
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
    write_change_log(episode_workdir / "changes.diff.srt", report["changes"])
    _write_json(episode_workdir / "cost.json", cost_meter.as_dict())

    return PipelineResult(output_path, episode_workdir, cost_meter, report)


def _known_audio_duration_ms(audio_path: Path) -> int | None:
    duration = audio_seconds(audio_path)
    return int(round(duration * 1000)) if isfinite(duration) and duration > 0 else None


_LEFTOVER_DUPLICATE_MAX_TOKENS = 4
# Findings about a cue alone that are moot once the cue is gone.
_LEFTOVER_CUE_FLAG_KINDS = frozenset({
    "missing_audio_timing_held", "missing_audio_source_cue_held", "missing_audio_source_cue_restored",
    "unmatched_cue", "dropped_line_candidate", "divergence_unresolved", "low_confidence_adjudication",
    "cue_without_speech_activity", "overlap_stacked", "overlap_flag_only",
})


def _remove_leftover_duplicate_cues(
    cues: list[Cue],
    source_cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    flags: list[QCFlag],
) -> tuple[list[Cue], list[QCFlag]]:
    """Remove an unspoken source cue that repeats words of the cue shown at its time.

    The script says "Claro," and then "É claro que a empresa ..."; the actor
    says the phrase once. The first cue has no audio and stayed at its source
    time inside the second, which shows the same word: a doubled line and an
    overlap. A short source cue without any owned word whose complete wording
    is spoken inside an overlapping, acoustically timed cue is a duplicate.
    """
    source_by_id = {cue.index: cue for cue in source_cues}

    def spoken_signature(cue: Cue) -> list[str]:
        owned = sorted(
            (words[index] for index in alignment.cue_word_indices.get(cue.index, []) if 0 <= index < len(words)),
            key=lambda word: (word.start, word.end),
        )
        return alphanumeric_signature(" ".join(word.text for word in owned))

    def contains(whole: list[str], part: list[str]) -> bool:
        # The aligner's spelling tolerance: "Vamos" is spoken as "vamo".
        return any(
            all(
                token == candidate or fuzz.ratio(token, candidate, score_cutoff=85) >= 85
                for token, candidate in zip(part, whole[start:start + len(part)])
            )
            for start in range(len(whole) - len(part) + 1)
        )

    removed: dict[int, Cue] = {}
    merge_flags: list[QCFlag] = []
    for cue in cues:
        source = source_by_id.get(cue.index)
        signature = alphanumeric_signature(speech_text_for_alignment(cue))
        if (
            source is None or alignment.cue_word_indices.get(cue.index)
            or (cue.start_ms, cue.end_ms, cue.text) != (source.start_ms, source.end_ms, source.text)
            or not 1 <= len(signature) <= _LEFTOVER_DUPLICATE_MAX_TOKENS
            or cue_has_bracketed_screen_text(cue) or is_song_caption_cue(cue)
        ):
            continue
        for other in cues:
            overlap_ms = min(cue.end_ms, other.end_ms) - max(cue.start_ms, other.start_ms)
            if (
                other.index == cue.index or other.index in removed
                or overlap_ms * 2 < cue.duration_ms
                or cue_has_bracketed_screen_text(other) or is_song_caption_cue(other)
                or speakers_known_different(cue.speaker_id, other.speaker_id)
                or not contains(alphanumeric_signature(speech_text_for_alignment(other)), signature)
                or not contains(spoken_signature(other), signature)
            ):
                continue
            removed[cue.index] = cue
            merge_flags.append(QCFlag(
                kind="duplicate_cue_merged", cue_ids=[other.index, cue.index],
                message=(
                    "A source cue without audio repeats words spoken in the cue shown at the same time; "
                    "the unspoken duplicate was removed."
                ),
                old_text=f"{other.text}\n\n{cue.text}", new_text=other.text,
                start=other.start_ms / 1000.0, end=other.end_ms / 1000.0,
            ))
            break
    if not removed:
        return cues, flags
    kept_flags: list[QCFlag] = []
    for flag in flags:
        if flag.kind in _LEFTOVER_CUE_FLAG_KINDS and removed.keys() & set(flag.cue_ids):
            remaining = [cue_id for cue_id in flag.cue_ids if cue_id not in removed]
            if not remaining or flag.kind in {"overlap_stacked", "overlap_flag_only"}:
                continue
            flag = flag.model_copy(update={"cue_ids": remaining})
        kept_flags.append(flag)
    return [cue for cue in cues if cue.index not in removed], [*kept_flags, *merge_flags]


def _remove_silent_generated_adlibs(
    cues: list[Cue],
    flags: list[QCFlag],
    activity_flags: list[QCFlag],
    audible_cue_ids: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag], list[QCFlag]]:
    generated_cue_ids = {
        cue_id
        for flag in flags
        if flag.kind == "adlib_inserted"
        for cue_id in flag.cue_ids
    }
    if not generated_cue_ids:
        return cues, flags, activity_flags

    silent_cue_ids = {
        cue_id
        for flag in activity_flags
        if (
            flag.kind == "cue_without_speech_activity"
            and flag.confidence is not None
            and flag.confidence <= 0.0
        )
        for cue_id in flag.cue_ids
        if cue_id in generated_cue_ids
    } - (audible_cue_ids or set())
    if not silent_cue_ids:
        return cues, flags, activity_flags

    removed_cues = [cue for cue in cues if cue.index in silent_cue_ids]
    remaining_cues = [cue for cue in cues if cue.index not in silent_cue_ids]
    retained_flags = [
        flag
        for flag in flags
        if not (
            flag.kind in {"adlib_inserted", "adlib_timing_estimated"}
            and any(cue_id in silent_cue_ids for cue_id in flag.cue_ids)
        )
    ]
    retained_activity_flags = [
        flag
        for flag in activity_flags
        if not any(cue_id in silent_cue_ids for cue_id in flag.cue_ids)
    ]
    removal_flags = [
        QCFlag(
            kind="adlib_removed_without_speech_activity",
            cue_ids=[cue.index],
            message="Generated ad-lib cue was removed because VAD found no speech activity in its displayed interval.",
            old_text=cue.text,
            new_text="",
            start=cue.start_ms / 1000.0,
            end=cue.end_ms / 1000.0,
        )
        for cue in removed_cues
    ]
    return remaining_cues, [*retained_flags, *removal_flags], retained_activity_flags


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


def _long_audio_llm_skip_flag(
    audio_path: Path,
    provider_config: dict[str, object],
) -> QCFlag | None:
    return None


def _long_audio_punctuation_skip_flag(
    audio_path: Path,
    provider_config: dict[str, object],
) -> QCFlag | None:
    llm_config = llm_config_for_pass(provider_config, "punctuation")
    provider = str(llm_config.get("provider", "gemini")).lower()
    if provider == "fixture":
        return None
    max_duration_seconds = _float_config(
        llm_config,
        "max_audio_duration_seconds",
        30 * 60.0,
        "llm.punctuation.max_audio_duration_seconds",
    )
    if max_duration_seconds <= 0:
        raise ValueError("llm.punctuation.max_audio_duration_seconds must be positive")
    duration_seconds = audio_seconds(audio_path)
    if duration_seconds <= 0 or duration_seconds <= max_duration_seconds:
        return None
    return QCFlag(
        kind="punctuation_skipped_for_long_audio",
        cue_ids=[],
        message=(
            "LLM punctuation was skipped because the episode audio is "
            f"{duration_seconds:g} seconds, above the configured "
            f"{max_duration_seconds:g} second punctuation limit. Existing text, "
            "line structure, and cue timing were preserved."
        ),
        severity="warning",
    )


def _record_llm_usage_events(
    cost_meter: CostMeter,
    adapter: object,
    provider_config: dict[str, object],
    pass_name: str | None = None,
) -> list[QCFlag]:
    llm_config = llm_config_for_pass(provider_config, pass_name)
    if not isinstance(llm_config, dict):
        return []
    provider = str(llm_config.get("provider", "gemini")).lower()
    model = str(llm_config.get("model") or _default_llm_model(provider))
    unmetered_reasons: set[str] = set()
    first_item = len(cost_meter.items)
    for event in drain_usage_events(adapter):
        event_config, event_provider, event_model = llm_config, provider, model
        route = event.get("adjudication_route") if isinstance(event, dict) else None
        if route == "fallback":
            fallback_config = adjudication_fallback_config(llm_config) if pass_name == "adjudication" else None
            if fallback_config is None:
                unmetered_reasons.add("fallback usage has no configured review model")
                continue
            event_config = fallback_config
            event_provider = str(fallback_config["provider"])
            event_model = str(fallback_config["model"])
        elif route not in (None, "primary"):
            unmetered_reasons.add("unknown adjudication usage route")
            continue
        reason = record_llm_usage(cost_meter, event_provider, event_model, event_config, event)
        if reason is not None:
            unmetered_reasons.add(reason)
    estimate_flags = []
    if any(item.kind == "tokens_cache_metadata_estimate" for item in cost_meter.items[first_item:]):
        estimate_flags.append(QCFlag(
            kind="cost_estimate_uncertain", cue_ids=[], severity="warning",
            message="Gemini reported cached token counts above total input tokens. Cost includes a conservative full-input estimate; raw reported counts are retained in the cost artifact.",
        ))
    if not unmetered_reasons:
        return estimate_flags
    pass_label = pass_name or "llm"
    return [*estimate_flags,
        QCFlag(
            kind="cost_unmetered",
            cue_ids=[],
            message=(
                f"{pass_label} usage for {model} was not added to cost totals: "
                f"{', '.join(sorted(unmetered_reasons))}. Configure token prices or "
                "use a provider response that reports token usage."
            ),
            severity="warning",
        )
    ]


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
        return "gpt-5.6-luna"
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


def _expanded_adlib_cue_flags(
    flags: list[QCFlag],
    cue_id_expansions: dict[int, list[int]],
) -> list[QCFlag]:
    if not cue_id_expansions:
        return list(flags)
    return [
        flag.model_copy(
            update={
                "cue_ids": [
                    expanded_id
                    for cue_id in flag.cue_ids
                    for expanded_id in cue_id_expansions.get(cue_id, [cue_id])
                ]
            }
        )
        if flag.kind == "adlib_inserted"
        else flag
        for flag in flags
    ]


def _hold_incomplete_source_insertions(
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    *,
    source_cue_count: int | None = None,
    alignment_unresolved: bool = False,
    missing_audio_cue_ids: set[int] | None = None,
    protected_source_regions: dict[str, set[int]] | None = None,
    song_caption_cue_ids: set[int] | None = None,
) -> tuple[list[DivergenceSpan], list[AdjudicationDecision], list[QCFlag]]:
    max_duration = _generation_float_config(
        provider_config,
        "max_generated_adlib_duration_seconds",
        20.0,
    )
    provider_spans: list[DivergenceSpan] = []
    held_decisions: list[AdjudicationDecision] = []
    flags: list[QCFlag] = []
    missing_audio = missing_audio_cue_ids or set()
    for span in spans:
        hold = _protected_source_region_hold(span, protected_source_regions or {})
        if hold is None:
            caption_hold = _song_caption_source_hold(span, song_caption_cue_ids or set(), missing_audio)
            if caption_hold is not None:
                held_decisions.append(caption_hold[0])
                flags.extend(caption_hold[1])
                continue
        if hold is None:
            hold = _missing_audio_source_hold(span, missing_audio)
        if hold is None:
            hold = (
                _unresolved_alignment_adjudication_hold(span)
                if alignment_unresolved
                else None
            )
        if hold is None:
            hold = _oversized_adjudication_span_hold(span, source_cue_count)
        if hold is None:
            hold = _incomplete_source_hold(span, max_duration)
        if hold is None:
            provider_spans.append(span)
            continue
        decision, flag = hold
        held_decisions.append(decision)
        flags.append(flag)
    return provider_spans, held_decisions, flags


def _protected_source_region_hold(
    span: DivergenceSpan, validated_regions: dict[str, set[int]],
) -> tuple[AdjudicationDecision, QCFlag] | None:
    if span.case_id not in validated_regions:
        return None
    if (not span.case_id.startswith(PROTECTED_SOURCE_PREFIX)
        or set(span.cue_ids) != validated_regions[span.case_id]
        or not span.srt_token_indices or not span.srt_text.strip()
        or span.asr_word_indices or span.asr_text.strip()):
        raise ValueError("Invalid protected source branch; resume from align.")
    return AdjudicationDecision(
        case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
        confidence=1.0, reason="Preserved the complete protected source region; earlier repeated speech has its own fresh audio question.",
    ), QCFlag(
        kind="protected_source_region_held", cue_ids=list(span.cue_ids), severity="info",
        message="Preserved the later song captions at their original text and timing; the distinct earlier speech requires independent audio approval.",
        old_text=span.srt_text, start=span.start, end=span.end,
    )


_SONG_CAPTION_NOTE_KIND = "song_lyric_source_kept"


def _song_caption_cue_ids(cues: list[Cue]) -> set[int]:
    return {cue.index for cue in cues if is_song_caption_cue(cue)}


def _song_caption_note(cue_id: int, *, absent_from_voice_track: bool) -> QCFlag:
    return QCFlag(
        kind=_SONG_CAPTION_NOTE_KIND, cue_ids=[cue_id], severity="info",
        message=(
            "Song caption is not part of the voice track; its source text and timing were kept."
            if absent_from_voice_track
            else "Song caption kept its source text; audio inside its span is dialogue, not a new lyric."
        ),
    )


def _alignment_with_song_caption_guard(
    alignment: AlignmentResult, cues: list[Cue], words: list[Word],
) -> AlignmentResult:
    """Keep song captions out of adjudication and out of the error list.

    A voice-only dub contains no music. A caption without local speech is the
    expected case, not a missing-audio failure: it keeps its source text and
    timing (it stays in the protected missing-audio set) and is reported with
    one informational note instead of an error-level hold.
    """
    caption_ids = _song_caption_cue_ids(cues)
    if not caption_ids:
        return alignment
    unsung = caption_ids & set(alignment.diagnostics.missing_audio_cue_ids)
    noted: set[int] = set()
    flags: list[QCFlag] = []
    for flag in alignment.flags:
        if flag.kind == "missing_audio_timing_held" and flag.cue_ids and set(flag.cue_ids) <= unsung:
            for cue_id in flag.cue_ids:
                if cue_id not in noted:
                    noted.add(cue_id)
                    flags.append(_song_caption_note(cue_id, absent_from_voice_track=True))
            continue
        flags.append(flag)
    return alignment.model_copy(update={
        "divergence_spans": protect_song_captions(alignment.divergence_spans, cues, words),
        "flags": flags,
    })


def _song_caption_source_hold(
    span: DivergenceSpan, song_caption_cue_ids: set[int], missing_audio_cue_ids: set[int],
) -> tuple[AdjudicationDecision, list[QCFlag]] | None:
    """Never send a song caption's tokens to an adjudicator.

    Dialogue words next to a caption once replaced it ("♪Queria♪"). A source
    span that still touches a caption after the alignment guard cannot be
    divided safely, so its complete source text is kept.
    """
    caption_ids = [cue_id for cue_id in dict.fromkeys(span.cue_ids) if cue_id in song_caption_cue_ids]
    if not caption_ids or not span.srt_token_indices:
        return None
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text, confidence=1.0,
        reason="Song captions keep their source text; dialogue audio never rewrites a caption.",
    )
    # A caption absent from the voice track already carries its per-cue note.
    return decision, [
        _song_caption_note(cue_id, absent_from_voice_track=False)
        for cue_id in caption_ids if cue_id not in missing_audio_cue_ids
    ]


def _without_song_caption_silence_duplicates(flags: list[QCFlag]) -> list[QCFlag]:
    """A noted song caption is expected to be silent in a voice-only track."""
    noted = {cue_id for flag in flags if flag.kind == _SONG_CAPTION_NOTE_KIND for cue_id in flag.cue_ids}
    if not noted:
        return flags
    silence_kinds = {"dropped_line_candidate", "cue_without_speech_activity", "cue_on_silence"}
    return [
        flag for flag in flags
        if not (flag.kind in silence_kinds and flag.cue_ids and set(flag.cue_ids) <= noted)
    ]


def _missing_audio_source_hold(
    span: DivergenceSpan,
    missing_audio_cue_ids: set[int],
) -> tuple[AdjudicationDecision, QCFlag] | None:
    referenced_cue_ids = set(span.cue_ids)
    # Replacement anchors describe neighboring evidence, not additional source
    # text being edited. Only a pure insertion may belong inside either anchor.
    if not span.cue_ids and not span.srt_token_indices:
        if span.left_anchor_cue_id is not None:
            referenced_cue_ids.add(span.left_anchor_cue_id)
        if span.right_anchor_cue_id is not None:
            referenced_cue_ids.add(span.right_anchor_cue_id)
    protected_cue_ids = referenced_cue_ids & missing_audio_cue_ids
    if (
        not protected_cue_ids
        or (not span.srt_text.strip() and not span.asr_text.strip())
    ):
        return None
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt",
        final_text=span.srt_text,
        confidence=0.0,
        speaker=span.speaker_ids[0] if len(span.speaker_ids) == 1 else None,
        character="unknown",
        reason=(
            "The source cue had no trustworthy local speech evidence; source text and "
            "timing were retained instead of assigning unrelated audio."
        ),
    )
    flag = QCFlag(
        kind="missing_audio_source_cue_held",
        cue_ids=sorted(protected_cue_ids),
        message=(
            "Source-backed text was not sent to adjudication because its audio is "
            "missing or timing-inconsistent; retained the source cue for review."
        ),
        severity="error",
        old_text=span.srt_text,
        new_text=span.asr_text,
        start=span.start,
        end=span.end,
    )
    return decision, flag


def _restore_missing_audio_source_cues(
    rebuilt: list[Cue],
    source_cues: list[Cue],
    protected_cue_ids: set[int],
    *,
    reason: str = "missing_audio",
) -> tuple[list[Cue], list[QCFlag]]:
    if not protected_cue_ids:
        return list(rebuilt), []

    source_by_id = {
        cue.index: cue
        for cue in source_cues
        if cue.index in protected_cue_ids
    }
    restored_ids: set[int] = set()
    restored: list[Cue] = []
    for cue in rebuilt:
        source = source_by_id.get(cue.index)
        if source is None:
            restored.append(cue)
            continue
        # A low-confidence decision already preserves its own source token span.
        # Holding its timing must not erase an independently approved insertion
        # elsewhere in the same cue. Missing-audio holds remain fully verbatim.
        exact_source = (
            cue.with_timing(source.start_ms, source.end_ms)
            if reason in {"low_confidence", "timing_evidence"}
            else source.model_copy(update={
                "speaker_id": cue.speaker_id,
                "character": cue.character,
            })
        )
        restored.append(exact_source)
        # Report only what this pass really undid. A timing hold retains its
        # approved wording on purpose, and its timing is normally still the
        # source timing. A locked cue whose punctuation or line breaks alone
        # were touched downstream is put back without a review error.
        timing_restored = cue.start_ms != source.start_ms or cue.end_ms != source.end_ms
        wording_restored = (
            reason not in {"low_confidence", "timing_evidence"}
            and alphanumeric_signature(cue.text) != alphanumeric_signature(source.text)
        )
        if timing_restored or wording_restored:
            restored_ids.add(cue.index)

    present_ids = {cue.index for cue in restored}
    for cue in source_cues:
        if cue.index not in source_by_id or cue.index in present_ids:
            continue
        restored.append(cue.model_copy(deep=True))
        restored_ids.add(cue.index)

    if not restored_ids:
        return restored, []
    return restored, [
        QCFlag(
            kind=f"{reason}_source_cue_restored",
            cue_ids=sorted(restored_ids),
            message=(
                "An uncertain source cue was restored to its original timing; "
                "independently approved text edits were retained."
                if reason in {"low_confidence", "timing_evidence"}
                else "Protected song captions were restored to their exact source text and timing."
                if reason == "protected_region"
                else "An uncertain source cue was restored to its exact editorial text and "
                     "timing after downstream processing attempted to alter it."
            ),
            severity="error",
        )
    ]


def _set_adapter_episode_context(adapter: object, cues: list[Cue], *, words: list[Word] | None = None) -> None:
    setter = getattr(adapter, "set_episode_context", None)
    if callable(setter):
        setter(cues)
    word_setter = getattr(adapter, "set_episode_words", None)
    if words is not None and callable(word_setter):
        word_setter(words)


def _incomplete_source_hold(
    span: DivergenceSpan,
    max_duration: float,
) -> tuple[AdjudicationDecision, QCFlag] | None:
    if span.cue_ids or span.srt_token_indices:
        return None
    if not span.asr_word_indices and not span.asr_text.strip():
        return None
    duration = (
        span.end - span.start
        if span.start is not None
        and span.end is not None
        and isfinite(span.start)
        and isfinite(span.end)
        and span.start >= 0
        and span.end > span.start
        else None
    )
    if duration is not None and duration <= max_duration:
        return None
    timing_reason = (
        f"lasted {duration:.1f}s, above the {max_duration:g}s automatic ad-lib limit"
        if duration is not None
        else "had no valid timing for bounded automatic ad-lib generation"
    )
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt",
        final_text=span.srt_text,
        confidence=0.0,
        speaker=span.speaker_ids[0] if len(span.speaker_ids) == 1 else None,
        character="unknown",
        reason=(
            "ASR-only dialogue exceeded a safe automatic ad-lib bound; the source "
            "subtitles may be incomplete, so no text was generated."
        ),
    )
    flag = QCFlag(
        kind="generated_adlib_rejected_incomplete_source",
        cue_ids=[],
        message=(
            f"An ASR-only span {timing_reason}. The source SRT may be incomplete; "
            "supply the full customer subtitles instead of auto-generating an episode section."
        ),
        severity="error",
        old_text=span.srt_text,
        new_text=span.asr_text,
        start=span.start,
        end=span.end,
    )
    return decision, flag


def _unresolved_alignment_adjudication_hold(
    span: DivergenceSpan,
) -> tuple[AdjudicationDecision, QCFlag] | None:
    if not span.cue_ids and not span.srt_token_indices:
        return None
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt",
        final_text=span.srt_text,
        confidence=0.0,
        speaker=span.speaker_ids[0] if len(span.speaker_ids) == 1 else None,
        character="unknown",
        reason=(
            "Alignment was unresolved within the bounded cell budget; source text "
            "was retained instead of permitting an unanchored adjudication rewrite."
        ),
    )
    flag = QCFlag(
        kind="unresolved_alignment_adjudication_held",
        cue_ids=span.cue_ids,
        message=(
            "Source-backed divergence was not sent to adjudication because alignment "
            "was unresolved; retained source text for manual review."
        ),
        severity="error",
        old_text=span.srt_text,
        new_text=span.asr_text,
        start=span.start,
        end=span.end,
    )
    return decision, flag


def _oversized_adjudication_span_hold(
    span: DivergenceSpan,
    source_cue_count: int | None,
) -> tuple[AdjudicationDecision, QCFlag] | None:
    if source_cue_count is None or source_cue_count < 25 or not span.cue_ids:
        return None
    cue_count = len(set(span.cue_ids))
    ceiling = max(ceil(source_cue_count * 0.20), 5)
    if cue_count <= ceiling:
        return None
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt",
        final_text=span.srt_text,
        confidence=0.0,
        speaker=span.speaker_ids[0] if len(span.speaker_ids) == 1 else None,
        character="unknown",
        reason=(
            "Source-backed divergence covered too many cues for safe automatic "
            "adjudication; source text was retained for review."
        ),
    )
    flag = QCFlag(
        kind="oversized_adjudication_span_held",
        cue_ids=span.cue_ids,
        message=(
            f"Divergence span covered {cue_count} of {source_cue_count} source cues, "
            f"above the automatic adjudication ceiling of {ceiling}; retained source "
            "text instead of sending a collapsed episode section to adjudication."
        ),
        severity="error",
        start=span.start,
        end=span.end,
    )
    return decision, flag


def _apply_incomplete_source_holds_to_decisions(
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    decisions: list[AdjudicationDecision],
    adjudication_flags: list[QCFlag],
    *,
    source_cue_count: int | None = None,
    alignment_unresolved: bool = False,
    missing_audio_cue_ids: set[int] | None = None,
    protected_source_regions: dict[str, set[int]] | None = None,
    song_caption_cue_ids: set[int] | None = None,
) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    _, held_decisions, incomplete_source_flags = _hold_incomplete_source_insertions(
        spans,
        provider_config,
        source_cue_count=source_cue_count,
        alignment_unresolved=alignment_unresolved,
        missing_audio_cue_ids=missing_audio_cue_ids,
        protected_source_regions=protected_source_regions,
        song_caption_cue_ids=song_caption_cue_ids,
    )
    if not held_decisions:
        return decisions, adjudication_flags

    held_by_case = {decision.case_id: decision for decision in held_decisions}
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    decisions_by_case.update(held_by_case)
    ordered_decisions = [
        decisions_by_case[span.case_id]
        for span in spans
        if span.case_id in decisions_by_case
    ]
    retained_flags = [
        flag
        for flag in adjudication_flags
        if flag.kind
        not in {
            "generated_adlib_rejected_incomplete_source",
            "oversized_adjudication_span_held",
            "unresolved_alignment_adjudication_held",
            "missing_audio_source_cue_held",
            "protected_source_region_held",
            _SONG_CAPTION_NOTE_KIND,
        }
    ]
    return ordered_decisions, [*retained_flags, *incomplete_source_flags]


def _unsafe_incomplete_source_resume_case_ids(
    spans: list[DivergenceSpan],
    provider_config: dict[str, object],
    decisions: list[AdjudicationDecision],
    rebuilt: list[Cue],
    source_cues: list[Cue],
    *,
    alignment_unresolved: bool = False,
    missing_audio_cue_ids: set[int] | None = None,
    protected_source_regions: dict[str, set[int]] | None = None,
) -> list[str]:
    _, held_decisions, _ = _hold_incomplete_source_insertions(
        spans,
        provider_config,
        source_cue_count=_spoken_source_cue_count(source_cues),
        alignment_unresolved=alignment_unresolved,
        missing_audio_cue_ids=missing_audio_cue_ids,
        protected_source_regions=protected_source_regions,
        song_caption_cue_ids=_song_caption_cue_ids(source_cues),
    )
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    spans_by_case = {span.case_id: span for span in spans}
    source_ids = {cue.index for cue in source_cues}
    unsafe_cases: list[str] = []
    for held in held_decisions:
        decision = decisions_by_case.get(held.case_id)
        span = spans_by_case.get(held.case_id)
        stale_decision = (
            decision is None
            or decision.verdict != "keep_srt"
        )
        stale_rebuild = (
            span is not None
            and _rebuilt_contains_generated_span(rebuilt, source_ids, span)
        )
        if held.case_id in (protected_source_regions or {}):
            source_by_id = {cue.index: cue for cue in source_cues}
            rebuilt_by_id = {cue.index: cue for cue in rebuilt}
            stale_decision = stale_decision or decision.final_text != held.final_text
            stale_rebuild = any(
                cue_id not in rebuilt_by_id
                or (rebuilt_by_id[cue_id].lines, rebuilt_by_id[cue_id].start_ms, rebuilt_by_id[cue_id].end_ms)
                != (source_by_id[cue_id].lines, source_by_id[cue_id].start_ms, source_by_id[cue_id].end_ms)
                for cue_id in protected_source_regions[held.case_id]
            )
        if stale_decision or stale_rebuild:
            unsafe_cases.append(held.case_id)
    return unsafe_cases


def _rebuilt_contains_generated_span(
    rebuilt: list[Cue],
    source_ids: set[int],
    span: DivergenceSpan,
) -> bool:
    generated = [cue for cue in rebuilt if cue.index not in source_ids]
    if span.start is None or span.end is None or span.end <= span.start:
        return bool(generated)
    span_start_ms = round(span.start * 1000)
    span_end_ms = round(span.end * 1000)
    return any(
        cue.end_ms > span_start_ms and cue.start_ms < span_end_ms
        for cue in generated
    )


def _adlib_cue_ids_by_case(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    unmatched_cue_ids: list[int],
) -> tuple[dict[str, int], list[QCFlag]]:
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    cues_by_id = {cue.index: cue for cue in cues}
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
        rejection_flag = _generated_adlib_rejection_flag(cues, span, decision.final_text)
        if rejection_flag is not None:
            flags.append(rejection_flag)
            continue
        anchored_cue_id = _anchored_adlib_cue_id(cues_by_id, span, decision.final_text)
        if anchored_cue_id is not None:
            cue_ids[span.case_id] = anchored_cue_id
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


def _absorb_redecoded_insertions(
    alignment: AlignmentResult, decisions: list[AdjudicationDecision], words: list[Word],
) -> tuple[AlignmentResult, list[AdjudicationDecision]]:
    """Never insert words that only repeat the adjacent owned words at the same time.

    A provider can decode one utterance twice ('manda umas flores. Manda umas
    flores', "Você..." / "Você..." with overlapping timestamps). The aligner
    gives a single touching copy back to the cue of its exactly matched twin;
    a copied phrase, or a copy left as its own case, reached the adjudicator
    and came back as approved new dialogue. Such a copy is an ASR artefact,
    not speech: nothing is inserted or reported, and its words time the cue
    that owns the twin, because both copies are that cue's utterance.
    """
    owners: dict[int, set[int]] = {}
    for cue_id, indices in alignment.cue_word_indices.items():
        for index in indices:
            owners.setdefault(index, set()).add(cue_id)
    protected = set(alignment.diagnostics.missing_audio_cue_ids)
    by_case = {decision.case_id: decision for decision in decisions}
    cue_word_indices = {cue_id: list(indices) for cue_id, indices in alignment.cue_word_indices.items()}
    dropped: dict[str, AdjudicationDecision] = {}
    absorbed = False
    for span in alignment.divergence_spans:
        indices = span.asr_word_indices
        if (
            span.cue_ids or span.srt_token_indices or not indices
            or indices != list(range(indices[0], indices[-1] + 1))
            or indices[0] < 0 or indices[-1] >= len(words)
        ):
            continue
        keys = [normalize_token(words[index].text) for index in indices]
        decision = by_case.get(span.case_id)
        approved = decision is not None and decision.verdict != "keep_srt" and bool(decision.final_text.strip())
        if not all(keys) or (
            approved and alphanumeric_signature(decision.final_text) != alphanumeric_signature(span.asr_text)
        ):
            continue
        length = len(indices)
        # The case and the owned words around it are two touching copies of
        # one phrase; alignment may have matched any part of either copy.
        for first in range(indices[0] - length, indices[0] + 1):
            if first < 0 or first + 2 * length > len(words):
                continue
            twin = [index for index in range(first, first + 2 * length) if not indices[0] <= index <= indices[-1]]
            twin_owners = [owners.get(index, set()) for index in twin]
            if (
                any(len(owner) != 1 for owner in twin_owners)
                or len(set().union(*twin_owners)) != 1
                or any(
                    normalize_token(words[index].text) != normalize_token(words[index + length].text)
                    for index in range(first, first + length)
                )
                or not _words_touch(words[first + length - 1], words[first + length])
            ):
                continue
            (owner,) = twin_owners[0]
            if owner not in protected:
                cue_word_indices[owner] = sorted({*cue_word_indices.get(owner, []), *indices})
                absorbed = True
            if approved:
                dropped[span.case_id] = decision.model_copy(update={
                    "verdict": "keep_srt", "final_text": span.srt_text,
                    "reason": (
                        "The words repeat the adjacent words at the same time (the provider decoded one "
                        f"utterance twice); nothing was inserted. Proposed {decision.verdict}: {decision.final_text!r}."
                    ),
                })
            break
    if not absorbed and not dropped:
        return alignment, decisions
    return (
        alignment.model_copy(update={"cue_word_indices": cue_word_indices}),
        [dropped.get(decision.case_id, decision) for decision in decisions],
    )


def _reconciled_cue_ids(flags: list[QCFlag]) -> set[int]:
    return {cue_id for flag in flags if flag.kind == "adlib_reconciled" for cue_id in flag.cue_ids}


def _release_reconciled_cues(
    alignment: AlignmentResult, flags: list[QCFlag],
) -> tuple[AlignmentResult, list[QCFlag]]:
    """A source cue that received its own spoken words is no missing-audio hold.

    Lines spoken in another order than written leave one cue unmatched at its
    source position while its words appear elsewhere as an insertion. Once the
    accepted insertion reused that cue, the cue has acoustic evidence: it
    leaves the protected and unmatched sets, is timed from its words and is
    ordered by time. Verify repeats this from the persisted reconciliation
    flags, so a resumed run cannot restore the stale source timing.
    """
    released = _reconciled_cue_ids(flags) & (
        set(alignment.diagnostics.missing_audio_cue_ids) | set(alignment.unmatched_cue_ids)
    )
    if not released:
        return alignment, flags
    kept: list[QCFlag] = []
    for flag in flags:
        if flag.kind in {"missing_audio_timing_held", "missing_audio_source_cue_held"} and released.intersection(flag.cue_ids):
            remaining = [cue_id for cue_id in flag.cue_ids if cue_id not in released]
            if not remaining:
                continue
            flag = flag.model_copy(update={"cue_ids": remaining})
        kept.append(flag)
    return alignment.model_copy(update={
        "unmatched_cue_ids": [cue_id for cue_id in alignment.unmatched_cue_ids if cue_id not in released],
        "diagnostics": alignment.diagnostics.model_copy(update={
            "missing_audio_cue_ids": [
                cue_id for cue_id in alignment.diagnostics.missing_audio_cue_ids if cue_id not in released
            ],
        }),
    }), kept


def _validate_inline_adlib_ownership(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    decisions: list[AdjudicationDecision],
    adlib_cue_ids_by_case: dict[str, int],
    profile: StyleProfile,
    *,
    protected_cue_ids: set[int] | None = None,
    max_intra_cue_gap: float = 1.5,
) -> tuple[dict[str, int], list[QCFlag]]:
    """Keep cross-actor inline additions only when the complete cue can split.

    A clear insertion can coexist with a decision to retain uncertain source
    wording. Probe the same pure transformations used below so that an exact
    insertion alone cannot promise speaker ownership for a nonexact whole cue.
    """
    result = dict(adlib_cue_ids_by_case)
    cues_by_id = {cue.index: cue for cue in cues}
    candidates = [
        span for span in alignment.divergence_spans
        if span.case_id in result
        and not span.cue_ids
        and span.left_anchor_cue_id == span.right_anchor_cue_id == result[span.case_id]
        and result[span.case_id] in cues_by_id
        and speakers_known_different(span.left_anchor_speaker_id, span.right_anchor_speaker_id)
    ]
    if not candidates:
        return result, []
    probe_alignment = _alignment_with_decision_words(
        alignment, decisions, alignment.divergence_spans, result, source_cues=cues, words=words,
        protected_cue_ids=protected_cue_ids,
        max_intra_cue_gap=max_intra_cue_gap,
    )
    probe_cues, _ = apply_adjudication_decisions(
        cues, alignment.divergence_spans, decisions, profile, result,
        protected_cue_ids=protected_cue_ids,
        words=words,
        max_intra_cue_gap=max_intra_cue_gap,
        token_matches=alignment.token_matches,
    )
    _, _, _, expansions = split_speaker_turn_cues(
        probe_cues, words, probe_alignment, profile, protected_cue_ids=protected_cue_ids,
    )
    next_cue_id = max([*cues_by_id, *result.values()], default=0) + 1
    flags: list[QCFlag] = []
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    for span in candidates:
        source_id = result[span.case_id]
        if source_id in expansions:
            continue
        result[span.case_id] = next_cue_id
        flags.append(QCFlag(
            kind="adlib_speaker_ownership_held",
            cue_ids=[source_id, next_cue_id], severity="warning",
            message=(
                "The complete retained source cue could not be safely separated into actor turns. "
                "Recognized inserted speech remains a separate cue for review instead of being "
                "attached to uncertain speaker ownership."
            ),
            old_text=cues_by_id[source_id].text,
            new_text=decisions_by_case[span.case_id].final_text,
            start=span.start, end=span.end,
        ))
        next_cue_id += 1
    return result, flags


def _generated_adlib_rejection_flag(
    source_cues: list[Cue],
    span: DivergenceSpan,
    final_text: str,
    source_margin_seconds: float = 5.0,
) -> QCFlag | None:
    if _is_repetitive_generated_text(final_text):
        return QCFlag(
            kind="adlib_rejected_repetitive_content",
            cue_ids=[],
            message="ASR-only text was held for review because it is highly repetitive and may be music or non-dialogue audio.",
            severity="error",
            new_text=final_text,
            start=span.start,
            end=span.end,
        )
    if not source_cues or (span.start is None and span.end is None):
        return None
    source_start = min(cue.start_ms for cue in source_cues) / 1000.0
    source_end = max(cue.end_ms for cue in source_cues) / 1000.0
    span_start = span.start if span.start is not None else span.end
    span_end = span.end if span.end is not None else span.start
    assert span_start is not None and span_end is not None
    if span_end >= source_start - source_margin_seconds and span_start <= source_end + source_margin_seconds:
        return None
    return QCFlag(
        kind="adlib_rejected_outside_source_span",
        cue_ids=[],
        message="ASR-only text was held for review because it falls outside the source subtitle span and safety margin.",
        severity="error",
        new_text=final_text,
        start=span.start,
        end=span.end,
    )


def _is_repetitive_generated_text(text: str) -> bool:
    tokens = alphanumeric_signature(text)
    if len(tokens) < 12 or len(set(tokens)) / len(tokens) > 0.35:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    return len(set(trigrams)) < len(trigrams)


# Word times are decimal seconds: 234.84 - 234.64 must count as a 0.2 s gap.
_GAP_EPSILON_SECONDS = 1e-6


def _anchored_adlib_cue_id(
    cues_by_id: dict[int, Cue],
    span: DivergenceSpan,
    final_text: str,
    max_gap_seconds: float = 0.2,
    max_continuation_gap_seconds: float = 1.0,
) -> int | None:
    left_id = span.left_anchor_cue_id
    right_id = span.right_anchor_cue_id
    # Dialogue spoken over a song is its own cue; it never joins the caption.
    if left_id in cues_by_id and is_song_caption_cue(cues_by_id[left_id]):
        left_id = None
    if right_id in cues_by_id and is_song_caption_cue(cues_by_id[right_id]):
        right_id = None
    # One existing cue cannot own an insertion spoken by multiple actors.
    # Keep it generated so the word-aware segmentation stage can split turns.
    # Labels of unrelated scopes (two MAI chunks) do not prove several actors.
    if has_known_different_speakers(span.speaker_ids):
        return None
    left_speaker = span.left_anchor_speaker_id
    right_speaker = span.right_anchor_speaker_id
    if left_speaker is None and left_id in cues_by_id:
        left_speaker = cues_by_id[left_id].speaker_id
    if right_speaker is None and right_id in cues_by_id:
        right_speaker = cues_by_id[right_id].speaker_id
    if (
        left_id is not None
        and left_id == right_id
        and left_id in cues_by_id
        and span.insertion_token_offset is not None
    ):
        if speakers_known_different(span.left_anchor_speaker_id, span.right_anchor_speaker_id):
            # A source cue may already contain two actors. A word-anchored
            # insertion can complete one actor's clause at that transition;
            # the mandatory speaker splitter then separates the two turns.
            cue = cues_by_id[left_id]
            indices = span.asr_word_indices
            if (
                len(set(span.speaker_ids)) == 1
                and span.speaker_ids[0] in {span.left_anchor_speaker_id, span.right_anchor_speaker_id}
                and indices and indices[0] >= 0
                and all(current == previous + 1 for previous, current in zip(indices, indices[1:]))
                and alphanumeric_signature(final_text)
                and alphanumeric_signature(final_text) == alphanumeric_signature(span.asr_text)
                and 0 < span.insertion_token_offset < len(alphanumeric_signature(cue.plain_text))
                and all(value is not None and isfinite(value) for value in (
                    span.start, span.end, span.left_anchor_end, span.right_anchor_start,
                ))
                and cue.start_ms / 1000 - 1.5 <= span.left_anchor_end <= span.start
                and span.start < span.end <= span.right_anchor_start <= cue.end_ms / 1000 + 1.5
            ):
                return left_id
            return None
        if (
            _anchor_speaker_is_compatible(span.speaker_ids, left_speaker)
            and _anchor_speaker_is_compatible(span.speaker_ids, right_speaker)
        ):
            return left_id

    if (
        right_id is not None
        and right_id in cues_by_id
        and len(alphanumeric_signature(final_text)) <= 3
        and span.end is not None
        and span.right_anchor_start is not None
        and _anchor_speaker_is_compatible(span.speaker_ids, right_speaker)
        and (
            span.left_anchor_end is None
            or span.start is None
            or span.start >= span.left_anchor_end - 0.05
        )
    ):
        gap = span.right_anchor_start - span.end
        if -0.05 <= gap <= max_gap_seconds + _GAP_EPSILON_SECONDS:
            return right_id
        if (
            0 <= gap <= max_continuation_gap_seconds
            and re.search(r"[,;:]\s*$", final_text)
            and _anchor_speaker_is_confirmed(
                span.speaker_ids,
                right_speaker,
            )
        ):
            return right_id

    if (
        left_id is not None
        and left_id in cues_by_id
        and len(alphanumeric_signature(final_text)) <= 3
        and span.start is not None
        and span.left_anchor_end is not None
        and _anchor_speaker_is_compatible(span.speaker_ids, left_speaker)
    ):
        gap = span.start - span.left_anchor_end
        left_has_terminal_punctuation = bool(
            re.search(r"[.!?\u2026]\s*$", cues_by_id[left_id].plain_text)
        )
        narrow_confirmed_continuation = (
            len(alphanumeric_signature(final_text)) == 1
            and -0.05 <= gap <= 0.05
            and _anchor_speaker_is_confirmed(
                span.speaker_ids,
                left_speaker,
            )
        )
        if (
            -0.05 <= gap <= max_gap_seconds + _GAP_EPSILON_SECONDS
            and (
                not left_has_terminal_punctuation
                or narrow_confirmed_continuation
            )
        ):
            return left_id
    return None


def _anchor_speaker_is_compatible(speaker_ids: list[str], anchor_speaker_id: str | None) -> bool:
    # Only a provably different actor is incompatible; a label from another
    # chunk scope has an unknown relation to the anchor.
    return (
        not speaker_ids or anchor_speaker_id is None or anchor_speaker_id in speaker_ids
        or not any(speakers_known_different(speaker_id, anchor_speaker_id) for speaker_id in speaker_ids)
    )


def _anchor_speaker_is_confirmed(speaker_ids: list[str], anchor_speaker_id: str | None) -> bool:
    return anchor_speaker_id is not None and set(speaker_ids) == {anchor_speaker_id}


def _reconciled_adlib_source_cue(
    cues: list[Cue],
    candidate_ids: set[int],
    span: DivergenceSpan,
    final_text: str,
) -> Cue | None:
    candidates = [cue for cue in cues if cue.index in candidate_ids and not is_song_caption_cue(cue)]
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


def _alignment_with_decision_words(
    alignment, decisions, spans, adlib_cue_ids_by_case=None, *, source_cues=None, words=None,
    protected_cue_ids=None, max_intra_cue_gap=1.5, held_case_ids=None,
):
    # ``held_case_ids`` collects every decided case whose word mapping is held,
    # so the caller holds its text edit too instead of showing new words at
    # the old source time.
    timed_decisions = {
        decision.case_id: decision
        for decision in decisions
        if decision.verdict in {"keep_srt", "use_audio", "hybrid"}
    }
    if not timed_decisions and not any(is_joint_region(span) for span in spans):
        return alignment

    adlib_cue_ids_by_case = adlib_cue_ids_by_case or {}
    prefix_replacement_targets = single_token_prefix_replacement_targets(
        source_cues or [], spans, decisions
    )
    external_target_protection = set(protected_cue_ids or ()) | set(alignment.diagnostics.missing_audio_cue_ids)
    # Confidence holds apply to their own spans. Retain existing evidence for
    # independent accepted edits while protecting new neighboring destinations.
    protected_cue_ids = set(alignment.diagnostics.missing_audio_cue_ids)
    cue_word_indices = {cue_id: list(indices) for cue_id, indices in alignment.cue_word_indices.items()}
    mapping_flags = list(alignment.flags)

    def hold_mapping(flag: QCFlag) -> None:
        mapping_flags.append(flag)
        if held_case_ids is not None:
            held_case_ids.add(span.case_id)

    for span in spans:
        decision = timed_decisions.get(span.case_id)
        if decision is None:
            if is_joint_region(span):
                mapping_flags.append(QCFlag(
                    kind="adjudication_word_mapping_held", cue_ids=list(span.cue_ids), severity="warning",
                    message="The joint region has no fresh decision; its source text and timing were preserved.",
                    old_text=span.srt_text, new_text=span.asr_text, start=span.start, end=span.end,
                ))
            continue
        if is_joint_region(span) and (
            decision.verdict == "keep_srt" or not source_cues or words is None
            or set(span.cue_ids) & protected_cue_ids
        ):
            hold_mapping(QCFlag(
                kind="adjudication_word_mapping_held", cue_ids=list(span.cue_ids), severity="warning",
                message="The joint region was not approved with complete word evidence; its source text and timing were preserved.",
                confidence=decision.confidence, old_text=span.srt_text,
                new_text=decision.final_text, start=span.start, end=span.end,
            ))
            continue
        if decision.verdict in {"use_audio", "hybrid"} and not decision.final_text.strip() and not is_joint_region(span):
            # Rejected acoustic words cannot time the retained source residue.
            # Remove prior evidence only when the exact source deletion is valid.
            edits = indexed_multi_cue_replacements(source_cues, span, "") if source_cues else None
            if edits is not None:
                rejected_indices = set(span.asr_word_indices)
                for cue_id in edits:
                    if cue_id not in protected_cue_ids:
                        cue_word_indices[cue_id] = [
                            index for index in cue_word_indices.get(cue_id, [])
                            if index not in rejected_indices
                        ]
            continue
        adlib_cue_id = adlib_cue_ids_by_case.get(span.case_id)
        if adlib_cue_id is not None:
            cue_word_indices[adlib_cue_id] = sorted(
                set(cue_word_indices.get(adlib_cue_id, []) + span.asr_word_indices)
            )
            continue
        replacement_target = prefix_replacement_targets.get(span.case_id)
        word_indices_by_cue = _span_word_indices_by_cue(span, replacement_target=replacement_target)
        edits = None
        whole_plan = None
        # Filled when the text pieces were placed from word timing: the same
        # phrases then time the cues that show them.
        placed_word_indices: dict[int, list[int]] = {}
        if source_cues and decision.verdict in {"use_audio", "hybrid"}:
            try:
                whole_plan = whole_cue_replacement_plan(
                    source_cues, span, decision.final_text, words, max_intra_cue_gap=max_intra_cue_gap,
                    token_matches=alignment.token_matches,
                )
            except ReplacementOwnershipError as exc:
                hold_mapping(QCFlag(
                    kind="adjudication_word_mapping_held", cue_ids=list(span.cue_ids),
                    severity="warning", message=str(exc), confidence=decision.confidence,
                    old_text=span.asr_text, new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
        if whole_plan is not None:
            edits = whole_plan.edits
        if (
            whole_plan is None and source_cues
            and decision.verdict in {"use_audio", "hybrid"}
            and (decision.final_text.strip() or is_joint_region(span))
            and (
                len(set(span.cue_ids)) > 1
                or (span.right_anchor_cue_id is not None and span.right_anchor_cue_id not in span.cue_ids)
            )
            and span.srt_token_indices
        ):
            try:
                edits = indexed_multi_cue_replacements(
                    source_cues, span, decision.final_text, replacement_target=replacement_target, words=words,
                    ownership=placed_word_indices,
                )
                if edits is None and is_joint_region(span):
                    raise ReplacementOwnershipError("The joint source region could not be reconstructed; source evidence was held for review.")
            except ReplacementOwnershipError as exc:
                hold_mapping(QCFlag(
                    kind="adjudication_word_mapping_held",
                    cue_ids=list(span.cue_ids),
                    severity="warning", message=str(exc), confidence=decision.confidence,
                    old_text=span.asr_text, new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
            if edits is None and len(set(span.cue_ids)) > 1:
                continue
        if edits is not None:
            if protected_replacement_targets(span, edits) & external_target_protection:
                hold_mapping(QCFlag(
                    kind="adjudication_word_mapping_held", cue_ids=list(edits), severity="warning",
                    message=(
                        f"Adjudication {span.case_id} would transfer a word into a protected source cue. "
                        "The complete replacement and its existing evidence ownership need review."
                    ),
                    confidence=decision.confidence, old_text=span.asr_text,
                    new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
            mapped_indices = (
                whole_plan.word_indices_by_cue if whole_plan is not None
                else placed_word_indices or _indexed_replacement_word_indices(span, edits, words=words)
            )
            if mapped_indices is None:
                hold_mapping(QCFlag(
                    kind="adjudication_word_mapping_held", cue_ids=list(edits), severity="warning",
                    message=(
                        f"Adjudication {span.case_id} has no unique acoustic word boundary supported "
                        "by retained lexical anchors or sentence separators. Existing evidence ownership "
                        "and source text were preserved; the proposed wording needs review."
                    ),
                    confidence=decision.confidence, old_text=span.asr_text,
                    new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
            word_indices_by_cue = mapped_indices
            span_word_indices = set(span.asr_word_indices)
            for cue_id in edits:
                if cue_id not in protected_cue_ids:
                    cue_word_indices[cue_id] = [
                        index for index in cue_word_indices.get(cue_id, []) if index not in span_word_indices
                    ]
        for cue_id, spoken_indices in word_indices_by_cue.items():
            if cue_id in protected_cue_ids:
                continue
            if decision.verdict == "keep_srt":
                existing = cue_word_indices.get(cue_id, [])
                if not existing:
                    continue
            combined = sorted(set(cue_word_indices.get(cue_id, []) + spoken_indices))
            if combined:
                cue_word_indices[cue_id] = combined
    return alignment.model_copy(update={"cue_word_indices": cue_word_indices, "flags": mapping_flags})


def _indexed_replacement_word_indices(
    span: DivergenceSpan,
    edits: dict[int, tuple[int, int, str]],
    *,
    words: list[Word] | None = None,
) -> dict[int, list[int]] | None:
    """Map replacement cuts through lexical edits, never through token ratios.

    A changed phrase can contain a different number of tokens than its ASR
    rendering. Only cuts with a unique optimal lexical alignment and a nearby
    exact anchor or matching sentence separator can transfer word ownership.
    Every cut must also lie between complete ASR words.
    """
    if not span.asr_word_indices:
        return {cue_id: [] for cue_id in edits}
    if len(set(span.asr_word_indices)) != len(span.asr_word_indices):
        return None
    if words is None:
        word_texts = span.asr_text.split()
        if len(word_texts) != len(span.asr_word_indices):
            return None
        confidences = [span.confidence] * len(word_texts)
    else:
        if any(index < 0 or index >= len(words) for index in span.asr_word_indices):
            return None
        word_texts = [words[index].text for index in span.asr_word_indices]
        confidences = [words[index].confidence for index in span.asr_word_indices]

    asr_tokens: list[str] = []
    token_confidences: list[float | None] = []
    word_boundaries = {0: 0}
    for position, (text, confidence) in enumerate(zip(word_texts, confidences, strict=True)):
        signature = alphanumeric_signature(text)
        if not signature:
            return None
        asr_tokens.extend(signature)
        token_confidences.extend([confidence] * len(signature))
        word_boundaries[len(asr_tokens)] = position + 1
    if asr_tokens != alphanumeric_signature(span.asr_text):
        return None

    final_tokens: list[str] = []
    cuts: list[tuple[int, int, bool]] = []
    for cue_id, (_, _, text) in edits.items():
        final_tokens.extend(alphanumeric_signature(text))
        cuts.append((cue_id, len(final_tokens), _ends_with_sentence_separator(text)))
    if not final_tokens:
        return None

    exact_text = final_tokens == asr_tokens
    forward = [] if exact_text else lexical_edit_costs(final_tokens, asr_tokens)
    backward = [] if exact_text else lexical_edit_costs(final_tokens[::-1], asr_tokens[::-1])
    total_cost = 0 if exact_text else forward[-1][-1]
    result: dict[int, list[int]] = {}
    previous_word_boundary = 0
    previous_asr_boundary = 0
    previous_final_boundary = 0
    for cue_id, final_boundary, sentence_boundary in cuts:
        if final_boundary == 0:
            candidates = [0]
        elif final_boundary == len(final_tokens):
            candidates = [len(asr_tokens)]
        elif exact_text:
            candidates = [final_boundary] if final_boundary in word_boundaries else []
        else:
            candidates = []
            separator_candidates = []
            for asr_boundary, word_boundary in word_boundaries.items():
                if (
                    not 0 < asr_boundary < len(asr_tokens)
                    or forward[final_boundary][asr_boundary]
                    + backward[len(final_tokens) - final_boundary][len(asr_tokens) - asr_boundary]
                    != total_cost
                ):
                    continue
                left_anchor = (
                    final_tokens[final_boundary - 1] == asr_tokens[asr_boundary - 1]
                    and anchor_confidence_is_acceptable(token_confidences[asr_boundary - 1])
                )
                right_anchor = (
                    final_tokens[final_boundary] == asr_tokens[asr_boundary]
                    and anchor_confidence_is_acceptable(token_confidences[asr_boundary])
                )
                matching_separator = sentence_boundary and _ends_with_sentence_separator(word_texts[word_boundary - 1])
                if left_anchor or right_anchor or matching_separator:
                    candidates.append(asr_boundary)
                if matching_separator:
                    separator_candidates.append(asr_boundary)
            if separator_candidates:
                candidates = separator_candidates
        if len(candidates) != 1:
            return None
        word_boundary = word_boundaries[candidates[0]]
        if word_boundary < previous_word_boundary:
            return None
        word_start, word_end = previous_word_boundary, word_boundary
        part_tokens = final_tokens[previous_final_boundary:final_boundary]
        if part_tokens:
            # An exact retained phrase can exclude ASR-only context inside an
            # otherwise valid partition. Do not lend its neighbors' timestamps
            # to the retained phrase, or choose among repeated exact windows.
            exact_windows = [
                (start, start + len(part_tokens))
                for start in range(previous_asr_boundary, candidates[0] - len(part_tokens) + 1)
                if asr_tokens[start:start + len(part_tokens)] == part_tokens
            ]
            if exact_windows:
                if (
                    len(exact_windows) != 1
                    or exact_windows[0][0] not in word_boundaries
                    or exact_windows[0][1] not in word_boundaries
                ):
                    return None
                word_start, word_end = (word_boundaries[position] for position in exact_windows[0])
        result[cue_id] = span.asr_word_indices[word_start:word_end]
        previous_word_boundary = word_boundary
        previous_asr_boundary = candidates[0]
        previous_final_boundary = final_boundary
    return result


def _ends_with_sentence_separator(text: str) -> bool:
    return re.search(r"[.!?\u2026][\"'\u2019\u201d\u00bb\)\]]*\s*$", text) is not None


def _span_word_indices_by_cue(
    span: DivergenceSpan,
    *,
    replacement_target: int | None = None,
) -> dict[int, list[int]]:
    cue_ids = list(dict.fromkeys(span.cue_ids))
    if not cue_ids:
        return {}
    if replacement_target in cue_ids:
        return {replacement_target: list(span.asr_word_indices)}
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
