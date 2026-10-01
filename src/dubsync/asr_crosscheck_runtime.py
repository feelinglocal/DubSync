"""Opt-in secondary transcription, with independent cache and saved evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .adjudication_policy import DeterministicAdjudicationPolicy
from .asr_crosscheck import SpanCrossCheck, enforce_crosscheck_decisions, preaccepted_decision
from .asr_crosscheck_config import cross_check_context
from .audio import AudioNormalizationLimits, normalize_audio
from .cache import JsonDiskCache, _sha256_file, write_json_atomic, write_text_atomic
from .cost import CostMeter, asr_dollars_per_hour
from .models import AdjudicationDecision, DivergenceSpan, QCFlag, Word
from .providers import ASRAdapter, CachedASRAdapter, ProviderError, adapter_from_config, repair_word_stream


@dataclass(frozen=True)
class PreparedCrossCheck:
    config: dict[str, object]
    adapter: ASRAdapter | None


def prepare_cross_check(config: dict[str, object], *, resume: bool) -> PreparedCrossCheck:
    """Validate local settings before primary spend; resumes never build a provider."""
    asr = dict(config["asr"])
    fixture = asr.get("fixture_path")
    if fixture:
        try:
            path = Path(str(fixture))
            payload = json.loads(path.read_text(encoding="utf-8"))
            repair_word_stream(payload.get("words", payload) if isinstance(payload, dict) else payload,
                               source="ASR cross-check fixture")
            asr["fixture_sha256"] = _sha256_file(path)
        except (OSError, ValueError, ProviderError) as exc:
            if resume:
                raise ValueError("Cannot resume ASR cross-check with a missing or invalid fixture; resume from asr.") from exc
            raise ValueError("ASR cross-check fixture is missing or invalid.") from exc
    prepared_config = {"asr": asr}
    adapter = None if resume else adapter_from_config(prepared_config)
    if adapter is not None and not fixture and not getattr(adapter, "api_key", None):
        key = "ELEVENLABS_API_KEY" if asr["provider"] == "elevenlabs" else "OPENROUTER_API_KEY"
        raise ProviderError(f"{key} is required for the selected ASR cross-check.", code="configuration")
    return PreparedCrossCheck(prepared_config, adapter)


def run_cross_check(
    prepared: PreparedCrossCheck, *, source_audio: Path, audio_for_asr: Path,
    episode_workdir: Path, cost_meter: CostMeter, resume: bool,
    audio_limits: AudioNormalizationLimits | None = None,
) -> tuple[list[Word], dict[str, object]]:
    """Transcribe once or require a matching saved artifact, without fallback."""
    path = episode_workdir / "asr_cross_check.json"
    if resume:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            words = [Word.model_validate(item) for item in payload["words"]]
            context = cross_check_context(words, prepared.config)
            metadata = payload["metadata"]
            provenance = metadata["audio_provenance"]
            saved_audio = episode_workdir / "audio.16k.wav" if provenance["normalized"] else source_audio
            valid = (metadata["context"] == context
                     and provenance["source_sha256"] == _sha256_file(source_audio)
                     and provenance["asr_input_sha256"] == _sha256_file(saved_audio))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Cannot resume ASR cross-check with missing or invalid secondary evidence; resume from asr.") from exc
        if not valid:
            raise ValueError("Cannot resume ASR cross-check with changed secondary evidence or settings; resume from asr.")
        return words, context

    asr = prepared.config["asr"]
    secondary_audio = audio_for_asr
    # A primary fixture does not normalize audio. A live secondary still needs
    # normalized PCM; its audio never replaces the primary timing input.
    if not asr.get("fixture_path") and audio_for_asr == source_audio:
        secondary_audio = normalize_audio(source_audio, episode_workdir / "audio.16k.wav", limits=audio_limits)
    model = str(asr.get("model_id", asr.get("model")))
    adapter = CachedASRAdapter(
        prepared.adapter, JsonDiskCache(episode_workdir / "asr-cross-check-cache"), model, asr,
        cost_meter=cost_meter, cost_provider=model,
        dollars_per_hour=0 if asr.get("fixture_path") else asr_dollars_per_hour(str(asr["provider"]), asr),
    )
    start_item = len(cost_meter.items)
    try:
        words = adapter.transcribe(secondary_audio)
    except Exception:
        write_json_atomic(episode_workdir / "asr_cross_check_failure.json", {
            "provider": asr["provider"], "model": model,
            "usage": adapter.last_usage, "cost": cost_meter.as_dict(),
        })
        write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())
        raise
    context = cross_check_context(words, prepared.config)
    metadata = {
        "provider": asr["provider"], "model": model, "context": context,
        "usage": adapter.last_usage, "cache_hit": adapter.last_cache_hit,
        "cost_items": [item.model_dump() for item in cost_meter.items[start_item:]],
        "audio_provenance": {"source_sha256": _sha256_file(source_audio),
                             "asr_input_sha256": adapter.last_cache_key.audio_sha256,
                             "normalized": secondary_audio != source_audio},
        "repair_flags": [flag.model_dump() for flag in adapter.last_repair_flags],
    }
    if adapter.last_evidence is not None:
        metadata["provider_evidence"] = adapter.last_evidence
    write_json_atomic(path, {"words": [word.model_dump() for word in words], "metadata": metadata})
    # Persist both provider charges even if a later alignment/rebuild fails.
    write_text_atomic(episode_workdir / "cost.json", cost_meter.to_json())
    return words, context


def preaccept_cross_checked_spans(
    spans: Sequence[DivergenceSpan], checks: Sequence[SpanCrossCheck], *,
    policy: DeterministicAdjudicationPolicy, uncertain_word_indices: set[int],
) -> tuple[list[DivergenceSpan], list[AdjudicationDecision]]:
    """Respect source policy and acoustic ownership before bypassing a review."""
    by_id = {check.case_id: check for check in checks}
    pending, accepted = [], []
    for span in spans:
        decision = None
        if (span.case_id in by_id and not uncertain_word_indices.intersection(span.asr_word_indices)
                and policy.decide(span) is None):
            decision = preaccepted_decision(span, by_id[span.case_id], language=policy.language,
                                            source_names=policy.source_names)
        if decision is None:
            pending.append(span)
        else:
            accepted.append(decision)
    return pending, accepted


def cross_check_decision_gate(
    spans: Sequence[DivergenceSpan], decisions: Sequence[AdjudicationDecision],
    checks: Sequence[SpanCrossCheck], *, policy: DeterministicAdjudicationPolicy,
) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    selected = enforce_crosscheck_decisions(spans, decisions, checks, policy=policy)
    by_id = {span.case_id: span for span in spans}
    flags = []
    for decision in selected:
        if not decision.reason.startswith("Dual ASR cross-check hold:"):
            continue
        span = by_id[decision.case_id]
        flags.append(QCFlag(
            kind="low_confidence_adjudication", cue_ids=span.cue_ids, confidence=0,
            message=f"{decision.reason} Review this audio before changing the source wording.",
            old_text=span.srt_text, new_text=span.asr_text, start=span.start, end=span.end,
        ))
    return selected, flags
