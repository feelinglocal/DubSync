"""Clip-only adjudication routing with explicit source-preserving failure paths."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import ConfigDict, ValidationError

from .adjudication import _snippet_covers_span
from .models import AdjudicationDecision, AudioSnippet, Cue, CueContext, DivergenceSpan, Word
from .providers import ProviderError
from .tokenize import alphanumeric_signature


HYBRID_POLICY_VERSION = 1
Reviewer = Callable[..., tuple[list[dict[str, object]], list[dict[str, object]]]]


def _confidence_gate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value <= 1:
        raise ProviderError("Hybrid confidence_gate must be finite and greater than 0 and at most 1")
    return float(value)


def _indexed_decisions(
    spans: Sequence[DivergenceSpan], raw: object, *, prefix: str,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    """Bind by exact ID only; never infer identity from response order."""
    ids = {span.case_id for span in spans}
    if len(ids) != len(spans):
        raise ValueError("Hybrid adjudication requires unique input case IDs")
    faults: dict[str, list[str]] = {}
    if not isinstance(raw, list):
        return {}, {cid: [f"{prefix}_invalid_batch"] for cid in ids}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    batch_fault = None
    for payload in raw:
        if not isinstance(payload, dict):
            batch_fault = f"{prefix}_invalid_decision"
            continue
        cid = payload.get("case_id")
        if not isinstance(cid, str) or cid not in ids:
            batch_fault = f"{prefix}_unexpected_case_id"
            continue
        grouped[cid].append(payload)
    decisions = {}
    for span in spans:
        cid = span.case_id
        payloads = grouped.get(cid, [])
        if len(payloads) != 1:
            faults[cid] = [f"{prefix}_duplicate_decision" if payloads else f"{prefix}_missing_decision"]
            continue
        payload = payloads[0]
        confidence = payload.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not isfinite(confidence) or not 0 <= confidence <= 1):
            faults[cid] = [f"{prefix}_invalid_decision"]
            continue
        try:
            parsed = AdjudicationDecision.model_validate(payload, strict=True)
        except ValidationError:
            faults[cid] = [f"{prefix}_invalid_decision"]
            continue
        decisions[cid] = parsed.model_dump(mode="json")
    if batch_fault:
        for cid in ids:
            faults[cid] = list(dict.fromkeys([batch_fault, *faults.get(cid, [])]))
    return decisions, faults


def triage_decisions(
    spans: Sequence[DivergenceSpan], decisions: object, confidence_gate: float = .7,
) -> dict[str, list[str]]:
    """Return review reasons without rewriting wording or guessing ownership."""
    gate = _confidence_gate(confidence_gate)
    indexed, reasons = _indexed_decisions(spans, decisions, prefix="primary")
    for span in spans:
        cid = span.case_id
        if cid in reasons:
            continue
        decision = indexed[cid]
        if decision["confidence"] < gate:
            reasons[cid] = ["low_primary_confidence"]
        elif decision["verdict"] == "keep_srt":
            if alphanumeric_signature(span.srt_text) != alphanumeric_signature(span.asr_text):
                reasons[cid] = ["source_differs_from_owned_asr"]
        elif alphanumeric_signature(decision["final_text"]) != alphanumeric_signature(span.asr_text):
            reasons[cid] = ["wording_differs_from_owned_asr"]
    return reasons


class _ReadOnlyCue(Cue):
    model_config = ConfigDict(frozen=True)
    lines: tuple[str, ...]


class _ReadOnlyWord(Word):
    model_config = ConfigDict(frozen=True)


class _ReadOnlyCueContext(CueContext):
    model_config = ConfigDict(frozen=True)


class _ReadOnlySpan(DivergenceSpan):
    model_config = ConfigDict(frozen=True)
    cue_ids: tuple[int, ...]
    speaker_ids: tuple[str, ...]
    srt_token_indices: tuple[int, ...]
    asr_word_indices: tuple[int, ...]
    context_before: tuple[_ReadOnlyCueContext, ...]
    context_after: tuple[_ReadOnlyCueContext, ...]


def _frozen(model, cls):
    # model_dump excludes prompt scene fields; preserve those annotations too.
    values = {key: deepcopy(getattr(model, key)) for key in type(model).model_fields}
    if isinstance(model, DivergenceSpan):
        for field in ("context_before", "context_after"):
            values[field] = [_frozen(item, _ReadOnlyCueContext) for item in values[field]]
    return cls.model_validate(values)


def _complete_clip(span: DivergenceSpan, snippet: AudioSnippet | None) -> bool:
    if not _snippet_covers_span(snippet, span):
        return False
    if not 0 <= span.start < span.end:
        return False
    path = Path(snippet.path)
    return path.is_file() and path.stat().st_size > 0


def _held(span: DivergenceSpan, reason: str, proposed: dict[str, object] | None = None) -> dict[str, object]:
    detail = ""
    if proposed is not None:
        detail = f" Proposed {proposed['verdict']} ({proposed['confidence']}): {proposed['final_text']!r}."
    return {
        "case_id": span.case_id, "verdict": "keep_srt", "final_text": span.srt_text,
        "confidence": 0.0, "speaker": None, "character": "unknown",
        "reason": f"[hybrid:held] {reason}; source text and timing preserved for review.{detail}",
    }


def _routed(decision: dict[str, object], route: str) -> dict[str, object]:
    return {**deepcopy(decision), "reason": f"[hybrid:{route}] {decision['reason']}"}


class HybridAdjudicationAdapter:
    def __init__(self, primary: object, reviewer: Reviewer, confidence_gate: float = .7):
        self.primary = primary
        self.reviewer = reviewer
        self.confidence_gate = _confidence_gate(confidence_gate)
        self._lock = RLock()
        self._usage_events: list[dict[str, object]] = []
        self._routes: list[dict[str, object]] = []
        self._episode_context: tuple[Cue, ...] = ()
        self._episode_words: tuple[Word, ...] = ()

    def set_episode_context(self, cues: list[Cue]) -> None:
        frozen = tuple(_frozen(cue, _ReadOnlyCue) for cue in cues)
        with self._lock:
            self._episode_context = frozen
        setter = getattr(self.primary, "set_episode_context", None)
        if callable(setter):
            setter([cue.model_copy(deep=True) for cue in cues])

    def set_episode_words(self, words: list[Word]) -> None:
        frozen = tuple(_frozen(word, _ReadOnlyWord) for word in words)
        with self._lock:
            self._episode_words = frozen
        setter = getattr(self.primary, "set_episode_words", None)
        if callable(setter):
            setter([word.model_copy(deep=True) for word in words])

    def set_audio_context(self, path: str | Path, **kwargs: Any) -> None:
        setter = getattr(self.primary, "set_audio_context", None)
        if callable(setter):
            setter(path, **kwargs)

    def close(self) -> None:
        close = getattr(self.primary, "close", None)
        if callable(close):
            close()

    def audio_context_report(self) -> dict[str, object]:
        method = getattr(self.primary, "audio_context_report", None)
        report = deepcopy(method()) if callable(method) else {"enabled": False}
        return {**report, "hybrid_routes": self.route_report()}

    def _record_usage(self, events: list[object], route: str) -> None:
        with self._lock:
            for event in events:
                copied = deepcopy(event) if isinstance(event, dict) else {"usage": deepcopy(event)}
                self._usage_events.append({**copied, "adjudication_route": route})

    def _drain_primary_usage(self) -> None:
        drain = getattr(self.primary, "drain_usage_events", None)
        if callable(drain):
            with self._lock:
                self._record_usage(drain(), "primary")

    def drain_usage_events(self) -> list[dict[str, object]]:
        self._drain_primary_usage()
        with self._lock:
            events, self._usage_events = self._usage_events, []
            return events

    def route_report(self) -> dict[str, object]:
        with self._lock:
            routes = deepcopy(self._routes)
        counts = Counter(item["route"] for item in routes)
        return {
            "policy_version": HYBRID_POLICY_VERSION, "confidence_gate": self.confidence_gate,
            "counts": {"primary": counts["primary"], "fallback": counts["fallback"], "held": counts["held"],
                       "review_requested": sum(item["review_requested"] for item in routes)},
            "decisions": routes,
        }

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        return self.adjudicate_with_audio(spans, {})

    def adjudicate_with_audio(
        self, spans: list[DivergenceSpan], audio_snippets: dict[str, AudioSnippet],
    ) -> list[dict[str, object]]:
        if len({span.case_id for span in spans}) != len(spans):
            raise ValueError("Hybrid adjudication requires unique input case IDs")
        # Providers receive detached values; reviewer context cannot mutate another call.
        batch = [span.model_copy(deep=True) for span in spans]
        batch_snapshot = tuple(_frozen(span, _ReadOnlySpan) for span in batch)
        results: dict[str, dict[str, object]] = {}
        traces: dict[str, dict[str, object]] = {}

        def record(span, result, route, reasons, reviewed=False):
            results[span.case_id] = result
            traces[span.case_id] = {"case_id": span.case_id, "route": route,
                                    "reasons": list(reasons), "review_requested": reviewed}

        available = []
        for span in batch:
            if _complete_clip(span, audio_snippets.get(span.case_id)):
                available.append(span)
            else:
                record(span, _held(span, "case_audio_unavailable"), "held", ["case_audio_unavailable"])
        exact_clips = {span.case_id: audio_snippets[span.case_id].model_copy(deep=True) for span in available}
        primary_method = getattr(self.primary, "adjudicate_with_audio", None)
        indexed, reasons = {}, {}
        if available:
            try:
                if not callable(primary_method):
                    raise ProviderError("Primary adapter does not support case audio")
                raw = primary_method([span.model_copy(deep=True) for span in available], deepcopy(exact_clips))
                indexed, _ = _indexed_decisions(available, raw, prefix="primary")
                reasons = triage_decisions(available, raw, self.confidence_gate)
            except ProviderError:
                reasons = {span.case_id: ["primary_provider_failure"] for span in available}
            finally:
                self._drain_primary_usage()
        selected = [span for span in available if span.case_id in reasons]
        for span in available:
            if span.case_id not in reasons:
                record(span, _routed(indexed[span.case_id], "primary"), "primary", [])
        if selected:
            selected_clips = {span.case_id: exact_clips[span.case_id] for span in selected}
            with self._lock:
                context, words = self._episode_context, self._episode_words
            try:
                review_raw, events = self.reviewer(
                    spans=tuple(_frozen(span, _ReadOnlySpan) for span in selected),
                    audio_snippets=selected_clips,
                    reasons=deepcopy(reasons),
                    primary_decisions={span.case_id: deepcopy(indexed[span.case_id]) for span in selected if span.case_id in indexed},
                    batch_spans=batch_snapshot,
                    episode_context=deepcopy(context), episode_words=deepcopy(words),
                )
                self._record_usage(events, "fallback")
                reviewed, invalid = _indexed_decisions(selected, review_raw, prefix="review")
            except ProviderError:
                reviewed = {}
                invalid = {span.case_id: ["review_provider_failure"] for span in selected}
            for span in selected:
                cid = span.case_id
                proposal = reviewed.get(cid)
                failure = invalid.get(cid, [])
                if not failure and proposal["confidence"] < self.confidence_gate:
                    failure = ["review_low_confidence"]
                if failure:
                    record(span, _held(span, ", ".join(failure), proposal), "held", reasons[cid] + failure, True)
                else:
                    record(span, _routed(proposal, "fallback"), "fallback", reasons[cid], True)
        with self._lock:
            self._routes.extend(traces[span.case_id] for span in batch)
        return [results[span.case_id] for span in batch]
