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
from .adjudication_policy import DeterministicAdjudicationPolicy
from .models import AdjudicationDecision, AudioSnippet, Cue, CueContext, DivergenceSpan, Word
from .providers import ProviderError
from .text_metrics import token_texts
from .tokenize import alphanumeric_signature, normalize_token, number_value


HYBRID_POLICY_VERSION = 5
Reviewer = Callable[..., tuple[list[dict[str, object]], list[dict[str, object]]]]
_NEGATIONS = frozenset({
    "no", "not", "never", "nothing", "nobody", "neither", "nor", "without",
    "não", "nao", "nunca", "jamais", "nem", "ninguém", "ninguem", "nada", "sem",
    "nenhum", "nenhuma", "nenhuns", "nenhumas",
    "nicht", "nein", "nie", "niemals", "nichts", "niemand", "kein", "keine",
    "keinen", "keinem", "keiner", "keines", "ohne", "ningún", "ningun", "nadie",
    "non", "pas", "jamais", "rien", "aucun", "sans", "ない", "ません", "ぬ",
})
# A review reply without exactly one valid decision for a case (truncated or
# empty JSON, a missing envelope, an omitted, duplicate or schema-invalid
# decision) carries no model opinion. Like the direct route's invalid replies
# it is asked once more, and then held as a transient fault that is never cached.
_REVIEW_REPLY_FAULTS = frozenset({
    "review_invalid_batch", "review_invalid_decision", "review_unexpected_case_id",
    "review_duplicate_decision", "review_missing_decision", "review_contradictory_audio_evidence",
    "review_invalid_source_keep",
})


def _confidence_gate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value <= 1:
        raise ProviderError("Hybrid confidence_gate must be finite and greater than 0 and at most 1")
    return float(value)


def _indexed_decisions(
    spans: Sequence[DivergenceSpan], raw: object, *, prefix: str, tolerate_stray: bool = False,
    policy: DeterministicAdjudicationPolicy | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    """Bind by exact ID only; never infer identity from response order.

    With ``tolerate_stray`` an entry for an id outside this batch (the review
    prompt lists read-only sibling cases) faults only the cases that lack
    exactly one valid decision of their own, instead of the whole batch.
    """
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
        has_evidence = payload.get("evidence") is not None or payload.get("heard_text") is not None
        if not has_evidence and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not isfinite(confidence) or not 0 <= confidence <= 1):
            faults[cid] = [f"{prefix}_invalid_decision"]
            continue
        try:
            parsed = AdjudicationDecision.model_validate(payload, strict=True)
        except ValidationError:
            faults[cid] = [f"{prefix}_invalid_decision"]
            continue
        if parsed.verdict != "keep_srt" and not _evidence_supports_wording(parsed, policy):
            faults[cid] = [f"{prefix}_contradictory_audio_evidence"]
            continue
        if parsed.verdict == "keep_srt" and parsed.final_text != span.srt_text:
            faults[cid] = [f"{prefix}_invalid_source_keep"]
            continue
        decisions[cid] = parsed.model_dump(mode="json")
    if batch_fault:
        for cid in ids:
            if tolerate_stray and cid not in faults:
                continue
            faults[cid] = list(dict.fromkeys([batch_fault, *faults.get(cid, [])]))
    return decisions, faults


def transient_hold(trace: object) -> bool:
    """A held route trace that a later run must ask again: an outage or an unusable review reply."""
    return (isinstance(trace, dict) and trace.get("route") == "held" and "case_id" in trace
            and any(str(reason).endswith("_provider_failure") or reason in _REVIEW_REPLY_FAULTS
                    for reason in trace.get("reasons", [])))


def _evidence_supports_wording(
    decision: AdjudicationDecision, policy: DeterministicAdjudicationPolicy | None = None,
) -> bool:
    """A clear hearing must support any changed words, allowing editorial forms."""
    if decision.evidence != "heard_clearly":
        return True
    if alphanumeric_signature(decision.final_text) == alphanumeric_signature(decision.heard_text or ""):
        return True
    if policy is None:
        return False
    equivalent = policy.decide(DivergenceSpan(
        case_id=decision.case_id, cue_ids=[], srt_text=decision.final_text,
        asr_text=decision.heard_text or "",
    ))
    return equivalent is not None and equivalent.verdict == "keep_srt"


def _risk_reasons(
    span: DivergenceSpan, *, language: str | None = None,
    source_names: frozenset[tuple[str, ...]] = frozenset(),
    policy: DeterministicAdjudicationPolicy | None = None,
) -> list[str]:
    source, spoken = token_texts(span.srt_text), token_texts(span.asr_text)
    if alphanumeric_signature(span.srt_text) == alphanumeric_signature(span.asr_text):
        return []
    tokens = [*source, *spoken]
    keys = [token.casefold() for token in tokens]
    risks = []
    known_name = any(tuple(keys[start:start + len(name)]) == name
                     for name in source_names for start in range(len(keys)))
    # A mid-phrase titlecase word may be a new name absent from the script's
    # recurring-name lexicon. This routes to review; it never changes text.
    possible_name = any(token.istitle() and token.casefold() != phrase[0].casefold()
                        for phrase in (source, spoken) if phrase for token in phrase[1:])
    # So may the first word when it does not open a sentence of the cue, or
    # when the script never writes it in lower case (a vocative, a one-off name).
    possible_name = possible_name or (policy is not None and any(
        policy.span_initial_name(span, phrase[0]) for phrase in (source, spoken) if phrase))
    if known_name or possible_name:
        risks.append("risky_name")
    if any(any(char.isnumeric() for char in token) or number_value(normalize_token(token), language) is not None
           for token in tokens):
        risks.append("risky_number")
    if any(key in _NEGATIONS for key in keys) or any(
        marker in text.casefold() for text in (span.srt_text, span.asr_text)
        for marker in ("n't", "n’t", "ない", "ません")
    ):
        risks.append("risky_negation")
    if len(source) == len(spoken) == 1:
        risks.append("risky_single_word_substitution")
    return risks


def _has_stray_entries(spans: Sequence[DivergenceSpan], raw: object) -> bool:
    ids = {span.case_id for span in spans}
    return isinstance(raw, list) and any(
        not isinstance(payload, dict) or payload.get("case_id") not in ids for payload in raw
    )


def triage_decisions(
    spans: Sequence[DivergenceSpan], decisions: object, confidence_gate: float = .7,
    *, language: str | None = None, source_names: frozenset[tuple[str, ...]] = frozenset(),
    policy: DeterministicAdjudicationPolicy | None = None,
) -> dict[str, list[str]]:
    """Return review reasons without rewriting wording or guessing ownership."""
    gate = _confidence_gate(confidence_gate)
    indexed, reasons = _indexed_decisions(spans, decisions, prefix="primary", policy=policy)
    for span in spans:
        cid = span.case_id
        if cid in reasons:
            continue
        decision = indexed[cid]
        if decision.get("evidence") not in (None, "heard_clearly"):
            reasons[cid] = ["primary_audio_unclear"]
        elif decision["confidence"] < gate:
            reasons[cid] = ["low_primary_confidence"]
        elif decision["verdict"] == "keep_srt":
            if not _evidence_supports_wording(AdjudicationDecision.model_validate(decision), policy):
                reasons[cid] = ["source_keep_hearing_unresolved"]
            if alphanumeric_signature(span.srt_text) != alphanumeric_signature(span.asr_text):
                reasons.setdefault(cid, []).append("source_differs_from_owned_asr")
        elif alphanumeric_signature(decision["final_text"]) != alphanumeric_signature(span.asr_text):
            reasons[cid] = ["wording_differs_from_owned_asr"]
        if cid not in reasons:
            risks = _risk_reasons(span, language=language, source_names=source_names, policy=policy)
            if risks:
                reasons[cid] = risks
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
    evidence = {}
    if proposed is not None:
        detail = f" Proposed {proposed['verdict']} ({proposed['confidence']}): {proposed['final_text']!r}."
        # Retain validated reviewer uncertainty so the engine and customer
        # reports distinguish a real listening hold from a synthetic failure.
        if proposed.get("evidence") in {"heard_unclear", "not_audible"}:
            evidence = {key: proposed[key] for key in ("evidence", "heard_text")}
    return {
        "case_id": span.case_id, "verdict": "keep_srt", "final_text": span.srt_text,
        "confidence": 0.0, "speaker": None, "character": "unknown",
        "reason": f"[hybrid:held] {reason}; source text and timing preserved for review.{detail}",
        **evidence,
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
        self.language: str | None = None
        self.register_policy = "spoken"
        self._source_names: frozenset[tuple[str, ...]] = frozenset()
        self._wording_policy = DeterministicAdjudicationPolicy()

    def set_adjudication_context(self, *, language: str | None = None, register_policy: str = "spoken") -> None:
        if language is not None and not isinstance(language, str):
            raise ValueError("adjudication language must be a string or None")
        policy = DeterministicAdjudicationPolicy(
            self._episode_context, language=language, register_policy=register_policy)
        with self._lock:
            self.language, self.register_policy = language, register_policy
            self._source_names = policy.source_names
            self._wording_policy = policy
        setter = getattr(self.primary, "set_adjudication_context", None)
        if callable(setter):
            setter(language=language, register_policy=register_policy)

    def set_episode_context(self, cues: list[Cue]) -> None:
        frozen = tuple(_frozen(cue, _ReadOnlyCue) for cue in cues)
        with self._lock:
            self._episode_context = frozen
            self._wording_policy = DeterministicAdjudicationPolicy(
                frozen, language=self.language, register_policy=self.register_policy)
            self._source_names = self._wording_policy.source_names
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
        with self._lock:
            context, words = self._episode_context, self._episode_words
            language, register_policy = self.language, self.register_policy
            source_names, wording_policy = self._source_names, self._wording_policy
        results: dict[str, dict[str, object]] = {}
        traces: dict[str, dict[str, object]] = {}

        def record(span, result, route, reasons, reviewed=False):
            results[span.case_id] = result
            traces[span.case_id] = {"case_id": span.case_id, "route": route,
                                    "reasons": list(reasons), "review_requested": reviewed}

        available = []
        for span in batch:
            pair_complete = (not span.case_id.startswith("source-pair-timing-v2-")
                             or span.case_id + "-candidate" in audio_snippets)
            if _complete_clip(span, audio_snippets.get(span.case_id)) and pair_complete:
                available.append(span)
            else:
                record(span, _held(span, "case_audio_unavailable"), "held", ["case_audio_unavailable"])
        exact_clips = {span.case_id: audio_snippets[span.case_id].model_copy(deep=True) for span in available}
        exact_clips.update({span.case_id + "-candidate": audio_snippets[span.case_id + "-candidate"].model_copy(deep=True)
                            for span in available if span.case_id.startswith("source-pair-timing-v2-")})
        primary_method = getattr(self.primary, "adjudicate_with_audio", None)
        indexed, reasons = {}, {}
        if available:
            try:
                if not callable(primary_method):
                    raise ProviderError("Primary adapter does not support case audio")
                raw = primary_method([span.model_copy(deep=True) for span in available], deepcopy(exact_clips))
                indexed, _ = _indexed_decisions(available, raw, prefix="primary", policy=wording_policy)
                reasons = triage_decisions(available, raw, self.confidence_gate,
                                          language=language, source_names=source_names, policy=wording_policy)
            except ProviderError:
                reasons = {span.case_id: ["primary_provider_failure"] for span in available}
            finally:
                self._drain_primary_usage()
        selected = [span for span in available if span.case_id in reasons]
        for span in available:
            if span.case_id not in reasons:
                record(span, _routed(indexed[span.case_id], "primary"), "primary", [])
        if selected:
            reviewed: dict[str, dict[str, object]] = {}
            invalid: dict[str, list[str]] = {}
            stray_ignored: set[str] = set()
            pending = selected
            # One retry, for the cases whose reply carried no usable decision.
            for _attempt in range(2):
                pending_clips = {span.case_id: exact_clips[span.case_id] for span in pending}
                pending_clips.update({span.case_id + "-candidate": exact_clips[span.case_id + "-candidate"]
                                      for span in pending if span.case_id.startswith("source-pair-timing-v2-")})
                try:
                    review_raw, events = self.reviewer(
                        spans=tuple(_frozen(span, _ReadOnlySpan) for span in pending),
                        audio_snippets=pending_clips,
                        reasons={span.case_id: deepcopy(reasons[span.case_id]) for span in pending},
                        primary_decisions={span.case_id: deepcopy(indexed[span.case_id])
                                           for span in pending if span.case_id in indexed},
                        batch_spans=batch_snapshot,
                        episode_context=deepcopy(context), episode_words=deepcopy(words),
                        language=language, register_policy=register_policy,
                    )
                    self._record_usage(events, "fallback")
                    accepted, faults = _indexed_decisions(
                        pending, review_raw, prefix="review", tolerate_stray=True, policy=wording_policy)
                    if _has_stray_entries(pending, review_raw):
                        stray_ignored.update(accepted)
                except ProviderError:
                    accepted, faults = {}, {span.case_id: ["review_provider_failure"] for span in pending}
                reviewed.update(accepted)
                for cid in accepted:
                    invalid.pop(cid, None)
                invalid.update(faults)
                pending = [span for span in pending
                           if _REVIEW_REPLY_FAULTS.intersection(faults.get(span.case_id, []))]
                if not pending:
                    break
            for span in selected:
                cid = span.case_id
                proposal = reviewed.get(cid)
                failure = invalid.get(cid, [])
                if not failure and proposal.get("evidence") not in (None, "heard_clearly"):
                    failure = ["review_audio_unclear"]
                if not failure and proposal["confidence"] < self.confidence_gate:
                    failure = ["review_low_confidence"]
                if failure:
                    record(span, _held(span, ", ".join(failure), proposal), "held", reasons[cid] + failure, True)
                else:
                    accepted_reasons = reasons[cid] + (["review_stray_decision_ignored"] if cid in stray_ignored else [])
                    record(span, _routed(proposal, "fallback"), "fallback", accepted_reasons, True)
        with self._lock:
            self._routes.extend(traces[span.case_id] for span in batch)
        return [results[span.case_id] for span in batch]
