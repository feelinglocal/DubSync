from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from pydantic import ValidationError

from .models import AdjudicationDecision, AudioSnippet, DivergenceSpan, QCFlag
from .providers import ProviderError
from .tokenize import alphanumeric_signature


_MAX_ADJUDICATION_BATCH_SPANS = 25
_MAX_UNPACKED_SCENE_BATCHES = 16


class LLMAdapter(Protocol):
    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        raise NotImplementedError


class SnippetAwareLLMAdapter(Protocol):
    def adjudicate_with_audio(
        self,
        spans: list[DivergenceSpan],
        audio_snippets: dict[str, AudioSnippet],
    ) -> list[dict[str, object]]:
        raise NotImplementedError


AudioSnippetBatchLoader = Callable[
    [list[DivergenceSpan]],
    AbstractContextManager[dict[str, AudioSnippet]],
]


class StaticLLMAdapter:
    def __init__(self, responses: dict[str, dict[str, object]]):
        self._responses = responses

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        return [self._responses.get(span.case_id, {}) for span in spans]


class KeepSRTAdapter:
    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        return [
            {
                "case_id": span.case_id,
                "verdict": "keep_srt",
                "final_text": span.srt_text,
                "confidence": span.confidence,
                "speaker": span.speaker_ids[0] if span.speaker_ids else None,
                "character": "unknown",
                "reason": "LLM disabled; preserved source SRT for human review.",
            }
            for span in spans
        ]


class AdjudicationEngine:
    def __init__(
        self,
        llm: LLMAdapter,
        confidence_gate: float = 0.7,
        scene_gap_seconds: float = 4.0,
        audio_snippets: dict[str, AudioSnippet] | None = None,
        audio_snippet_batches: AudioSnippetBatchLoader | None = None,
    ):
        self.llm = llm
        self.confidence_gate = confidence_gate
        self.scene_gap_seconds = scene_gap_seconds
        self.audio_snippets = dict(audio_snippets or {})
        self.audio_snippet_batches = audio_snippet_batches

    def adjudicate(self, spans: list[DivergenceSpan]) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
        decisions_by_case: dict[str, AdjudicationDecision] = {}
        llm_spans: list[DivergenceSpan] = []
        for span in spans:
            heuristic_decision = _heuristic_decision(span)
            if heuristic_decision is None:
                llm_spans.append(span)
            else:
                decisions_by_case[span.case_id] = heuristic_decision

        invalid_spans: list[DivergenceSpan] = []
        provider_failed_spans: list[DivergenceSpan] = []
        if llm_spans:
            for batch in self._scene_batches(llm_spans):
                try:
                    raw_decisions = self._adjudicate_batch(batch)
                except (ProviderError, OSError):
                    provider_failed_spans.extend(batch)
                    continue
                llm_decisions, batch_invalid_spans = self._validate_raw(raw_decisions, batch)
                decisions_by_case = {**decisions_by_case, **llm_decisions}
                invalid_spans.extend(batch_invalid_spans)
            if invalid_spans:
                retry_invalid_spans: list[DivergenceSpan] = []
                for batch in self._scene_batches(invalid_spans):
                    try:
                        raw_retry_decisions = self._adjudicate_batch(batch)
                    except (ProviderError, OSError):
                        provider_failed_spans.extend(batch)
                        continue
                    retry_decisions, batch_invalid_spans = self._validate_raw(raw_retry_decisions, batch)
                    decisions_by_case = {**decisions_by_case, **retry_decisions}
                    retry_invalid_spans.extend(batch_invalid_spans)
                invalid_spans = retry_invalid_spans

        decisions: list[AdjudicationDecision] = []
        flags: list[QCFlag] = []
        provider_failed_case_ids = {span.case_id for span in provider_failed_spans}

        for span in spans:
            decision = decisions_by_case.get(span.case_id)
            if decision is None:
                provider_failed = span.case_id in provider_failed_case_ids
                decision = AdjudicationDecision(
                    case_id=span.case_id,
                    verdict="keep_srt",
                    final_text=span.srt_text,
                    confidence=0.0,
                    speaker=span.speaker_ids[0] if span.speaker_ids else None,
                    character="unknown",
                    reason=(
                        "Adjudication provider failed; preserved source SRT."
                        if provider_failed
                        else "Invalid LLM response; preserved source SRT."
                    ),
                )
                flags.append(
                    QCFlag(
                        kind=(
                            "llm_provider_unavailable"
                            if provider_failed
                            else "invalid_llm_response"
                        ),
                        cue_ids=span.cue_ids,
                        message=(
                            "LLM adjudication provider failed; source SRT was preserved."
                            if provider_failed
                            else "LLM response failed schema validation."
                        ),
                        severity="error",
                        old_text=span.srt_text,
                        new_text=span.asr_text,
                        start=span.start,
                        end=span.end,
                    )
                )

            decision, confidence_flag = confidence_gated_decision(
                span, decision, self.confidence_gate
            )
            if confidence_flag is not None:
                flags.append(confidence_flag)
            decisions.append(decision)

        return decisions, flags

    def _adjudicate_batch(self, batch: list[DivergenceSpan]) -> list[dict[str, object]]:
        snippets = {span.case_id: self.audio_snippets[span.case_id] for span in batch if span.case_id in self.audio_snippets}
        if self.audio_snippet_batches is not None:
            with self.audio_snippet_batches(batch) as loaded_snippets:
                return self._call_adjudication_adapter(
                    batch,
                    {**snippets, **loaded_snippets},
                )
        return self._call_adjudication_adapter(batch, snippets)

    def _call_adjudication_adapter(
        self,
        batch: list[DivergenceSpan],
        snippets: dict[str, AudioSnippet],
    ) -> list[dict[str, object]]:
        if snippets and hasattr(self.llm, "adjudicate_with_audio"):
            return getattr(self.llm, "adjudicate_with_audio")(batch, snippets)
        return self.llm.adjudicate(batch)

    def _validate_raw(
        self,
        raw: object,
        spans: list[DivergenceSpan],
    ) -> tuple[dict[str, AdjudicationDecision], list[DivergenceSpan]]:
        if not isinstance(raw, list):
            return {}, list(spans)

        by_case = {span.case_id: span for span in spans}
        decisions: dict[str, AdjudicationDecision] = {}
        invalid_spans: dict[str, DivergenceSpan] = {}

        for index, payload in enumerate(raw):
            span = self._span_for_payload(payload, index, spans, by_case)
            if span is None:
                continue

            try:
                decision = AdjudicationDecision.model_validate(payload)
            except (ValidationError, TypeError, ValueError):
                invalid_spans[span.case_id] = span
                continue

            if decision.case_id != span.case_id:
                invalid_spans[span.case_id] = span
                continue

            decisions[span.case_id] = decision

        for span in spans:
            if span.case_id not in decisions and span.case_id not in invalid_spans:
                invalid_spans[span.case_id] = span

        return decisions, list(invalid_spans.values())

    def _scene_batches(self, spans: list[DivergenceSpan]) -> list[list[DivergenceSpan]]:
        if not spans:
            return []

        batches: list[list[DivergenceSpan]] = [[spans[0]]]
        previous = spans[0]
        for span in spans[1:]:
            if _starts_new_scene(previous, span, self.scene_gap_seconds):
                batches.append([span])
            else:
                batches[-1].append(span)
            previous = span
        annotated_scenes = [
            [
                span.model_copy(
                    update={
                        "prompt_scene_id": scene_id,
                        "prompt_scene_position": position,
                    }
                )
                for position, span in enumerate(batch, start=1)
            ]
            for scene_id, batch in enumerate(batches, start=1)
        ]
        scene_chunks = [
            chunk
            for batch in annotated_scenes
            for chunk in _split_span_batch_by_size(batch, _MAX_ADJUDICATION_BATCH_SPANS)
        ]
        if len(scene_chunks) <= _MAX_UNPACKED_SCENE_BATCHES:
            return scene_chunks
        return _pack_scene_chunks(scene_chunks, _MAX_ADJUDICATION_BATCH_SPANS)

    @staticmethod
    def _span_for_payload(
        payload: object,
        index: int,
        spans: list[DivergenceSpan],
        by_case: dict[str, DivergenceSpan],
    ) -> DivergenceSpan | None:
        if isinstance(payload, dict):
            span = by_case.get(str(payload.get("case_id")))
            if span is not None:
                return span

        if index < len(spans):
            return spans[index]
        return None


def confidence_gated_decision(
    span: DivergenceSpan,
    decision: AdjudicationDecision,
    confidence_gate: float,
) -> tuple[AdjudicationDecision, QCFlag | None]:
    """Keep uncertain proposed wording reviewable without applying it to the SRT."""
    if decision.confidence >= confidence_gate:
        return decision, None
    flag = QCFlag(
        kind="low_confidence_adjudication",
        cue_ids=span.cue_ids,
        message=(
            "Adjudication confidence is below the configured gate; source SRT was preserved. "
            f"Proposed verdict: {decision.verdict}. Reason: {decision.reason}"
        ),
        confidence=decision.confidence,
        old_text=span.srt_text,
        new_text=decision.final_text,
        start=span.start,
        end=span.end,
    )
    if decision.verdict == "keep_srt":
        return decision, flag
    return decision.model_copy(update={
        "verdict": "keep_srt",
        "final_text": span.srt_text,
        "reason": "Adjudication confidence is below the configured gate; preserved source SRT for review.",
    }), flag


def _pack_scene_chunks(
    scene_chunks: list[list[DivergenceSpan]],
    max_size: int,
) -> list[list[DivergenceSpan]]:
    packed: list[list[DivergenceSpan]] = []
    current: list[DivergenceSpan] = []
    for chunk in scene_chunks:
        if current and len(current) + len(chunk) > max_size:
            packed = [*packed, current]
            current = []
        current = [*current, *chunk]
    return [*packed, current] if current else packed


def _heuristic_decision(span: DivergenceSpan) -> AdjudicationDecision | None:
    srt_signature = alphanumeric_signature(span.srt_text)
    asr_signature = alphanumeric_signature(span.asr_text)
    if not srt_signature or not asr_signature:
        return None

    if srt_signature == asr_signature:
        return _keep_srt_decision(span, "Punctuation/casing-only difference; preserved source SRT.")

    return None


def _starts_new_scene(previous: DivergenceSpan, current: DivergenceSpan, scene_gap_seconds: float) -> bool:
    if previous.end is None or current.start is None:
        return False
    return current.start - previous.end > scene_gap_seconds


def _split_span_batch_by_size(
    spans: list[DivergenceSpan],
    max_size: int,
) -> list[list[DivergenceSpan]]:
    if len(spans) <= max_size:
        return [spans]
    batches: list[list[DivergenceSpan]] = []
    remaining = list(spans)
    while len(remaining) > max_size:
        split_at = _widest_internal_span_gap_index(remaining[: max_size + 1])
        if split_at <= 0 or split_at > max_size:
            split_at = max_size
        batches.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        batches.append(remaining)
    return batches


def _widest_internal_span_gap_index(spans: list[DivergenceSpan]) -> int:
    best_index = len(spans) - 1
    best_gap: float | None = None
    for index, (previous, current) in enumerate(zip(spans, spans[1:]), start=1):
        if previous.end is None or current.start is None:
            continue
        gap = current.start - previous.end
        if best_gap is None or gap >= best_gap:
            best_gap = gap
            best_index = index
    return best_index


def _keep_srt_decision(span: DivergenceSpan, reason: str) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt",
        final_text=span.srt_text,
        confidence=span.confidence,
        speaker=span.speaker_ids[0] if span.speaker_ids else None,
        character="unknown",
        reason=reason,
    )
