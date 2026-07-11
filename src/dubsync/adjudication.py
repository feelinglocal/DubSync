from __future__ import annotations

from typing import Protocol

from rapidfuzz import fuzz
from pydantic import ValidationError

from .models import AdjudicationDecision, AudioSnippet, DivergenceSpan, QCFlag
from .tokenize import alphanumeric_signature


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
    ):
        self.llm = llm
        self.confidence_gate = confidence_gate
        self.scene_gap_seconds = scene_gap_seconds
        self.audio_snippets = dict(audio_snippets or {})

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
        if llm_spans:
            for batch in self._scene_batches(llm_spans):
                llm_decisions, batch_invalid_spans = self._validate_raw(self._adjudicate_batch(batch), batch)
                decisions_by_case = {**decisions_by_case, **llm_decisions}
                invalid_spans.extend(batch_invalid_spans)
            if invalid_spans:
                retry_invalid_spans: list[DivergenceSpan] = []
                for batch in self._scene_batches(invalid_spans):
                    retry_decisions, batch_invalid_spans = self._validate_raw(
                        self._adjudicate_batch(batch),
                        batch,
                    )
                    decisions_by_case = {**decisions_by_case, **retry_decisions}
                    retry_invalid_spans.extend(batch_invalid_spans)
                invalid_spans = retry_invalid_spans

        decisions: list[AdjudicationDecision] = []
        flags: list[QCFlag] = []

        for span in spans:
            decision = decisions_by_case.get(span.case_id)
            if decision is None:
                decision = AdjudicationDecision(
                    case_id=span.case_id,
                    verdict="keep_srt",
                    final_text=span.srt_text,
                    confidence=0.0,
                    speaker=span.speaker_ids[0] if span.speaker_ids else None,
                    character="unknown",
                    reason="Invalid LLM response; preserved source SRT.",
                )
                flags.append(
                    QCFlag(
                        kind="invalid_llm_response",
                        cue_ids=span.cue_ids,
                        message="LLM response failed schema validation.",
                        severity="error",
                        old_text=span.srt_text,
                        new_text=span.asr_text,
                        start=span.start,
                        end=span.end,
                    )
                )

            if decision.confidence < self.confidence_gate:
                flags.append(
                    QCFlag(
                        kind="low_confidence_adjudication",
                        cue_ids=span.cue_ids,
                        message="Adjudication confidence is below the configured gate.",
                        confidence=decision.confidence,
                        old_text=span.srt_text,
                        new_text=decision.final_text,
                        start=span.start,
                        end=span.end,
                    )
                )
            decisions.append(decision)

        return decisions, flags

    def _adjudicate_batch(self, batch: list[DivergenceSpan]) -> list[dict[str, object]]:
        snippets = {span.case_id: self.audio_snippets[span.case_id] for span in batch if span.case_id in self.audio_snippets}
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
        return batches

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


def _heuristic_decision(span: DivergenceSpan) -> AdjudicationDecision | None:
    srt_signature = alphanumeric_signature(span.srt_text)
    asr_signature = alphanumeric_signature(span.asr_text)
    if not srt_signature or not asr_signature:
        return None

    if srt_signature == asr_signature:
        return _keep_srt_decision(span, "Punctuation/casing-only difference; preserved source SRT.")

    if len(srt_signature) == len(asr_signature):
        srt_joined = " ".join(srt_signature)
        asr_joined = " ".join(asr_signature)
        if fuzz.ratio(srt_joined, asr_joined) >= 92:
            return _keep_srt_decision(span, "Tiny ASR spelling/noise difference; preserved source SRT.")

    return None


def _starts_new_scene(previous: DivergenceSpan, current: DivergenceSpan, scene_gap_seconds: float) -> bool:
    if previous.end is None or current.start is None:
        return False
    return current.start - previous.end > scene_gap_seconds


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
