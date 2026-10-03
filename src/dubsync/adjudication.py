from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from difflib import SequenceMatcher
from math import isfinite
from typing import Protocol

from pydantic import ValidationError

from .adjudication_policy import DETERMINISTIC_KEEP_CONFIDENCE, DeterministicAdjudicationPolicy
from .models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan, QCFlag
from .providers import ProviderError
from .tokenize import alphanumeric_signature


_MAX_ADJUDICATION_BATCH_SPANS = 25
_MAX_UNPACKED_SCENE_BATCHES = 16
REQUIRED_AUDIO_HEARING_POLICY_VERSION = 1
# Words this close to a cue's retained word are spoken at that cue's time (the
# default timing.max_intra_cue_gap): a clip reaching this close is at the cue.
_RETAINED_EDGE_REACH_SECONDS = 1.5


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


class _RequiredAudioBatchError(ProviderError):
    def __init__(self, unavailable_case_ids: set[str]):
        super().__init__("Adjudication failed for the cases with available audio.")
        self.unavailable_case_ids = unavailable_case_ids


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
                "confidence": DETERMINISTIC_KEEP_CONFIDENCE,
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
        max_batch_spans: int = _MAX_ADJUDICATION_BATCH_SPANS,
        max_concurrent_batches: int = 1,
        retry_timed_out_batches: bool = False,
        require_audio_snippets: bool = False,
        required_audio_case_ids: set[str] | None = None,
        source_cues: Sequence[Cue] | None = None,
        language: str | None = None,
        register_policy: str = "spoken",
    ):
        self.llm = llm
        self.confidence_gate = confidence_gate
        self.scene_gap_seconds = scene_gap_seconds
        self.audio_snippets = dict(audio_snippets or {})
        self.audio_snippet_batches = audio_snippet_batches
        for name, value, maximum in (
            ("max_batch_spans", max_batch_spans, _MAX_ADJUDICATION_BATCH_SPANS),
            ("max_concurrent_batches", max_concurrent_batches, 4),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"adjudication.{name} must be an integer between 1 and {maximum}")
        self.max_batch_spans = max_batch_spans
        self.max_concurrent_batches = max_concurrent_batches
        if not isinstance(retry_timed_out_batches, bool):
            raise ValueError("adjudication.retry_timed_out_batches must be boolean")
        self.retry_timed_out_batches = retry_timed_out_batches
        if not isinstance(require_audio_snippets, bool):
            raise ValueError("adjudication.require_audio_snippets must be boolean")
        self.require_audio_snippets = require_audio_snippets
        self.required_audio_case_ids = frozenset(required_audio_case_ids or ())
        self.deterministic_policy = DeterministicAdjudicationPolicy(source_cues, language, register_policy)

    def adjudicate(self, spans: list[DivergenceSpan]) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
        decisions_by_case: dict[str, AdjudicationDecision] = {}
        llm_spans: list[DivergenceSpan] = []
        deterministic_case_ids: set[str] = set()
        for span in spans:
            # Explicit hearing questions concern audibility/acoustic ownership,
            # even when the two written strings happen to be identical.
            heuristic_decision = (None if span.case_id in self.required_audio_case_ids
                                  else self.deterministic_policy.decide(span))
            if heuristic_decision is None:
                llm_spans.append(span)
            else:
                decisions_by_case[span.case_id] = heuristic_decision
                deterministic_case_ids.add(span.case_id)

        invalid_spans: list[DivergenceSpan] = []
        provider_failed_spans: list[DivergenceSpan] = []
        timed_out_spans: list[DivergenceSpan] = []
        audio_unavailable_case_ids: set[str] = set()
        if llm_spans:
            for batch, raw_decisions, timed_out, unavailable_ids in self._adjudicate_batches(llm_spans):
                audio_unavailable_case_ids.update(unavailable_ids)
                if raw_decisions is None:
                    if timed_out and self.retry_timed_out_batches and len(batch) > 1:
                        timed_out_spans.extend(batch)
                    else:
                        provider_failed_spans.extend(batch)
                    continue
                llm_decisions, batch_invalid_spans = self._validate_raw(raw_decisions, batch)
                decisions_by_case = {**decisions_by_case, **llm_decisions}
                invalid_spans.extend(batch_invalid_spans)
            # One bounded recovery pass reduces reasoning load after a known
            # timeout. Authentication/rate errors and singleton timeouts are
            # never expanded into another round of provider calls.
            for batch, raw_decisions, _, unavailable_ids in self._adjudicate_batches(timed_out_spans, max_batch_spans=1):
                audio_unavailable_case_ids.update(unavailable_ids)
                if raw_decisions is None:
                    provider_failed_spans.extend(batch)
                    continue
                recovered, batch_invalid_spans = self._validate_raw(raw_decisions, batch)
                decisions_by_case = {**decisions_by_case, **recovered}
                invalid_spans.extend(batch_invalid_spans)
            if invalid_spans:
                retry_invalid_spans: list[DivergenceSpan] = []
                for batch, raw_retry_decisions, _, unavailable_ids in self._adjudicate_batches(invalid_spans):
                    audio_unavailable_case_ids.update(unavailable_ids)
                    if raw_retry_decisions is None:
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
            # A hold created here already carries its own specific finding.
            # It is not a model opinion, so it is never reported a second
            # time as a low-confidence answer.
            held_by_engine = False
            if span.case_id in audio_unavailable_case_ids:
                held_by_engine = True
                decision = AdjudicationDecision.model_validate(_unavailable_audio_decision(span))
                flags.append(QCFlag(
                    kind="adjudication_audio_unavailable", cue_ids=span.cue_ids,
                    message="Required case audio was unavailable or incomplete; source text and timing were preserved.",
                    severity="error", confidence=0.0, old_text=span.srt_text,
                    new_text=span.asr_text, start=span.start, end=span.end,
                ))
            if decision is None:
                held_by_engine = True
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

            if not held_by_engine and span.case_id not in deterministic_case_ids:
                decision, confidence_flag = confidence_gated_decision(
                    span, decision, self.confidence_gate, policy=self.deterministic_policy,
                )
                if confidence_flag is not None:
                    flags.append(confidence_flag)
            decisions.append(decision)

        return decisions, flags

    def _adjudicate_batches(self, spans: list[DivergenceSpan], *, max_batch_spans: int | None = None):
        batches = self._scene_batches(spans, max_batch_spans=max_batch_spans)
        if self.max_concurrent_batches == 1 or len(batches) <= 1:
            for batch in batches:
                yield self._attempt_batch(batch)
            return
        # Results are consumed in stable case order. The context and snippet
        # owners synchronize resource acquisition and keep each batch isolated.
        # Leaving this scope waits for every in-flight call before cleanup.
        results = {}
        next_batch = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batches, thread_name_prefix="dubsync-adjudicate") as executor:
            pending = {}
            try:
                while next_batch < len(batches) or pending:
                    while next_batch < len(batches) and len(pending) < self.max_concurrent_batches:
                        pending[executor.submit(self._attempt_batch, batches[next_batch])] = next_batch
                        next_batch += 1
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # Observe all completed failures before scheduling more paid
                    # work, even if an earlier case is still in flight.
                    for future in done:
                        results[pending.pop(future)] = future.result()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        for index in range(len(batches)):
            yield results[index]

    def _attempt_batch(self, batch: list[DivergenceSpan]):
        try:
            raw, unavailable_ids = self._adjudicate_batch(batch)
            return batch, raw, False, unavailable_ids
        except (ProviderError, OSError) as exc:
            unavailable_ids = exc.unavailable_case_ids if isinstance(exc, _RequiredAudioBatchError) else set()
            return batch, None, _is_request_timeout(exc), unavailable_ids

    def _adjudicate_batch(self, batch: list[DivergenceSpan]) -> tuple[object, set[str]]:
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
    ) -> tuple[object, set[str]]:
        audio_method = getattr(self.llm, "adjudicate_with_audio", None)
        if self.require_audio_snippets or any(span.case_id in self.required_audio_case_ids for span in batch):
            unavailable = [span for span in batch
                           if (self.require_audio_snippets or span.case_id in self.required_audio_case_ids)
                           and not (callable(audio_method) and _snippet_covers_span(snippets.get(span.case_id), span))]
            unavailable_ids = {span.case_id for span in unavailable}
            available = [span for span in batch if span.case_id not in unavailable_ids]
            held = [_unavailable_audio_decision(span) for span in unavailable]
            # Never turn failed or partial snippet extraction into a text-only
            # approval. Each missing case remains held even with a zero gate.
            try:
                selected_snippets = {span.case_id: snippets[span.case_id] for span in available if span.case_id in snippets}
                raw = (audio_method(available, selected_snippets)
                       if selected_snippets and callable(audio_method)
                       else self.llm.adjudicate(available)) if available else []
            except (ProviderError, OSError) as exc:
                raise _RequiredAudioBatchError(unavailable_ids) from exc
            return ([*raw, *held] if isinstance(raw, list) else raw), unavailable_ids
        if snippets and callable(audio_method):
            return audio_method(batch, snippets), set()
        return self.llm.adjudicate(batch), set()

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
            if decision.verdict == "keep_srt" and decision.final_text != span.srt_text:
                invalid_spans[span.case_id] = span
                continue

            decisions[span.case_id] = decision

        for span in spans:
            if span.case_id not in decisions and span.case_id not in invalid_spans:
                invalid_spans[span.case_id] = span

        return decisions, list(invalid_spans.values())

    def _scene_batches(self, spans: list[DivergenceSpan], *, max_batch_spans: int | None = None) -> list[list[DivergenceSpan]]:
        if not spans:
            return []

        batch_limit = self.max_batch_spans if max_batch_spans is None else max_batch_spans
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
            for chunk in _split_span_batch_by_size(batch, batch_limit)
        ]
        if len(scene_chunks) <= _MAX_UNPACKED_SCENE_BATCHES:
            return scene_chunks
        return _pack_scene_chunks(scene_chunks, batch_limit)

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


def _snippet_covers_span(snippet: AudioSnippet | None, span: DivergenceSpan) -> bool:
    if not isinstance(snippet, AudioSnippet) or snippet.case_id != span.case_id:
        return False
    if span.start is None or span.end is None:
        return False
    return (
        all(isfinite(value) for value in (snippet.start, snippet.end, span.start, span.end))
        and 0 <= snippet.start < snippet.end
        and span.start <= span.end
        and snippet.start <= span.start + 0.001
        and snippet.end >= span.end - 0.001
        and _snippet_hears_retained_cues(snippet, span)
    )


def _snippet_hears_retained_cues(snippet: AudioSnippet, span: DivergenceSpan) -> bool:
    """Each edited cue that keeps words beside the case is heard where they end.

    A case is timed by its ASR words alone. When they are seconds away from
    the retained words of the cue whose text it edits, a clip around them does
    not contain that cue, and its answer cannot judge the cue's source text.
    The clip must contain or reach within one intra-cue gap of a retained edge
    of every such cue (either edge of a cue whose words surround the case).
    """
    if not span.srt_token_indices:
        return True
    edges: dict[int, list[float]] = {}
    for cue_id, time in (
        (span.left_anchor_cue_id, span.left_anchor_end), (span.right_anchor_cue_id, span.right_anchor_start),
    ):
        if cue_id in span.cue_ids and time is not None and isfinite(time):
            edges.setdefault(cue_id, []).append(time)
    reach = _RETAINED_EDGE_REACH_SECONDS + 0.001
    return all(
        any(snippet.start - reach <= time <= snippet.end + reach for time in times) for times in edges.values()
    )


def _unavailable_audio_decision(span: DivergenceSpan) -> dict[str, object]:
    return {
        "case_id": span.case_id, "verdict": "keep_srt", "final_text": span.srt_text,
        "confidence": 0.0, "speaker": None, "character": "unknown",
        "reason": "Required case audio was unavailable or incomplete; preserved source SRT for review.",
    }


def _is_request_timeout(exc: BaseException) -> bool:
    current = exc
    for _ in range(4):
        if isinstance(current, TimeoutError) or getattr(current, "code", None) == 504:
            return True
        if type(current).__name__ in {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
            return True
        current = current.__cause__
        if current is None:
            break
    return False


# The audio cannot tell a kanji spelling from kana of the same sound: 大井 and
# おい, a name and an interjection. A clear hearing that spells two or more of
# the source's kanji in kana is therefore no evidence for the change; the
# customer's spelling stays and the proposal is reviewed. One kanji (様 -> さん)
# is a real wording change and still applies.
_KANA_RESPELLING_HOLD = (
    "The proposed wording spells source kanji in kana; the audio cannot tell a same-sounding "
    "respelling (such as a name) from a different word"
)
_MIN_RESPELLED_KANJI = 2


def _japanese_script(character: str) -> str:
    name = unicodedata.name(character, "")
    if character in "々〆" or name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return "kanji"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "kana"
    return "other"


def respells_kanji_in_kana(source: str, proposed: str) -> bool:
    """Whether ``proposed`` replaces a run of two or more source kanji with kana only."""
    source = unicodedata.normalize("NFKC", source)
    proposed = unicodedata.normalize("NFKC", proposed)
    matcher = SequenceMatcher(None, source, proposed, autojunk=False)
    for operation, first, last, start, end in matcher.get_opcodes():
        if operation != "replace":
            continue
        replaced, replacement = source[first:last], proposed[start:end]
        if (
            len(replaced) >= _MIN_RESPELLED_KANJI
            and all(_japanese_script(character) == "kanji" for character in replaced)
            and all(_japanese_script(character) == "kana" for character in replacement)
        ):
            return True
    return False


def confidence_gated_decision(
    span: DivergenceSpan,
    decision: AdjudicationDecision,
    confidence_gate: float,
    *, policy: DeterministicAdjudicationPolicy | None = None,
) -> tuple[AdjudicationDecision, QCFlag | None]:
    """Keep uncertain proposed wording reviewable without applying it to the SRT."""
    from .hybrid_adjudication import _evidence_supports_wording

    uncertain_audio = decision.evidence in {"heard_unclear", "not_audible"}
    if uncertain_audio and not any(alphanumeric_signature(text) for text in (
        span.srt_text, span.asr_text, decision.final_text, decision.heard_text or "",
    )):
        # Source, ASR and hearing all agree that no word is spoken (an ASR
        # punctuation mark between cues). The absence is confirmed, not held.
        return decision.model_copy(update={"verdict": "keep_srt", "final_text": span.srt_text}), None
    uncertain_source_keep = (
        decision.verdict == "keep_srt" and decision.final_text == span.srt_text
        and decision.evidence == "heard_clearly" and not _evidence_supports_wording(decision, policy)
    )
    kana_respelling = (
        not uncertain_audio and decision.verdict != "keep_srt"
        and respells_kanji_in_kana(span.srt_text, decision.final_text)
    )
    if (
        not uncertain_audio and not uncertain_source_keep and not kana_respelling
        and decision.confidence >= confidence_gate
    ):
        return decision, None
    hold_reason = ("Reported hearing differs from the preserved source wording and their equivalence is unresolved"
                   if uncertain_source_keep else "Adjudication audio evidence is unclear or inaudible" if uncertain_audio
                   else _KANA_RESPELLING_HOLD if kana_respelling
                   else "Adjudication confidence is below the configured gate")
    flag = QCFlag(
        kind="low_confidence_adjudication",
        cue_ids=span.cue_ids,
        message=(
            f"{hold_reason}; source SRT was preserved. "
            f"Proposed verdict: {decision.verdict}. Reason: {decision.reason}"
        ),
        # Hearing can be clear while spelling/wording equivalence is unknown.
        # Preserve the decision's actual evidence across serialization; the
        # independent review flag records this semantic uncertainty.
        confidence=0.0 if uncertain_source_keep else decision.confidence,
        old_text=span.srt_text,
        new_text=decision.heard_text if uncertain_source_keep else decision.final_text,
        start=span.start,
        end=span.end,
    )
    if decision.verdict == "keep_srt":
        return decision, flag
    return decision.model_copy(update={
        "verdict": "keep_srt",
        "final_text": span.srt_text,
        "reason": f"{hold_reason}; preserved source SRT for review.",
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
    return DeterministicAdjudicationPolicy().decide(span)


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
        confidence=DETERMINISTIC_KEEP_CONFIDENCE,
        speaker=span.speaker_ids[0] if span.speaker_ids else None,
        character="unknown",
        reason=reason,
    )
