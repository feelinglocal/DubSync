"""Conservative wording corroboration from two independent ASR word streams.

The secondary stream is evidence only. Nothing in this module changes words,
timestamps, cue ownership, or segmentation. Matching uses exact lexical tokens
(case and punctuation ignored), never the aligner's semantic/number aliases.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal, Sequence

from .models import AdjudicationDecision, DivergenceSpan, Word
from .text_metrics import contains_character_level_script, token_texts

if TYPE_CHECKING:
    from .adjudication_policy import DeterministicAdjudicationPolicy

CROSS_CHECK_POLICY_VERSION = 4
CrossCheckLabel = Literal[
    "both_agree", "secondary_matches_script", "primary_only_insertion", "conflict", "ambiguous",
]
_START_TOLERANCE = 0.4
_END_TOLERANCE = 0.7


@dataclass(frozen=True)
class _Token:
    text: str
    word_index: int
    start: float
    end: float


@dataclass(frozen=True)
class StreamAgreement:
    primary_words: tuple[Word, ...]
    secondary_words: tuple[Word, ...]
    primary_tokens: tuple[_Token, ...]
    secondary_tokens: tuple[_Token, ...]
    token_matches: tuple[int | None, ...]
    ambiguous_token_indices: frozenset[int]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "primary_tokens": len(self.primary_tokens),
            "secondary_tokens": len(self.secondary_tokens),
            "agreed_tokens": sum(index is not None for index in self.token_matches),
            "ambiguous_tokens": len(self.ambiguous_token_indices),
        }


@dataclass(frozen=True)
class SpanCrossCheck:
    case_id: str
    label: CrossCheckLabel
    primary_word_indices: tuple[int, ...]
    secondary_word_indices: tuple[int, ...] = ()
    secondary_text: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "primary_word_indices": list(self.primary_word_indices),
            "secondary_word_indices": list(self.secondary_word_indices),
            "secondary_text": self.secondary_text,
            "reason": self.reason,
        }


def _keys(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in token_texts(text))


def _tokens(words: Sequence[Word]) -> tuple[_Token, ...]:
    return tuple(_Token(key, index, word.start, word.end)
                 for index, word in enumerate(words) for key in _keys(word.text))


def _valid_time(token: _Token) -> bool:
    return all(math.isfinite(value) for value in (token.start, token.end)) and 0 <= token.start <= token.end


def _near(left: _Token, right: _Token) -> bool:
    return (
        _valid_time(left) and _valid_time(right)
        and abs(left.start - right.start) <= _START_TOLERANCE + 1e-9
        and abs(left.end - right.end) <= _END_TOLERANCE + 1e-9
    )


def compare_word_streams(primary_words: Sequence[Word], secondary_words: Sequence[Word]) -> StreamAgreement:
    """Match time-local tokens only when the match is unique in both directions.

    Repeated words with multiple plausible matches and crossed word order remain
    ambiguous. Time-indexed lexical buckets keep work bounded on long episodes.
    """
    primary, secondary = _tokens(primary_words), _tokens(secondary_words)
    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, token in enumerate(secondary):
        if _valid_time(token):
            buckets[token.text].append((token.start, index))
    for bucket in buckets.values():
        bucket.sort()
    candidates: list[list[int]] = []
    reverse_counts: Counter[int] = Counter()
    for token in primary:
        bucket = buckets.get(token.text, [])
        left = bisect_left(bucket, (token.start - _START_TOLERANCE, -1))
        right = bisect_right(bucket, (token.start + _START_TOLERANCE, len(secondary)))
        matches = [index for _, index in bucket[left:right] if _near(token, secondary[index])]
        candidates.append(matches)
        reverse_counts.update(matches)
    ambiguous = {index for index, matches in enumerate(candidates)
                 if len(matches) > 1 or (matches and reverse_counts[matches[0]] > 1)}
    mapping = [matches[0] if len(matches) == 1 and index not in ambiguous else None
               for index, matches in enumerate(candidates)]
    # Reject every participant of an inversion, including a longer reordered run.
    matched = [(index, target) for index, target in enumerate(mapping) if target is not None]
    max_target = -1
    for index, target in matched:
        if target < max_target:
            ambiguous.add(index)
        max_target = max(max_target, target)
    min_target = len(secondary)
    for index, target in reversed(matched):
        if target > min_target:
            ambiguous.add(index)
        min_target = min(min_target, target)
    for index in ambiguous:
        mapping[index] = None
    return StreamAgreement(tuple(primary_words), tuple(secondary_words), primary, secondary,
                           tuple(mapping), frozenset(ambiguous))


def _secondary_window(agreement: StreamAgreement, first: int, last: int) -> tuple[int, ...]:
    primary, secondary = agreement.primary_tokens, agreement.secondary_tokens
    start, end = primary[first].start, primary[last].end
    if not math.isfinite(start) or not math.isfinite(end):
        return ()
    before = agreement.token_matches[first - 1] if first else -1
    after = agreement.token_matches[last + 1] if last + 1 < len(primary) else len(secondary)
    if before is not None and after is not None and before < after:
        indices = tuple(range(before + 1, after))
    else:
        # Without two reliable anchors, only a tight local group is evidence.
        indices = tuple(index for index, token in enumerate(secondary)
                        if _valid_time(token) and start - 0.1 <= (token.start + token.end) / 2 <= end + 0.1)
    if not indices or indices != tuple(range(indices[0], indices[-1] + 1)):
        return ()
    if (abs(secondary[indices[0]].start - start) > _START_TOLERANCE
            or abs(secondary[indices[-1]].end - end) > _END_TOLERANCE):
        return ()
    return indices


def classify_spans(spans: Sequence[DivergenceSpan], agreement: StreamAgreement) -> list[SpanCrossCheck]:
    """Describe corroboration for already aligned primary divergence spans."""
    results: list[SpanCrossCheck] = []
    tokens_by_word: dict[int, list[int]] = defaultdict(list)
    for index, token in enumerate(agreement.primary_tokens):
        tokens_by_word[token.word_index].append(index)
    for span in spans:
        owned = tuple(span.asr_word_indices)
        if (not owned or owned != tuple(range(owned[0], owned[-1] + 1))
                or owned[0] < 0 or owned[-1] >= len(agreement.primary_words)):
            results.append(SpanCrossCheck(span.case_id, "ambiguous", owned, reason="Primary ownership is not a contiguous word group."))
            continue
        token_indices = tuple(index for word in owned for index in tokens_by_word[word])
        primary_keys = tuple(agreement.primary_tokens[index].text for index in token_indices)
        if not token_indices or primary_keys != _keys(span.asr_text):
            results.append(SpanCrossCheck(span.case_id, "ambiguous", owned, reason="Span text does not exactly describe its primary words."))
            continue
        matched = tuple(agreement.token_matches[index] for index in token_indices)
        if any(index in agreement.ambiguous_token_indices for index in token_indices):
            results.append(SpanCrossCheck(span.case_id, "ambiguous", owned, reason="Repeated or reordered tokens have more than one plausible mapping."))
            continue
        all_agree = (all(index is not None for index in matched)
                     and matched == tuple(range(matched[0], matched[-1] + 1))) if matched[0] is not None and matched[-1] is not None else False
        secondary_indices = matched if all_agree else _secondary_window(agreement, token_indices[0], token_indices[-1])
        secondary_words = tuple(dict.fromkeys(agreement.secondary_tokens[index].word_index for index in secondary_indices))
        secondary_text = " ".join(agreement.secondary_words[index].text for index in secondary_words)
        secondary_keys = tuple(agreement.secondary_tokens[index].text for index in secondary_indices)
        if all_agree:
            label, reason = "both_agree", "Both models produced the same exact wording in one unambiguous local group."
        elif not _keys(span.srt_text) and len(secondary_keys) < len(primary_keys):
            label, reason = "primary_only_insertion", "The proposed insertion is not fully corroborated by the secondary model."
        elif secondary_keys and secondary_keys == _keys(span.srt_text):
            label, reason = "secondary_matches_script", "The secondary wording matches the script and contradicts the primary."
        else:
            label, reason = "conflict", "The models do not provide complete, unambiguous wording agreement."
        results.append(SpanCrossCheck(span.case_id, label, owned, secondary_words, secondary_text, reason))
    return results


def preaccepted_decision(
    span: DivergenceSpan, check: SpanCrossCheck, *, language: str | None = None,
    source_names: frozenset[tuple[str, ...]] = frozenset(),
) -> AdjudicationDecision | None:
    """Pre-accept only low-risk multiword substitutions within one source cue.

    Agreement alone cannot resolve names, quantities, negation, near-spelling
    inflections, or ownership across cues. Those cases still need audio review.
    The caller must apply its deterministic/source-preservation policy first.
    """
    from .hybrid_adjudication import _risk_reasons

    # Character-level comparison tokens are not independently identified
    # words: a single place/person name can span several such tokens. Keep
    # agreement labels and audio review available, but do not bypass review
    # for this tokenization without a validated word-boundary/name policy.
    if any(contains_character_level_script(text) for text in (span.srt_text, span.asr_text)):
        return None
    source, spoken = _keys(span.srt_text), _keys(span.asr_text)
    if (span.case_id != check.case_id or check.label != "both_agree"
            or tuple(span.asr_word_indices) != check.primary_word_indices
            or len(span.cue_ids) != 1 or min(len(source), len(spoken)) < 2
            or source == spoken or _risk_reasons(span, language=language, source_names=source_names)):
        return None
    for tag, first, last, other_first, other_last in SequenceMatcher(None, source, spoken, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        # A recognizer pair can share the same dropped suffix or vowel error.
        # Keep close lexical variants reviewable instead of treating agreement
        # as independent proof of inflection/spelling correctness.
        if any(SequenceMatcher(None, old, new, autojunk=False).ratio() >= 0.6
               for old in source[first:last] for new in spoken[other_first:other_last]):
            return None
    if len(source) > len(spoken) * 2 or len(spoken) > len(source) * 2:
        # Large omissions/additions can be a span ownership failure even when
        # the words themselves are present in both transcriptions.
        return None
    return AdjudicationDecision(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
                                confidence=1, reason="Dual ASR cross-check: exact primary wording corroborated.",
                                evidence="heard_clearly", heard_text=span.asr_text)


def enforce_crosscheck_decisions(
    spans: Sequence[DivergenceSpan],
    decisions: Sequence[AdjudicationDecision],
    checks: Sequence[SpanCrossCheck],
    *, policy: DeterministicAdjudicationPolicy | None = None,
) -> list[AdjudicationDecision]:
    """Apply cross-check holds after review; legacy confidence is not clear hearing.

    Uncorroborated insertions and substitutions contradicted by secondary/script
    agreement require explicit clear audio review before changing the SRT.
    Secondary silence alone cannot disprove speech heard by an audio reviewer.
    """
    span_by_id = {span.case_id: span for span in spans}
    by_id = {decision.case_id: decision for decision in decisions}
    for check in checks:
        span = span_by_id.get(check.case_id)
        if span is None:
            continue
        decision = by_id.get(check.case_id)
        # Source-authority and deterministic holds have already prevented the
        # edit. Do not replace their explanation or manufacture a new warning.
        if decision is not None and decision.verdict == "keep_srt":
            continue
        clearly_heard = decision is not None and decision.evidence == "heard_clearly" and (
            _keys(decision.heard_text or "") == _keys(decision.final_text)
        )
        if decision is not None and decision.evidence == "heard_clearly" and not clearly_heard and policy is not None:
            equivalent = policy.decide(DivergenceSpan(
                case_id=span.case_id, cue_ids=[], srt_text=decision.final_text, asr_text=decision.heard_text or "",
            ))
            clearly_heard = equivalent is not None and equivalent.verdict == "keep_srt"
        insertion_hold = not _keys(span.srt_text) and check.label != "both_agree" and not clearly_heard
        conflict_hold = (check.label == "secondary_matches_script" and (
            decision is None or (decision.verdict != "keep_srt" and not clearly_heard)
        ))
        if insertion_hold or conflict_hold:
            by_id[check.case_id] = AdjudicationDecision(
                case_id=check.case_id, verdict="keep_srt", final_text=span.srt_text, confidence=0,
                evidence="heard_unclear", heard_text="",
                reason=f"Dual ASR cross-check hold: {check.reason}",
            )
    return list(by_id.values())
