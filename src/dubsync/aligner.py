from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Literal

from rapidfuzz import fuzz

from .alignment_windows import (
    RETRY_MARGINS as ALIGNMENT_RETRY_MARGINS,
    band_cell_count as _band_cell_count,
    band_windows as _band_windows,
    interval_cell_count as _interval_cell_count,
    iter_interval_cells as _iter_interval_cells,
    reachability_centers as _reachability_centers,
    retry_margins as _retry_margins,
    row_offset as _row_offset,
    unique_exact_pairs as _unique_exact_pairs,
)
from .models import (
    AlignmentDiagnostics,
    AlignmentResult,
    AnchorRegion,
    Cue,
    DivergenceSpan,
    QCFlag,
    TokenMatch,
    Word,
)
from .subtitle_annotations import cue_has_bracketed_screen_text
from .tokenize import SRTToken, normalized_words, tokenize_cues

MATCH_THRESHOLD = 0.85
MIN_ANCHOR_TOKENS = 3
BAND_MARGIN = 64
ALIGNMENT_CELL_BUDGET = 2_000_000
LOCAL_TRANSPOSITION_RADIUS = 2
IMPLAUSIBLE_MATCHED_WORD_SECONDS = 2.0
IMPLAUSIBLE_WORD_TO_CUE_RATIO = 2.0
NEG_INF = -1_000_000_000.0
TIME_PRIOR_MAX_BONUS = 0.2
TIME_PRIOR_MIN_RADIUS_SECONDS = 2.0
ALIGNMENT_OUTLIER_SECONDS = 12.0
MISSING_AUDIO_GUARD_VERSION = 3
_BACK_NONE, _BACK_MATCH, _BACK_DELETE, _BACK_INSERT = range(4)

@dataclass(frozen=True)
class _Op:
    kind: str
    srt_index: int | None = None
    asr_index: int | None = None
    score: float = 0.0


@dataclass(frozen=True)
class _TimeTransform:
    rate: float
    offset_seconds: float
    anchor_count: int


@dataclass(frozen=True)
class _TimingPriors:
    token_priors: list[tuple[float, float]]
    word_centers: list[float]
    transform: _TimeTransform | None = None


@dataclass(frozen=True)
class _AlignmentRun:
    ops: list[_Op]
    unbanded_fallback: bool = False
    band_limited: bool = False
    unresolved: bool = False


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return fuzz.ratio(left, right, score_cutoff=MATCH_THRESHOLD * 100.0) / 100.0


def _timing_priors(
    cues: list[Cue],
    tokens: list[SRTToken],
    words: list[Word],
    words_norm: list[str],
    preliminary_ops: list[_Op],
) -> _TimingPriors | None:
    """Build a bounded time tie breaker when both timelines are usable.

    Unique exact matches from the preliminary text-only alignment estimate a robust
    global offset/rate for shifted episodes. The prior still contributes only a small
    positive bonus and never penalizes a text match outside its local window.
    """

    cue_ids = [cue.index for cue in cues]
    cue_intervals = [(cue.start_ms / 1000.0, cue.end_ms / 1000.0) for cue in cues]
    word_intervals = [(word.start, word.end) for word in words]
    if len(set(cue_ids)) != len(cue_ids):
        return None
    if not _ordered_usable_intervals(cue_intervals) or not _ordered_usable_intervals(
        word_intervals
    ):
        return None

    token_indices_by_cue: dict[int, list[int]] = {}
    for token_index, token in enumerate(tokens):
        token_indices_by_cue.setdefault(token.cue_id, []).append(token_index)

    cue_by_id = {cue.index: cue for cue in cues}
    token_priors: list[tuple[float, float] | None] = [None] * len(tokens)
    for cue_id, token_indices in token_indices_by_cue.items():
        cue = cue_by_id.get(cue_id)
        if cue is None:
            return None
        start = cue.start_ms / 1000.0
        duration = (cue.end_ms - cue.start_ms) / 1000.0
        radius = max(TIME_PRIOR_MIN_RADIUS_SECONDS, duration)
        count = len(token_indices)
        for position, token_index in enumerate(token_indices):
            expected = start + duration * ((position + 0.5) / count)
            token_priors[token_index] = (expected, radius)

    if any(prior is None for prior in token_priors):
        return None
    word_centers = [(start + end) / 2.0 for start, end in word_intervals]
    complete_priors = [prior for prior in token_priors if prior is not None]
    transform = _anchor_time_transform(
        tokens,
        words_norm,
        preliminary_ops,
        complete_priors,
        word_centers,
    )
    if transform is None:
        return _TimingPriors(token_priors=complete_priors, word_centers=word_centers)
    transformed_priors = [
        (
            transform.offset_seconds + transform.rate * expected,
            max(TIME_PRIOR_MIN_RADIUS_SECONDS, transform.rate * radius),
        )
        for expected, radius in complete_priors
    ]
    return _TimingPriors(
        token_priors=transformed_priors,
        word_centers=word_centers,
        transform=transform,
    )


def _anchor_time_transform(
    tokens: list[SRTToken],
    words_norm: list[str],
    preliminary_ops: list[_Op],
    token_time_priors: list[tuple[float, float]],
    word_time_centers: list[float],
) -> _TimeTransform | None:
    token_counts = Counter(token.normalized for token in tokens if token.normalized)
    word_counts = Counter(word for word in words_norm if word)
    anchors = [
        (token_time_priors[op.srt_index][0], word_time_centers[op.asr_index])
        for op in preliminary_ops
        if op.kind == "match"
        and op.srt_index is not None
        and op.asr_index is not None
        and token_counts[tokens[op.srt_index].normalized] == 1
        and word_counts[words_norm[op.asr_index]] == 1
    ]
    if not anchors:
        return None
    if any(
        left_cue >= right_cue or left_audio > right_audio
        for (left_cue, left_audio), (right_cue, right_audio) in zip(
            anchors,
            anchors[1:],
        )
    ):
        return None

    return _fit_time_transform(anchors)


def _fit_time_transform(anchors: list[tuple[float, float]]) -> _TimeTransform | None:
    if not anchors:
        return None

    offset_only = median(audio_time - cue_time for cue_time, audio_time in anchors)
    chosen_rate = 1.0
    chosen_offset = offset_only
    chosen_residual = median(
        abs((cue_time + offset_only) - audio_time) for cue_time, audio_time in anchors
    )

    cue_span = anchors[-1][0] - anchors[0][0]
    if len(anchors) >= 2 and cue_span >= 5.0:
        sampled = _sample_anchors(anchors, limit=128)
        minimum_separation = max(1.0, cue_span * 0.1)
        slopes = [
            (right_audio - left_audio) / (right_cue - left_cue)
            for left_index, (left_cue, left_audio) in enumerate(sampled)
            for right_cue, right_audio in sampled[left_index + 1 :]
            if right_cue - left_cue >= minimum_separation
        ]
        if slopes:
            candidate_rate = median(slopes)
            if 0.5 <= candidate_rate <= 2.0 and math.isfinite(candidate_rate):
                candidate_offset = median(
                    audio_time - candidate_rate * cue_time
                    for cue_time, audio_time in anchors
                )
                candidate_residual = median(
                    abs((candidate_rate * cue_time + candidate_offset) - audio_time)
                    for cue_time, audio_time in anchors
                )
                if candidate_residual < chosen_residual:
                    chosen_rate = candidate_rate
                    chosen_offset = candidate_offset
                    chosen_residual = candidate_residual

    audio_span = anchors[-1][1] - anchors[0][1]
    if chosen_residual > max(2.0, abs(audio_span) * 0.05):
        return None
    return _TimeTransform(
        rate=chosen_rate,
        offset_seconds=chosen_offset,
        anchor_count=len(anchors),
    )


def _sample_anchors(
    anchors: list[tuple[float, float]],
    limit: int,
) -> list[tuple[float, float]]:
    if len(anchors) <= limit:
        return anchors
    last_index = len(anchors) - 1
    return [anchors[round(position * last_index / (limit - 1))] for position in range(limit)]


def _has_repeated_alignment_candidates(tokens: list[SRTToken], words_norm: list[str]) -> bool:
    token_counts = Counter(token.normalized for token in tokens if token.normalized)
    word_counts = Counter(word for word in words_norm if word)
    return any(
        token_counts[value] > 1 or word_counts[value] > 1
        for value in token_counts.keys() & word_counts.keys()
    )


def _ordered_usable_intervals(intervals: list[tuple[float, float]]) -> bool:
    if not intervals:
        return False
    if any(
        not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start
        for start, end in intervals
    ):
        return False
    return all(left[0] <= right[0] for left, right in zip(intervals, intervals[1:]))


def _timing_bonus(
    token_index: int,
    word_index: int,
    token_time_priors: list[tuple[float, float]] | None,
    word_time_centers: list[float] | None,
) -> float:
    if token_time_priors is None or word_time_centers is None:
        return 0.0
    expected, radius = token_time_priors[token_index]
    distance = abs(expected - word_time_centers[word_index])
    if distance >= radius:
        return 0.0
    return TIME_PRIOR_MAX_BONUS * (1.0 - distance / radius)


def _align_tokens(
    tokens: list[SRTToken],
    words_norm: list[str],
    band_margin: int = BAND_MARGIN,
    *,
    token_time_priors: list[tuple[float, float]] | None = None,
    word_time_centers: list[float] | None = None,
) -> list[_Op]:
    return _align_tokens_detailed(
        tokens,
        words_norm,
        band_margin=band_margin,
        token_time_priors=token_time_priors,
        word_time_centers=word_time_centers,
    ).ops


def _align_tokens_detailed(
    tokens: list[SRTToken],
    words_norm: list[str],
    band_margin: int = BAND_MARGIN,
    *,
    token_time_priors: list[tuple[float, float]] | None = None,
    word_time_centers: list[float] | None = None,
) -> _AlignmentRun:
    n = len(tokens)
    m = len(words_norm)
    reachability = _reachability_centers(
        tokens,
        words_norm,
        token_time_priors,
        word_time_centers,
    )
    band_limited = False
    remaining_cells = ALIGNMENT_CELL_BUDGET
    for attempt_index, margin in enumerate(_retry_margins(band_margin, max(n, m))):
        cell_count = _band_cell_count(n, m, margin, reachability)
        if cell_count > remaining_cells:
            band_limited = True
            continue
        remaining_cells -= cell_count
        ops = _align_tokens_once(
            tokens,
            words_norm,
            margin,
            reachability,
            token_time_priors=token_time_priors,
            word_time_centers=word_time_centers,
        )
        if ops is None:
            band_limited = band_limited or margin >= ALIGNMENT_RETRY_MARGINS[-1]
            continue
        run = _AlignmentRun(
            ops=ops,
            unbanded_fallback=attempt_index > 0 and margin >= max(n, m),
        )
        if margin < max(n, m) and _misses_unique_exact_pair(ops, tokens, words_norm):
            continue
        return run
    return _AlignmentRun(
        ops=_fully_divergent_ops(n, m),
        band_limited=True,
        unresolved=True,
    )


def _fully_divergent_ops(token_count: int, word_count: int) -> list[_Op]:
    return [
        *(_Op("delete", token_index, None, 0.0) for token_index in range(token_count)),
        *(_Op("insert", None, word_index, 0.0) for word_index in range(word_count)),
    ]


def _align_tokens_once(
    tokens: list[SRTToken],
    words_norm: list[str],
    band_margin: int,
    reachability: dict[int, tuple[int, ...]],
    *,
    token_time_priors: list[tuple[float, float]] | None = None,
    word_time_centers: list[float] | None = None,
) -> list[_Op] | None:
    n = len(tokens)
    m = len(words_norm)
    gap = -0.75
    first_intervals = _band_windows(0, n, m, band_margin, reachability.get(0, ()))
    previous_scores = [NEG_INF] * _interval_cell_count(first_intervals)
    first_back = bytearray(len(previous_scores))
    zero_offset = _row_offset(first_intervals, 0)
    if zero_offset is None:
        return None
    previous_scores[zero_offset] = 0.0
    for offset, j in _iter_interval_cells(first_intervals):
        if j == 0:
            continue
        previous_offset = _row_offset(first_intervals, j - 1)
        if previous_offset is None:
            continue
        previous_score = previous_scores[previous_offset]
        if previous_score <= NEG_INF / 2:
            continue
        previous_scores[offset] = previous_score + gap
        first_back[offset] = _BACK_INSERT
    previous_intervals = first_intervals
    back_rows: list[tuple[list[tuple[int, int]], bytearray]] = [(first_intervals, first_back)]

    for i in range(1, n + 1):
        current_intervals = _band_windows(i, n, m, band_margin, reachability.get(i, ()))
        current_scores = [NEG_INF] * _interval_cell_count(current_intervals)
        current_back = bytearray(len(current_scores))
        for offset, j in _iter_interval_cells(current_intervals):
            best_score = NEG_INF
            best_op = _BACK_NONE
            previous_offset = _row_offset(previous_intervals, j)
            if previous_offset is not None:
                best_score = previous_scores[previous_offset] + gap
                best_op = _BACK_DELETE
            current_previous_offset = _row_offset(current_intervals, j - 1) if j > 0 else None
            if (
                current_previous_offset is not None
                and current_scores[current_previous_offset] + gap > best_score
            ):
                best_score = current_scores[current_previous_offset] + gap
                best_op = _BACK_INSERT
            previous_diagonal_offset = _row_offset(previous_intervals, j - 1) if j > 0 else None
            if previous_diagonal_offset is not None:
                similarity = _similarity(tokens[i - 1].normalized, words_norm[j - 1])
                match_score = 2.0 * similarity if similarity >= MATCH_THRESHOLD else -0.6
                candidate = previous_scores[previous_diagonal_offset] + match_score
                if similarity >= MATCH_THRESHOLD:
                    candidate += _timing_bonus(
                        i - 1,
                        j - 1,
                        token_time_priors,
                        word_time_centers,
                    )
                if candidate > best_score:
                    best_score = candidate
                    best_op = _BACK_MATCH
            if best_op:
                current_scores[offset] = best_score
                current_back[offset] = best_op
        previous_intervals = current_intervals
        previous_scores = current_scores
        back_rows.append((current_intervals, current_back))

    final_offset = _row_offset(previous_intervals, m)
    if final_offset is None or previous_scores[final_offset] <= NEG_INF / 2:
        return None

    ops: list[_Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        intervals, row_back = back_rows[i]
        offset = _row_offset(intervals, j)
        op = _BACK_NONE if offset is None else row_back[offset]
        if op == _BACK_MATCH:
            score = _similarity(tokens[i - 1].normalized, words_norm[j - 1])
            kind = "match" if tokens[i - 1].normalized == words_norm[j - 1] else "replace"
            ops.append(_Op(kind, i - 1, j - 1, score))
            i -= 1
            j -= 1
        elif op == _BACK_DELETE:
            ops.append(_Op("delete", i - 1, None, 0.0))
            i -= 1
        elif op == _BACK_INSERT:
            ops.append(_Op("insert", None, j - 1, 0.0))
            j -= 1
        else:
            return None
    ops.reverse()
    return ops


def _misses_unique_exact_pair(
    ops: list[_Op],
    tokens: list[SRTToken],
    words_norm: list[str],
) -> bool:
    expected = set(_unique_exact_pairs(tokens, words_norm))
    if not expected:
        return False
    matched = {
        (op.srt_index, op.asr_index)
        for op in ops
        if op.kind == "match" and op.srt_index is not None and op.asr_index is not None
    }
    return any(
        pair not in matched and not _is_locally_explained_transposition(pair, matched)
        for pair in expected
    )


def _is_locally_explained_transposition(
    missed_pair: tuple[int, int],
    matched_pairs: set[tuple[int, int]],
) -> bool:
    """Allow one locally reordered word without discarding an otherwise sound run."""

    missed_srt, missed_asr = missed_pair
    for srt_delta in range(-LOCAL_TRANSPOSITION_RADIUS, LOCAL_TRANSPOSITION_RADIUS + 1):
        if srt_delta == 0:
            continue
        for asr_delta in range(-LOCAL_TRANSPOSITION_RADIUS, LOCAL_TRANSPOSITION_RADIUS + 1):
            if asr_delta == 0 or srt_delta * asr_delta >= 0:
                continue
            crossed_srt = missed_srt + srt_delta
            crossed_asr = missed_asr + asr_delta
            if (crossed_srt, crossed_asr) not in matched_pairs:
                continue

            srt_start, srt_end = sorted((missed_srt, crossed_srt))
            asr_start, asr_end = sorted((missed_asr, crossed_asr))
            has_left_anchor = _has_nearby_monotonic_anchor(matched_pairs, srt_start, asr_start, -1)
            has_right_anchor = _has_nearby_monotonic_anchor(matched_pairs, srt_end, asr_end, 1)
            if has_left_anchor and has_right_anchor:
                return True
    return False


def _has_nearby_monotonic_anchor(
    matched_pairs: set[tuple[int, int]],
    srt_index: int,
    asr_index: int,
    direction: int,
) -> bool:
    for srt_delta in range(1, LOCAL_TRANSPOSITION_RADIUS + 1):
        for asr_delta in range(1, LOCAL_TRANSPOSITION_RADIUS + 1):
            candidate = (srt_index + direction * srt_delta, asr_index + direction * asr_delta)
            if candidate in matched_pairs:
                return True
    return False


def _span_text_from_tokens(tokens: list[SRTToken], indices: list[int]) -> str:
    return " ".join(tokens[index].text for index in indices)


def _span_text_from_words(words: list[Word], indices: list[int]) -> str:
    return " ".join(words[index].text for index in indices)


def _build_divergences(ops: list[_Op], tokens: list[SRTToken], words: list[Word]) -> list[DivergenceSpan]:
    spans: list[DivergenceSpan] = []
    srt_indices: list[int] = []
    asr_indices: list[int] = []
    previous_match: _Op | None = None

    def boundary_start(next_match: _Op | None) -> float | None:
        if asr_indices:
            return min(words[index].start for index in asr_indices)
        if previous_match is not None and previous_match.asr_index is not None:
            return words[previous_match.asr_index].end
        if next_match is not None and next_match.asr_index is not None:
            return words[next_match.asr_index].start
        return None

    def boundary_end(next_match: _Op | None) -> float | None:
        if asr_indices:
            return max(words[index].end for index in asr_indices)
        if next_match is not None and next_match.asr_index is not None:
            return words[next_match.asr_index].start
        if previous_match is not None and previous_match.asr_index is not None:
            return words[previous_match.asr_index].end
        return None

    def flush(next_match: _Op | None = None) -> None:
        if not srt_indices and not asr_indices:
            return
        cue_ids = sorted({tokens[index].cue_id for index in srt_indices})
        confidences = [
            words[index].confidence
            for index in asr_indices
            if words[index].confidence is not None
        ]
        speaker_ids = sorted({words[index].speaker_id for index in asr_indices if words[index].speaker_id})
        start = boundary_start(next_match)
        end = boundary_end(next_match)
        pure_insertion = bool(asr_indices) and not srt_indices
        left_anchor_cue_id = (
            tokens[previous_match.srt_index].cue_id
            if pure_insertion and previous_match is not None and previous_match.srt_index is not None
            else None
        )
        right_anchor_cue_id = (
            tokens[next_match.srt_index].cue_id
            if pure_insertion and next_match is not None and next_match.srt_index is not None
            else None
        )
        insertion_token_offset = None
        if (
            left_anchor_cue_id is not None
            and left_anchor_cue_id == right_anchor_cue_id
            and next_match is not None
            and next_match.srt_index is not None
        ):
            insertion_token_offset = sum(
                1
                for token in tokens[: next_match.srt_index]
                if token.cue_id == right_anchor_cue_id
            )
        left_anchor_word = (
            words[previous_match.asr_index]
            if pure_insertion and previous_match is not None and previous_match.asr_index is not None
            else None
        )
        right_anchor_word = (
            words[next_match.asr_index]
            if pure_insertion and next_match is not None and next_match.asr_index is not None
            else None
        )
        case_number = len(spans) + 1
        spans.append(
            DivergenceSpan(
                case_id=f"case-{case_number}",
                cue_ids=cue_ids,
                srt_text=_span_text_from_tokens(tokens, srt_indices),
                asr_text=_span_text_from_words(words, asr_indices),
                start=start,
                end=end,
                confidence=min(confidences) if confidences else 0.0,
                speaker_ids=speaker_ids,
                srt_token_indices=list(srt_indices),
                asr_word_indices=list(asr_indices),
                left_anchor_cue_id=left_anchor_cue_id,
                right_anchor_cue_id=right_anchor_cue_id,
                insertion_token_offset=insertion_token_offset,
                left_anchor_end=left_anchor_word.end if left_anchor_word is not None else None,
                right_anchor_start=right_anchor_word.start if right_anchor_word is not None else None,
                left_anchor_speaker_id=left_anchor_word.speaker_id if left_anchor_word is not None else None,
                right_anchor_speaker_id=right_anchor_word.speaker_id if right_anchor_word is not None else None,
            )
        )
        srt_indices.clear()
        asr_indices.clear()

    for op in ops:
        if op.kind == "match":
            flush(op)
            previous_match = op
            continue
        if op.srt_index is not None:
            srt_indices.append(op.srt_index)
        if op.asr_index is not None:
            asr_indices.append(op.asr_index)
    flush()
    return spans


def _build_anchor_regions(
    ops: list[_Op],
    tokens: list[SRTToken],
    words: list[Word],
    min_tokens: int = MIN_ANCHOR_TOKENS,
) -> list[AnchorRegion]:
    regions: list[AnchorRegion] = []
    run: list[_Op] = []

    def flush() -> None:
        if len(run) < min_tokens:
            run.clear()
            return
        srt_indices = [op.srt_index for op in run if op.srt_index is not None]
        asr_indices = [op.asr_index for op in run if op.asr_index is not None]
        if len(srt_indices) < min_tokens or len(asr_indices) < min_tokens:
            run.clear()
            return
        anchor_number = len(regions) + 1
        regions.append(
            AnchorRegion(
                anchor_id=f"anchor-{anchor_number}",
                cue_ids=sorted({tokens[index].cue_id for index in srt_indices}),
                srt_token_indices=srt_indices,
                asr_word_indices=asr_indices,
                srt_text=_span_text_from_tokens(tokens, srt_indices),
                asr_text=_span_text_from_words(words, asr_indices),
                start=min(words[index].start for index in asr_indices),
                end=max(words[index].end for index in asr_indices),
                score=round(sum(op.score for op in run) / len(run), 4),
            )
        )
        run.clear()

    for op in ops:
        if op.kind == "match":
            run.append(op)
        else:
            flush()
    flush()
    return regions


def align_cues_to_words(cues: list[Cue], words: list[Word]) -> AlignmentResult:
    tokens = tokenize_cues(cues)
    words_norm = normalized_words(words)
    excluded_screen_text_cue_ids = [
        cue.index for cue in cues if cue_has_bracketed_screen_text(cue)
    ]
    if not tokens:
        return AlignmentResult(
            diagnostics=AlignmentDiagnostics(
                excluded_screen_text_cue_ids=excluded_screen_text_cue_ids,
                missing_audio_guard_version=MISSING_AUDIO_GUARD_VERSION,
            )
        )
    tokenized_cue_ids = {token.cue_id for token in tokens}

    preliminary_run = _align_tokens_detailed(tokens, words_norm)
    ops = preliminary_run.ops
    unbanded_fallback = preliminary_run.unbanded_fallback
    band_limited = preliminary_run.band_limited
    unresolved = preliminary_run.unresolved
    prior_attempted = _has_repeated_alignment_candidates(tokens, words_norm)
    prior_used = False
    transform: _TimeTransform | None = None
    if prior_attempted:
        timing_priors = _timing_priors(
            cues,
            tokens,
            words,
            words_norm,
            preliminary_run.ops,
        )
        if timing_priors is not None:
            prior_used = True
            transform = timing_priors.transform
            prior_run = _align_tokens_detailed(
                tokens,
                words_norm,
                token_time_priors=timing_priors.token_priors,
                word_time_centers=timing_priors.word_centers,
            )
            ops = prior_run.ops
            unbanded_fallback = unbanded_fallback or prior_run.unbanded_fallback
            band_limited = band_limited or prior_run.band_limited
            unresolved = prior_run.unresolved
    if not unresolved:
        ops = _prefer_unique_full_cue_windows(ops, tokens, words_norm)
    provisional_matches = _token_matches_from_ops(ops, tokens)
    flags = [
        *_alignment_outlier_flags(provisional_matches, cues, tokens, words),
        *_implausible_matched_word_duration_flags(provisional_matches, cues, words),
    ]
    rejected_outlier_cue_ids = _rejectable_outlier_cue_ids(flags)
    if rejected_outlier_cue_ids:
        ops = _without_cue_matches(ops, tokens, rejected_outlier_cue_ids)

    matches = _token_matches_from_ops(ops, tokens)
    cue_word_indices: dict[int, list[int]] = {cue.index: [] for cue in cues}
    for match in matches:
        cue_word_indices.setdefault(match.cue_id, []).append(match.asr_word_index)

    divergence_spans = _build_divergences(ops, tokens, words)
    anchor_regions = _build_anchor_regions(ops, tokens, words)
    unmatched_cue_ids = [
        cue.index
        for cue in cues
        if cue.index in tokenized_cue_ids and not cue_word_indices.get(cue.index)
    ]
    anchor_coverage = len(matches) / len(tokens)
    missing_audio_cue_ids = sorted(
        rejected_outlier_cue_ids
        | _source_only_unmatched_cue_ids(unmatched_cue_ids, divergence_spans)
        | _source_only_zero_window_cue_ids(divergence_spans)
    )
    cue_by_id = {cue.index: cue for cue in cues}
    for cue_id in missing_audio_cue_ids:
        cue = cue_by_id.get(cue_id)
        flags.append(
            QCFlag(
                kind="missing_audio_timing_held",
                cue_ids=[cue_id],
                message=(
                    "No trustworthy local speech evidence was available for this source cue; "
                    "its source text and timing are locked instead of borrowing another passage."
                ),
                severity="error",
                old_text=cue.text if cue is not None else None,
                start=cue.start_ms / 1000.0 if cue is not None else None,
                end=cue.end_ms / 1000.0 if cue is not None else None,
            )
        )
    if band_limited:
        episode_start = min((cue.start_ms for cue in cues), default=0) / 1000.0
        episode_end = max((cue.end_ms for cue in cues), default=0) / 1000.0
        flags.append(
            QCFlag(
                kind="alignment_band_limited",
                cue_ids=[],
                message=(
                    "Alignment retry reached its bounded margin or cell budget before every "
                    "unique exact anchor could be proven; review this artifact instead of "
                    "running an unbounded alignment."
                ),
                severity="warning",
                start=episode_start,
                end=episode_end,
            )
        )
    if unresolved:
        episode_start = min((cue.start_ms for cue in cues), default=0) / 1000.0
        episode_end = max((cue.end_ms for cue in cues), default=0) / 1000.0
        flags.append(
            QCFlag(
                kind="alignment_unresolved",
                cue_ids=[],
                message=(
                    "Alignment could not be resolved within the bounded cell budget; "
                    "the reviewable whole-span artifact was retained instead of silently "
                    "accepting a collapsed alignment."
                ),
                severity="error",
                start=episode_start,
                end=episode_end,
            )
        )

    return AlignmentResult(
        token_matches=matches,
        anchor_regions=anchor_regions,
        cue_word_indices={cue_id: sorted(set(indices)) for cue_id, indices in cue_word_indices.items() if indices},
        anchor_coverage=round(anchor_coverage, 4),
        divergence_spans=divergence_spans,
        unmatched_cue_ids=unmatched_cue_ids,
        flags=flags,
        diagnostics=AlignmentDiagnostics(
            prior_attempted=prior_attempted,
            prior_used=prior_used,
            transform_applied=transform is not None,
            transform_rate=transform.rate if transform is not None else None,
            transform_offset_seconds=(
                transform.offset_seconds if transform is not None else None
            ),
            transform_anchor_count=transform.anchor_count if transform is not None else 0,
            unbanded_fallback=unbanded_fallback,
            band_limited=band_limited,
            unresolved=unresolved,
            excluded_screen_text_cue_ids=excluded_screen_text_cue_ids,
            missing_audio_cue_ids=missing_audio_cue_ids,
            missing_audio_guard_version=MISSING_AUDIO_GUARD_VERSION,
        ),
    )


def _token_matches_from_ops(
    ops: list[_Op],
    tokens: list[SRTToken],
) -> list[TokenMatch]:
    return [
        TokenMatch(
            cue_id=tokens[op.srt_index].cue_id,
            srt_token_index=op.srt_index,
            asr_word_index=op.asr_index,
            score=round(op.score, 4),
        )
        for op in ops
        if op.kind == "match"
        and op.srt_index is not None
        and op.asr_index is not None
    ]


def _rejectable_outlier_cue_ids(
    flags: list[QCFlag],
) -> set[int]:
    """Reject only timing failures strong enough to fail closed."""

    return {
        cue_id
        for flag in flags
        if flag.severity == "error"
        and flag.kind
        in {
            "alignment_outlier",
            "alignment_model_unavailable",
            "implausible_matched_word_duration",
        }
        for cue_id in flag.cue_ids
    }


def _without_cue_matches(
    ops: list[_Op],
    tokens: list[SRTToken],
    cue_ids: set[int],
) -> list[_Op]:
    filtered: list[_Op] = []
    for op in ops:
        if (
            op.kind == "match"
            and op.srt_index is not None
            and op.asr_index is not None
            and tokens[op.srt_index].cue_id in cue_ids
        ):
            filtered.extend(
                [
                    _Op("delete", op.srt_index, None, 0.0),
                    _Op("insert", None, op.asr_index, 0.0),
                ]
            )
            continue
        filtered.append(op)
    return filtered


def _source_only_unmatched_cue_ids(
    unmatched_cue_ids: list[int],
    spans: list[DivergenceSpan],
) -> set[int]:
    locked: set[int] = set()
    for cue_id in unmatched_cue_ids:
        relevant = [span for span in spans if cue_id in span.cue_ids]
        if relevant and all(
            not span.asr_word_indices and not span.asr_text.strip()
            for span in relevant
        ):
            locked.add(cue_id)
    return locked


def _source_only_zero_window_cue_ids(spans: list[DivergenceSpan]) -> set[int]:
    locked: set[int] = set()
    for span in spans:
        has_valid_window = (
            span.start is not None
            and span.end is not None
            and math.isfinite(span.start)
            and math.isfinite(span.end)
            and span.end > span.start
        )
        if (
            span.cue_ids
            and span.srt_text.strip()
            and not span.asr_word_indices
            and not span.asr_text.strip()
            and not has_valid_window
        ):
            locked.update(span.cue_ids)
    return locked


def _prefer_unique_full_cue_windows(
    ops: list[_Op],
    tokens: list[SRTToken],
    words_norm: list[str],
) -> list[_Op]:
    """Repair repeated-token theft when a later cue has a provable full window."""

    if not tokens or not words_norm:
        return ops

    exact_pairs = _unique_full_cue_exact_pairs(tokens, words_norm)
    if not exact_pairs:
        return ops

    matched_pairs = {
        (op.srt_index, op.asr_index)
        for op in ops
        if op.kind == "match"
        and op.srt_index is not None
        and op.asr_index is not None
    }
    desired_pairs = _monotonic_exact_pair_map(matched_pairs, exact_pairs)
    if desired_pairs == matched_pairs:
        return ops
    return _ops_from_exact_pairs(len(tokens), len(words_norm), desired_pairs)


def _unique_full_cue_exact_pairs(
    tokens: list[SRTToken],
    words_norm: list[str],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for token_indices in _token_indices_by_cue(tokens).values():
        target = [tokens[index].normalized for index in token_indices]
        if len(target) < 2:
            continue
        windows = _exact_sequence_windows(words_norm, target, limit=2)
        if len(windows) != 1:
            continue
        start = windows[0]
        pairs.update(
            (token_index, start + offset)
            for offset, token_index in enumerate(token_indices)
        )
    return pairs


def _token_indices_by_cue(tokens: list[SRTToken]) -> dict[int, list[int]]:
    by_cue: dict[int, list[int]] = {}
    for token_index, token in enumerate(tokens):
        by_cue.setdefault(token.cue_id, []).append(token_index)
    return by_cue


def _exact_sequence_windows(
    haystack: list[str],
    needle: list[str],
    *,
    limit: int,
) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    windows: list[int] = []
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] != needle:
            continue
        windows.append(start)
        if len(windows) >= limit:
            return windows
    return windows


def _monotonic_exact_pair_map(
    matched_pairs: set[tuple[int | None, int | None]],
    preferred_pairs: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    preferred = sorted(preferred_pairs)
    if any(
        left_srt >= right_srt or left_word >= right_word
        for (left_srt, left_word), (right_srt, right_word) in zip(
            preferred,
            preferred[1:],
        )
    ):
        return {
            (srt_index, asr_index)
            for srt_index, asr_index in matched_pairs
            if srt_index is not None and asr_index is not None
        }

    selected = set(preferred)
    used_srt = {srt_index for srt_index, _ in selected}
    used_words = {word_index for _, word_index in selected}
    existing_pairs = sorted(
        (srt_index, asr_index)
        for srt_index, asr_index in matched_pairs
        if srt_index is not None and asr_index is not None
    )
    for srt_index, word_index in existing_pairs:
        if (srt_index, word_index) in selected:
            continue
        if srt_index in used_srt or word_index in used_words:
            continue
        previous_words = [
            preferred_word
            for preferred_srt, preferred_word in preferred
            if preferred_srt < srt_index
        ]
        next_words = [
            preferred_word
            for preferred_srt, preferred_word in preferred
            if preferred_srt > srt_index
        ]
        if previous_words and word_index <= previous_words[-1]:
            continue
        if next_words and word_index >= next_words[0]:
            continue
        selected.add((srt_index, word_index))
        used_srt.add(srt_index)
        used_words.add(word_index)
    return selected


def _ops_from_exact_pairs(
    token_count: int,
    word_count: int,
    pairs: set[tuple[int, int]],
) -> list[_Op]:
    ops: list[_Op] = []
    token_cursor = 0
    word_cursor = 0
    for token_index, word_index in sorted(pairs):
        if token_index < token_cursor or word_index < word_cursor:
            continue
        while token_cursor < token_index:
            ops.append(_Op("delete", token_cursor, None, 0.0))
            token_cursor += 1
        while word_cursor < word_index:
            ops.append(_Op("insert", None, word_cursor, 0.0))
            word_cursor += 1
        ops.append(_Op("match", token_index, word_index, 1.0))
        token_cursor = token_index + 1
        word_cursor = word_index + 1
    while token_cursor < token_count:
        ops.append(_Op("delete", token_cursor, None, 0.0))
        token_cursor += 1
    while word_cursor < word_count:
        ops.append(_Op("insert", None, word_cursor, 0.0))
        word_cursor += 1
    return ops


def _alignment_outlier_flags(
    matches: list[TokenMatch],
    cues: list[Cue],
    tokens: list[SRTToken],
    words: list[Word],
) -> list[QCFlag]:
    cue_by_id = {cue.index: cue for cue in cues}
    matches_by_cue: dict[int, list[TokenMatch]] = {}
    for match in matches:
        matches_by_cue.setdefault(match.cue_id, []).append(match)

    observations: list[tuple[list[TokenMatch], Cue, Word, float, float]] = []
    for cue_id, cue_matches in matches_by_cue.items():
        cue = cue_by_id.get(cue_id)
        if cue is None:
            continue
        matched_words = [words[match.asr_word_index] for match in cue_matches]
        word_centers = [(word.start + word.end) / 2.0 for word in matched_words]
        word_center = median(word_centers)
        representative_index = min(
            range(len(matched_words)),
            key=lambda index: abs(word_centers[index] - word_center),
        )
        cue_center = (cue.start_ms + cue.end_ms) / 2000.0
        observations.append(
            (
                cue_matches,
                cue,
                matched_words[representative_index],
                cue_center,
                word_center,
            )
        )
    if len(observations) < 2:
        return []
    observations = sorted(observations, key=lambda item: item[3])

    transform = _fit_time_transform(
        [(cue_center, word_center) for _, _, _, cue_center, word_center in observations]
    )
    if transform is None:
        threshold_seconds = _alignment_outlier_threshold_seconds(cues)
        if len(observations) == 2:
            offsets = [
                word_center - cue_center
                for _, _, _, cue_center, word_center in observations
            ]
            locally_anchored = [
                index
                for index, offset in enumerate(offsets)
                if abs(offset) <= 3.0
            ]
            if len(locally_anchored) == 1:
                anchor_index = locally_anchored[0]
                suspect_index = 1 - anchor_index
                residual = offsets[suspect_index] - offsets[anchor_index]
                cue_matches, cue, word, _, _ = observations[suspect_index]
                suspect_token_count = sum(
                    1 for token in tokens if token.cue_id == cue.index
                )
                if abs(residual) > threshold_seconds and suspect_token_count <= 2:
                    return [
                        _alignment_outlier_flag(
                            residual,
                            cue_matches,
                            cue,
                            word,
                            tokens,
                            severity="error",
                        )
                    ]
                if abs(residual) <= threshold_seconds:
                    return []
        episode_start = min((cue.start_ms for cue in cues), default=0) / 1000.0
        episode_end = max((cue.end_ms for cue in cues), default=0) / 1000.0
        return [
            QCFlag(
                kind="alignment_model_unavailable",
                cue_ids=sorted(cue.index for _, cue, *_rest in observations),
                message=(
                    "Matched anchors could not produce a stable episode timing model; "
                    "review alignment order and timing before accepting this artifact."
                ),
                severity="warning",
                start=episode_start,
                end=episode_end,
            )
        ]

    threshold_seconds = _alignment_outlier_threshold_seconds(cues)
    outlier_by_cue: dict[int, tuple[float, list[TokenMatch], Cue, Word]] = {}
    for cue_matches, cue, word, cue_center, word_center in observations:
        expected_word_center = transform.rate * cue_center + transform.offset_seconds
        residual = word_center - expected_word_center
        if abs(residual) <= threshold_seconds:
            continue
        current = outlier_by_cue.get(cue.index)
        if current is None or abs(residual) > abs(current[0]):
            outlier_by_cue[cue.index] = (residual, cue_matches, cue, word)

    hard_reject_seconds = max(3.0, threshold_seconds)
    return [
        _alignment_outlier_flag(
            residual,
            cue_matches,
            cue,
            word,
            tokens,
            severity="error" if abs(residual) > hard_reject_seconds else "warning",
        )
        for residual, cue_matches, cue, word in outlier_by_cue.values()
    ]


def _implausible_matched_word_duration_flags(
    matches: list[TokenMatch],
    cues: list[Cue],
    words: list[Word],
) -> list[QCFlag]:
    cue_by_id = {cue.index: cue for cue in cues}
    matches_by_cue: dict[int, list[TokenMatch]] = {}
    for match in matches:
        matches_by_cue.setdefault(match.cue_id, []).append(match)

    flags: list[QCFlag] = []
    for cue_id, cue_matches in matches_by_cue.items():
        if len(cue_matches) != 1:
            continue
        cue = cue_by_id.get(cue_id)
        if cue is None:
            continue
        match = cue_matches[0]
        word = words[match.asr_word_index]
        word_duration = max(0.0, word.end - word.start)
        cue_duration = max(0.0, (cue.end_ms - cue.start_ms) / 1000.0)
        if word_duration <= IMPLAUSIBLE_MATCHED_WORD_SECONDS:
            continue
        if word_duration <= cue_duration * IMPLAUSIBLE_WORD_TO_CUE_RATIO:
            continue
        flags.append(
            QCFlag(
                kind="implausible_matched_word_duration",
                cue_ids=[cue_id],
                message=(
                    f"The only matched ASR word spans {word_duration:.2f}s versus the "
                    f"source cue's {cue_duration:.2f}s; its timing is rejected and the "
                    "source cue is held for review."
                ),
                severity="error",
                confidence=round(match.score, 4),
                old_text=cue.text,
                new_text=word.text,
                start=cue.start_ms / 1000.0,
                end=cue.end_ms / 1000.0,
            )
        )
    return flags


def _alignment_outlier_flag(
    residual: float,
    cue_matches: list[TokenMatch],
    cue: Cue,
    word: Word,
    tokens: list[SRTToken],
    *,
    severity: Literal["warning", "error"],
) -> QCFlag:
    representative_match = cue_matches[0]
    token_text = tokens[representative_match.srt_token_index].text
    return QCFlag(
        kind="alignment_outlier",
        cue_ids=[cue.index],
        message=(
            f"Matched cue near token '{token_text}' is {abs(residual):.1f}s from "
            "the episode timing model; weak timing evidence is held for review."
        ),
        severity=severity,
        confidence=round(min(match.score for match in cue_matches), 4),
        old_text=cue.text,
        new_text=word.text,
        start=cue.start_ms / 1000.0,
        end=cue.end_ms / 1000.0,
    )


def _alignment_outlier_threshold_seconds(cues: list[Cue]) -> float:
    starts = [cue.start_ms for cue in cues]
    ends = [cue.end_ms for cue in cues]
    if not starts or not ends:
        return ALIGNMENT_OUTLIER_SECONDS
    duration_seconds = max(0.0, (max(ends) - min(starts)) / 1000.0)
    if duration_seconds <= 0.0:
        return ALIGNMENT_OUTLIER_SECONDS
    return min(ALIGNMENT_OUTLIER_SECONDS, duration_seconds * 0.1)
