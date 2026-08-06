from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from statistics import median

from rapidfuzz import fuzz

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
from .tokenize import SRTToken, normalized_words, tokenize_cues

MATCH_THRESHOLD = 0.85
MIN_ANCHOR_TOKENS = 3
BAND_MARGIN = 64
NEG_INF = -1_000_000_000.0
TIME_PRIOR_MAX_BONUS = 0.2
TIME_PRIOR_MIN_RADIUS_SECONDS = 2.0
ALIGNMENT_OUTLIER_SECONDS = 12.0


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


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return fuzz.ratio(left, right) / 100.0


def _band_window(
    row: int,
    token_count: int,
    word_count: int,
    margin: int,
    prior_centers: tuple[int, ...] = (),
) -> tuple[int, int]:
    if token_count <= 0:
        return 0, word_count
    center = round(row * word_count / token_count)
    start = max(0, center - margin)
    end = min(word_count, center + margin)
    if row == 0:
        start = 0
    if row == token_count:
        end = word_count
    for prior_center in prior_centers:
        start = min(start, max(0, prior_center - margin))
        end = max(end, min(word_count, prior_center + margin))
    return start, end


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
    gap = -0.75
    dp: list[dict[int, float]] = [{0: 0.0}]
    back: list[dict[int, str]] = [{0: ""}]
    reachability = _reachability_centers(
        tokens,
        words_norm,
        token_time_priors,
        word_time_centers,
    )

    _, first_row_end = _band_window(
        0,
        n,
        m,
        band_margin,
        reachability.get(0, ()),
    )
    for j in range(1, first_row_end + 1):
        dp[0][j] = dp[0][j - 1] + gap
        back[0][j] = "insert"

    for i in range(1, n + 1):
        previous = dp[i - 1]
        row: dict[int, float] = {}
        row_back: dict[int, str] = {}
        start, end = _band_window(
            i,
            n,
            m,
            band_margin,
            reachability.get(i, ()),
        )
        for j in range(start, end + 1):
            best_score = NEG_INF
            best_op = ""
            if j in previous:
                best_score = previous[j] + gap
                best_op = "delete"
            if j > 0 and (j - 1) in row and row[j - 1] + gap > best_score:
                best_score = row[j - 1] + gap
                best_op = "insert"
            if j > 0 and (j - 1) in previous:
                similarity = _similarity(tokens[i - 1].normalized, words_norm[j - 1])
                match_score = 2.0 * similarity if similarity >= MATCH_THRESHOLD else -0.6
                candidate = previous[j - 1] + match_score
                if similarity >= MATCH_THRESHOLD:
                    candidate += _timing_bonus(
                        i - 1,
                        j - 1,
                        token_time_priors,
                        word_time_centers,
                    )
                if candidate > best_score:
                    best_score = candidate
                    best_op = "match"
            if best_op:
                row[j] = best_score
                row_back[j] = best_op
        dp.append(row)
        back.append(row_back)

    if m not in dp[n]:
        if band_margin >= max(n, m):
            raise RuntimeError("alignment band failed to find a global path")
        retry = _align_tokens_detailed(
            tokens,
            words_norm,
            band_margin=max(n, m),
            token_time_priors=token_time_priors,
            word_time_centers=word_time_centers,
        )
        return _AlignmentRun(ops=retry.ops, unbanded_fallback=True)

    ops: list[_Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i].get(j, "")
        if op == "match":
            score = _similarity(tokens[i - 1].normalized, words_norm[j - 1])
            kind = "match" if tokens[i - 1].normalized == words_norm[j - 1] else "replace"
            ops.append(_Op(kind, i - 1, j - 1, score))
            i -= 1
            j -= 1
        elif op == "delete":
            ops.append(_Op("delete", i - 1, None, 0.0))
            i -= 1
        elif op == "insert":
            ops.append(_Op("insert", None, j - 1, 0.0))
            j -= 1
        else:
            raise RuntimeError("alignment backtrack reached an empty operation")
    ops.reverse()
    if band_margin < max(n, m) and _misses_unique_exact_pair(ops, tokens, words_norm):
        retry = _align_tokens_detailed(
            tokens,
            words_norm,
            band_margin=max(n, m),
            token_time_priors=token_time_priors,
            word_time_centers=word_time_centers,
        )
        return _AlignmentRun(ops=retry.ops, unbanded_fallback=True)
    return _AlignmentRun(ops=ops)


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
    return not expected.issubset(matched)


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
        confidences = [words[index].confidence for index in asr_indices]
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
    if not tokens:
        return AlignmentResult()

    preliminary_run = _align_tokens_detailed(tokens, words_norm)
    ops = preliminary_run.ops
    unbanded_fallback = preliminary_run.unbanded_fallback
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
    matches: list[TokenMatch] = []
    cue_word_indices: dict[int, list[int]] = {cue.index: [] for cue in cues}

    for op in ops:
        if op.kind != "match" or op.srt_index is None or op.asr_index is None:
            continue
        token = tokens[op.srt_index]
        matches.append(
            TokenMatch(
                cue_id=token.cue_id,
                srt_token_index=op.srt_index,
                asr_word_index=op.asr_index,
                score=round(op.score, 4),
            )
        )
        cue_word_indices.setdefault(token.cue_id, []).append(op.asr_index)

    divergence_spans = _build_divergences(ops, tokens, words)
    anchor_regions = _build_anchor_regions(ops, tokens, words)
    unmatched_cue_ids = [cue.index for cue in cues if not cue_word_indices.get(cue.index)]
    anchor_coverage = len(matches) / len(tokens)
    flags = _alignment_outlier_flags(matches, cues, tokens, words)

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
        ),
    )


def _unique_exact_pairs(
    tokens: list[SRTToken],
    words_norm: list[str],
) -> list[tuple[int, int]]:
    token_positions: dict[str, list[int]] = {}
    word_positions: dict[str, list[int]] = {}
    for token_index, token in enumerate(tokens):
        if token.normalized:
            token_positions.setdefault(token.normalized, []).append(token_index)
    for word_index, word in enumerate(words_norm):
        if word:
            word_positions.setdefault(word, []).append(word_index)
    pairs = sorted(
        (token_indices[0], word_positions[value][0])
        for value, token_indices in token_positions.items()
        if len(token_indices) == 1
        and value in word_positions
        and len(word_positions[value]) == 1
    )
    if any(left_word >= right_word for (_, left_word), (_, right_word) in zip(pairs, pairs[1:])):
        return []
    return pairs


def _reachability_centers(
    tokens: list[SRTToken],
    words_norm: list[str],
    token_time_priors: list[tuple[float, float]] | None,
    word_time_centers: list[float] | None,
) -> dict[int, tuple[int, ...]]:
    centers: dict[int, set[int]] = {}
    for token_index, word_index in _unique_exact_pairs(tokens, words_norm):
        centers.setdefault(token_index, set()).add(word_index)
        centers.setdefault(token_index + 1, set()).add(word_index + 1)

    if token_time_priors is not None and word_time_centers:
        for token_index, (expected, _) in enumerate(token_time_priors):
            word_index = _closest_word_index(word_time_centers, expected)
            centers.setdefault(token_index, set()).add(word_index)
            centers.setdefault(token_index + 1, set()).add(word_index + 1)
    return {row: tuple(sorted(row_centers)) for row, row_centers in centers.items()}


def _closest_word_index(word_time_centers: list[float], expected: float) -> int:
    insertion = bisect_left(word_time_centers, expected)
    if insertion <= 0:
        return 0
    if insertion >= len(word_time_centers):
        return len(word_time_centers) - 1
    before = insertion - 1
    return (
        before
        if abs(word_time_centers[before] - expected)
        <= abs(word_time_centers[insertion] - expected)
        else insertion
    )


def _alignment_outlier_flags(
    matches: list[TokenMatch],
    cues: list[Cue],
    tokens: list[SRTToken],
    words: list[Word],
) -> list[QCFlag]:
    cue_by_id = {cue.index: cue for cue in cues}
    observations: list[tuple[TokenMatch, Cue, Word, float, float]] = []
    for match in matches:
        cue = cue_by_id.get(match.cue_id)
        if cue is None:
            continue
        word = words[match.asr_word_index]
        cue_center = (cue.start_ms + cue.end_ms) / 2000.0
        word_center = (word.start + word.end) / 2.0
        observations.append((match, cue, word, cue_center, word_center))
    if len(observations) < 3:
        return []

    transform = _fit_time_transform(
        [(cue_center, word_center) for _, _, _, cue_center, word_center in observations]
    )
    if transform is None:
        return []

    outlier_by_cue: dict[int, tuple[float, TokenMatch, Cue, Word]] = {}
    for match, cue, word, cue_center, word_center in observations:
        expected_word_center = transform.rate * cue_center + transform.offset_seconds
        residual = word_center - expected_word_center
        if abs(residual) <= ALIGNMENT_OUTLIER_SECONDS:
            continue
        current = outlier_by_cue.get(match.cue_id)
        if current is None or abs(residual) > abs(current[0]):
            outlier_by_cue[match.cue_id] = (residual, match, cue, word)

    flags: list[QCFlag] = []
    for residual, match, cue, word in outlier_by_cue.values():
        token_text = tokens[match.srt_token_index].text
        flags.append(
            QCFlag(
                kind="alignment_outlier",
                cue_ids=[match.cue_id],
                message=(
                    f"Matched token '{token_text}' is {abs(residual):.1f}s from the "
                    "episode timing model; "
                    "hold this alignment for review."
                ),
                severity="warning",
                confidence=round(match.score, 4),
                old_text=cue.text,
                new_text=word.text,
                start=cue.start_ms / 1000.0,
                end=cue.end_ms / 1000.0,
            )
        )
    return flags
