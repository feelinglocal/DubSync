from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator

from .tokenize import SRTToken

RETRY_MARGINS = (64, 256, 1024)


def band_windows(
    row: int,
    token_count: int,
    word_count: int,
    margin: int,
    prior_centers: tuple[int, ...] = (),
) -> list[tuple[int, int]]:
    if token_count <= 0:
        return [(0, word_count)]
    center = round(row * word_count / token_count)
    intervals = [(max(0, center - margin), min(word_count, center + margin))]
    if row == 0:
        intervals.append((0, min(word_count, margin)))
    if row == token_count:
        intervals.append((max(0, word_count - margin), word_count))
    intervals.extend(
        (max(0, prior_center - margin), min(word_count, prior_center + margin))
        for prior_center in prior_centers
    )
    return _merge_intervals(intervals)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((start, end) for start, end in intervals if end >= start)
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def retry_margins(initial_margin: int, full_width: int) -> list[int]:
    bounded_initial = max(0, initial_margin)
    candidates = [
        bounded_initial,
        *(candidate for candidate in RETRY_MARGINS if candidate > bounded_initial),
    ]
    margins: list[int] = []
    for candidate in candidates:
        margin = min(candidate, full_width)
        if margin not in margins:
            margins.append(margin)
    return margins


def band_cell_count(
    token_count: int,
    word_count: int,
    margin: int,
    reachability: dict[int, tuple[int, ...]],
) -> int:
    return sum(
        sum(
            end - start + 1
            for start, end in band_windows(
                row,
                token_count,
                word_count,
                margin,
                reachability.get(row, ()),
            )
        )
        for row in range(token_count + 1)
    )


def interval_cell_count(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def iter_interval_cells(intervals: list[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    offset = 0
    for start, end in intervals:
        for value in range(start, end + 1):
            yield offset, value
            offset += 1


def row_offset(intervals: list[tuple[int, int]], column: int) -> int | None:
    offset = 0
    for start, end in intervals:
        if start <= column <= end:
            return offset + column - start
        offset += end - start + 1
    return None


def unique_exact_pairs(
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


def reachability_centers(
    tokens: list[SRTToken],
    words_norm: list[str],
    token_time_priors: list[tuple[float, float]] | None,
    word_time_centers: list[float] | None,
) -> dict[int, tuple[int, ...]]:
    centers: dict[int, set[int]] = {}
    for token_index, word_index in unique_exact_pairs(tokens, words_norm):
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
