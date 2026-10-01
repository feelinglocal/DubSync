from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator

from .tokenize import SRTToken

RETRY_MARGINS = (64, 256, 1024)
# An anchor rectangle whose word and token extents differ by more than this
# holds a long pure deletion or insertion run that a diagonal band cannot cross.
ANCHOR_FILL_SKEW = 48
# Fill centres closer than the smallest band width so their windows merge.
ANCHOR_FILL_STEP = 96
# Larger skewed rectangles only get a connected staircase, keeping cells bounded.
ANCHOR_FILL_MAX_CELLS = 250_000
# A band anchor needs a neighbour within this many tokens that agrees on its
# token-to-word offset; an isolated coincidental pair must not bend the band.
ANCHOR_SUPPORT_TOKENS = 64
ANCHOR_SUPPORT_OFFSET = 32


def band_windows(
    row: int,
    token_count: int,
    word_count: int,
    margin: int,
    prior_centers: tuple[int, ...] = (),
    *,
    diagonal: bool = True,
) -> list[tuple[int, int]]:
    if token_count <= 0:
        return [(0, word_count)]
    intervals: list[tuple[int, int]] = []
    if diagonal:
        center = round(row * word_count / token_count)
        intervals.append((max(0, center - margin), min(word_count, center + margin)))
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
    *,
    diagonal: bool = True,
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
                diagonal=diagonal,
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
    # Words unique on both sides by coincidence can sit far out of order. Keep
    # the longest monotonic chain instead of discarding every anchor for them.
    return _longest_increasing_word_chain(pairs)


def _longest_increasing_word_chain(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    tail_words: list[int] = []
    tail_positions: list[int] = []
    previous = [-1] * len(pairs)
    for position, (_, word_index) in enumerate(pairs):
        length = bisect_left(tail_words, word_index)
        if length == len(tail_words):
            tail_words.append(word_index)
            tail_positions.append(position)
        else:
            tail_words[length] = word_index
            tail_positions[length] = position
        previous[position] = tail_positions[length - 1] if length else -1
    chain: list[tuple[int, int]] = []
    position = tail_positions[-1] if tail_positions else -1
    while position != -1:
        chain.append(pairs[position])
        position = previous[position]
    chain.reverse()
    return chain


def supported_anchor_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep monotonic anchors that a nearby anchor confirms.

    Real dialogue yields an anchor every few tokens on a smooth path. A word
    unique on both sides by coincidence is usually isolated; trusting it would
    force the whole band through the wrong passage.
    """

    kept: list[tuple[int, int]] = []
    for position, (token_index, word_index) in enumerate(pairs):
        offset = word_index - token_index
        supported = False
        for step in (-1, 1):
            neighbor = position + step
            while 0 <= neighbor < len(pairs) and abs(pairs[neighbor][0] - token_index) <= ANCHOR_SUPPORT_TOKENS:
                if abs((pairs[neighbor][1] - pairs[neighbor][0]) - offset) <= ANCHOR_SUPPORT_OFFSET:
                    supported = True
                    break
                neighbor += step
            if supported:
                break
        if supported:
            kept.append((token_index, word_index))
    return kept


def anchor_path_centers(
    token_count: int,
    word_count: int,
    pairs: list[tuple[int, int]],
) -> dict[int, set[int]]:
    """Band centres along the piecewise-linear path through monotonic anchors.

    The path replaces the global diagonal, so a long unspoken head, an offset
    source or a drifted frame rate cannot push the true path out of the band.
    Rectangles between anchors that hold a long pure deletion or insertion run
    are filled (or, when large, given a connected staircase) so the banded DP
    always has a path from the first to the last cell.
    """

    points = [(0, 0)]
    for token_index, word_index in pairs:
        points.append((token_index, word_index))
        points.append((token_index + 1, word_index + 1))
    points.append((token_count, word_count))
    centers: dict[int, set[int]] = {}
    for (row_start, column_start), (row_end, column_end) in zip(points, points[1:]):
        if row_end < row_start or column_end < column_start:
            continue
        rows = row_end - row_start
        columns = column_end - column_start
        skewed = abs(columns - rows) > ANCHOR_FILL_SKEW
        full_fill = skewed and (rows + 1) * (columns + 1) <= ANCHOR_FILL_MAX_CELLS

        def center_at(row: int) -> int:
            if rows == 0:
                return column_start
            return column_start + round((row - row_start) * columns / rows)

        for row in range(row_start, row_end + 1):
            row_centers = centers.setdefault(row, set())
            if full_fill or rows == 0:
                row_centers.update(range(column_start, column_end + 1, ANCHOR_FILL_STEP))
                row_centers.add(column_end)
            elif skewed:
                left = center_at(row)
                right = center_at(min(row + 1, row_end))
                row_centers.update(range(left, right + 1, ANCHOR_FILL_STEP))
                row_centers.add(right)
            else:
                row_centers.add(center_at(row))
    return centers


def reachability_centers(
    tokens: list[SRTToken],
    words_norm: list[str],
    token_time_priors: list[tuple[float, float]] | None,
    word_time_centers: list[float] | None,
    *,
    anchors: list[tuple[int, int]] | None = None,
) -> dict[int, tuple[int, ...]]:
    if anchors is None:
        anchors = supported_anchor_pairs(unique_exact_pairs(tokens, words_norm))
    centers = anchor_path_centers(len(tokens), len(words_norm), anchors)

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
