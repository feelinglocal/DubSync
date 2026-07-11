from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import AlignmentResult, AnchorRegion, Cue, DivergenceSpan, TokenMatch, Word
from .tokenize import SRTToken, normalized_words, tokenize_cues

MATCH_THRESHOLD = 0.85
MIN_ANCHOR_TOKENS = 3
BAND_MARGIN = 64
NEG_INF = -1_000_000_000.0


@dataclass(frozen=True)
class _Op:
    kind: str
    srt_index: int | None = None
    asr_index: int | None = None
    score: float = 0.0


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return fuzz.ratio(left, right) / 100.0


def _band_window(row: int, token_count: int, word_count: int, margin: int) -> tuple[int, int]:
    if token_count <= 0:
        return 0, word_count
    center = round(row * word_count / token_count)
    start = max(0, center - margin)
    end = min(word_count, center + margin)
    if row == 0:
        start = 0
    if row == token_count:
        end = word_count
    return start, end


def _align_tokens(tokens: list[SRTToken], words_norm: list[str], band_margin: int = BAND_MARGIN) -> list[_Op]:
    n = len(tokens)
    m = len(words_norm)
    gap = -0.75
    dp: list[dict[int, float]] = [{0: 0.0}]
    back: list[dict[int, str]] = [{0: ""}]

    _, first_row_end = _band_window(0, n, m, band_margin)
    for j in range(1, first_row_end + 1):
        dp[0][j] = dp[0][j - 1] + gap
        back[0][j] = "insert"

    for i in range(1, n + 1):
        previous = dp[i - 1]
        row: dict[int, float] = {}
        row_back: dict[int, str] = {}
        start, end = _band_window(i, n, m, band_margin)
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
        return _align_tokens(tokens, words_norm, band_margin=max(n, m))

    ops: list[_Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i].get(j, "")
        if op == "match":
            score = _similarity(tokens[i - 1].normalized, words_norm[j - 1])
            ops.append(_Op("match" if score >= MATCH_THRESHOLD else "replace", i - 1, j - 1, score))
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
    return ops


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

    ops = _align_tokens(tokens, words_norm)
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

    return AlignmentResult(
        token_matches=matches,
        anchor_regions=anchor_regions,
        cue_word_indices={cue_id: sorted(set(indices)) for cue_id, indices in cue_word_indices.items() if indices},
        anchor_coverage=round(anchor_coverage, 4),
        divergence_spans=divergence_spans,
        unmatched_cue_ids=unmatched_cue_ids,
    )
