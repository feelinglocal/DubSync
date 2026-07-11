from __future__ import annotations

from .models import Cue, QCFlag
from .tokenize import alphanumeric_signature


def detect_source_errors(cues: list[Cue]) -> list[QCFlag]:
    flagged: dict[tuple[int, ...], QCFlag] = {}
    tokenized = {cue.index: alphanumeric_signature(cue.plain_text) for cue in cues}

    for position in range(1, len(cues)):
        previous = cues[position - 1]
        current = cues[position]
        prev_tokens = tokenized[previous.index]
        cur_tokens = tokenized[current.index]
        if _is_adjacent_duplicate_fragment(prev_tokens, cur_tokens):
            window = cues[max(0, position - 2) : position + 1]
            cue_ids = tuple(cue.index for cue in window)
            flagged[cue_ids] = QCFlag(
                kind="source_error",
                cue_ids=list(cue_ids),
                message="Adjacent cues contain a duplicated phrase fragment; source SRT may be scrambled.",
                old_text="\n\n".join(cue.text for cue in window),
                start=window[0].start_ms / 1000.0,
                end=window[-1].end_ms / 1000.0,
            )

    return list(flagged.values())


def _is_adjacent_duplicate_fragment(previous: list[str], current: list[str]) -> bool:
    if len(current) < 2 or len(previous) < 2:
        return False
    if current == previous[: len(current)]:
        return True
    if current == previous[-len(current) :]:
        return True
    return current[:2] == previous[:2] or current[:2] == previous[-2:]
