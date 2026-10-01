from __future__ import annotations

from .models import Cue, QCFlag
from .subtitle_annotations import speech_text_for_alignment
from .tokenize import alphanumeric_signature

# Cues closer than this are one continuous subtitle stream (a frame or less).
TOUCHING_CUE_GAP_MS = 100
_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
_SONG_MARKERS = "♪♫"


def detect_source_errors(cues: list[Cue]) -> list[QCFlag]:
    """Flag adjacent cues whose text was duplicated and scrambled by an editor.

    Dialogue legitimately repeats: echoed questions, two characters greeting,
    a restart after a pause, song lines and screen captions. A scrambled file
    instead repeats an unfinished fragment of the touching neighbour verbatim
    at its edge ("erwecken kann," right after "erwecken kann, / hohe Klasse").
    """

    flagged: dict[tuple[int, ...], QCFlag] = {}
    tokenized = {cue.index: alphanumeric_signature(speech_text_for_alignment(cue)) for cue in cues}

    for position in range(1, len(cues)):
        previous = cues[position - 1]
        current = cues[position]
        if not _is_scrambled_neighbour(previous, current, tokenized[previous.index], tokenized[current.index]):
            continue
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


def _is_scrambled_neighbour(previous: Cue, current: Cue, previous_tokens: list[str], current_tokens: list[str]) -> bool:
    if current.start_ms - previous.end_ms > TOUCHING_CUE_GAP_MS:
        return False
    if any(marker in cue.text for cue in (previous, current) for marker in _SONG_MARKERS):
        return False
    if not _is_edge_duplicate(previous_tokens, current_tokens):
        return False
    shorter = current if len(current_tokens) < len(previous_tokens) else previous
    # A complete sentence repeated or echoed is dialogue; a scrambled copy is
    # an unfinished fragment of its neighbour.
    return not speech_text_for_alignment(shorter).rstrip().endswith(_SENTENCE_END)


def _is_edge_duplicate(previous: list[str], current: list[str]) -> bool:
    shorter, longer = (current, previous) if len(current) < len(previous) else (previous, current)
    if len(shorter) < 2 or len(shorter) == len(longer):
        return False
    return longer[: len(shorter)] == shorter or longer[-len(shorter) :] == shorter
