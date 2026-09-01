from __future__ import annotations

from .models import Cue
from .text_metrics import token_texts


def bracketed_screen_text_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return simple balanced ``[...]`` spans, or none for a malformed layout.

    Treating an entire malformed bracket layout as spoken text is deliberately
    conservative: a stray or nested bracket must never make ordinary dialogue
    disappear from alignment.
    """

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, character in enumerate(text):
        if character == "[":
            if start is not None:
                return ()
            start = index
        elif character == "]":
            if start is None:
                return ()
            spans.append((start, index + 1))
            start = None
    if start is not None:
        return ()
    return tuple(spans)


def text_without_bracketed_screen_text(text: str) -> str:
    spans = bracketed_screen_text_spans(text)
    if not spans:
        return text

    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append("".join("\n" if character == "\n" else " " for character in text[start:end]))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def speech_lines_for_alignment(cue: Cue) -> list[str]:
    """Return the original line-wise spoken residue of an annotated cue."""

    return [
        line.strip()
        for line in text_without_bracketed_screen_text(cue.text).splitlines()
        if line.strip()
    ]


def speech_text_for_alignment(cue: Cue) -> str:
    return " ".join(speech_lines_for_alignment(cue))


def cue_has_spoken_text(cue: Cue) -> bool:
    return bool(speech_text_for_alignment(cue))


def cue_has_bracketed_screen_text(cue: Cue) -> bool:
    return bool(bracketed_screen_text_spans(cue.text))


def is_bracketed_screen_text_cue(cue: Cue) -> bool:
    text = cue.plain_text.strip()
    return bool(text) and not speech_text_for_alignment(cue)


def cue_contains_bracketed_screen_text(cue: Cue) -> bool:
    """Backward-compatible alias for callers using the earlier predicate name."""

    return cue_has_bracketed_screen_text(cue)


def alignment_token_character_spans(cue: Cue) -> tuple[tuple[int, int], ...] | None:
    """Map bracket-stripped alignment tokens back into ``cue.text``.

    ``None`` means the annotated source layout cannot be reconstructed exactly.
    Callers must hold the edit rather than fall back to replacing the whole cue.
    """

    if not cue_has_bracketed_screen_text(cue):
        return None

    searchable_text = text_without_bracketed_screen_text(cue.text)
    source_tokens = token_texts(searchable_text)
    alignment_tokens = token_texts(speech_text_for_alignment(cue))
    if source_tokens != alignment_tokens:
        return None

    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in source_tokens:
        start = searchable_text.find(token, cursor)
        if start < 0:
            return None
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return tuple(spans)
