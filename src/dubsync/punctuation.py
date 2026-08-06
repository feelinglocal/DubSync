from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from .models import Cue, QCFlag
from .text_metrics import display_width, wrap_visual_width


class PunctuationValidationError(ValueError):
    pass


class PunctuationAdapter(Protocol):
    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        raise NotImplementedError


class StaticPunctuationAdapter:
    def __init__(self, responses: dict[str | int, str]):
        self.responses = {int(cue_id): text for cue_id, text in responses.items()}

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        return {cue.index: self.responses[cue.index] for cue in cues if cue.index in self.responses}


def validate_punctuation_only(before: str, after: str) -> str:
    if _word_freeze_signature(before) != _word_freeze_signature(after):
        raise PunctuationValidationError("alphanumeric content changed during punctuation pass")
    if _quotation_mark_signature(before) != _quotation_mark_signature(after):
        raise PunctuationValidationError("quotation mark signature changed during punctuation pass")
    return after


def apply_punctuation_pass(
    cues: list[Cue],
    adapter: PunctuationAdapter,
    scene_gap_seconds: float = 4.0,
    max_chars_per_line: int | None = None,
    max_lines_per_cue: int | None = None,
    source_cues: list[Cue] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    source_by_id = {
        cue.index: cue for cue in (cues if source_cues is None else source_cues)
    }
    prompt_cues = [
        cue.with_lines(source_by_id[cue.index].lines)
        if _source_words_unchanged(cue, source_by_id.get(cue.index))
        else cue
        for cue in cues
    ]
    proposed: dict[int, str] = {}
    for batch in _scene_batches(prompt_cues, scene_gap_seconds):
        proposed.update(adapter.punctuate(batch))
    if not proposed:
        return cues, []

    updated: list[Cue] = []
    flags: list[QCFlag] = []
    for cue in cues:
        next_text = proposed.get(cue.index)
        if next_text is None:
            updated.append(cue)
            continue
        try:
            validate_punctuation_only(cue.plain_text, next_text.replace("\n", " "))
        except PunctuationValidationError as exc:
            updated.append(cue)
            flags.append(
                QCFlag(
                    kind="invalid_punctuation_change",
                    cue_ids=[cue.index],
                    message=str(exc),
                    severity="error",
                    old_text=cue.text,
                    new_text=next_text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        source_cue = source_by_id.get(cue.index)
        source_words_unchanged = _source_words_unchanged(cue, source_cue)
        lines = (
            _restore_source_line_breaks(source_cue.lines, next_text)
            if source_words_unchanged
            else (next_text.splitlines() or [next_text])
        )
        width_exceeded = max_chars_per_line is not None and any(
            display_width(line) > max_chars_per_line for line in lines
        )
        line_count_exceeded = max_lines_per_cue is not None and len(lines) > max_lines_per_cue
        restored_matches_source_structure = source_words_unchanged and (
            _line_word_boundaries(lines) == _line_word_boundaries(source_cue.lines)
        )
        preserve_source_breaks = restored_matches_source_structure and not line_count_exceeded
        if (width_exceeded or line_count_exceeded) and not preserve_source_breaks:
            plain_text = next_text.replace("\n", " ")
            lines = [plain_text]
            if max_chars_per_line is not None:
                lines = wrap_visual_width(plain_text, max_chars_per_line) or [plain_text]
        updated.append(cue.with_lines(lines))
    return updated, flags


def _source_words_unchanged(cue: Cue, source_cue: Cue | None) -> bool:
    return source_cue is not None and (
        _word_freeze_signature(source_cue.plain_text)
        == _word_freeze_signature(cue.plain_text)
    )


def _scene_batches(cues: list[Cue], scene_gap_seconds: float) -> list[list[Cue]]:
    if not cues:
        return []
    batches: list[list[Cue]] = [[cues[0]]]
    previous = cues[0]
    gap_ms = scene_gap_seconds * 1000
    for cue in cues[1:]:
        if cue.start_ms - previous.end_ms > gap_ms:
            batches.append([cue])
        else:
            batches[-1].append(cue)
        previous = cue
    return batches


def _word_freeze_signature(text: str) -> list[str]:
    return [
        unicodedata.normalize("NFC", token).casefold()
        for token in re.findall(r"[\w]+", text, re.UNICODE)
    ]


_ALWAYS_QUOTATION_MARKS = frozenset('"\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a\u201a')
_CONTEXTUAL_SINGLE_QUOTATION_MARKS = frozenset("'\u2018\u2019")


def _quotation_mark_signature(text: str) -> tuple[str, ...]:
    signature: list[str] = []
    for index, character in enumerate(text):
        if character in _ALWAYS_QUOTATION_MARKS:
            signature.append(character)
            continue
        if character not in _CONTEXTUAL_SINGLE_QUOTATION_MARKS:
            continue
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if _is_word_character(previous) and _is_word_character(following):
            continue
        signature.append(character)
    return tuple(signature)


def _is_word_character(character: str) -> bool:
    if not character:
        return False
    category = unicodedata.category(character)
    return character == "_" or category[0] in {"L", "M", "N"}


def _restore_source_line_breaks(source_lines: list[str], proposed_text: str) -> list[str]:
    proposed_lines = proposed_text.splitlines() or [proposed_text]
    source_boundaries = _line_word_boundaries(source_lines)
    if _line_word_boundaries(proposed_lines) == source_boundaries:
        return proposed_lines

    flattened = " ".join(proposed_text.split())
    word_spans = list(re.finditer(r"[\w]+", flattened, re.UNICODE))
    if not word_spans:
        return proposed_lines

    split_positions = list(
        dict.fromkeys(
            _line_split_position(flattened, word_spans, boundary)
            for boundary in source_boundaries
            if 0 < boundary < len(word_spans)
        )
    )
    if not split_positions:
        return [flattened]

    restored: list[str] = []
    start = 0
    for end in split_positions:
        line = flattened[start:end].strip()
        if line:
            restored.append(line)
        start = end
    final_line = flattened[start:].strip()
    if final_line:
        restored.append(final_line)
    return restored or proposed_lines


def _line_word_boundaries(lines: list[str]) -> list[int]:
    boundaries: list[int] = []
    word_count = 0
    for line in lines[:-1]:
        word_count += len(re.findall(r"[\w]+", line, re.UNICODE))
        boundaries.append(word_count)
    return boundaries


def _line_split_position(text: str, word_spans: list[re.Match[str]], boundary: int) -> int:
    current_word_end = word_spans[boundary - 1].end()
    next_word_start = word_spans[boundary].start()
    between_words = text[current_word_end:next_word_start]
    whitespace_positions = [
        index for index, character in enumerate(between_words) if character.isspace()
    ]
    if whitespace_positions:
        return current_word_end + whitespace_positions[-1]
    return next_word_start
