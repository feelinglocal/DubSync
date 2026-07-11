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
    return after


def apply_punctuation_pass(
    cues: list[Cue],
    adapter: PunctuationAdapter,
    scene_gap_seconds: float = 4.0,
    max_chars_per_line: int | None = None,
    max_lines_per_cue: int | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    proposed: dict[int, str] = {}
    for batch in _scene_batches(cues, scene_gap_seconds):
        proposed.update(adapter.punctuate(batch))
    if not proposed:
        return cues, []

    updated: list[Cue] = []
    flags: list[QCFlag] = []
    for cue in cues:
        next_text = proposed.get(cue.index)
        if next_text is None or next_text == cue.text:
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

        lines = next_text.splitlines() or [next_text]
        width_exceeded = max_chars_per_line is not None and any(
            display_width(line) > max_chars_per_line for line in lines
        )
        line_count_exceeded = max_lines_per_cue is not None and len(lines) > max_lines_per_cue
        if width_exceeded or line_count_exceeded:
            plain_text = next_text.replace("\n", " ")
            lines = [plain_text]
            if max_chars_per_line is not None:
                lines = wrap_visual_width(plain_text, max_chars_per_line) or [plain_text]
        updated.append(cue.with_lines(lines))
    return updated, flags


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
