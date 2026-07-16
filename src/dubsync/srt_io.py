from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .models import Cue

TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


class SRTParseError(ValueError):
    pass


@dataclass(frozen=True)
class SRTParseLimits:
    max_lines: int
    max_cues: int
    max_line_chars: int

    def __post_init__(self) -> None:
        if min(self.max_lines, self.max_cues, self.max_line_chars) <= 0:
            raise ValueError("SRT parse limits must be greater than zero")


def parse_timestamp(value: str) -> int:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + millis


def format_timestamp(ms: int) -> str:
    if ms < 0:
        raise ValueError("timestamp cannot be negative")
    seconds, millis = divmod(int(ms), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _iter_lines(text: str, limits: SRTParseLimits | None) -> Iterator[tuple[int, str]]:
    start = 0
    cursor = 0
    line_number = 0
    text_length = len(text)
    while cursor < text_length:
        character = text[cursor]
        if character not in {"\r", "\n"}:
            cursor += 1
            continue
        line_number += 1
        line = text[start:cursor]
        if line_number == 1:
            line = line.lstrip("\ufeff")
        _validate_line_limit(line, line_number=line_number, limits=limits)
        yield line_number, line
        cursor += 2 if character == "\r" and cursor + 1 < text_length and text[cursor + 1] == "\n" else 1
        start = cursor

    if start < text_length:
        line_number += 1
        line = text[start:]
        if line_number == 1:
            line = line.lstrip("\ufeff")
        _validate_line_limit(line, line_number=line_number, limits=limits)
        yield line_number, line


def _validate_line_limit(
    line: str,
    *,
    line_number: int,
    limits: SRTParseLimits | None,
) -> None:
    if limits is None:
        return
    if line_number > limits.max_lines:
        raise SRTParseError(f"subtitle exceeds {limits.max_lines} lines")
    if len(line) > limits.max_line_chars:
        raise SRTParseError(
            f"subtitle line {line_number} exceeds {limits.max_line_chars} characters"
        )


def _split_blocks(text: str, limits: SRTParseLimits | None = None) -> Iterator[list[str]]:
    current: list[str] = []
    for _, line in _iter_lines(text, limits):
        if line.strip() == "":
            if current:
                yield current
                current = []
            continue
        current.append(line)
    if current:
        yield current


def parse_srt_text(text: str, *, limits: SRTParseLimits | None = None) -> list[Cue]:
    cues: list[Cue] = []
    for block_number, block in enumerate(_split_blocks(text, limits), start=1):
        if limits is not None and block_number > limits.max_cues:
            raise SRTParseError(f"subtitle exceeds {limits.max_cues} cues")
        if len(block) < 3:
            raise SRTParseError(f"block {block_number} is incomplete")
        try:
            index = int(block[0].strip())
        except ValueError as exc:
            raise SRTParseError(f"block {block_number} has invalid cue index") from exc

        match = TIMESTAMP_RE.match(block[1].strip())
        if match is None:
            raise SRTParseError(f"cue {index} has invalid timestamp line")

        lines = [line.rstrip() for line in block[2:]]
        cues.append(
            Cue(
                index=index,
                start_ms=parse_timestamp(match.group("start")),
                end_ms=parse_timestamp(match.group("end")),
                lines=lines,
            )
        )
    return cues


def write_srt(cues: list[Cue], *, renumber: bool = False) -> str:
    blocks: list[str] = []
    for output_index, cue in enumerate(cues, start=1):
        cue_index = output_index if renumber else cue.index
        lines = [
            str(cue_index),
            f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}",
            *[line.rstrip() for line in cue.lines],
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
