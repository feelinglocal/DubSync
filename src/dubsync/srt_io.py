from __future__ import annotations

import re

from .models import Cue

TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


class SRTParseError(ValueError):
    pass


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


def _split_blocks(text: str) -> list[list[str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in normalized.split("\n"):
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_srt_text(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block_number, block in enumerate(_split_blocks(text), start=1):
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
