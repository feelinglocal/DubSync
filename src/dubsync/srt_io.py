from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .models import Cue

TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


class SRTParseError(ValueError):
    pass


class SubtitleEncodingWarning(UserWarning):
    """A legacy subtitle was decoded without changing the authored file."""


def decode_srt_bytes(data: bytes, *, encoding: str | None = None) -> tuple[str, str | None]:
    """Decode common subtitle encodings, reporting every legacy conversion.

    BOMs are authoritative, and a BOM-marked UTF-16/32 file decodes without
    loss, so it needs no notice. An unmarked Western file can use Windows-1252,
    but ambiguous Japanese byte streams require an explicit encoding instead
    of silently becoming plausible-looking Western text.
    """
    codec = encoding
    unicode_bom = False
    if codec is None:
        if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
            codec = "utf-32"
            unicode_bom = True
        elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
            codec = "utf-16"
            unicode_bom = True
        else:
            try:
                decoded_utf8 = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                pass
            else:
                if "\x00" in decoded_utf8:
                    raise SRTParseError("subtitle contains NUL characters; check its encoding and save it as UTF-8")
                return decoded_utf8, None
            # A Shift-JIS file often decodes successfully as Windows-1252.
            # Reject that ambiguity; callers can pass encoding="shift_jis".
            ambiguous = False
            for candidate in ("shift_jis", "euc_jp"):
                try:
                    decoded = data.decode(candidate)
                except UnicodeDecodeError:
                    continue
                if any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in decoded):
                    ambiguous = True
                    break
            if ambiguous or b"\x00" in data:
                raise SRTParseError(
                    "subtitle encoding is not UTF-8 and cannot be identified safely; "
                    "save it as UTF-8 or specify its encoding explicitly"
                )
            codec = "cp1252"
    try:
        text = data.decode(codec)
    except (LookupError, UnicodeDecodeError) as exc:
        raise SRTParseError(
            f"subtitle is not valid {codec}; save it as UTF-8 or specify its encoding explicitly"
        ) from exc
    if "\x00" in text:
        raise SRTParseError("subtitle contains NUL characters; check its encoding and save it as UTF-8")
    notice = None
    if not unicode_bom:
        label = "Windows-1252" if codec.lower().replace("-", "") in {"cp1252", "windows1252"} else codec
        notice = f"Subtitle decoded as {label}; verify accented characters. UTF-8 is recommended for interchange."
    return text.lstrip("\ufeff"), notice


def parse_srt_bytes(
    data: bytes, *, encoding: str | None = None, limits: SRTParseLimits | None = None,
) -> list[Cue]:
    text, notice = decode_srt_bytes(data, encoding=encoding)
    if notice is not None:
        warnings.warn(notice, SubtitleEncodingWarning, stacklevel=2)
    return parse_srt_text(text, limits=limits)


def read_srt(path: Path, *, encoding: str | None = None, limits: SRTParseLimits | None = None) -> list[Cue]:
    return parse_srt_bytes(path.read_bytes(), encoding=encoding, limits=limits)


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
    for line_number, line in _iter_lines(text, limits):
        if line.strip() == "":
            if current:
                yield current
                current = []
            continue
        # Missing blank separators are common in hand-edited SRTs. A complete
        # index/timestamp header starts a new cue even inside the current block.
        # A number spoken in dialogue remains text unless a timestamp follows.
        if len(current) >= 3 and TIMESTAMP_RE.match(line.strip()) and re.fullmatch(r"[+-]?\d+", current[-1].strip()):
            next_index = current.pop()
            yield current
            current = [next_index, line]
        elif len(current) >= 2 and TIMESTAMP_RE.match(line.strip()):
            # Without its cue number the cue cannot be told apart from text:
            # kept, it would merge two cues and show the timestamp as dialogue.
            raise SRTParseError(
                f"subtitle line {line_number} is a cue timestamp inside the previous cue; "
                "add a blank line and the cue number before it"
            )
        else:
            current.append(line)
    if current:
        yield current


def parse_srt_text(text: str, *, limits: SRTParseLimits | None = None) -> list[Cue]:
    cues: list[Cue] = []
    for block_number, block in enumerate(_split_blocks(text, limits), start=1):
        if limits is not None and block_number > limits.max_cues:
            raise SRTParseError(f"subtitle exceeds {limits.max_cues} cues")
        if len(block) < 2:
            raise SRTParseError(f"block {block_number} is incomplete")
        try:
            index = int(block[0].strip())
        except ValueError as exc:
            raise SRTParseError(f"block {block_number} has invalid cue index") from exc

        match = TIMESTAMP_RE.match(block[1].strip())
        if match is None:
            raise SRTParseError(f"cue {index} has invalid timestamp line")

        lines = [line.rstrip() for line in block[2:]] or [""]
        cues.append(
            Cue(
                index=index,
                start_ms=parse_timestamp(match.group("start")),
                end_ms=parse_timestamp(match.group("end")),
                lines=lines,
            )
        )
    return cues


def validate_cue_timings_for_export(cues: list[Cue]) -> None:
    """Reject unusable display intervals without guessing speech or losing text.

    Input parsing remains permissive for inspection and evidence-based repair.
    All SRT exports, including resumed runs, must satisfy this hard contract;
    recording a QC error alone cannot make a malformed cue importable.
    """
    for cue in cues:
        if cue.start_ms < 0:
            raise ValueError(
                f"cue {cue.index} has a negative start ({cue.start_ms} ms); "
                "review its timing before exporting subtitles"
            )
        if cue.end_ms <= cue.start_ms:
            raise ValueError(
                f"cue {cue.index} has a non-positive duration "
                f"({cue.start_ms} --> {cue.end_ms} ms); "
                "review its timing before exporting subtitles"
            )


def write_srt(cues: list[Cue], *, renumber: bool = False) -> str:
    validate_cue_timings_for_export(cues)
    blocks: list[str] = []
    for output_index, cue in enumerate(cues, start=1):
        if not cue.plain_text:
            raise ValueError(f"cue {cue.index} has no subtitle text")
        cue_index = output_index if renumber else cue.index
        # Break only where a reader breaks (CR/LF); str.splitlines would also
        # turn other separators inside an authored line into an extra line.
        lines = [
            str(cue_index),
            f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}",
            *[part.rstrip() for line in cue.lines for part in re.split(r"\r\n?|\n", line) if part.strip()],
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
