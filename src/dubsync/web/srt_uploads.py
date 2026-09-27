from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from ..models import Cue
from ..srt_io import SRTParseError, SRTParseLimits, parse_srt_text

UPLOAD_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class ValidatedSRTUpload:
    data: bytes
    cues: tuple[Cue, ...]


@dataclass(frozen=True)
class _SRTByteScanState:
    completed_lines: int = 0
    current_line_bytes: int = 0
    previous_byte_was_cr: bool = False


async def read_validated_srt_upload(
    upload: UploadFile | StarletteUploadFile,
    *,
    max_bytes: int,
    max_line_bytes: int,
    parse_limits: SRTParseLimits,
    label: str,
) -> ValidatedSRTUpload:
    chunks: list[bytes] = []
    total_bytes = 0
    scan_state = _SRTByteScanState()
    try:
        while chunk := await upload.read(UPLOAD_READ_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            try:
                scan_state = _scan_srt_bytes(
                    chunk,
                    state=scan_state,
                    max_lines=parse_limits.max_lines,
                    max_line_bytes=max_line_bytes,
                )
            except SRTParseError as exc:
                raise _invalid_srt(label, exc) from exc
            chunks.append(chunk)
    finally:
        await upload.close()

    if total_bytes == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    try:
        _finish_srt_byte_scan(scan_state, max_lines=parse_limits.max_lines)
        text = b"".join(chunks).decode("utf-8-sig")
        cues = parse_srt_text(text, limits=parse_limits)
        if not cues:
            raise SRTParseError("no subtitle cues were found")
    except UnicodeDecodeError as exc:
        raise _invalid_srt(label, ValueError("file is not valid UTF-8")) from exc
    except SRTParseError as exc:
        raise _invalid_srt(label, exc) from exc
    return ValidatedSRTUpload(data=b"".join(chunks), cues=tuple(cues))


def _scan_srt_bytes(
    chunk: bytes,
    *,
    state: _SRTByteScanState,
    max_lines: int,
    max_line_bytes: int,
) -> _SRTByteScanState:
    completed_lines = state.completed_lines
    current_line_bytes = state.current_line_bytes
    previous_byte_was_cr = state.previous_byte_was_cr
    for byte in chunk:
        if byte == 13:
            completed_lines += 1
            _validate_completed_line_count(completed_lines, max_lines=max_lines)
            current_line_bytes = 0
            previous_byte_was_cr = True
            continue
        if byte == 10:
            if previous_byte_was_cr:
                previous_byte_was_cr = False
                continue
            completed_lines += 1
            _validate_completed_line_count(completed_lines, max_lines=max_lines)
            current_line_bytes = 0
            continue
        previous_byte_was_cr = False
        current_line_bytes += 1
        if current_line_bytes > max_line_bytes:
            raise SRTParseError(
                f"subtitle line {completed_lines + 1} exceeds {max_line_bytes} bytes"
            )
    return _SRTByteScanState(
        completed_lines=completed_lines,
        current_line_bytes=current_line_bytes,
        previous_byte_was_cr=previous_byte_was_cr,
    )


def _finish_srt_byte_scan(state: _SRTByteScanState, *, max_lines: int) -> None:
    if state.current_line_bytes:
        _validate_completed_line_count(state.completed_lines + 1, max_lines=max_lines)


def _validate_completed_line_count(line_count: int, *, max_lines: int) -> None:
    if line_count > max_lines:
        raise SRTParseError(f"subtitle exceeds {max_lines} lines")


def _invalid_srt(label: str, error: Exception) -> HTTPException:
    detail = str(error).splitlines()[0] or "invalid SRT"
    return HTTPException(status_code=422, detail=f"Could not read the {label}: {detail}")
