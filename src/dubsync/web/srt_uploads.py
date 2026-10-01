from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from ..models import Cue
from ..srt_io import SRTParseError, SRTParseLimits, decode_srt_bytes, parse_srt_text

UPLOAD_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class ValidatedSRTUpload:
    data: bytes
    cues: tuple[Cue, ...]
    encoding_notice: str | None = None


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
    bom_encoded = False
    try:
        while chunk := await upload.read(UPLOAD_READ_CHUNK_BYTES):
            if not chunks:
                bom_encoded = chunk.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff"))
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            try:
                scan_state = scan_state if bom_encoded else _scan_srt_bytes(
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
        data = b"".join(chunks)
        text, encoding_notice = decode_srt_bytes(data)
        if bom_encoded:
            scan_state = _scan_srt_bytes(
                text.encode("utf-8"), state=_SRTByteScanState(),
                max_lines=parse_limits.max_lines, max_line_bytes=max_line_bytes,
            )
        _finish_srt_byte_scan(scan_state, max_lines=parse_limits.max_lines)
        cues = parse_srt_text(text, limits=parse_limits)
        if not cues:
            raise SRTParseError("no subtitle cues were found")
    except SRTParseError as exc:
        raise _invalid_srt(label, exc) from exc
    # Preserve the authored upload; downstream read_srt reports the conversion.
    return ValidatedSRTUpload(data=data, cues=tuple(cues), encoding_notice=encoding_notice)


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
