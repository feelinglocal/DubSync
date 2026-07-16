from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from .jobs import JobMode

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
AUDIO_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/wave",
    "audio/x-m4a",
    "audio/x-wav",
    "application/octet-stream",
}
MAX_BATCH_ITEMS = 10
MAX_UPLOAD_FILENAME_CHARS = 255
WINDOWS_UNSAFE_FILENAME_CHARS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_STEMS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def batch_upload_plan(
    mode: JobMode,
    audio_uploads: list[UploadFile],
    subtitle_uploads: list[UploadFile],
) -> list[tuple[str, UploadFile, str, UploadFile | None]]:
    if not 1 <= len(audio_uploads) <= MAX_BATCH_ITEMS:
        raise HTTPException(status_code=422, detail="Select between 1 and 10 audio files.")
    if mode == "generate" and subtitle_uploads:
        raise HTTPException(status_code=422, detail="Generate batches accept audio files only.")
    if mode == "sync" and len(subtitle_uploads) != len(audio_uploads):
        raise HTTPException(status_code=422, detail="Every audio file must have one matching SRT.")

    audio_entries: list[tuple[str, str, UploadFile, str]] = []
    audio_keys: set[str] = set()
    for upload in audio_uploads:
        source_name = validate_source_filename(upload)
        key = _pairing_key(source_name)
        if key in audio_keys:
            raise HTTPException(status_code=422, detail="Audio filenames must be unique.")
        audio_keys.add(key)
        audio_entries.append((source_name, key, upload, validate_audio(upload)))

    if mode == "generate":
        return [
            (source_name, upload, extension, None)
            for source_name, _, upload, extension in audio_entries
        ]

    subtitle_by_key: dict[str, UploadFile] = {}
    for upload in subtitle_uploads:
        source_name = validate_source_filename(upload)
        validate_subtitle(upload)
        key = _pairing_key(source_name)
        if key in subtitle_by_key:
            raise HTTPException(status_code=422, detail="Subtitle filenames must be unique.")
        subtitle_by_key[key] = upload
    if set(subtitle_by_key) != audio_keys:
        raise HTTPException(
            status_code=422,
            detail="Audio and SRT filenames must have matching names before their extensions.",
        )
    return [
        (source_name, upload, extension, subtitle_by_key[key])
        for source_name, key, upload, extension in audio_entries
    ]


def validate_audio(upload: UploadFile) -> str:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in AUDIO_EXTENSIONS or upload.content_type not in AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio file.")
    return extension


def validate_subtitle(upload: UploadFile) -> None:
    if Path(upload.filename or "").suffix.lower() != ".srt":
        raise HTTPException(status_code=415, detail="Subtitle must be an SRT file.")
    if upload.content_type not in {"application/octet-stream", "application/x-subrip", "text/plain"}:
        raise HTTPException(status_code=415, detail="Unsupported subtitle file.")


def validate_source_filename(upload: UploadFile) -> str:
    filename = upload.filename or ""
    normalized_filename = unicodedata.normalize("NFKC", filename)
    if (
        not filename
        or len(filename) > MAX_UPLOAD_FILENAME_CHARS
        or _windows_filename_units(normalized_filename) > MAX_UPLOAD_FILENAME_CHARS
        or normalized_filename in {".", ".."}
        or any(character in WINDOWS_UNSAFE_FILENAME_CHARS for character in normalized_filename)
        or any(unicodedata.category(character).startswith("C") for character in normalized_filename)
        or _contains_percent_encoded_unsafe_character(normalized_filename)
    ):
        raise HTTPException(status_code=422, detail="Unsafe or overlong upload filename.")
    source_name = Path(filename).stem
    normalized_source = unicodedata.normalize("NFKC", source_name)
    if (
        not normalized_source.strip(" .")
        or normalized_source != normalized_source.rstrip(" .")
        or normalized_source.casefold() in WINDOWS_RESERVED_STEMS
        or _windows_filename_units(f"{source_name}-dubsync-synced.srt")
        > MAX_UPLOAD_FILENAME_CHARS
    ):
        raise HTTPException(status_code=422, detail="Unsafe or overlong upload filename.")
    return source_name


def _pairing_key(source_name: str) -> str:
    return unicodedata.normalize("NFKC", source_name).upper()


def _contains_percent_encoded_unsafe_character(value: str) -> bool:
    for match in re.finditer(r"%([0-9A-Fa-f]{2})", value):
        byte = int(match.group(1), 16)
        if byte < 32 or byte == 127 or chr(byte) in WINDOWS_UNSAFE_FILENAME_CHARS:
            return True
    return False


def _windows_filename_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2
