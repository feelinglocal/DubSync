from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

from .audio import resolve_ffmpeg_timeout_seconds
from .models import AudioSnippet, DivergenceSpan


class AudioSnippetError(RuntimeError):
    pass


DEFAULT_MAX_AUDIO_SNIPPET_BYTES = 32 * 1024 * 1024
SNIPPET_WAV_ALLOWANCE_BYTES = 64 * 1024
SNIPPET_CAP_SENTINEL_BYTES = 4096


def extract_audio_snippets(
    audio_path: Path,
    spans: list[DivergenceSpan],
    output_dir: Path,
    pad_seconds: float = 2.0,
    max_duration_seconds: float = 20.0,
    ffmpeg: str = "ffmpeg",
    ffmpeg_timeout_seconds: float | None = None,
    max_total_bytes: int | None = None,
) -> list[AudioSnippet]:
    try:
        resolved_timeout = resolve_ffmpeg_timeout_seconds(ffmpeg_timeout_seconds)
    except ValueError as exc:
        raise AudioSnippetError(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_max_bytes = _max_snippet_bytes(max_total_bytes)
    used_bytes = 0
    snippets: list[AudioSnippet] = []
    for span in spans:
        if span.start is None or span.end is None or span.end <= span.start:
            continue
        start, end = _snippet_window(span.start, span.end, pad_seconds, max_duration_seconds)
        output_path = output_dir / f"{_safe_case_id(span.case_id)}.wav"
        predicted_bytes = math.ceil((end - start) * 32_000) + SNIPPET_WAV_ALLOWANCE_BYTES
        remaining_bytes = resolved_max_bytes - used_bytes
        if predicted_bytes > remaining_bytes:
            raise AudioSnippetError("Audio snippets would exceed the job storage budget")
        _cut_wav_snippet(
            audio_path,
            output_path,
            start,
            end,
            ffmpeg,
            resolved_timeout,
            predicted_bytes,
        )
        used_bytes += output_path.stat().st_size
        snippets.append(
            AudioSnippet(
                case_id=span.case_id,
                path=str(output_path),
                mime_type="audio/wav",
                start=round(start, 3),
                end=round(end, 3),
            )
        )
    return snippets


def _snippet_window(
    span_start: float,
    span_end: float,
    pad_seconds: float,
    max_duration_seconds: float,
) -> tuple[float, float]:
    pad = max(0.0, pad_seconds)
    start = max(0.0, span_start - pad)
    end = span_end + pad
    max_duration = max(0.0, max_duration_seconds)
    if max_duration and end - start > max_duration:
        midpoint = (span_start + span_end) / 2.0
        start = max(0.0, midpoint - (max_duration / 2.0))
        end = start + max_duration
    return start, max(start, end)


def _cut_wav_snippet(
    audio_path: Path,
    output_path: Path,
    start: float,
    end: float,
    ffmpeg: str,
    timeout_seconds: float,
    output_cap_bytes: int,
) -> None:
    duration = max(0.001, end - start)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    partial_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-fs",
        str(output_cap_bytes),
        "-f",
        "wav",
        partial_path,
    ]
    try:
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AudioSnippetError("ffmpeg was not found on PATH while extracting audio snippets") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioSnippetError(
                f"ffmpeg timed out after {timeout_seconds:g} seconds while extracting audio snippet"
            ) from exc
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            raise AudioSnippetError(f"ffmpeg failed while extracting audio snippet: {message}") from exc
        if not partial_path.exists() or partial_path.stat().st_size <= 0:
            raise AudioSnippetError("ffmpeg did not produce an audio snippet")
        if partial_path.stat().st_size >= max(1, output_cap_bytes - SNIPPET_CAP_SENTINEL_BYTES):
            raise AudioSnippetError("Audio snippet reached the job storage budget")
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)


def _safe_case_id(case_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip(".-")
    return sanitized or "case"


def _max_snippet_bytes(value: int | None) -> int:
    raw_value: int | str = (
        value
        if value is not None
        else os.getenv("DUBSYNC_MAX_AUDIO_SNIPPET_BYTES", str(DEFAULT_MAX_AUDIO_SNIPPET_BYTES))
    )
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise AudioSnippetError("DUBSYNC_MAX_AUDIO_SNIPPET_BYTES must be an integer") from exc
    if resolved <= 0:
        raise AudioSnippetError("DUBSYNC_MAX_AUDIO_SNIPPET_BYTES must be greater than zero")
    return resolved
