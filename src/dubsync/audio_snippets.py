from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import AudioSnippet, DivergenceSpan


class AudioSnippetError(RuntimeError):
    pass


def extract_audio_snippets(
    audio_path: Path,
    spans: list[DivergenceSpan],
    output_dir: Path,
    pad_seconds: float = 2.0,
    max_duration_seconds: float = 20.0,
    ffmpeg: str = "ffmpeg",
) -> list[AudioSnippet]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snippets: list[AudioSnippet] = []
    for span in spans:
        if span.start is None or span.end is None or span.end <= span.start:
            continue
        start, end = _snippet_window(span.start, span.end, pad_seconds, max_duration_seconds)
        output_path = output_dir / f"{_safe_case_id(span.case_id)}.wav"
        _cut_wav_snippet(audio_path, output_path, start, end, ffmpeg)
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


def _cut_wav_snippet(audio_path: Path, output_path: Path, start: float, end: float, ffmpeg: str) -> None:
    duration = max(0.001, end - start)
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
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioSnippetError("ffmpeg was not found on PATH while extracting audio snippets") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise AudioSnippetError(f"ffmpeg failed while extracting audio snippet: {message}") from exc


def _safe_case_id(case_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip(".-")
    return sanitized or "case"
