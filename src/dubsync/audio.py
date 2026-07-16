from __future__ import annotations

import math
import os
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioNormalizeError(RuntimeError):
    pass


DEFAULT_FFMPEG_TIMEOUT_SECONDS = 1800.0
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 4 * 60 * 60
DEFAULT_FFPROBE_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_NORMALIZED_AUDIO_BYTES = 512 * 1024 * 1024
PCM_BYTES_PER_SECOND = 16_000 * 2
WAV_SIZE_ALLOWANCE_BYTES = 1024 * 1024
CAP_SENTINEL_BYTES = 64 * 1024


@dataclass(frozen=True)
class AudioNormalizationLimits:
    max_duration_seconds: float = DEFAULT_MAX_AUDIO_DURATION_SECONDS
    probe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_NORMALIZED_AUDIO_BYTES
    job_directory: Path | None = None
    max_job_storage_bytes: int | None = None
    min_free_storage_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_duration_seconds", self.max_duration_seconds),
            ("probe_timeout_seconds", self.probe_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")
        if self.max_job_storage_bytes is not None and self.max_job_storage_bytes <= 0:
            raise ValueError("max_job_storage_bytes must be greater than zero")
        if self.min_free_storage_bytes < 0:
            raise ValueError("min_free_storage_bytes cannot be negative")

    @classmethod
    def from_env(cls) -> "AudioNormalizationLimits":
        return cls(
            max_duration_seconds=_positive_env_float(
                "DUBSYNC_MAX_AUDIO_DURATION_SECONDS",
                DEFAULT_MAX_AUDIO_DURATION_SECONDS,
            ),
            probe_timeout_seconds=_positive_env_float(
                "DUBSYNC_FFPROBE_TIMEOUT_SECONDS",
                DEFAULT_FFPROBE_TIMEOUT_SECONDS,
            ),
            max_output_bytes=_positive_env_int(
                "DUBSYNC_MAX_NORMALIZED_AUDIO_BYTES",
                DEFAULT_MAX_NORMALIZED_AUDIO_BYTES,
            ),
        )


def resolve_ffmpeg_timeout_seconds(timeout_seconds: float | None = None) -> float:
    raw_value: float | str = (
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("DUBSYNC_FFMPEG_TIMEOUT_SECONDS", str(DEFAULT_FFMPEG_TIMEOUT_SECONDS))
    )
    try:
        resolved = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("DUBSYNC_FFMPEG_TIMEOUT_SECONDS must be a number") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("DUBSYNC_FFMPEG_TIMEOUT_SECONDS must be finite and greater than zero")
    return resolved


def normalize_audio(
    source: Path,
    dest: Path,
    ffmpeg: str = "ffmpeg",
    *,
    timeout_seconds: float | None = None,
    ffprobe: str = "ffprobe",
    limits: AudioNormalizationLimits | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_timeout = resolve_ffmpeg_timeout_seconds(timeout_seconds)
        resolved_limits = limits or AudioNormalizationLimits.from_env()
    except ValueError as exc:
        raise AudioNormalizeError(str(exc)) from exc
    duration = probe_audio_duration(
        source,
        ffprobe=ffprobe,
        timeout_seconds=resolved_limits.probe_timeout_seconds,
    )
    if duration > resolved_limits.max_duration_seconds:
        raise AudioNormalizeError(
            f"Audio is longer than {resolved_limits.max_duration_seconds:g} seconds."
        )
    predicted_bytes = predicted_normalized_audio_bytes(duration)
    output_cap = min(predicted_bytes, resolved_limits.max_output_bytes)
    if predicted_bytes > resolved_limits.max_output_bytes:
        raise AudioNormalizeError("Normalized audio would exceed the storage limit.")
    if resolved_limits.job_directory is not None and resolved_limits.max_job_storage_bytes is not None:
        current_job_bytes = tree_size_bytes(resolved_limits.job_directory)
        if current_job_bytes + predicted_bytes > resolved_limits.max_job_storage_bytes:
            raise AudioNormalizeError("Audio processing would exceed the per-job storage limit.")
    if resolved_limits.min_free_storage_bytes:
        try:
            free_bytes = shutil.disk_usage(dest.parent).free
        except OSError as exc:
            raise AudioNormalizeError("Available processing storage could not be verified.") from exc
        if free_bytes < predicted_bytes + resolved_limits.min_free_storage_bytes:
            raise AudioNormalizeError("Not enough free storage is available to process this audio.")

    partial = dest.with_name(f"{dest.name}.partial")
    partial.unlink(missing_ok=True)
    dest.unlink(missing_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        f"{resolved_limits.max_duration_seconds + 1:g}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-fs",
        str(output_cap),
        "-f",
        "wav",
        str(partial),
    ]
    try:
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=resolved_timeout,
            )
        except FileNotFoundError as exc:
            raise AudioNormalizeError("ffmpeg was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioNormalizeError(
                f"ffmpeg timed out after {resolved_timeout:g} seconds while normalizing audio"
            ) from exc
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            raise AudioNormalizeError(f"ffmpeg failed: {message}") from exc
        if not partial.exists() or partial.stat().st_size <= 0:
            raise AudioNormalizeError("ffmpeg did not produce normalized audio.")
        if partial.stat().st_size >= max(1, output_cap - CAP_SENTINEL_BYTES):
            raise AudioNormalizeError("Normalized audio reached the storage limit.")
        normalized_duration = probe_audio_duration(
            partial,
            ffprobe=ffprobe,
            timeout_seconds=resolved_limits.probe_timeout_seconds,
        )
        if normalized_duration > resolved_limits.max_duration_seconds + 0.1:
            raise AudioNormalizeError("Normalized audio exceeded the duration limit.")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def probe_audio_duration(
    source: Path,
    *,
    ffprobe: str = "ffprobe",
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
) -> float:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise AudioNormalizeError("ffprobe timeout must be finite and greater than zero")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise AudioNormalizeError("ffprobe was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioNormalizeError(
            f"ffprobe timed out after {timeout_seconds:g} seconds while inspecting audio"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioNormalizeError("Audio duration could not be inspected.") from exc
    durations = _duration_values(result.stdout)
    if not durations:
        raise AudioNormalizeError("Audio duration could not be inspected.")
    return max(durations)


def predicted_normalized_audio_bytes(duration_seconds: float) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise AudioNormalizeError("Audio duration must be finite and greater than zero.")
    return math.ceil(duration_seconds * PCM_BYTES_PER_SECOND) + WAV_SIZE_ALLOWANCE_BYTES


def tree_size_bytes(directory: Path) -> int:
    total = 0
    if not directory.exists():
        return 0
    for path in directory.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _duration_values(raw: str) -> list[float]:
    values: list[object] = []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        values = [raw.strip()]
    else:
        if isinstance(payload, dict):
            format_payload = payload.get("format")
            if isinstance(format_payload, dict):
                values.append(format_payload.get("duration"))
            streams = payload.get("streams")
            if isinstance(streams, list):
                values.extend(
                    stream.get("duration")
                    for stream in streams
                    if isinstance(stream, dict)
                )
        elif isinstance(payload, (int, float, str)):
            values = [payload]
    durations: list[float] = []
    for value in values:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    return durations


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
