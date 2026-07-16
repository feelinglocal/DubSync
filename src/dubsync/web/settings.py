from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from ..srt_io import SRTParseLimits


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path
    providers_path: Path
    style_path: Path | None
    static_dir: Path = Path("web/dist")
    max_upload_bytes: int = 512 * 1024 * 1024
    max_batch_upload_bytes: int = 512 * 1024 * 1024
    max_retained_storage_bytes: int = 4 * 1024 * 1024 * 1024
    max_audio_duration_seconds: float = 4 * 60 * 60
    ffprobe_timeout_seconds: float = 15.0
    max_normalized_audio_bytes: int = 512 * 1024 * 1024
    max_job_work_bytes: int = 64 * 1024 * 1024
    max_job_storage_bytes: int = 1024 * 1024 * 1024
    min_free_storage_bytes: int = 2 * 1024 * 1024 * 1024
    max_srt_bytes: int = 2 * 1024 * 1024
    max_srt_lines: int = 60_000
    max_srt_cues: int = 20_000
    max_srt_line_bytes: int = 16 * 1024
    max_srt_line_chars: int = 4_096
    retention_hours: int = 24
    processing_inline: bool = False
    max_submissions_per_hour: int = 5
    max_outstanding_child_jobs: int = 10
    worker_threads: int = 1
    cleanup_interval_seconds: float = 300.0
    active_job_timeout_hours: float = 24.0
    job_access_code: str | None = None
    require_job_access_code: bool = False

    @property
    def max_jobs_per_hour(self) -> int:
        """Backward-compatible code alias for the submission rate limit."""
        return self.max_submissions_per_hour

    @property
    def srt_parse_limits(self) -> SRTParseLimits:
        return SRTParseLimits(
            max_lines=self.max_srt_lines,
            max_cues=self.max_srt_cues,
            max_line_chars=self.max_srt_line_chars,
        )

    @classmethod
    def from_env(cls) -> "WebSettings":
        load_dotenv(Path.cwd() / ".env", override=False)
        style_value = os.getenv("DUBSYNC_STYLE_PATH", "style_profile.yaml").strip()
        access_code = os.getenv("DUBSYNC_JOB_ACCESS_CODE", "").strip() or None
        if access_code is not None and len(access_code) < 12:
            raise ValueError("DUBSYNC_JOB_ACCESS_CODE must contain at least 12 characters")
        return cls(
            data_dir=Path(os.getenv("DUBSYNC_DATA_DIR", "runtime-data")),
            providers_path=Path(os.getenv("DUBSYNC_PROVIDERS_PATH", "provider.yaml")),
            style_path=Path(style_value) if style_value else None,
            static_dir=Path(os.getenv("DUBSYNC_STATIC_DIR", "web/dist")),
            max_upload_bytes=_env_int("DUBSYNC_MAX_UPLOAD_BYTES", 512 * 1024 * 1024),
            max_batch_upload_bytes=_env_int("DUBSYNC_MAX_BATCH_UPLOAD_BYTES", 512 * 1024 * 1024),
            max_retained_storage_bytes=_env_int(
                "DUBSYNC_MAX_RETAINED_STORAGE_BYTES",
                4 * 1024 * 1024 * 1024,
            ),
            max_audio_duration_seconds=_env_float(
                "DUBSYNC_MAX_AUDIO_DURATION_SECONDS",
                4 * 60 * 60,
            ),
            ffprobe_timeout_seconds=_env_float("DUBSYNC_FFPROBE_TIMEOUT_SECONDS", 15.0),
            max_normalized_audio_bytes=_env_int(
                "DUBSYNC_MAX_NORMALIZED_AUDIO_BYTES",
                512 * 1024 * 1024,
            ),
            max_job_work_bytes=_env_int("DUBSYNC_MAX_JOB_WORK_BYTES", 64 * 1024 * 1024),
            max_job_storage_bytes=_env_int(
                "DUBSYNC_MAX_JOB_STORAGE_BYTES",
                1024 * 1024 * 1024,
            ),
            min_free_storage_bytes=_env_int(
                "DUBSYNC_MIN_FREE_STORAGE_BYTES",
                2 * 1024 * 1024 * 1024,
            ),
            max_srt_bytes=_env_int("DUBSYNC_MAX_SRT_BYTES", 2 * 1024 * 1024),
            max_srt_lines=_env_int("DUBSYNC_MAX_SRT_LINES", 60_000),
            max_srt_cues=_env_int("DUBSYNC_MAX_SRT_CUES", 20_000),
            max_srt_line_bytes=_env_int("DUBSYNC_MAX_SRT_LINE_BYTES", 16 * 1024),
            max_srt_line_chars=_env_int("DUBSYNC_MAX_SRT_LINE_CHARS", 4_096),
            retention_hours=_env_int("DUBSYNC_RETENTION_HOURS", 24),
            processing_inline=_env_bool("DUBSYNC_PROCESSING_INLINE", False),
            max_submissions_per_hour=_env_int_with_legacy(
                "DUBSYNC_MAX_SUBMISSIONS_PER_HOUR",
                "DUBSYNC_MAX_JOBS_PER_HOUR",
                5,
            ),
            max_outstanding_child_jobs=_env_int("DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS", 10),
            worker_threads=_env_int("DUBSYNC_WORKER_THREADS", 1),
            cleanup_interval_seconds=_env_float("DUBSYNC_CLEANUP_INTERVAL_SECONDS", 300.0),
            active_job_timeout_hours=_env_float("DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS", 24.0),
            job_access_code=access_code,
            require_job_access_code=_env_bool(
                "DUBSYNC_REQUIRE_JOB_ACCESS_CODE",
                os.getenv("RENDER", "").strip().lower() == "true",
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_int_with_legacy(name: str, legacy_name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return _env_int(name, default)
    return _env_int(legacy_name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value
