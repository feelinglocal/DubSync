from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path
    providers_path: Path
    style_path: Path | None
    static_dir: Path = Path("web/dist")
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_srt_bytes: int = 20 * 1024 * 1024
    retention_hours: int = 24
    processing_inline: bool = False
    max_jobs_per_hour: int = 5
    worker_threads: int = 1
    cleanup_interval_seconds: float = 300.0
    job_access_code: str | None = None
    require_job_access_code: bool = False

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
            max_upload_bytes=_env_int("DUBSYNC_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024),
            max_srt_bytes=_env_int("DUBSYNC_MAX_SRT_BYTES", 20 * 1024 * 1024),
            retention_hours=_env_int("DUBSYNC_RETENTION_HOURS", 24),
            processing_inline=_env_bool("DUBSYNC_PROCESSING_INLINE", False),
            max_jobs_per_hour=_env_int("DUBSYNC_MAX_JOBS_PER_HOUR", 5),
            worker_threads=_env_int("DUBSYNC_WORKER_THREADS", 1),
            cleanup_interval_seconds=_env_float("DUBSYNC_CLEANUP_INTERVAL_SECONDS", 300.0),
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
