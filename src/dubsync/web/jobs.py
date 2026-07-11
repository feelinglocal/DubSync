from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

from dubsync.pipeline import sync_episode
from dubsync.transcription import generate_srt_from_audio

from .settings import WebSettings

logger = logging.getLogger(__name__)

JobMode = Literal["sync", "generate"]
JobStatus = Literal["queued", "processing", "complete", "failed"]


@dataclass(frozen=True)
class JobRecord:
    id: str
    token_hash: str
    mode: JobMode
    status: JobStatus
    progress: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    directory: Path
    audio_path: Path
    srt_path: Path | None
    fps: float
    language: str
    style: str
    output_srt: Path | None = None
    qc_json: Path | None = None
    qc_html: Path | None = None
    changes_srt: Path | None = None
    cost_usd: float | None = None
    cue_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProcessedArtifacts:
    output_srt: Path
    qc_json: Path
    qc_html: Path
    cost_usd: float
    cue_count: int
    changes_srt: Path | None = None


class JobStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "jobs.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('sync', 'generate')),
                    status TEXT NOT NULL CHECK(status IN ('queued', 'processing', 'complete', 'failed')),
                    progress INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    directory TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    srt_path TEXT,
                    fps REAL NOT NULL,
                    language TEXT NOT NULL,
                    style TEXT NOT NULL,
                    output_srt TEXT,
                    qc_json TEXT,
                    qc_html TEXT,
                    changes_srt TEXT,
                    cost_usd REAL,
                    cue_count INTEGER,
                    error TEXT
                )
                """
            )

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def create(self, job: JobRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, token_hash, mode, status, progress, created_at, updated_at, expires_at,
                    directory, audio_path, srt_path, fps, language, style
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.token_hash,
                    job.mode,
                    job.status,
                    job.progress,
                    _iso(job.created_at),
                    _iso(job.updated_at),
                    _iso(job.expires_at),
                    str(job.directory),
                    str(job.audio_path),
                    str(job.srt_path) if job.srt_path else None,
                    job.fps,
                    job.language,
                    job.style,
                ),
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _record(row) if row is not None else None

    def pending(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'processing') AND expires_at > ? ORDER BY created_at",
                (_iso(datetime.now(UTC)),),
            ).fetchall()
        return [_record(row) for row in rows]

    def mark_processing(self, job_id: str) -> JobRecord:
        return self._update(job_id, status="processing", progress=25, error=None)

    def mark_complete(self, job_id: str, artifacts: ProcessedArtifacts) -> JobRecord:
        return self._update(
            job_id,
            status="complete",
            progress=100,
            output_srt=str(artifacts.output_srt),
            qc_json=str(artifacts.qc_json),
            qc_html=str(artifacts.qc_html),
            changes_srt=str(artifacts.changes_srt) if artifacts.changes_srt else None,
            cost_usd=artifacts.cost_usd,
            cue_count=artifacts.cue_count,
            error=None,
        )

    def mark_failed(self, job_id: str) -> JobRecord:
        return self._update(
            job_id,
            status="failed",
            progress=100,
            error="Processing failed. Check the input files and try again.",
        )

    def delete_expired(self, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs WHERE expires_at <= ?", (_iso(cutoff),)).fetchall()
            connection.executemany("DELETE FROM jobs WHERE id = ?", [(row["id"],) for row in rows])
        for row in rows:
            self._remove_job_directory(Path(row["directory"]))
        return len(rows)

    def _remove_job_directory(self, directory: Path) -> None:
        resolved = directory.resolve()
        if resolved.parent != self.data_dir or not resolved.name.startswith("job-"):
            logger.error("Refusing to delete unexpected job directory: %s", resolved)
            return
        shutil.rmtree(resolved, ignore_errors=True)

    def _update(self, job_id: str, **values: object) -> JobRecord:
        allowed = {
            "status",
            "progress",
            "output_srt",
            "qc_json",
            "qc_html",
            "changes_srt",
            "cost_usd",
            "cue_count",
            "error",
        }
        if not values or not set(values).issubset(allowed):
            raise ValueError("Unsupported job update")
        values = {**values, "updated_at": _iso(datetime.now(UTC))}
        assignments = ", ".join(f"{key} = ?" for key in values)
        parameters = [*values.values(), job_id]
        with self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", parameters)
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


Processor = Callable[[JobRecord, WebSettings], ProcessedArtifacts]


class JobService:
    def __init__(self, settings: WebSettings, processor: Processor):
        self.settings = settings
        self.store = JobStore(settings.data_dir)
        self.processor = processor
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_threads, thread_name_prefix="dubsync-job")
        self.cleanup_stop = threading.Event()
        self.cleanup_thread: threading.Thread | None = None

    def start(self) -> None:
        self.store.delete_expired()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, name="dubsync-cleanup", daemon=True)
        self.cleanup_thread.start()
        for job in self.store.pending():
            self.submit(replace(job, status="queued", progress=5))

    def submit(self, job: JobRecord) -> None:
        if self.settings.processing_inline:
            self._process(job)
            return
        self.executor.submit(self._process, job)

    def shutdown(self) -> None:
        self.cleanup_stop.set()
        if self.cleanup_thread is not None:
            self.cleanup_thread.join(timeout=max(1.0, self.settings.cleanup_interval_seconds * 2))
        self.executor.shutdown(wait=True, cancel_futures=False)

    def _cleanup_loop(self) -> None:
        while not self.cleanup_stop.wait(self.settings.cleanup_interval_seconds):
            try:
                self.store.delete_expired()
            except Exception:
                logger.exception("DubSync retention cleanup failed")

    def _process(self, job: JobRecord) -> None:
        try:
            processing = self.store.mark_processing(job.id)
            artifacts = self.processor(processing, self.settings)
            self.store.mark_complete(job.id, artifacts)
        except Exception:
            logger.exception("DubSync job %s failed", job.id)
            self.store.mark_failed(job.id)


def default_processor(job: JobRecord, settings: WebSettings) -> ProcessedArtifacts:
    output_name = "generated.srt" if job.mode == "generate" else "synced.srt"
    output_path = job.directory / output_name
    workdir = job.directory / "work"
    language = None if job.language == "auto" else job.language
    if job.mode == "sync":
        if job.srt_path is None:
            raise ValueError("Sync job is missing its SRT input")
        result = sync_episode(
            job.srt_path,
            job.audio_path,
            output_path,
            workdir,
            style_path=settings.style_path,
            providers_path=settings.providers_path,
            fps=job.fps,
            language=language,
        )
    else:
        result = generate_srt_from_audio(
            job.audio_path,
            output_path,
            workdir,
            style_path=settings.style_path,
            providers_path=settings.providers_path,
            fps=job.fps,
            language=language,
        )
    summary = result.report.get("summary", {})
    cue_count = int(summary.get("cue_count", 0)) if isinstance(summary, dict) else 0
    changes = result.episode_workdir / "changes.diff.srt"
    return ProcessedArtifacts(
        output_srt=result.output_srt,
        qc_json=result.episode_workdir / "qc_report.json",
        qc_html=result.episode_workdir / "qc_report.html",
        changes_srt=changes if changes.exists() else None,
        cost_usd=result.cost_meter.total_usd,
        cue_count=cue_count,
    )


def new_job_record(
    *,
    job_id: str,
    token_hash: str,
    mode: JobMode,
    directory: Path,
    audio_path: Path,
    srt_path: Path | None,
    fps: float,
    language: str,
    style: str,
    retention_hours: int,
) -> JobRecord:
    now = datetime.now(UTC)
    return JobRecord(
        id=job_id,
        token_hash=token_hash,
        mode=mode,
        status="queued",
        progress=5,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(hours=retention_hours),
        directory=directory,
        audio_path=audio_path,
        srt_path=srt_path,
        fps=fps,
        language=language,
        style=style,
    )


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        token_hash=row["token_hash"],
        mode=row["mode"],
        status=row["status"],
        progress=row["progress"],
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        expires_at=_datetime(row["expires_at"]),
        directory=Path(row["directory"]),
        audio_path=Path(row["audio_path"]),
        srt_path=Path(row["srt_path"]) if row["srt_path"] else None,
        fps=row["fps"],
        language=row["language"],
        style=row["style"],
        output_srt=Path(row["output_srt"]) if row["output_srt"] else None,
        qc_json=Path(row["qc_json"]) if row["qc_json"] else None,
        qc_html=Path(row["qc_html"]) if row["qc_html"] else None,
        changes_srt=Path(row["changes_srt"]) if row["changes_srt"] else None,
        cost_usd=row["cost_usd"],
        cue_count=row["cue_count"],
        error=row["error"],
    )


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)
