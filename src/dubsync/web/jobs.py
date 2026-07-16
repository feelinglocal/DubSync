from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

from dubsync.audio import AudioNormalizationLimits, tree_size_bytes
from dubsync.pipeline import sync_episode
from dubsync.transcription import generate_srt_from_audio

from .generation_styles import ResolvedGenerationStyle
from .settings import WebSettings

logger = logging.getLogger(__name__)

JobMode = Literal["sync", "generate"]
JobStatus = Literal["queued", "processing", "complete", "failed"]
STALE_JOB_ERROR = "Processing timed out or was interrupted. Please submit the files again."
STORAGE_RESERVATION_FILENAME = ".storage-reservation"
STORAGE_RESERVATION_TEMP_FILENAME = ".storage-reservation.tmp"


class OutstandingJobLimitError(RuntimeError):
    """Raised when accepting children would exceed the active queue capacity."""


class JobStorageLimitError(RuntimeError):
    """Raised when a job cannot fit inside its bounded processing allocation."""


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
    source_name: str | None = None
    batch_id: str | None = None
    batch_position: int | None = None
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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

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
                    source_name TEXT,
                    batch_id TEXT,
                    batch_position INTEGER,
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
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for name, column_type in (
                ("source_name", "TEXT"),
                ("batch_id", "TEXT"),
                ("batch_position", "INTEGER"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {column_type}")

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def storage_usage_bytes(self) -> int:
        total = 0
        for directory in self.data_dir.glob("job-*"):
            resolved = self._managed_job_directory(directory)
            if resolved is None:
                continue
            actual = self.job_storage_bytes(resolved)
            total += max(actual, self._storage_reservation_bytes(resolved))
        return total

    def job_storage_bytes(self, directory: Path) -> int:
        resolved = self._managed_job_directory(directory)
        if resolved is None:
            raise JobStorageLimitError("Unexpected job storage directory")
        total = tree_size_bytes(resolved)
        for metadata_name in (STORAGE_RESERVATION_FILENAME, STORAGE_RESERVATION_TEMP_FILENAME):
            metadata = resolved / metadata_name
            if metadata.is_file() and not metadata.is_symlink():
                total -= metadata.stat().st_size
        return max(0, total)

    def reserve_job_storage(
        self,
        directory: Path,
        *,
        additional_bytes: int,
        max_job_storage_bytes: int,
    ) -> int:
        if additional_bytes < 0 or max_job_storage_bytes <= 0:
            raise ValueError("Storage reservation limits must be positive")
        resolved = self._managed_job_directory(directory)
        if resolved is None:
            raise JobStorageLimitError("Unexpected job storage directory")
        reservation = self.job_storage_bytes(resolved) + additional_bytes
        if reservation > max_job_storage_bytes:
            raise JobStorageLimitError("Job processing would exceed its storage limit")
        temporary = resolved / STORAGE_RESERVATION_TEMP_FILENAME
        target = resolved / STORAGE_RESERVATION_FILENAME
        temporary.write_text(str(reservation), encoding="ascii")
        temporary.replace(target)
        return reservation

    def release_job_storage(self, directory: Path) -> None:
        resolved = self._managed_job_directory(directory)
        if resolved is None:
            logger.error("Refusing to release storage for unexpected job directory: %s", directory)
            return
        (resolved / STORAGE_RESERVATION_FILENAME).unlink(missing_ok=True)
        (resolved / STORAGE_RESERVATION_TEMP_FILENAME).unlink(missing_ok=True)

    def assert_job_storage_within_limit(self, directory: Path, *, max_bytes: int) -> None:
        actual = self.job_storage_bytes(directory)
        reserved = self._storage_reservation_bytes(directory)
        effective_limit = min(max_bytes, reserved) if reserved else max_bytes
        if actual > effective_limit:
            raise JobStorageLimitError("Job output exceeded its storage reservation")

    def _storage_reservation_bytes(self, directory: Path) -> int:
        path = directory / STORAGE_RESERVATION_FILENAME
        if not path.exists():
            return 0
        if path.is_symlink() or not path.is_file():
            raise OSError("Invalid job storage reservation")
        try:
            reservation = int(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise OSError("Invalid job storage reservation") from exc
        if reservation < 0:
            raise OSError("Invalid job storage reservation")
        return reservation

    def _managed_job_directory(self, directory: Path) -> Path | None:
        if directory.is_symlink():
            return None
        try:
            resolved = directory.resolve()
        except OSError:
            return None
        if (
            resolved.parent != self.data_dir
            or not resolved.name.startswith("job-")
            or not resolved.is_dir()
        ):
            return None
        return resolved

    def reconcile_orphaned_job_directories(self) -> int:
        with self._connect() as connection:
            rows = connection.execute("SELECT directory FROM jobs").fetchall()
        referenced = {
            resolved
            for row in rows
            if (resolved := self._managed_job_directory(Path(row["directory"]))) is not None
        }
        removed = 0
        for directory in self.data_dir.glob("job-*"):
            resolved = self._managed_job_directory(directory)
            if (
                resolved is not None
                and resolved not in referenced
                and self._remove_job_directory(directory)
            ):
                removed += 1
        return removed

    def create(self, job: JobRecord) -> None:
        self.create_many((job,))

    def create_many(
        self,
        jobs: tuple[JobRecord, ...] | list[JobRecord],
        *,
        max_outstanding: int | None = None,
    ) -> None:
        if not jobs:
            raise ValueError("At least one job is required")
        if max_outstanding is not None and max_outstanding < 1:
            raise ValueError("max_outstanding must be at least 1")
        with self._connect() as connection:
            if max_outstanding is not None:
                connection.execute("BEGIN IMMEDIATE")
                active_count = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')"
                ).fetchone()[0]
                if active_count + len(jobs) > max_outstanding:
                    raise OutstandingJobLimitError
            connection.executemany(
                """
                INSERT INTO jobs (
                    id, token_hash, mode, status, progress, created_at, updated_at, expires_at,
                    directory, audio_path, srt_path, fps, language, style,
                    source_name, batch_id, batch_position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
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
                        job.source_name,
                        job.batch_id,
                        job.batch_position,
                    )
                    for job in jobs
                ],
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _record(row) if row is not None else None

    def pending(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'processing')
                ORDER BY created_at, batch_id, batch_position
                """,
            ).fetchall()
        return [_record(row) for row in rows]

    def mark_processing(self, job_id: str) -> JobRecord:
        return self._update(
            job_id,
            expected_statuses=("queued", "processing"),
            status="processing",
            progress=25,
            error=None,
        )

    def mark_complete(
        self,
        job_id: str,
        artifacts: ProcessedArtifacts,
        *,
        expires_at: datetime | None = None,
    ) -> JobRecord:
        return self._update(
            job_id,
            expected_statuses=("processing",),
            status="complete",
            progress=100,
            output_srt=str(artifacts.output_srt),
            qc_json=str(artifacts.qc_json),
            qc_html=str(artifacts.qc_html),
            changes_srt=str(artifacts.changes_srt) if artifacts.changes_srt else None,
            cost_usd=artifacts.cost_usd,
            cue_count=artifacts.cue_count,
            error=None,
            **({"expires_at": _iso(expires_at)} if expires_at is not None else {}),
        )

    def mark_failed(self, job_id: str, *, expires_at: datetime | None = None) -> JobRecord:
        return self._update(
            job_id,
            expected_statuses=("queued", "processing"),
            status="failed",
            progress=100,
            error="Processing failed. Check the input files and try again.",
            **({"expires_at": _iso(expires_at)} if expires_at is not None else {}),
        )

    def fail_stale_active(
        self,
        *,
        stale_before: datetime,
        failed_at: datetime,
        expires_at: datetime,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', progress = 100, error = ?, updated_at = ?, expires_at = ?
                WHERE status IN ('queued', 'processing') AND updated_at <= ?
                """,
                (
                    STALE_JOB_ERROR,
                    _iso(failed_at),
                    _iso(expires_at),
                    _iso(stale_before),
                ),
            )
        return cursor.rowcount

    def delete_expired(self, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('complete', 'failed') AND expires_at <= ?
                """,
                (_iso(cutoff),),
            ).fetchall()
        removed_ids = [
            row["id"]
            for row in rows
            if self._remove_job_directory(Path(row["directory"]))
        ]
        if removed_ids:
            with self._connect() as connection:
                connection.executemany("DELETE FROM jobs WHERE id = ?", [(job_id,) for job_id in removed_ids])
        return len(removed_ids)

    def delete_many(self, job_ids: tuple[str, ...] | list[str]) -> int:
        unique_ids = tuple(dict.fromkeys(job_ids))
        if not unique_ids:
            return 0
        placeholders = ", ".join("?" for _ in unique_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, directory FROM jobs WHERE id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        removed_ids = [
            row["id"]
            for row in rows
            if self._remove_job_directory(Path(row["directory"]))
        ]
        if removed_ids:
            with self._connect() as connection:
                connection.executemany("DELETE FROM jobs WHERE id = ?", [(job_id,) for job_id in removed_ids])
        return len(removed_ids)

    def _remove_job_directory(self, directory: Path) -> bool:
        if directory.is_symlink():
            logger.error("Refusing to delete symlinked job directory: %s", directory)
            return False
        try:
            resolved = directory.resolve()
        except OSError:
            logger.exception("Could not resolve job directory: %s", directory)
            return False
        if resolved.parent != self.data_dir or not resolved.name.startswith("job-"):
            logger.error("Refusing to delete unexpected job directory: %s", resolved)
            return False
        if not resolved.exists():
            return True
        if not resolved.is_dir():
            logger.error("Refusing to delete non-directory job path: %s", resolved)
            return False
        try:
            shutil.rmtree(resolved)
        except FileNotFoundError:
            return True
        except OSError:
            logger.exception("Could not delete job directory: %s", resolved)
            return False
        return True

    def _update(
        self,
        job_id: str,
        *,
        expected_statuses: tuple[JobStatus, ...] | None = None,
        **values: object,
    ) -> JobRecord:
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
            "expires_at",
        }
        if not values or not set(values).issubset(allowed):
            raise ValueError("Unsupported job update")
        values = {**values, "updated_at": _iso(datetime.now(UTC))}
        assignments = ", ".join(f"{key} = ?" for key in values)
        parameters: list[object] = [*values.values(), job_id]
        status_filter = ""
        if expected_statuses:
            placeholders = ", ".join("?" for _ in expected_statuses)
            status_filter = f" AND status IN ({placeholders})"
            parameters.extend(expected_statuses)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?{status_filter}",
                parameters,
            )
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


Processor = Callable[[JobRecord, WebSettings], ProcessedArtifacts]


class JobService:
    def __init__(self, settings: WebSettings, processor: Processor):
        if settings.worker_threads != 1:
            raise ValueError("DUBSYNC_WORKER_THREADS must be exactly 1")
        self.settings = settings
        self.store = JobStore(settings.data_dir)
        self.processor = processor
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_threads, thread_name_prefix="dubsync-job")
        self.cleanup_stop = threading.Event()
        self.cleanup_thread: threading.Thread | None = None

    def start(self) -> None:
        self._run_maintenance()
        orphan_count = self.store.reconcile_orphaned_job_directories()
        if orphan_count:
            logger.warning("Removed %d orphaned DubSync job storage directories", orphan_count)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, name="dubsync-cleanup", daemon=True)
        self.cleanup_thread.start()
        for job in self.store.pending():
            self.submit(replace(job, status="queued", progress=5))

    def submit(self, job: JobRecord) -> None:
        if self.settings.processing_inline:
            self._process(job)
            return
        self.executor.submit(self._process, job)

    def submit_batch(self, jobs: tuple[JobRecord, ...] | list[JobRecord]) -> None:
        ordered_jobs = tuple(jobs)
        if not ordered_jobs:
            raise ValueError("At least one job is required")
        if self.settings.processing_inline:
            self._process_batch(ordered_jobs)
            return
        self.executor.submit(self._process_batch, ordered_jobs)

    def shutdown(self) -> None:
        self.cleanup_stop.set()
        if self.cleanup_thread is not None:
            self.cleanup_thread.join(timeout=max(1.0, self.settings.cleanup_interval_seconds * 2))
        self.executor.shutdown(wait=True, cancel_futures=False)

    def _cleanup_loop(self) -> None:
        while not self.cleanup_stop.wait(self.settings.cleanup_interval_seconds):
            try:
                self._run_maintenance()
            except Exception:
                logger.exception("DubSync retention cleanup failed")

    def _run_maintenance(self) -> None:
        now = datetime.now(UTC)
        stale_before = now - timedelta(hours=self.settings.active_job_timeout_hours)
        stale_count = self.store.fail_stale_active(
            stale_before=stale_before,
            failed_at=now,
            expires_at=now + timedelta(hours=self.settings.retention_hours),
        )
        if stale_count:
            logger.warning("Dead-lettered %d stale DubSync job(s)", stale_count)
        self.store.delete_expired(now)

    def _process(self, job: JobRecord) -> None:
        terminal_expiry = lambda: datetime.now(UTC) + timedelta(hours=self.settings.retention_hours)
        try:
            processing = self.store.mark_processing(job.id)
            if processing.status != "processing":
                return
            artifacts = self.processor(processing, self.settings)
            self.store.assert_job_storage_within_limit(
                job.directory,
                max_bytes=self.settings.max_job_storage_bytes,
            )
            self.store.mark_complete(job.id, artifacts, expires_at=terminal_expiry())
        except Exception:
            logger.exception("DubSync job %s failed", job.id)
            _remove_generated_job_files(job)
            self.store.mark_failed(job.id, expires_at=terminal_expiry())
        finally:
            self.store.release_job_storage(job.directory)

    def _process_batch(self, jobs: tuple[JobRecord, ...]) -> None:
        for job in jobs:
            self._process(job)


def default_processor(job: JobRecord, settings: WebSettings) -> ProcessedArtifacts:
    output_name = "generated.srt" if job.mode == "generate" else "synced.srt"
    output_path = job.directory / output_name
    workdir = job.directory / "work"
    language = None if job.language == "auto" else job.language
    audio_limits = AudioNormalizationLimits(
        max_duration_seconds=settings.max_audio_duration_seconds,
        probe_timeout_seconds=settings.ffprobe_timeout_seconds,
        max_output_bytes=settings.max_normalized_audio_bytes,
        job_directory=job.directory,
        max_job_storage_bytes=settings.max_job_storage_bytes,
        min_free_storage_bytes=settings.min_free_storage_bytes,
    )
    if job.mode == "sync":
        if job.srt_path is None:
            raise ValueError("Sync job is missing its SRT input")
        result = sync_episode(
            job.srt_path,
            job.audio_path,
            output_path,
            workdir,
            style_path=None,
            providers_path=settings.providers_path,
            fps=job.fps,
            language=language,
            audio_limits=audio_limits,
        )
    else:
        generate_options: dict[str, object] = {
            "style_path": settings.style_path,
            "providers_path": settings.providers_path,
            "fps": job.fps,
            "language": language,
            "audio_limits": audio_limits,
        }
        if job.style.strip().lower() != "standard":
            resolved_style = ResolvedGenerationStyle.model_validate_json(job.style)
            uses_configured_default = resolved_style.source == "preset" and resolved_style.preset == "standard"
            if uses_configured_default and settings.style_path is None:
                generate_options = {
                    **generate_options,
                    "style_path": None,
                    "style_profile": resolved_style.profile,
                }
            elif not uses_configured_default:
                generate_options = {
                    **generate_options,
                    "style_path": None,
                    "style_profile": resolved_style.profile,
                    "generation_constraints": resolved_style.constraints,
                }
        result = generate_srt_from_audio(job.audio_path, output_path, workdir, **generate_options)
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


def _remove_generated_job_files(job: JobRecord) -> None:
    preserved = {
        job.audio_path.resolve(),
        *((job.srt_path.resolve(),) if job.srt_path is not None else ()),
        (job.directory / "style-example.srt").resolve(),
        (job.directory / STORAGE_RESERVATION_FILENAME).resolve(),
        (job.directory / STORAGE_RESERVATION_TEMP_FILENAME).resolve(),
    }
    try:
        children = tuple(job.directory.iterdir())
    except OSError:
        logger.exception("Could not inspect failed job output directory: %s", job.directory)
        return
    for child in children:
        try:
            resolved = child.resolve()
            if resolved in preserved:
                continue
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child)
        except OSError:
            logger.exception("Could not remove failed job output: %s", child)


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
    source_name: str | None = None,
    batch_id: str | None = None,
    batch_position: int | None = None,
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
        source_name=source_name,
        batch_id=batch_id,
        batch_position=batch_position,
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
        source_name=row["source_name"],
        batch_id=row["batch_id"],
        batch_position=row["batch_position"],
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
