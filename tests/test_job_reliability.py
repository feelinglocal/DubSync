from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import CancelledError as FutureCancelledError, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import (
    JobRecord,
    JobService,
    OutstandingJobLimitError,
    ProcessedArtifacts,
    new_job_record,
)
from dubsync.web.settings import WebSettings


def _settings(
    tmp_path: Path,
    *,
    active_job_timeout_hours: float = 1.0,
    cleanup_interval_seconds: float = 60.0,
    processing_inline: bool = True,
) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data",
        providers_path=providers,
        style_path=None,
        max_upload_bytes=1024 * 1024,
        max_batch_upload_bytes=2 * 1024 * 1024,
        max_srt_bytes=1024 * 1024,
        retention_hours=2,
        processing_inline=processing_inline,
        max_submissions_per_hour=100,
        max_outstanding_child_jobs=10,
        worker_threads=1,
        cleanup_interval_seconds=cleanup_interval_seconds,
        active_job_timeout_hours=active_job_timeout_hours,
    )


def _job(settings: WebSettings, *, job_id: str, status: str, updated_at: datetime) -> JobRecord:
    directory = settings.data_dir / f"job-{job_id}"
    directory.mkdir(parents=True)
    audio = directory / "audio.wav"
    audio.write_bytes(b"audio")
    job = new_job_record(
        job_id=job_id,
        token_hash="token-hash",
        mode="generate",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30.0,
        language="auto",
        style="standard",
        retention_hours=settings.retention_hours,
    )
    return replace(
        job,
        status=status,
        created_at=updated_at,
        updated_at=updated_at,
        expires_at=updated_at + timedelta(minutes=1),
    )


def _unexpected_processor(_job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
    raise AssertionError("cancelled intake must not start processing")


def _raise_cancelled(*_args, **_kwargs) -> None:
    raise asyncio.CancelledError


def test_single_intake_cancellation_rolls_back_its_persisted_row_and_directory(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    app = create_app(settings=settings, processor=_unexpected_processor)
    monkeypatch.setattr(app.state.jobs, "submit", _raise_cancelled)

    with TestClient(app) as client:
        with pytest.raises((asyncio.CancelledError, FutureCancelledError)):
            client.post(
                "/api/jobs",
                data={"mode": "generate", "fps": "30"},
                files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
            )

    assert app.state.jobs.store.pending() == []
    assert list(settings.data_dir.glob("job-*")) == []


def test_batch_intake_cancellation_rolls_back_all_persisted_rows_and_directories(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    app = create_app(settings=settings, processor=_unexpected_processor)
    monkeypatch.setattr(app.state.jobs, "submit_batch", _raise_cancelled)

    with TestClient(app) as client:
        with pytest.raises((asyncio.CancelledError, FutureCancelledError)):
            client.post(
                "/api/batches",
                data={"mode": "generate", "fps": "30"},
                files=[
                    ("audio", ("one.wav", b"one", "audio/wav")),
                    ("audio", ("two.wav", b"two", "audio/wav")),
                ],
            )

    assert app.state.jobs.store.pending() == []
    assert list(settings.data_dir.glob("job-*")) == []


def test_service_start_reconciles_only_safely_managed_orphan_job_directories(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    service = JobService(settings, _unexpected_processor)

    orphan = settings.data_dir / "job-orphan"
    orphan.mkdir()
    (orphan / "partial-upload.wav").write_bytes(b"orphan")

    retained = _job(
        settings,
        job_id="retained",
        status="complete",
        updated_at=datetime.now(UTC),
    )
    service.store.create(retained)

    unsafe_file = settings.data_dir / "job-unsafe-file"
    unsafe_file.write_bytes(b"not a directory")
    outside = tmp_path / "outside-job-storage"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    symlink = settings.data_dir / "job-external-link"
    try:
        symlink.symlink_to(outside, target_is_directory=True)
    except OSError:
        symlink.mkdir()
        original_is_symlink = Path.is_symlink

        def simulated_symlink(path: Path) -> bool:
            return path == symlink or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", simulated_symlink)

    try:
        service.start()
    finally:
        service.shutdown()

    assert not orphan.exists()
    assert retained.directory.exists()
    assert service.store.get(retained.id) is not None
    assert unsafe_file.exists()
    assert symlink.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_service_start_dead_letters_stale_active_jobs_before_recovery(tmp_path, status: str):
    settings = _settings(tmp_path, active_job_timeout_hours=1.0)
    processed: list[str] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        processed.append(job.id)
        raise AssertionError("stale jobs must not be recovered")

    service = JobService(settings, processor)
    stale = _job(
        settings,
        job_id=f"stale-{status}",
        status=status,
        updated_at=datetime.now(UTC) - timedelta(hours=2),
    )
    service.store.create(stale)
    before_start = datetime.now(UTC)

    try:
        service.start()
        terminal = service.store.get(stale.id)
    finally:
        service.shutdown()

    assert processed == []
    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.progress == 100
    assert terminal.error is not None and "timed out" in terminal.error.lower()
    assert terminal.updated_at >= before_start
    assert terminal.expires_at >= before_start + timedelta(hours=settings.retention_hours)


def test_periodic_cleanup_dead_letters_active_jobs_that_become_stale(tmp_path):
    settings = _settings(
        tmp_path,
        active_job_timeout_hours=0.0001,
        cleanup_interval_seconds=0.01,
    )
    service = JobService(settings, lambda _job, _settings: pytest.fail("job was unexpectedly processed"))
    service.start()
    stale = _job(
        settings,
        job_id="periodic-stale",
        status="queued",
        updated_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    service.store.create(stale)

    deadline = time.monotonic() + 2.0
    terminal: JobRecord | None = None
    try:
        while time.monotonic() < deadline:
            terminal = service.store.get(stale.id)
            if terminal is not None and terminal.status == "failed":
                break
            time.sleep(0.01)
    finally:
        service.shutdown()

    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.error is not None and "timed out" in terminal.error.lower()
    assert terminal.expires_at > datetime.now(UTC)


def test_in_flight_worker_cannot_overwrite_a_stale_failure(tmp_path):
    settings = _settings(tmp_path, processing_inline=False)
    started = threading.Event()
    release = threading.Event()

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        started.set()
        assert release.wait(timeout=2.0)
        output = job.directory / "generated.srt"
        qc_json = job.directory / "qc.json"
        qc_html = job.directory / "qc.html"
        output.write_text("ready", encoding="utf-8")
        qc_json.write_text("{}", encoding="utf-8")
        qc_html.write_text("ready", encoding="utf-8")
        return ProcessedArtifacts(output, qc_json, qc_html, 0.0, 1)

    service = JobService(settings, processor)
    job = _job(settings, job_id="racing-worker", status="queued", updated_at=datetime.now(UTC))
    service.store.create(job)
    service.submit(job)
    assert started.wait(timeout=2.0)
    failed_at = datetime.now(UTC)

    dead_lettered = service.store.fail_stale_active(
        stale_before=failed_at + timedelta(seconds=1),
        failed_at=failed_at,
        expires_at=failed_at + timedelta(hours=settings.retention_hours),
    )
    release.set()
    service.shutdown()
    terminal = service.store.get(job.id)

    assert dead_lettered == 1
    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.error is not None and "timed out" in terminal.error.lower()


def test_job_service_fails_closed_and_removes_outputs_over_the_job_storage_limit(tmp_path):
    settings = replace(_settings(tmp_path), max_job_storage_bytes=32)

    def oversized_processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        output = job.directory / "generated.srt"
        qc_json = job.directory / "qc.json"
        qc_html = job.directory / "qc.html"
        output.write_bytes(b"x" * 64)
        qc_json.write_text("{}", encoding="utf-8")
        qc_html.write_text("ok", encoding="utf-8")
        return ProcessedArtifacts(output, qc_json, qc_html, 0.0, 1)

    service = JobService(settings, oversized_processor)
    job = _job(settings, job_id="oversized-output", status="queued", updated_at=datetime.now(UTC))
    service.store.create(job)

    service.submit(job)
    terminal = service.store.get(job.id)

    assert terminal is not None
    assert terminal.status == "failed"
    assert job.audio_path.exists()
    assert not (job.directory / "generated.srt").exists()
    assert not (job.directory / "qc.json").exists()


def test_web_settings_prefers_submission_limit_and_supports_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DUBSYNC_MAX_SUBMISSIONS_PER_HOUR", "7")
    monkeypatch.setenv("DUBSYNC_MAX_JOBS_PER_HOUR", "99")
    monkeypatch.setenv("DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS", "8")
    monkeypatch.setenv("DUBSYNC_MAX_RETAINED_STORAGE_BYTES", "123456")
    monkeypatch.setenv("DUBSYNC_MAX_AUDIO_DURATION_SECONDS", "5400")
    monkeypatch.setenv("DUBSYNC_MAX_NORMALIZED_AUDIO_BYTES", "234567")
    monkeypatch.setenv("DUBSYNC_MAX_JOB_WORK_BYTES", "345678")
    monkeypatch.setenv("DUBSYNC_MAX_JOB_STORAGE_BYTES", "456789")
    monkeypatch.setenv("DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS", "3.5")

    settings = WebSettings.from_env()

    assert settings.max_submissions_per_hour == 7
    assert settings.max_jobs_per_hour == 7
    assert settings.max_outstanding_child_jobs == 8
    assert settings.max_retained_storage_bytes == 123456
    assert settings.max_audio_duration_seconds == 5400
    assert settings.max_normalized_audio_bytes == 234567
    assert settings.max_job_work_bytes == 345678
    assert settings.max_job_storage_bytes == 456789
    assert settings.active_job_timeout_hours == 3.5

    monkeypatch.delenv("DUBSYNC_MAX_SUBMISSIONS_PER_HOUR")
    legacy_settings = WebSettings.from_env()

    assert legacy_settings.max_submissions_per_hour == 99
    assert legacy_settings.max_jobs_per_hour == 99


def test_web_settings_defaults_to_ten_outstanding_child_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS", raising=False)

    settings = WebSettings.from_env()

    assert settings.max_outstanding_child_jobs == 10


def test_outstanding_child_limit_is_atomic_across_concurrent_writers(tmp_path):
    settings = _settings(tmp_path)
    service = JobService(settings, lambda _job, _settings: pytest.fail("processor must not run"))
    jobs = [
        _job(settings, job_id=f"concurrent-{index}", status="queued", updated_at=datetime.now(UTC))
        for index in range(2)
    ]
    start = threading.Barrier(2)

    def create(job: JobRecord) -> str:
        start.wait(timeout=2.0)
        try:
            service.store.create_many([job], max_outstanding=1)
        except OutstandingJobLimitError:
            return "rejected"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, jobs))

    assert sorted(results) == ["accepted", "rejected"]
    assert len(service.store.pending()) == 1


def test_storage_usage_counts_only_files_inside_managed_job_directories(tmp_path):
    settings = _settings(tmp_path)
    service = JobService(settings, lambda _job, _settings: pytest.fail("processor must not run"))
    managed = settings.data_dir / "job-retained" / "work"
    managed.mkdir(parents=True)
    (managed / "audio.wav").write_bytes(b"audio")
    (managed.parent / "subtitle.srt").write_bytes(b"sub")
    unrelated = settings.data_dir / "not-a-job"
    unrelated.mkdir()
    (unrelated / "ignored.bin").write_bytes(b"ignored")

    (managed.parent / ".storage-reservation").write_text("100", encoding="ascii")

    assert service.store.storage_usage_bytes() == 100


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0"])
def test_web_settings_rejects_non_finite_or_non_positive_runtime_intervals(
    tmp_path,
    monkeypatch,
    value: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS", value)

    with pytest.raises(ValueError, match="DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS"):
        WebSettings.from_env()
