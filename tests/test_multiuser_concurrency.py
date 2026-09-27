from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dubsync.web.app as web_app_module
from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, JobService, ProcessedArtifacts, new_job_record
from dubsync.web.process_lock import ProcessLockError
from dubsync.web.security import hash_job_token
from dubsync.web.settings import WebSettings


ACCESS_CODE = "shared-access-code-1234"


def _settings(
    tmp_path: Path,
    *,
    processing_inline: bool = False,
    worker_threads: int = 1,
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
        retention_hours=24,
        processing_inline=processing_inline,
        max_submissions_per_hour=100,
        max_outstanding_child_jobs=20,
        worker_threads=worker_threads,
    )


def _artifacts(
    job: JobRecord,
    _settings: WebSettings | None = None,
) -> ProcessedArtifacts:
    output = job.directory / "generated.srt"
    output.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\nReady.\n",
        encoding="utf-8",
    )
    qc_json = job.directory / "qc_report.json"
    qc_json.write_text(
        json.dumps({"summary": {"cue_count": 1, "flags": 0, "style_violations": 0}}),
        encoding="utf-8",
    )
    qc_html = job.directory / "qc_report.html"
    qc_html.write_text("<h1>QC</h1>", encoding="utf-8")
    return ProcessedArtifacts(
        output_srt=output,
        qc_json=qc_json,
        qc_html=qc_html,
        cost_usd=0.01,
        cue_count=1,
    )


def _job(
    settings: WebSettings,
    job_id: str,
    *,
    batch_id: str | None = None,
    batch_position: int | None = None,
) -> JobRecord:
    directory = settings.data_dir / f"job-{job_id}"
    directory.mkdir(parents=True)
    audio = directory / "audio.wav"
    audio.write_bytes(job_id.encode())
    job = new_job_record(
        job_id=job_id,
        token_hash=hash_job_token(f"token-{job_id}"),
        mode="generate",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30,
        language="auto",
        style="standard",
        retention_hours=settings.retention_hours,
        source_name=f"{job_id}.wav",
    )
    return replace(
        job,
        batch_id=batch_id,
        batch_position=batch_position,
    )


def _wait_for_completed_jobs(service: JobService, jobs: tuple[JobRecord, ...]) -> None:
    deadline = time.monotonic() + 5.0
    while True:
        records = [service.store.get(job.id) for job in jobs]
        statuses = [record.status if record is not None else "missing" for record in records]
        if statuses == ["complete"] * len(jobs):
            return
        assert "failed" not in statuses and "missing" not in statuses, statuses
        assert time.monotonic() < deadline, f"Jobs did not finish before shutdown: {statuses}"
        time.sleep(0.01)


def test_overlapping_authenticated_intakes_are_isolated_and_both_accepted(
    tmp_path,
    monkeypatch,
):
    settings = replace(
        _settings(tmp_path),
        job_access_code=ACCESS_CODE,
        require_job_access_code=True,
    )
    app = create_app(settings=settings, processor=_artifacts)
    first_save_started = threading.Event()
    release_first_save = threading.Event()
    original_save = web_app_module._save_upload
    save_lock = threading.Lock()
    save_calls = 0

    async def overlapping_save(*args, **kwargs):
        nonlocal save_calls
        with save_lock:
            save_calls += 1
            is_first_save = save_calls == 1
        if is_first_save:
            first_save_started.set()
            await asyncio.to_thread(release_first_save.wait, 3.0)
        return await original_save(*args, **kwargs)

    def submit(client: TestClient, filename: str, payload: bytes):
        return client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": ACCESS_CODE},
            data={"mode": "generate", "fps": "30"},
            files={"audio": (filename, payload, "audio/wav")},
        )

    monkeypatch.setattr("dubsync.web.app._save_upload", overlapping_save)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(submit, client, "device-a.wav", b"device-a")
        assert first_save_started.wait(timeout=2.0)
        release_timer = threading.Timer(0.2, release_first_save.set)
        release_timer.start()
        try:
            second_response = submit(client, "device-b.wav", b"device-b")
            first_response = first_future.result(timeout=3.0)
        finally:
            release_first_save.set()
            release_timer.join(timeout=1.0)

        assert [first_response.status_code, second_response.status_code] == [202, 202]
        first_created = first_response.json()
        second_created = second_response.json()
        assert first_created["id"] != second_created["id"]
        assert first_created["token"] != second_created["token"]

        first_record = app.state.jobs.store.get(first_created["id"])
        second_record = app.state.jobs.store.get(second_created["id"])
        assert first_record is not None
        assert second_record is not None
        assert first_record.directory != second_record.directory
        assert first_record.directory.is_dir()
        assert second_record.directory.is_dir()
        assert first_record.audio_path.read_bytes() == b"device-a"
        assert second_record.audio_path.read_bytes() == b"device-b"

        assert client.get(
            f"/api/jobs/{first_created['id']}",
            headers={"Authorization": f"Bearer {second_created['token']}"},
        ).status_code == 404
        assert client.get(
            f"/api/jobs/{second_created['id']}",
            headers={"Authorization": f"Bearer {first_created['token']}"},
        ).status_code == 404
        assert client.get(
            f"/api/jobs/{first_created['id']}",
            headers={"Authorization": f"Bearer {first_created['token']}"},
        ).status_code == 200
        assert client.get(
            f"/api/jobs/{second_created['id']}",
            headers={"Authorization": f"Bearer {second_created['token']}"},
        ).status_code == 200

    assert save_calls == 2


def test_two_workers_overlap_independent_jobs_without_exceeding_the_bound(tmp_path):
    settings = _settings(tmp_path, worker_threads=2)
    counter_lock = threading.Lock()
    two_active = threading.Event()
    release = threading.Event()
    active = 0
    max_active = 0

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_active.set()
        try:
            release.wait(timeout=3.0)
            return _artifacts(job)
        finally:
            with counter_lock:
                active -= 1

    service = JobService(settings, processor)
    jobs = tuple(_job(settings, f"independent-{index}") for index in range(4))
    service.store.create_many(list(jobs))
    try:
        for job in jobs:
            service.submit(job)
        assert two_active.wait(timeout=2.0)
        release.set()
        _wait_for_completed_jobs(service, jobs)
    finally:
        release.set()
        service.shutdown()

    assert max_active == 2
    assert [service.store.get(job.id).status for job in jobs] == ["complete"] * 4


def test_two_workers_keep_each_batch_serial_while_batches_overlap(tmp_path):
    settings = _settings(tmp_path, worker_threads=2)
    counter_lock = threading.Lock()
    two_batches_active = threading.Event()
    release = threading.Event()
    global_active = 0
    max_global_active = 0
    active_by_batch: dict[str, int] = {}
    max_active_by_batch: dict[str, int] = {}

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal global_active, max_global_active
        assert job.batch_id is not None
        with counter_lock:
            global_active += 1
            max_global_active = max(max_global_active, global_active)
            active_by_batch[job.batch_id] = active_by_batch.get(job.batch_id, 0) + 1
            max_active_by_batch[job.batch_id] = max(
                max_active_by_batch.get(job.batch_id, 0),
                active_by_batch[job.batch_id],
            )
            if len([count for count in active_by_batch.values() if count > 0]) == 2:
                two_batches_active.set()
        try:
            release.wait(timeout=3.0)
            return _artifacts(job)
        finally:
            with counter_lock:
                global_active -= 1
                active_by_batch[job.batch_id] -= 1

    service = JobService(settings, processor)
    first_batch = tuple(
        _job(
            settings,
            f"batch-a-{index}",
            batch_id="batch-a",
            batch_position=index,
        )
        for index in range(2)
    )
    second_batch = tuple(
        _job(
            settings,
            f"batch-b-{index}",
            batch_id="batch-b",
            batch_position=index,
        )
        for index in range(2)
    )
    jobs = first_batch + second_batch
    service.store.create_many(list(jobs))
    try:
        service.submit_batch(first_batch)
        service.submit_batch(second_batch)
        assert two_batches_active.wait(timeout=2.0)
        release.set()
        _wait_for_completed_jobs(service, jobs)
    finally:
        release.set()
        service.shutdown()

    assert max_global_active == 2
    assert max_active_by_batch == {"batch-a": 1, "batch-b": 1}
    assert [service.store.get(job.id).status for job in jobs] == ["complete"] * 4


def test_duplicate_submissions_claim_one_job_only_once(tmp_path):
    settings = _settings(tmp_path, worker_threads=2)
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal calls
        with calls_lock:
            calls += 1
            started.set()
        release.wait(timeout=3.0)
        return _artifacts(job)

    service = JobService(settings, processor)
    job = _job(settings, "duplicate-claim")
    service.store.create(job)
    try:
        service.submit(job)
        service.submit(job)
        assert started.wait(timeout=2.0)
        time.sleep(0.1)
    finally:
        release.set()
        service.shutdown()

    assert calls == 1
    assert service.store.get(job.id).status == "complete"


def test_second_service_process_cannot_requeue_work_owned_by_the_active_service(tmp_path):
    settings = _settings(tmp_path, worker_threads=2)
    started = threading.Event()
    release = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal calls
        with calls_lock:
            calls += 1
            started.set()
        release.wait(timeout=3.0)
        return _artifacts(job)

    first = JobService(settings, processor)
    second = JobService(settings, processor)
    job = _job(settings, "process-owned")
    first.store.create(job)

    try:
        first.start()
        assert started.wait(timeout=2.0)
        assert first.store.get(job.id).status == "processing"

        with pytest.raises(ProcessLockError, match="already active"):
            second.start()

        time.sleep(0.1)
        assert calls == 1
        assert first.store.get(job.id).status == "processing"
    finally:
        release.set()
        second.shutdown()
        first.shutdown()

    assert first.store.get(job.id).status == "complete"


def test_startup_failure_stops_background_work_before_releasing_the_process_lock(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path, worker_threads=2)
    first = JobService(settings, _artifacts)
    second = JobService(settings, _artifacts)

    def fail_pending():
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(first.store, "pending", fail_pending)
    try:
        with pytest.raises(RuntimeError, match="injected startup failure"):
            first.start()

        assert first.cleanup_thread is not None
        assert not first.cleanup_thread.is_alive()
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            first.submit(_job(settings, "executor-is-closed"))

        second.start()
    finally:
        second.shutdown()
        first.shutdown()


def test_shutdown_holds_the_process_lock_until_cleanup_has_fully_stopped(tmp_path, monkeypatch):
    settings = replace(
        _settings(tmp_path, worker_threads=2),
        cleanup_interval_seconds=0.01,
    )
    first = JobService(settings, _artifacts)
    second = JobService(settings, _artifacts)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    shutdown_complete = threading.Event()

    first.start()

    def blocking_maintenance():
        cleanup_started.set()
        release_cleanup.wait(timeout=3.0)

    monkeypatch.setattr(first, "_run_maintenance", blocking_maintenance)
    assert cleanup_started.wait(timeout=2.0)

    shutdown_thread = threading.Thread(
        target=lambda: (first.shutdown(), shutdown_complete.set()),
        daemon=True,
    )
    shutdown_thread.start()
    try:
        assert not shutdown_complete.wait(timeout=1.1)
        with pytest.raises(ProcessLockError, match="already active"):
            second.start()
    finally:
        release_cleanup.set()
        shutdown_thread.join(timeout=2.0)

    assert shutdown_complete.is_set()
    try:
        second.start()
    finally:
        second.shutdown()
        first.shutdown()


def test_restart_recovery_keeps_children_of_one_batch_serial_with_two_workers(tmp_path):
    settings = _settings(tmp_path, worker_threads=2)
    first_started = threading.Event()
    same_batch_overlap = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal active, max_active
        assert job.batch_id == "recover-batch"
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
            if active > 1:
                same_batch_overlap.set()
            first_started.set()
        try:
            release.wait(timeout=3.0)
            return _artifacts(job)
        finally:
            with counter_lock:
                active -= 1

    service = JobService(settings, processor)
    jobs = tuple(
        replace(
            _job(
                settings,
                f"recover-{index}",
                batch_id="recover-batch",
                batch_position=index,
            ),
            **({"status": "processing", "progress": 25} if index == 1 else {}),
        )
        for index in range(3)
    )
    service.store.create_many(list(jobs))
    try:
        service.start()
        assert first_started.wait(timeout=2.0)
        assert not same_batch_overlap.wait(timeout=0.2)
        release.set()
        _wait_for_completed_jobs(service, jobs)
    finally:
        release.set()
        service.shutdown()

    assert max_active == 1
    assert [service.store.get(job.id).status for job in jobs] == ["complete"] * 3
