from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import dubsync.web.app as web_app_module
from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, JobService, ProcessedArtifacts, new_job_record
from dubsync.web.security import SlidingWindowRateLimiter
from dubsync.web.settings import WebSettings


def _settings(tmp_path: Path) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data",
        providers_path=providers,
        style_path=None,
        processing_inline=False,
        max_upload_bytes=100_000,
        max_srt_bytes=1_000,
        max_job_work_bytes=600_000,
        max_job_storage_bytes=2_000_000,
        max_retained_storage_bytes=20_000_000,
        min_free_storage_bytes=100,
    )


def _artifacts(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
    output = job.directory / "generated.srt"
    output.write_text("1\n00:00:00,000 --> 00:00:00,500\nReady.\n", encoding="utf-8")
    qc_json = job.directory / "qc.json"
    qc_json.write_text("{}", encoding="utf-8")
    qc_html = job.directory / "qc.html"
    qc_html.write_text("Ready", encoding="utf-8")
    return ProcessedArtifacts(output, qc_json, qc_html, 0.0, 1)


@pytest.mark.parametrize("reject_before_form", [False, True])
def test_disk_admission_includes_storage_promised_to_other_jobs(
    tmp_path, monkeypatch, reject_before_form: bool,
):
    settings = _settings(tmp_path)
    app = create_app(settings=settings, processor=_artifacts)
    monkeypatch.setattr(app.state.jobs, "submit", lambda _job: None)
    monkeypatch.setattr("shutil.disk_usage", lambda _path: SimpleNamespace(free=1_000_000))
    request = {
        "data": {"mode": "generate", "fps": "30"},
        "files": {"audio": ("dialogue.wav", b"audio", "audio/wav")},
    }

    with TestClient(app) as client:
        accepted = client.post("/api/jobs", **request)
        assert accepted.status_code == 202
        first_id = accepted.json()["id"]
        if reject_before_form:
            monkeypatch.setattr(
                "shutil.disk_usage", lambda _path: SimpleNamespace(free=600_250)
            )

            async def unexpected_form(*_args, **_kwargs):
                raise AssertionError("promised processing storage must be checked before upload parsing")

            monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)

        rejected = client.post("/api/jobs", **request)
        assert rejected.status_code == 507
        assert [job.id for job in app.state.jobs.store.pending()] == [first_id]
        assert len(list(settings.data_dir.glob("job-*"))) == 1
        first = app.state.jobs.store.get(first_id)
        assert first is not None and first.audio_path.read_bytes() == b"audio"


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_preserves_unstarted_jobs_for_restart_without_processing_them(tmp_path, batch: bool):
    settings = _settings(tmp_path)
    started = threading.Event()
    release = threading.Event()
    processed: list[str] = []

    def processor(job: JobRecord, config: WebSettings) -> ProcessedArtifacts:
        processed.append(job.id)
        if job.id == "first":
            started.set()
            assert release.wait(timeout=5)
        return _artifacts(job, config)

    service = JobService(settings, processor)
    service.start()
    jobs: list[JobRecord] = []
    for position, job_id in enumerate(("first", "second", "third")):
        directory = settings.data_dir / f"job-{job_id}"
        directory.mkdir()
        audio = directory / "audio.wav"
        audio.write_bytes(b"audio")
        job = new_job_record(
            job_id=job_id,
            token_hash="hash",
            mode="generate",
            directory=directory,
            audio_path=audio,
            srt_path=None,
            fps=30,
            language="auto",
            style="standard",
            retention_hours=24,
            batch_id="batch" if batch else None,
            batch_position=position if batch else None,
        )
        jobs.append(job)
    service.store.create_many(jobs)
    try:
        if batch:
            service.submit_batch(jobs)
        else:
            for job in jobs:
                service.submit(job)
        assert started.wait(timeout=5)
        with ThreadPoolExecutor(max_workers=1) as executor:
            shutdown = executor.submit(service.shutdown)
            try:
                assert service.cleanup_stop.wait(timeout=5)
            finally:
                release.set()
            shutdown.result(timeout=5)
        assert processed == ["first"]
        assert [job.id for job in service.store.pending()] == ["second", "third"]
        assert all(job.audio_path.read_bytes() == b"audio" for job in jobs)

        recovered = JobService(replace(settings, processing_inline=True), _artifacts)
        try:
            recovered.start()
            assert recovered.store.pending() == []
            assert all(recovered.store.get(job.id).status == "complete" for job in jobs)
        finally:
            recovered.shutdown()
    finally:
        release.set()
        service.shutdown()


def test_lifespan_failure_always_stops_service_and_releases_process_lock(tmp_path):
    app = create_app(settings=_settings(tmp_path), processor=_artifacts)

    async def failed_lifespan() -> None:
        async with app.router.lifespan_context(app):
            raise RuntimeError("application cancelled")

    try:
        with pytest.raises(RuntimeError, match="application cancelled"):
            asyncio.run(failed_lifespan())
        assert app.state.jobs.cleanup_stop.is_set()
        assert app.state.jobs.process_lock._handle is None
        assert not app.state.jobs.cleanup_thread.is_alive()
    finally:
        app.state.jobs.shutdown()


def test_rate_limit_reclaims_expired_clients_while_preserving_current_allowances(monkeypatch):
    now = 0.0
    monkeypatch.setattr("dubsync.web.security.time.monotonic", lambda: now)
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    for index in range(200):
        assert limiter.allow(f"expired-{index}")
    now = 50.0
    assert limiter.allow("active")
    now = 61.0
    assert limiter.allow("new")
    assert set(limiter._events) == {"active", "new"}
    assert limiter.allow("active")
    assert not limiter.allow("active")
    now = 111.0
    assert limiter.allow("active")
    assert not limiter.allow("active")


def test_rate_limit_keeps_a_refreshed_client_when_older_clients_expire(monkeypatch):
    now = 0.0
    monkeypatch.setattr("dubsync.web.security.time.monotonic", lambda: now)
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("refreshed")
    now = 1.0
    assert limiter.allow("expired")
    now = 50.0
    assert limiter.allow("refreshed")
    now = 62.0
    assert limiter.allow("new")
    assert set(limiter._events) == {"refreshed", "new"}
    assert limiter.allow("refreshed")
    assert not limiter.allow("refreshed")


def test_batch_archive_compression_does_not_block_health_requests(tmp_path, monkeypatch):
    app = create_app(settings=replace(_settings(tmp_path), processing_inline=True), processor=_artifacts)
    writing = threading.Event()
    release = threading.Event()
    original_write = web_app_module._write_batch_srt_archive

    def slow_write(*args, **kwargs):
        writing.set()
        assert release.wait(timeout=5)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(web_app_module, "_write_batch_srt_archive", slow_write)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        batch_response = client.post(
            "/api/batches",
            data={"mode": "generate", "fps": "30"},
            files=[("audio", ("dialogue.wav", b"audio", "audio/wav"))],
        )
        assert batch_response.status_code == 202
        batch = batch_response.json()
        credentials = {"jobs": [{"id": job["id"], "token": job["token"]} for job in batch["jobs"]]}
        download = executor.submit(
            client.post, f"/api/batches/{batch['id']}/downloads/srt", json=credentials
        )
        try:
            assert writing.wait(timeout=5)
            health = executor.submit(client.get, "/api/health")
            assert health.result(timeout=1).status_code == 200
        finally:
            release.set()
        assert download.result(timeout=5).status_code == 200
