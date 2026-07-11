from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, JobService, ProcessedArtifacts, default_processor, new_job_record
from dubsync.web.security import hash_job_token
from dubsync.web.settings import WebSettings


def _settings(tmp_path: Path, *, max_upload_bytes: int = 1024 * 1024) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data",
        providers_path=providers,
        style_path=None,
        max_upload_bytes=max_upload_bytes,
        retention_hours=24,
        processing_inline=True,
        max_jobs_per_hour=20,
    )


def _fake_processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
    output = job.directory / ("generated.srt" if job.mode == "generate" else "synced.srt")
    output.write_text("1\n00:00:00,000 --> 00:00:00,500\nReady.\n", encoding="utf-8")
    qc_json = job.directory / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {"cue_count": 1, "flags": 0, "style_violations": 0}}), encoding="utf-8")
    qc_html = job.directory / "qc_report.html"
    qc_html.write_text("<h1>QC</h1>", encoding="utf-8")
    return ProcessedArtifacts(output_srt=output, qc_json=qc_json, qc_html=qc_html, cost_usd=0.01, cue_count=1)


def test_create_generate_job_processes_and_protects_status_and_download(tmp_path):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "language": "auto", "style": "standard"},
            files={"audio": ("dialogue.wav", b"RIFF-test-audio", "audio/wav")},
        )

        assert response.status_code == 202
        created = response.json()
        assert created["mode"] == "generate"
        assert created["status"] == "complete"
        assert created["progress"] == 100
        assert created["result"]["cue_count"] == 1
        assert created["result"]["cost_usd"] == 0.01
        assert created["token"]

        assert client.get(f"/api/jobs/{created['id']}").status_code == 404
        status = client.get(
            f"/api/jobs/{created['id']}",
            headers={"Authorization": f"Bearer {created['token']}"},
        )
        assert status.status_code == 200
        assert status.json()["status"] == "complete"

        download = client.get(
            f"/api/jobs/{created['id']}/downloads/srt",
            headers={"Authorization": f"Bearer {created['token']}"},
        )
        assert download.status_code == 200
        assert "Ready." in download.text
        assert "attachment" in download.headers["content-disposition"]


def test_sync_mode_requires_srt_and_rejects_unsupported_or_oversized_audio(tmp_path):
    app = create_app(settings=_settings(tmp_path, max_upload_bytes=16), processor=_fake_processor)
    with TestClient(app) as client:
        missing_srt = client.post(
            "/api/jobs",
            data={"mode": "sync", "fps": "30"},
            files={"audio": ("dialogue.wav", b"short", "audio/wav")},
        )
        assert missing_srt.status_code == 422
        assert missing_srt.json()["detail"] == "An original SRT is required for sync mode."

        unsupported = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.exe", b"short", "application/octet-stream")},
        )
        assert unsupported.status_code == 415

        oversized = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"x" * 17, "audio/wav")},
        )
        assert oversized.status_code == 413


def test_health_and_security_headers_are_present(tmp_path):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_job_service_deletes_expired_job_without_waiting_for_another_request(tmp_path):
    settings = replace(_settings(tmp_path), cleanup_interval_seconds=0.05)
    service = JobService(settings, _fake_processor)
    service.start()
    try:
        directory = settings.data_dir / "job-expired"
        directory.mkdir()
        audio = directory / "audio.wav"
        audio.write_bytes(b"audio")
        job = new_job_record(
            job_id="expired",
            token_hash=hash_job_token("token"),
            mode="generate",
            directory=directory,
            audio_path=audio,
            srt_path=None,
            fps=30,
            language="auto",
            style="standard",
            retention_hours=24,
        )
        service.store.create(replace(job, expires_at=datetime.now(UTC) - timedelta(seconds=1)))

        deadline = time.monotonic() + 1
        while service.store.get("expired") is not None and time.monotonic() < deadline:
            time.sleep(0.02)

        assert service.store.get("expired") is None
        assert not directory.exists()
    finally:
        service.shutdown()


def test_default_processor_forwards_selected_language_to_generate_pipeline(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    directory = settings.data_dir / "job-language"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"fixture audio")
    job = new_job_record(
        job_id="language",
        token_hash=hash_job_token("token"),
        mode="generate",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30,
        language="de",
        style="standard",
        retention_hours=24,
    )
    calls = {}

    def fake_generate(_audio, output, _workdir, **kwargs):
        calls.update(kwargs)
        output.write_text("1\n00:00:00,000 --> 00:00:00,500\nBereit.\n", encoding="utf-8")
        artifacts = directory / "artifacts"
        artifacts.mkdir()
        (artifacts / "qc_report.json").write_text("{}", encoding="utf-8")
        (artifacts / "qc_report.html").write_text("<h1>QC</h1>", encoding="utf-8")
        return SimpleNamespace(
            output_srt=output,
            episode_workdir=artifacts,
            report={"summary": {"cue_count": 1}},
            cost_meter=SimpleNamespace(total_usd=0.01),
        )

    monkeypatch.setattr("dubsync.web.jobs.generate_srt_from_audio", fake_generate)

    default_processor(job, settings)

    assert calls["language"] == "de"


def test_web_settings_loads_dotenv_from_current_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DUBSYNC_RETENTION_HOURS=12\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUBSYNC_RETENTION_HOURS", raising=False)

    settings = WebSettings.from_env()

    assert settings.retention_hours == 12
