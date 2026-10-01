from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import JobStore, ProcessedArtifacts, default_processor, new_job_record
from dubsync.web.settings import WebSettings


def settings(tmp_path):
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n")
    return WebSettings(data_dir=tmp_path / "data", providers_path=providers, style_path=None,
                       processing_inline=True, max_submissions_per_hour=30)


def processor(job, _settings):
    output = job.directory / "synced.srt"
    output.write_text("1\n00:00:00,000 --> 00:00:01,000\nHallo\n")
    report = job.directory / "qc.json"
    report.write_text('{"summary":{"cue_count":1}}')
    html = job.directory / "qc.html"
    html.write_text("ok")
    return ProcessedArtifacts(output, report, html, 0, 1)


def submit(client, *, endpoint="/api/jobs", mode="sync", option=None):
    data = {"mode": mode, "fps": "30", "language": "de", "transcription_provider": "microsoft/mai-transcribe-2"}
    if option is not None:
        data["asr_cross_check"] = option
    files = [("audio", ("001.wav", b"audio", "audio/wav"))]
    if mode == "sync":
        files.append(("subtitle", ("001.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHallo\n", "text/plain")))
    return client.post(endpoint, data=data, files=files)


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/batches"])
@pytest.mark.parametrize("option,expected", [(None, False), ("false", False), ("true", True)])
def test_sync_cross_check_is_explicit_and_persisted(tmp_path, monkeypatch, endpoint, option, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    config = settings(tmp_path)
    with TestClient(create_app(settings=config, processor=processor)) as client:
        response = submit(client, endpoint=endpoint, option=option)
        assert response.status_code == 202, response.text
        payload = response.json()
        job = payload if endpoint == "/api/jobs" else payload["jobs"][0]
        assert job["asr_cross_check"] is expected
        assert JobStore(config.data_dir).get(job["id"]).asr_cross_check is expected


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/batches"])
def test_generation_rejects_cross_check_instead_of_spending_for_an_unsupported_mode(tmp_path, endpoint):
    with TestClient(create_app(settings=settings(tmp_path), processor=processor)) as client:
        response = submit(client, endpoint=endpoint, mode="generate", option="true")
        assert response.status_code == 422
        assert "sync" in response.json()["detail"].lower()


def test_cross_check_rejects_nonboolean_values(tmp_path):
    with TestClient(create_app(settings=settings(tmp_path), processor=processor)) as client:
        response = submit(client, option="yes")
        assert response.status_code == 422


def test_cross_check_requires_both_providers_before_accepting_a_job(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    with TestClient(create_app(settings=settings(tmp_path), processor=processor)) as client:
        assert client.get("/api/config").json()["asr_cross_check_available"] is False
        response = submit(client, option="true")
        assert response.status_code == 422
        assert "both" in response.json()["detail"].lower()


def test_public_config_reports_default_off_and_extra_catalog_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    with TestClient(create_app(settings=settings(tmp_path), processor=processor)) as client:
        payload = client.get("/api/config").json()
        assert payload["asr_cross_check_available"] is True
        assert payload["asr_cross_check_default"] is False
        assert payload["asr_cross_check_hourly_usd"] == {"scribe_v2": 0.22, "microsoft/mai-transcribe-2": 0.10}


def test_existing_job_database_migrates_to_cross_check_off(tmp_path):
    config = settings(tmp_path)
    store = JobStore(config.data_dir)
    job = new_job_record(job_id="old", token_hash="token", mode="sync", directory=tmp_path,
                         audio_path=tmp_path / "audio.wav", srt_path=tmp_path / "source.srt", fps=30,
                         language="de", style="source", retention_hours=24)
    store.create(job)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("ALTER TABLE jobs DROP COLUMN asr_cross_check")
    assert JobStore(config.data_dir).get("old").asr_cross_check is False


@pytest.mark.parametrize("enabled", [False, True])
def test_processor_passes_explicit_cross_check_preference_to_sync(tmp_path, monkeypatch, enabled):
    import dubsync.web.jobs as jobs
    calls = []
    def fake_sync(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(report={"summary": {"cue_count": 1}}, output_srt=tmp_path / "out.srt",
                               qc_json=tmp_path / "qc.json", qc_html=tmp_path / "qc.html",
                               episode_workdir=tmp_path, cost_meter=SimpleNamespace(total_usd=0))
    monkeypatch.setattr(jobs, "sync_episode", fake_sync)
    job = new_job_record(job_id="job", token_hash="token", mode="sync", directory=tmp_path,
                         audio_path=tmp_path / "audio.wav", srt_path=tmp_path / "source.srt", fps=30,
                         language="de", style="source", retention_hours=24, asr_cross_check=enabled)
    default_processor(job, settings(tmp_path))
    assert calls[0]["asr_cross_check"] is enabled
