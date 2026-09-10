from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import JobStore, ProcessedArtifacts, default_processor, new_job_record
from dubsync.web.settings import WebSettings


MAI_MODEL = "microsoft/mai-transcribe-2"
SCRIBE_MODEL = "scribe_v2"
SRT_BYTES = b"1\n00:00:00,000 --> 00:00:00,500\nReady.\n"


def _settings(tmp_path: Path) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data", providers_path=providers, style_path=None,
        processing_inline=True, max_submissions_per_hour=100,
    )


def _record(tmp_path: Path, **kwargs):
    return new_job_record(
        job_id="model-choice", token_hash="test-token", mode="generate", directory=tmp_path,
        audio_path=tmp_path / "audio.wav", srt_path=None, fps=30, language="auto",
        style="standard", retention_hours=24, **kwargs,
    )


@pytest.mark.parametrize("configured", [True, False])
def test_public_config_reports_model_availability_without_credentials(tmp_path, monkeypatch, configured):
    for env in ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(env, "private-test-credential" if configured else "  ")
    app = create_app(settings=_settings(tmp_path), processor=lambda *_args: None)
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_transcription_provider"] == SCRIBE_MODEL
    assert payload["transcription_models"] == [
        {"id": SCRIBE_MODEL, "label": "Scribe v2", "available": configured},
        {"id": MAI_MODEL, "label": "MAI-Transcribe 2", "available": configured},
    ]
    assert "private-test-credential" not in response.text
    assert "OPENROUTER_API_KEY" not in response.text
    assert "ELEVENLABS_API_KEY" not in response.text


@pytest.mark.parametrize(
    "provider,key,mai_available,scribe_available",
    [
        ("openrouter", "private-config-key", True, False),
        (MAI_MODEL, "private-config-key", True, False),
        ("elevenlabs", "private-config-key", False, True),
        ("openai", "private-config-key", False, False),
        ("fixture", "private-config-key", False, False),
        ("openrouter", "  ", False, False),
        ("elevenlabs", "", False, False),
    ],
)
def test_public_config_only_counts_configured_key_for_matching_model(
    tmp_path, monkeypatch, provider, key, mai_available, scribe_available,
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    settings = _settings(tmp_path)
    settings.providers_path.write_text(
        json.dumps({"asr": {"provider": provider, "api_key": key}}), encoding="utf-8",
    )
    app = create_app(settings=settings, processor=lambda *_args: None)
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.status_code == 200
    assert {model["id"]: model["available"] for model in response.json()["transcription_models"]} == {
        MAI_MODEL: mai_available, SCRIBE_MODEL: scribe_available,
    }
    assert "private-config-key" not in response.text
    assert "api_key" not in response.text


def test_public_config_keeps_fixture_models_available_without_cloud_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    settings = _settings(tmp_path)
    fixture = tmp_path / "words.json"
    fixture.write_text('{"words": []}', encoding="utf-8")
    settings.providers_path.write_text(
        f"asr:\n  provider: fixture\n  fixture_path: {fixture.as_posix()}\n", encoding="utf-8",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        payload = client.get("/api/config").json()
    assert all(model["available"] for model in payload["transcription_models"])


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/batches"])
@pytest.mark.parametrize("mode", ["sync", "generate"])
@pytest.mark.parametrize("selection", [None, "default", MAI_MODEL, SCRIBE_MODEL])
def test_model_choice_reaches_every_job_and_survives_status_reads(tmp_path, endpoint, mode, selection):
    captured = []

    def processor(job, _settings):
        captured.append(job)
        output = job.directory / "synced.srt"
        output.write_bytes(SRT_BYTES)
        qc = job.directory / "qc_report.json"
        qc.write_text(json.dumps({"summary": {"cue_count": 1}}), encoding="utf-8")
        html = job.directory / "qc_report.html"
        html.write_text("<p>Ready</p>", encoding="utf-8")
        return ProcessedArtifacts(output_srt=output, qc_json=qc, qc_html=html, cost_usd=0, cue_count=1)

    settings = _settings(tmp_path)
    app = create_app(settings=settings, processor=processor)
    data = {"mode": mode, "fps": "30"}
    if selection is not None:
        data["transcription_provider"] = selection
    stems = ["one", "two"] if endpoint.endswith("batches") else ["one"]
    files = [("audio", (f"{stem}.wav", b"audio", "audio/wav")) for stem in stems]
    if mode == "sync":
        files.extend(("subtitle", (f"{stem}.srt", SRT_BYTES, "application/x-subrip")) for stem in stems)
    expected = MAI_MODEL if selection == MAI_MODEL else SCRIBE_MODEL
    with TestClient(app) as client:
        response = client.post(endpoint, data=data, files=files)
        assert response.status_code == 202, response.text
        payload = response.json()
        children = payload["jobs"] if endpoint.endswith("batches") else [payload]
        assert len(children) == len(stems)
        for child in children:
            status = client.get(
                f"/api/jobs/{child['id']}", headers={"Authorization": f"Bearer {child['token']}"},
            )
            assert child["transcription_provider"] == expected
            assert status.json()["transcription_provider"] == expected
            assert app.state.jobs.store.get(child["id"]).transcription_provider == expected
    assert [job.transcription_provider for job in captured] == [expected] * len(stems)


@pytest.mark.parametrize("mode", ["sync", "generate"])
@pytest.mark.parametrize("selection,expected", [(MAI_MODEL, MAI_MODEL), (SCRIBE_MODEL, SCRIBE_MODEL), ("default", SCRIBE_MODEL)])
def test_worker_forwards_model_and_preserves_legacy_default_semantics(tmp_path, monkeypatch, mode, selection, expected):
    job = replace(
        _record(tmp_path, transcription_provider=selection), mode=mode,
        srt_path=tmp_path / "source.srt", style="source" if mode == "sync" else "standard",
    )
    calls = []

    def capture(*_args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("pipeline reached")

    monkeypatch.setattr("dubsync.web.jobs.sync_episode", capture)
    monkeypatch.setattr("dubsync.web.jobs.generate_srt_from_audio", capture)
    with pytest.raises(RuntimeError, match="pipeline reached"):
        default_processor(job, _settings(tmp_path))
    assert calls[0]["transcription_provider"] == expected


def test_worker_rejects_unknown_persisted_model_before_processing(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid model reached processing")

    monkeypatch.setattr("dubsync.web.jobs.generate_srt_from_audio", unexpected)
    with pytest.raises(ValueError, match="Invalid transcription provider"):
        default_processor(_record(tmp_path, transcription_provider="untrusted-model"), _settings(tmp_path))


def test_new_job_records_default_to_scribe(tmp_path):
    assert _record(tmp_path).transcription_provider == SCRIBE_MODEL


def test_reopening_job_store_preserves_explicit_mai_choice(tmp_path):
    store = JobStore(tmp_path / "data")
    store.create(_record(tmp_path, transcription_provider=MAI_MODEL))
    recovered = JobStore(tmp_path / "data").get("model-choice")
    assert recovered.transcription_provider == MAI_MODEL


def _legacy_database(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    database = settings.data_dir / "jobs.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript("""
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('sync', 'generate')),
                status TEXT NOT NULL CHECK(status IN ('queued', 'processing', 'complete', 'failed')),
                progress INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, directory TEXT NOT NULL, audio_path TEXT NOT NULL,
                srt_path TEXT, fps REAL NOT NULL, language TEXT NOT NULL, style TEXT NOT NULL,
                source_name TEXT, batch_id TEXT, batch_position INTEGER, output_srt TEXT,
                qc_json TEXT, qc_html TEXT, changes_srt TEXT, cost_usd REAL, cue_count INTEGER,
                error TEXT, transcription_provider TEXT NOT NULL DEFAULT 'default'
                    CHECK(transcription_provider IN ('default', 'gemini-3.5-transcribe'))
            );
            CREATE INDEX jobs_by_status ON jobs(status);
            CREATE TABLE job_notes (job_id TEXT REFERENCES jobs(id), note TEXT);
            CREATE TABLE job_audit (job_id TEXT);
            CREATE TRIGGER audit_job_update AFTER UPDATE ON jobs
                BEGIN INSERT INTO job_audit(job_id) VALUES (new.id); END;
            INSERT INTO jobs (id, token_hash, mode, status, progress, created_at, updated_at,
                expires_at, directory, audio_path, fps, language, style, transcription_provider)
            VALUES ('legacy', 'token', 'generate', 'queued', 5, '2026-09-05T00:00:00+00:00',
                '2026-09-05T00:00:00+00:00', '2026-09-06T00:00:00+00:00', 'job-legacy',
                'job-legacy/audio.wav', 30, 'auto', 'standard', 'default');
            INSERT INTO job_notes VALUES ('legacy', 'preserve me');
        """)
    return settings


def test_model_schema_migration_preserves_jobs_indexes_triggers_and_foreign_keys(tmp_path):
    settings = _legacy_database(tmp_path)
    database = settings.data_dir / "jobs.sqlite3"
    store = JobStore(settings.data_dir)
    assert store.get("legacy").transcription_provider == "default"
    store.create(_record(tmp_path, transcription_provider=MAI_MODEL))
    store.create(replace(_record(tmp_path, transcription_provider=SCRIBE_MODEL), id="scribe-choice"))
    JobStore(settings.data_dir)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT * FROM job_notes").fetchall() == [("legacy", "preserve me")]
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3
        objects = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert {"jobs_by_status", "audit_job_update"}.issubset(objects)
        connection.execute("UPDATE jobs SET progress = 10 WHERE id = 'legacy'")
        assert connection.execute("SELECT job_id FROM job_audit").fetchall() == [("legacy",)]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE jobs SET transcription_provider = 'unknown' WHERE id = 'legacy'")


def test_model_schema_migration_rolls_back_if_foreign_key_validation_fails(tmp_path):
    settings = _legacy_database(tmp_path)
    database = settings.data_dir / "jobs.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("INSERT INTO job_notes VALUES ('missing', 'existing orphan')")
    with pytest.raises(RuntimeError, match="foreign key validation"):
        JobStore(settings.data_dir)
    with closing(sqlite3.connect(database)) as connection:
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE name = 'jobs'").fetchone()[0]
        assert MAI_MODEL not in schema
        assert connection.execute("SELECT id, transcription_provider FROM jobs").fetchall() == [("legacy", "default")]
        assert connection.execute("SELECT COUNT(*) FROM job_notes").fetchone()[0] == 2
        objects = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert {"jobs_by_status", "audit_job_update"}.issubset(objects)
        assert "jobs_transcription_migration" not in objects


def test_model_schema_migration_preserves_views_of_jobs(tmp_path):
    settings = _legacy_database(tmp_path)
    database = settings.data_dir / "jobs.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE VIEW queued_jobs AS SELECT id FROM jobs WHERE status = 'queued'")
    JobStore(settings.data_dir)
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT id FROM queued_jobs").fetchall() == [("legacy",)]
