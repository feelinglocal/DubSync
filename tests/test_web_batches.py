from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, JobService, JobStore, ProcessedArtifacts, new_job_record
from dubsync.web.security import hash_job_token
from dubsync.web.settings import WebSettings


SRT_BYTES = b"1\n00:00:00,000 --> 00:00:00,500\nReady.\n"


def _settings(tmp_path: Path, *, processing_inline: bool = True) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data",
        providers_path=providers,
        style_path=None,
        max_upload_bytes=1024 * 1024,
        max_srt_bytes=1024 * 1024,
        retention_hours=24,
        processing_inline=processing_inline,
        max_jobs_per_hour=100,
        worker_threads=1,
        cleanup_interval_seconds=60,
    )


def _artifacts(job: JobRecord) -> ProcessedArtifacts:
    output = job.directory / "synced.srt"
    output.write_bytes(SRT_BYTES)
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


def _multipart(
    audio_files: list[tuple[str, bytes]],
    subtitle_files: list[tuple[str, bytes]],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        *[("audio", (name, content, "audio/wav")) for name, content in audio_files],
        *[("subtitle", (name, content, "application/x-subrip")) for name, content in subtitle_files],
    ]


def _post_batch(
    client: TestClient,
    audio_files: list[tuple[str, bytes]],
    subtitle_files: list[tuple[str, bytes]],
    *,
    mode: str = "sync",
    fps: float = 30,
    language: str = "auto",
    style: str = "standard",
):
    return client.post(
        "/api/batches",
        data={
            "mode": mode,
            "fps": str(fps),
            "language": language,
            "style": style,
            "access_code": "",
        },
        files=_multipart(audio_files, subtitle_files),
    )


def _database_row_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        return {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }


def test_batch_accepts_ten_case_insensitive_pairs_and_preserves_audio_order_and_source_names(tmp_path):
    audio_stems = ["007", "Scene-A", "003", "010", "002", "009", "004", "008", "006", "001"]
    audio_files = [(f"{stem}.WAV", f"audio:{stem.casefold()}".encode()) for stem in audio_stems]
    subtitle_files = [
        (f"{stem.swapcase()}.srt", f"subtitle:{stem.casefold()}".encode())
        for stem in reversed(audio_stems)
    ]
    processed: list[tuple[str, bytes, bytes]] = []

    def capture(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        assert job.srt_path is not None
        processed.append((job.source_name, job.audio_path.read_bytes(), job.srt_path.read_bytes()))
        return _artifacts(job)

    app = create_app(settings=_settings(tmp_path), processor=capture)
    with TestClient(app) as client:
        response = _post_batch(client, audio_files, subtitle_files)

        assert response.status_code == 202
        batch = response.json()
        assert batch["id"]
        assert [job["source_name"] for job in batch["jobs"]] == audio_stems
        assert [job["batch_position"] for job in batch["jobs"]] == list(range(10))
        assert {job["batch_id"] for job in batch["jobs"]} == {batch["id"]}
        assert all(job["token"] for job in batch["jobs"])
        assert len({job["token"] for job in batch["jobs"]}) == 10

        assert processed == [
            (stem, f"audio:{stem.casefold()}".encode(), f"subtitle:{stem.casefold()}".encode())
            for stem in audio_stems
        ]

        for position, child in enumerate(batch["jobs"]):
            persisted = client.get(
                f"/api/jobs/{child['id']}",
                headers={"Authorization": f"Bearer {child['token']}"},
            )
            assert persisted.status_code == 200
            assert persisted.json()["source_name"] == audio_stems[position]
            assert persisted.json()["batch_id"] == batch["id"]
            assert persisted.json()["batch_position"] == position

        cross_token = client.get(
            f"/api/jobs/{batch['jobs'][0]['id']}",
            headers={"Authorization": f"Bearer {batch['jobs'][1]['token']}"},
        )
        assert cross_token.status_code == 404

        named_child = next(job for job in batch["jobs"] if job["source_name"] == "001")
        download = client.get(
            f"/api/jobs/{named_child['id']}/downloads/srt",
            headers={"Authorization": f"Bearer {named_child['token']}"},
        )
        assert download.status_code == 200
        assert download.headers["content-disposition"] == 'attachment; filename="001-dubsync-synced.srt"'


def test_generate_batch_accepts_audio_only_and_applies_shared_options_in_selection_order(tmp_path):
    captured: list[JobRecord] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        captured.append(job)
        return _artifacts(job)

    app = create_app(settings=_settings(tmp_path), processor=processor)
    audio_files = [
        ("opening.wav", b"opening"),
        ("Middle.WAV", b"middle"),
        ("ending.wav", b"ending"),
    ]

    with TestClient(app) as client:
        response = _post_batch(
            client,
            audio_files,
            [],
            mode="generate",
            fps=25,
            language="id",
            style="standard",
        )

    assert response.status_code == 202
    children = response.json()["jobs"]
    assert [child["source_name"] for child in children] == ["opening", "Middle", "ending"]
    assert [job.source_name for job in captured] == ["opening", "Middle", "ending"]
    assert all(job.mode == "generate" for job in captured)
    assert all(job.srt_path is None for job in captured)
    assert all(job.fps == 25 for job in captured)
    assert all(job.language == "id" for job in captured)
    assert len({job.style for job in captured}) == 1


@pytest.mark.parametrize(
    ("audio_files", "subtitle_files"),
    [
        pytest.param([], [], id="empty"),
        pytest.param(
            [(f"{index:02}.wav", b"audio") for index in range(11)],
            [(f"{index:02}.srt", SRT_BYTES) for index in reversed(range(11))],
            id="eleven-pairs",
        ),
        pytest.param(
            [("Duplicate.wav", b"one"), ("duplicate.mp3", b"two")],
            [("DUPLICATE.srt", SRT_BYTES), ("duplicate.SRT", SRT_BYTES)],
            id="case-insensitive-duplicate-stems",
        ),
        pytest.param(
            [("Caf\u00e9.wav", b"one"), ("Cafe\u0301.mp3", b"two")],
            [("CAF\u00c9.srt", SRT_BYTES), ("cafe\u0301.SRT", SRT_BYTES)],
            id="unicode-normalized-duplicate-stems",
        ),
        pytest.param(
            [("complete.wav", b"one"), ("missing.wav", b"two")],
            [("COMPLETE.srt", SRT_BYTES)],
            id="missing-pair",
        ),
        pytest.param(
            [("alpha.wav", b"one"), ("bravo.wav", b"two")],
            [("BRAVO.srt", SRT_BYTES), ("charlie.srt", SRT_BYTES)],
            id="unmatched-pairs",
        ),
        pytest.param(
            [("../episode.wav", b"one")],
            [("episode.srt", SRT_BYTES)],
            id="path-bearing-filename",
        ),
        pytest.param(
            [("bad\x00name.wav", b"one")],
            [("bad\x00name.srt", SRT_BYTES)],
            id="control-character-filename",
        ),
        pytest.param(
            [("hidden\u202egpj.wav", b"one")],
            [("hidden\u202egpj.srt", SRT_BYTES)],
            id="bidi-control-filename",
        ),
        pytest.param(
            [("....wav", b"one")],
            [("....srt", SRT_BYTES)],
            id="dot-only-stem",
        ),
        pytest.param(
            [(f"{'a' * 300}.wav", b"one")],
            [(f"{'a' * 300}.srt", SRT_BYTES)],
            id="overlong-filename",
        ),
    ],
)
def test_batch_rejects_invalid_pair_sets_before_any_processor_runs(
    tmp_path,
    audio_files: list[tuple[str, bytes]],
    subtitle_files: list[tuple[str, bytes]],
):
    processed: list[str] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        processed.append(job.id)
        return _artifacts(job)

    app = create_app(settings=_settings(tmp_path), processor=processor)
    with TestClient(app) as client:
        response = _post_batch(client, audio_files, subtitle_files)

    assert response.status_code == 422
    assert processed == []
    assert [path for path in app.state.settings.data_dir.iterdir() if path.is_dir()] == []
    assert set(_database_row_counts(app.state.jobs.store.db_path).values()) == {0}


def test_batch_counts_as_one_rate_limit_event_instead_of_one_event_per_child(tmp_path):
    settings = replace(_settings(tmp_path), max_jobs_per_hour=1)
    processed: list[str] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        processed.append(job.source_name)
        return _artifacts(job)

    app = create_app(settings=settings, processor=processor)
    stems = [f"episode-{index:02}" for index in range(10)]
    audio_files = [(f"{stem}.wav", b"audio") for stem in stems]
    subtitle_files = [(f"{stem.upper()}.srt", SRT_BYTES) for stem in reversed(stems)]

    with TestClient(app) as client:
        accepted = _post_batch(client, audio_files, subtitle_files)
        limited = _post_batch(client, [("later.wav", b"audio")], [("LATER.srt", SRT_BYTES)])

    assert accepted.status_code == 202
    assert processed == stems
    assert limited.status_code == 429


def test_batch_processor_is_strictly_serial_and_continues_after_a_child_failure(tmp_path):
    lock = threading.Lock()
    active = 0
    max_active = 0
    processed: list[str] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            processed.append(job.source_name)
        try:
            time.sleep(0.04)
            if job.source_name == "fails":
                raise RuntimeError("intentional per-child failure")
            return _artifacts(job)
        finally:
            with lock:
                active -= 1

    settings = _settings(tmp_path, processing_inline=False)
    app = create_app(settings=settings, processor=processor)
    audio_files = [("first.wav", b"one"), ("fails.wav", b"two"), ("last.wav", b"three")]
    subtitle_files = [("LAST.srt", SRT_BYTES), ("FIRST.srt", SRT_BYTES), ("FAILS.srt", SRT_BYTES)]

    with TestClient(app) as client:
        response = _post_batch(client, audio_files, subtitle_files)
        assert response.status_code == 202
        children = response.json()["jobs"]

        deadline = time.monotonic() + 3
        statuses: list[str] = []
        while time.monotonic() < deadline:
            statuses = [
                client.get(
                    f"/api/jobs/{child['id']}",
                    headers={"Authorization": f"Bearer {child['token']}"},
                ).json()["status"]
                for child in children
            ]
            if all(status in {"complete", "failed"} for status in statuses):
                break
            time.sleep(0.02)

    assert statuses == ["complete", "failed", "complete"]
    assert processed == ["first", "fails", "last"]
    assert max_active == 1


def test_job_service_rejects_worker_thread_counts_other_than_one(tmp_path):
    settings = replace(_settings(tmp_path), worker_threads=2)
    service: JobService | None = None
    try:
        with pytest.raises(ValueError, match="1"):
            service = JobService(settings, _artifacts)
    finally:
        if service is not None:
            service.shutdown()


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_expired_cleanup_preserves_active_batch_children(tmp_path, status: str):
    settings = _settings(tmp_path)
    service = JobService(settings, lambda job, _settings: _artifacts(job))
    directory = settings.data_dir / f"job-active-{status}"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"audio")
    job = new_job_record(
        job_id=f"active-{status}",
        token_hash=hash_job_token("token"),
        mode="sync",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30,
        language="auto",
        style="source",
        retention_hours=24,
    )
    expired = replace(
        job,
        status=status,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    service.store.create(expired)
    try:
        assert service.store.delete_expired() == 0
        assert service.store.get(expired.id) is not None
        assert directory.exists()
    finally:
        service.shutdown()


@pytest.mark.parametrize("fails", [False, True], ids=["complete", "failed"])
def test_job_retention_begins_when_processing_reaches_a_terminal_state(tmp_path, fails: bool):
    settings = _settings(tmp_path)

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        if fails:
            raise RuntimeError("intentional terminal failure")
        return _artifacts(job)

    service = JobService(settings, processor)
    directory = settings.data_dir / f"job-terminal-{fails}"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"audio")
    queued = new_job_record(
        job_id=f"terminal-{fails}",
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
    service.store.create(replace(queued, expires_at=datetime.now(UTC) - timedelta(days=1)))
    terminal_at = datetime.now(UTC)
    try:
        service.submit(queued)
        terminal = service.store.get(queued.id)
        assert terminal is not None
        assert terminal.status == ("failed" if fails else "complete")
        assert terminal.expires_at >= terminal_at + timedelta(hours=23, minutes=59)
        assert service.store.delete_expired(terminal_at) == 0
        assert service.store.delete_expired(terminal.expires_at + timedelta(microseconds=1)) == 1
    finally:
        service.shutdown()


def test_batch_aggregate_upload_cap_removes_every_row_and_created_directory(tmp_path):
    settings = replace(
        _settings(tmp_path),
        max_upload_bytes=64,
        max_srt_bytes=64,
        max_batch_upload_bytes=20,
    )
    processed: list[str] = []

    def processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
        processed.append(job.id)
        return _artifacts(job)

    app = create_app(settings=settings, processor=processor)
    audio_files = [("one.wav", b"a" * 6), ("two.wav", b"b" * 6)]
    subtitle_files = [("TWO.srt", b"c" * 6), ("ONE.srt", b"d" * 6)]

    with TestClient(app) as client:
        response = _post_batch(client, audio_files, subtitle_files)

        assert response.status_code == 413
        assert processed == []
        assert [path for path in settings.data_dir.iterdir() if path.is_dir()] == []

        row_counts = _database_row_counts(app.state.jobs.store.db_path)
        assert row_counts
        assert set(row_counts.values()) == {0}


def test_batch_rejects_an_oversized_request_before_copying_uploads(tmp_path, monkeypatch):
    settings = replace(
        _settings(tmp_path),
        max_upload_bytes=4 * 1024 * 1024,
        max_batch_upload_bytes=32,
    )

    async def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("route-level upload copying must not start")

    monkeypatch.setattr("dubsync.web.app._save_upload", unexpected_copy)
    app = create_app(settings=settings, processor=lambda job, _settings: _artifacts(job))
    with TestClient(app) as client:
        response = _post_batch(
            client,
            [("large.wav", b"a" * (2 * 1024 * 1024))],
            [("large.srt", SRT_BYTES)],
        )

    assert response.status_code == 413
    assert [path for path in settings.data_dir.iterdir() if path.is_dir()] == []
    assert set(_database_row_counts(app.state.jobs.store.db_path).values()) == {0}


def test_job_store_migrates_legacy_rows_idempotently_and_keeps_old_downloads_working(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    directory = settings.data_dir / "job-legacy"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"audio")
    subtitle = directory / "original.srt"
    subtitle.write_bytes(SRT_BYTES)
    output = directory / "synced.srt"
    output.write_bytes(SRT_BYTES)
    qc_json = directory / "qc_report.json"
    qc_json.write_text("{}", encoding="utf-8")
    qc_html = directory / "qc_report.html"
    qc_html.write_text("<h1>QC</h1>", encoding="utf-8")
    now = datetime.now(UTC)
    database = settings.data_dir / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
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
        connection.execute(
            """
            INSERT INTO jobs (
                id, token_hash, mode, status, progress, created_at, updated_at, expires_at,
                directory, audio_path, srt_path, fps, language, style, output_srt,
                qc_json, qc_html, changes_srt, cost_usd, cue_count, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy",
                hash_job_token("legacy-token"),
                "sync",
                "complete",
                100,
                now.isoformat(),
                now.isoformat(),
                (now + timedelta(hours=24)).isoformat(),
                str(directory),
                str(audio),
                str(subtitle),
                30.0,
                "auto",
                "source",
                str(output),
                str(qc_json),
                str(qc_html),
                None,
                0.01,
                1,
                None,
            ),
        )

    app = create_app(settings=settings, processor=lambda job, _settings: _artifacts(job))
    with TestClient(app) as client:
        status = client.get(
            "/api/jobs/legacy",
            headers={"Authorization": "Bearer legacy-token"},
        )
        download = client.get(
            "/api/jobs/legacy/downloads/srt",
            headers={"Authorization": "Bearer legacy-token"},
        )

    assert status.status_code == 200
    assert status.json()["status"] == "complete"
    assert download.status_code == 200
    assert download.headers["content-disposition"] == 'attachment; filename="dubsync.synced.srt"'

    first = JobStore(settings.data_dir)
    second = JobStore(settings.data_dir)
    legacy = second.get("legacy")
    assert legacy is not None
    assert legacy.source_name is None
    assert legacy.batch_id is None
    assert legacy.batch_position is None
    with sqlite3.connect(first.db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()]
    assert columns.count("source_name") == 1
    assert columns.count("batch_id") == 1
    assert columns.count("batch_position") == 1
