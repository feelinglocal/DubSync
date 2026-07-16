from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, ProcessedArtifacts
from dubsync.web.settings import WebSettings


def _settings(tmp_path: Path) -> WebSettings:
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    return WebSettings(
        data_dir=tmp_path / "data",
        providers_path=providers,
        style_path=None,
        processing_inline=True,
        max_submissions_per_hour=20,
    )


def _unexpected_processor(_job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
    raise AssertionError("structurally unsafe SRT input must be rejected before processing")


def _cue(index: int, text: str = "Ready.") -> bytes:
    return (
        f"{index}\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"{text}\n\n"
    ).encode()


def test_single_sync_rejects_excess_srt_lines_before_creating_a_job(tmp_path):
    settings = replace(_settings(tmp_path), max_srt_lines=8)
    app = create_app(settings=settings, processor=_unexpected_processor)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "sync", "fps": "30"},
            files={
                "audio": ("001.wav", b"audio", "audio/wav"),
                "subtitle": ("001.srt", b"x\n" * 9, "application/x-subrip"),
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Could not read the SRT: subtitle exceeds 8 lines"}
    assert list(settings.data_dir.glob("job-*")) == []


def test_batch_sync_rejects_an_overlong_srt_line_before_creating_jobs(tmp_path):
    settings = replace(_settings(tmp_path), max_srt_line_bytes=40)
    app = create_app(settings=settings, processor=_unexpected_processor)

    with TestClient(app) as client:
        response = client.post(
            "/api/batches",
            data={"mode": "sync", "fps": "30"},
            files=[
                ("audio", ("001.wav", b"audio", "audio/wav")),
                ("subtitle", ("001.srt", _cue(1, "x" * 41), "application/x-subrip")),
            ],
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Could not read the SRT: subtitle line 3 exceeds 40 bytes"
    }
    assert list(settings.data_dir.glob("job-*")) == []


def test_generate_style_sample_rejects_excess_cues_before_creating_a_job(tmp_path):
    settings = replace(_settings(tmp_path), max_srt_cues=2)
    app = create_app(settings=settings, processor=_unexpected_processor)
    sample = b"".join(_cue(index) for index in range(1, 4))

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={
                "mode": "generate",
                "fps": "30",
                "style": json.dumps({"source": "sample"}),
            },
            files={
                "audio": ("001.wav", b"audio", "audio/wav"),
                "style_sample": ("style.srt", sample, "application/x-subrip"),
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Could not read the SRT style example: subtitle exceeds 2 cues"
    }
    assert list(settings.data_dir.glob("job-*")) == []


def test_default_web_srt_limits_bound_memory_for_ten_file_batches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "DUBSYNC_MAX_SRT_BYTES",
        "DUBSYNC_MAX_SRT_LINES",
        "DUBSYNC_MAX_SRT_CUES",
        "DUBSYNC_MAX_SRT_LINE_BYTES",
        "DUBSYNC_MAX_SRT_LINE_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = WebSettings.from_env()

    assert settings.max_srt_bytes == 2 * 1024 * 1024
    assert settings.max_srt_lines == 60_000
    assert settings.max_srt_cues == 20_000
    assert settings.max_srt_line_bytes == 16 * 1024
    assert settings.max_srt_line_chars == 4_096
