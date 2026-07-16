from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dubsync.web.intake_guard as intake_guard_module
from dubsync.web.app import create_app
from dubsync.web.jobs import JobRecord, ProcessedArtifacts
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
        max_submissions_per_hour=20,
    )


def _fake_processor(job: JobRecord, _settings: WebSettings) -> ProcessedArtifacts:
    output = job.directory / ("generated.srt" if job.mode == "generate" else "synced.srt")
    output.write_text("1\n00:00:00,000 --> 00:00:00,500\nReady.\n", encoding="utf-8")
    qc_json = job.directory / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {"cue_count": 1, "flags": 0, "style_violations": 0}}), encoding="utf-8")
    qc_html = job.directory / "qc_report.html"
    qc_html.write_text("<h1>QC</h1>", encoding="utf-8")
    return ProcessedArtifacts(output_srt=output, qc_json=qc_json, qc_html=qc_html, cost_usd=0.01, cue_count=1)


def test_render_rate_limit_uses_the_first_forwarded_client_address(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    settings = replace(_settings(tmp_path), max_submissions_per_hour=1)
    app = create_app(settings=settings, processor=_fake_processor)

    with TestClient(app) as client:
        accepted = client.post(
            "/api/jobs",
            headers={"X-Forwarded-For": "198.51.100.11, 203.0.113.8"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("first.wav", b"audio", "audio/wav")},
        )
        second_client = client.post(
            "/api/jobs",
            headers={"X-Forwarded-For": "198.51.100.12, 203.0.113.8"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("second.wav", b"audio", "audio/wav")},
        )
        limited = client.post(
            "/api/jobs",
            headers={"X-Forwarded-For": "198.51.100.11, 203.0.113.9"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("third.wav", b"audio", "audio/wav")},
        )

    assert accepted.status_code == 202
    assert second_client.status_code == 202
    assert limited.status_code == 429


@pytest.mark.parametrize("path", ["/api/jobs", "/api/batches"])
@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing-header"),
        pytest.param({"X-DubSync-Access-Code": "wrong-code"}, id="invalid-header"),
    ],
)
def test_job_intake_rejects_invalid_access_before_multipart_parsing_or_copying(
    tmp_path,
    monkeypatch,
    path: str,
    headers: dict[str, str],
):
    settings = replace(
        _settings(tmp_path),
        job_access_code="quoted-access-code-1234",
        require_job_access_code=True,
    )
    app = create_app(settings=settings, processor=_fake_processor)

    async def unexpected_form(*_args, **_kwargs):
        raise AssertionError("multipart parsing must not start")

    async def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("route-level upload copying must not start")

    monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
    monkeypatch.setattr("dubsync.web.app._save_upload", unexpected_copy)

    files: object
    if path == "/api/batches":
        files = [
            ("audio", ("dialogue.wav", b"audio", "audio/wav")),
            ("subtitle", ("dialogue.srt", b"1\n00:00:00,000 --> 00:00:00,500\nReady.\n", "application/x-subrip")),
        ]
    else:
        files = {"audio": ("dialogue.wav", b"audio", "audio/wav")}

    with TestClient(app) as client:
        response = client.post(
            path,
            headers=headers,
            data={"mode": "generate", "fps": "30"},
            files=files,
        )

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "A valid job access code is required."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/api/jobs", "/api/batches"])
def test_job_intake_rate_limit_is_enforced_before_multipart_parsing(tmp_path, monkeypatch, path: str):
    settings = replace(_settings(tmp_path), max_submissions_per_hour=1)
    app = create_app(settings=settings, processor=_fake_processor)
    if path == "/api/batches":
        files: object = [("audio", ("first.wav", b"audio", "audio/wav"))]
    else:
        files = {"audio": ("first.wav", b"audio", "audio/wav")}

    with TestClient(app) as client:
        accepted = client.post(path, data={"mode": "generate", "fps": "30"}, files=files)
        assert accepted.status_code == 202

        async def unexpected_form(*_args, **_kwargs):
            raise AssertionError("rate-limited requests must not be parsed")

        monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
        limited = client.post(path, data={"mode": "generate", "fps": "30"}, files=files)

    assert limited.status_code == 429
    assert limited.json() == {"detail": "Too many jobs. Try again later."}
    assert limited.headers["cache-control"] == "no-store"


def test_invalid_access_attempt_does_not_consume_the_valid_submission_allowance(tmp_path):
    settings = replace(
        _settings(tmp_path),
        max_submissions_per_hour=1,
        job_access_code="quoted-access-code-1234",
        require_job_access_code=True,
    )
    app = create_app(settings=settings, processor=_fake_processor)
    request = {
        "data": {"mode": "generate", "fps": "30"},
        "files": {"audio": ("dialogue.wav", b"audio", "audio/wav")},
    }

    with TestClient(app) as client:
        invalid = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "wrong-code"},
            **request,
        )
        accepted = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "quoted-access-code-1234"},
            **request,
        )
        limited = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "quoted-access-code-1234"},
            **request,
        )

    assert invalid.status_code == 403
    assert accepted.status_code == 202
    assert limited.status_code == 429


def test_manual_job_access_code_is_required_without_exposing_the_secret(tmp_path):
    settings = replace(
        _settings(tmp_path),
        job_access_code="quoted-access-code-1234",
        require_job_access_code=True,
    )
    app = create_app(settings=settings, processor=_fake_processor)
    with TestClient(app) as client:
        config = client.get("/api/config")
        missing = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )
        invalid = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "wrong-code"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )
        form_only = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "access_code": "quoted-access-code-1234"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )
        accepted = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "quoted-access-code-1234"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )

    assert config.status_code == 200
    assert config.json()["access_code_required"] is True
    assert config.json()["jobs_available"] is True
    assert "quoted-access-code-1234" not in config.text
    assert missing.status_code == 403
    assert invalid.status_code == 403
    assert form_only.status_code == 403
    assert accepted.status_code == 202


def test_production_job_intake_fails_closed_when_access_code_is_missing(tmp_path):
    settings = replace(_settings(tmp_path), require_job_access_code=True)
    app = create_app(settings=settings, processor=_fake_processor)
    with TestClient(app) as client:
        config = client.get("/api/config")
        response = client.post(
            "/api/jobs",
            headers={"X-DubSync-Access-Code": "anything"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )

    assert config.json()["access_code_required"] is True
    assert config.json()["jobs_available"] is False
    assert response.status_code == 503
    assert response.json()["detail"] == "Job access is not configured."


def test_single_upload_content_length_cap_rejects_before_parsing_or_copying(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path, max_upload_bytes=16), max_srt_bytes=16)
    app = create_app(settings=settings, processor=_fake_processor)

    async def unexpected_form(*_args, **_kwargs):
        raise AssertionError("oversized declared bodies must not be parsed")

    async def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("oversized declared bodies must not be copied")

    monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
    monkeypatch.setattr("dubsync.web.app._save_upload", unexpected_copy)

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"x" * (300 * 1024), "audio/wav")},
        )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Request body is too large."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_web_intake_rejects_overlong_audio_and_cleans_the_temporary_job(tmp_path):
    settings = replace(_settings(tmp_path), processing_inline=False)
    app = create_app(
        settings=settings,
        audio_duration_probe=lambda *_args, **_kwargs: settings.max_audio_duration_seconds + 1,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("too-long.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Audio is too long. Use audio no longer than 14400 seconds."
    assert list(settings.data_dir.glob("job-*")) == []


def test_single_upload_rejects_an_invalid_content_length_before_parsing(tmp_path, monkeypatch):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)

    async def unexpected_form(*_args, **_kwargs):
        raise AssertionError("an invalid Content-Length must be rejected before parsing")

    monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            headers={"Content-Length": "not-a-number"},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Invalid Content-Length header."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_single_upload_rejects_an_unbounded_content_length_as_safe_json(tmp_path, monkeypatch):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)

    async def unexpected_form(*_args, **_kwargs):
        raise AssertionError("an unbounded Content-Length must be rejected before parsing")

    monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/jobs",
            headers={"Content-Length": "9" * 5000},
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Request body is too large."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_single_upload_receive_cap_rejects_chunked_body_before_copying(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path, max_upload_bytes=16), max_srt_bytes=16)
    app = create_app(settings=settings, processor=_fake_processor)
    boundary = "dubsync-security-boundary"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="mode"\r\n\r\n'
        "generate\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="fps"\r\n\r\n'
        "30\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="dialogue.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    async def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("oversized streamed bodies must not be copied")

    seen_content_lengths: list[str | None] = []
    original_preflight = intake_guard_module.job_intake_preflight

    def recording_preflight(request, **kwargs):
        seen_content_lengths.append(request.headers.get("content-length"))
        return original_preflight(request, **kwargs)

    monkeypatch.setattr(intake_guard_module, "job_intake_preflight", recording_preflight)
    monkeypatch.setattr("dubsync.web.app._save_upload", unexpected_copy)

    def chunked_body():
        yield prefix
        yield b"x" * (300 * 1024)
        yield suffix

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Transfer-Encoding": "chunked",
            },
            content=chunked_body(),
        )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Request body is too large."}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert seen_content_lengths == [None]
