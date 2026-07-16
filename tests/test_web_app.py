from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import dubsync.web.app as web_app_module
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
        assert download.headers["content-disposition"] == (
            'attachment; filename="dialogue-dubsync-synced.srt"'
        )


def test_public_config_exposes_generation_style_presets_and_custom_limits(tmp_path):
    settings = replace(_settings(tmp_path), max_srt_bytes=3 * 1024 * 1024)
    app = create_app(settings=settings, processor=_fake_processor)

    with TestClient(app) as client:
        response = client.get("/api/config")

    payload = response.json()
    assert payload["max_srt_bytes"] == 3 * 1024 * 1024
    styles = payload["generation_styles"]
    assert styles["default_preset"] == "standard"
    assert [preset["id"] for preset in styles["presets"]] == ["standard", "streaming", "broadcast", "short_form"]
    assert styles["presets"][0]["values"] == {
        "max_lines_per_cue": 2,
        "max_chars_per_line": 26,
        "min_cue_duration_seconds": 0.5,
        "max_cue_duration_seconds": 5.0,
        "min_cps": 2.0,
        "max_cps": 30.0,
        "max_gap_seconds": 0.8,
        "lead_in_ms": 0,
        "tail_ms": 40,
    }
    assert styles["custom_limits"]["max_chars_per_line"] == {"min": 10, "max": 80, "step": 1}


def test_generate_job_resolves_preset_and_custom_styles_before_queueing(tmp_path):
    captured: list[JobRecord] = []

    def capture(job: JobRecord, settings: WebSettings) -> ProcessedArtifacts:
        captured.append(job)
        return _fake_processor(job, settings)

    app = create_app(settings=_settings(tmp_path), processor=capture)
    custom_values = {
        "max_lines_per_cue": 1,
        "max_chars_per_line": 34,
        "min_cue_duration_seconds": 0.7,
        "max_cue_duration_seconds": 4.5,
        "min_cps": 3.0,
        "max_cps": 21.0,
        "max_gap_seconds": 0.6,
        "lead_in_ms": 80,
        "tail_ms": 120,
    }

    with TestClient(app) as client:
        preset = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "25", "style": json.dumps({"source": "preset", "preset": "streaming"})},
            files={"audio": ("preset.wav", b"fixture audio", "audio/wav")},
        )
        custom = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "style": json.dumps({"source": "custom", "values": custom_values})},
            files={"audio": ("custom.wav", b"fixture audio", "audio/wav")},
        )

    assert preset.status_code == 202
    assert custom.status_code == 202
    preset_style = json.loads(captured[0].style)
    custom_style = json.loads(captured[1].style)
    assert preset_style["source"] == "preset"
    assert preset_style["preset"] == "streaming"
    assert preset_style["profile"]["fps"] == 25.0
    assert preset_style["profile"]["max_chars_per_line"] == 42
    assert preset_style["constraints"]["max_cps"] == 20.0
    assert custom_style["source"] == "custom"
    assert custom_style["profile"]["max_lines_per_cue"] == 1
    assert custom_style["profile"]["lead_in_ms"] == 80
    assert custom_style["constraints"]["max_cue_duration_seconds"] == 4.5


def test_generate_job_rejects_invalid_custom_style_ranges(tmp_path):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)
    values = {
        "max_lines_per_cue": 2,
        "max_chars_per_line": 26,
        "min_cue_duration_seconds": 6.0,
        "max_cue_duration_seconds": 2.0,
        "min_cps": 2.0,
        "max_cps": 30.0,
        "max_gap_seconds": 0.8,
        "lead_in_ms": 0,
        "tail_ms": 40,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "style": json.dumps({"source": "custom", "values": values})},
            files={"audio": ("dialogue.wav", b"fixture audio", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Custom minimum cue duration cannot exceed maximum cue duration."


def test_generate_job_derives_style_from_uploaded_srt_example(tmp_path):
    captured: list[JobRecord] = []

    def capture(job: JobRecord, settings: WebSettings) -> ProcessedArtifacts:
        captured.append(job)
        return _fake_processor(job, settings)

    app = create_app(settings=_settings(tmp_path), processor=capture)
    style_example = (
        "1\n00:00:00,000 --> 00:00:01,500\nA deliberately wider subtitle example line\n\n"
        "2\n00:00:02,000 --> 00:00:06,000\nFirst line\nSecond line\nThird line\n"
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "24", "style": json.dumps({"source": "sample"})},
            files={
                "audio": ("dialogue.wav", b"fixture audio", "audio/wav"),
                "style_sample": ("house-style.srt", style_example.encode(), "application/x-subrip"),
            },
        )

    assert response.status_code == 202
    resolved = json.loads(captured[0].style)
    assert resolved["source"] == "sample"
    assert resolved["profile"]["fps"] == 24.0
    assert resolved["profile"]["cue_count"] == 2
    assert resolved["profile"]["max_lines_per_cue"] == 3
    assert resolved["profile"]["max_chars_per_line"] == len("A deliberately wider subtitle example line")
    assert resolved["constraints"]["max_cue_duration_seconds"] == 4.0


def test_generate_job_requires_a_valid_srt_for_sample_style(tmp_path):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)

    with TestClient(app) as client:
        missing = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "style": json.dumps({"source": "sample"})},
            files={"audio": ("missing.wav", b"fixture audio", "audio/wav")},
        )
        malformed = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30", "style": json.dumps({"source": "sample"})},
            files={
                "audio": ("malformed.wav", b"fixture audio", "audio/wav"),
                "style_sample": ("broken.srt", b"not an srt", "application/x-subrip"),
            },
        )

    assert missing.status_code == 422
    assert missing.json()["detail"] == "An SRT style example is required."
    assert malformed.status_code == 422
    assert malformed.json()["detail"].startswith("Could not read the SRT style example:")


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


def test_retained_storage_quota_rejects_before_multipart_parsing(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), max_retained_storage_bytes=128)
    retained = settings.data_dir / "job-retained"
    retained.mkdir(parents=True)
    (retained / "payload.bin").write_bytes(b"x" * 96)
    app = create_app(settings=settings, processor=_fake_processor)

    async def unexpected_form(*_args, **_kwargs):
        raise AssertionError("storage-rejected requests must not be parsed")

    monkeypatch.setattr("starlette.requests.Request.form", unexpected_form)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 507
    assert response.json() == {
        "detail": "Storage capacity is temporarily unavailable. Wait for existing jobs to expire."
    }
    assert response.headers["cache-control"] == "no-store"


def test_predicted_normalized_storage_rejects_after_copy_and_cleans_the_job(tmp_path):
    settings = replace(
        _settings(tmp_path),
        max_retained_storage_bytes=1_000_000,
        max_normalized_audio_bytes=2_000_000,
        max_job_work_bytes=400,
        max_job_storage_bytes=3_000_000,
    )
    app = create_app(
        settings=settings,
        audio_duration_probe=lambda *_args, **_kwargs: 1.0,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files={"audio": ("compressed.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 507
    assert response.json() == {
        "detail": "Storage capacity is temporarily unavailable. Wait for existing jobs to expire."
    }
    assert list(settings.data_dir.glob("job-*")) == []


def test_concurrent_intake_is_rejected_before_a_second_upload_is_copied(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), processing_inline=False)
    app = create_app(settings=settings, processor=_fake_processor)
    first_save_started = threading.Event()
    release_first_save = threading.Event()
    save_calls = 0
    original_save = web_app_module._save_upload

    async def blocking_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            first_save_started.set()
            await asyncio.to_thread(release_first_save.wait, 3.0)
        return await original_save(*args, **kwargs)

    monkeypatch.setattr("dubsync.web.app._save_upload", blocking_save)
    request = {
        "data": {"mode": "generate", "fps": "30"},
        "files": {"audio": ("dialogue.wav", b"audio", "audio/wav")},
    }

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(client.post, "/api/jobs", **request)
        assert first_save_started.wait(timeout=2.0)
        try:
            second = client.post("/api/jobs", **request)
        finally:
            release_first_save.set()
        accepted = first.result(timeout=3.0)

    assert accepted.status_code == 202
    assert second.status_code == 429
    assert second.json() == {"detail": "Another upload is already being accepted. Try again shortly."}
    assert save_calls == 1


def test_single_upload_rejects_duplicate_file_parts_before_copying(tmp_path, monkeypatch):
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)
    save_calls = 0
    original_save = web_app_module._save_upload

    async def recording_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return await original_save(*args, **kwargs)

    monkeypatch.setattr("dubsync.web.app._save_upload", recording_save)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"mode": "generate", "fps": "30"},
            files=[
                ("audio", ("first.wav", b"first", "audio/wav")),
                ("audio", ("second.wav", b"second", "audio/wav")),
            ],
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Exactly one audio file is required."}
    assert save_calls == 0


def test_single_submit_failure_removes_the_persisted_row_and_directory_without_masking_error(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    app = create_app(settings=settings, processor=_fake_processor)

    def failed_submit(_job):
        raise RuntimeError("intentional submit failure")

    monkeypatch.setattr(app.state.jobs, "submit", failed_submit)
    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="intentional submit failure"):
            client.post(
                "/api/jobs",
                data={"mode": "generate", "fps": "30"},
                files={"audio": ("dialogue.wav", b"audio", "audio/wav")},
            )

        assert app.state.jobs.store.pending() == []
        assert [path for path in settings.data_dir.iterdir() if path.is_dir()] == []


def test_health_and_security_headers_are_present(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123def456")
    app = create_app(settings=_settings(tmp_path), processor=_fake_processor)
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "dubsync",
        "version": "0.2.0",
        "commit": "abc123def456",
    }
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_frontend_serves_crawler_assets_and_rejects_unknown_routes(tmp_path):
    static_dir = tmp_path / "site"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        """<!doctype html>
<html><head>
<meta name="description" content="Home description" />
<meta name="robots" content="index, follow" />
<link rel="canonical" href="https://dubsync.onrender.com/" />
<meta property="og:title" content="Home | DubSync" />
<meta property="og:description" content="Home description" />
<meta property="og:url" content="https://dubsync.onrender.com/" />
<meta name="twitter:title" content="Home | DubSync" />
<meta name="twitter:description" content="Home description" />
<script type="application/ld+json" data-home-schema>{"@type":"FAQPage"}</script>
<title>Home | DubSync</title>
</head><body></body></html>""",
        encoding="utf-8",
    )
    (static_dir / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    (static_dir / "sitemap.xml").write_text("<?xml version='1.0'?><urlset></urlset>", encoding="utf-8")
    (static_dir / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    brand_dir = static_dir / "brand"
    brand_dir.mkdir()
    (brand_dir / "dubsync-mark.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    (static_dir / ".build-secret").write_text("must not be public", encoding="utf-8")
    (static_dir / "api").write_text("must not shadow the API namespace", encoding="utf-8")
    settings = replace(_settings(tmp_path), static_dir=static_dir)
    app = create_app(settings=settings, processor=_fake_processor)

    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert 'data-home-schema' in home.text

        legal_metadata = {
            "/terms": ("Terms of Service | DubSync", "Terms for using DubSync"),
            "/privacy": ("Privacy Policy | DubSync", "How DubSync processes"),
            "/payments": ("Payments and Refunds | DubSync", "Manual billing"),
        }
        for route, (title, description_start) in legal_metadata.items():
            response = client.get(route)
            assert response.status_code == 200
            assert response.headers["x-robots-tag"] == "noindex, follow"
            assert f"<title>{title}</title>" in response.text
            assert f'content="{description_start}' in response.text
            assert f'href="https://dubsync.onrender.com{route}"' in response.text
            assert f'property="og:url" content="https://dubsync.onrender.com{route}"' in response.text
            assert 'content="noindex, follow"' in response.text
            assert 'data-home-schema' not in response.text
            assert '"@type":"FAQPage"' not in response.text

            slash_response = client.get(f"{route}/", follow_redirects=False)
            assert slash_response.status_code == 308
            assert slash_response.headers["location"] == route

        robots = client.get("/robots.txt")
        sitemap = client.get("/sitemap.xml")
        favicon = client.get("/favicon.svg")
        brand = client.get("/brand/dubsync-mark.svg")

        assert robots.status_code == 200
        assert robots.headers["content-type"].startswith("text/plain")
        assert robots.text.startswith("User-agent:")
        assert sitemap.status_code == 200
        assert sitemap.headers["content-type"].startswith(("application/xml", "text/xml"))
        assert sitemap.text.startswith("<?xml")
        assert favicon.status_code == 200
        assert favicon.headers["content-type"].startswith("image/svg+xml")
        assert brand.status_code == 200
        assert brand.headers["content-type"].startswith("image/svg+xml")
        assert client.get("/.build-secret").status_code == 404
        assert client.get("/api").status_code == 404
        assert client.get("/api/unknown").status_code == 404
        assert client.get("/not-a-real-page").status_code == 404


def test_frontend_refuses_allowlisted_file_that_resolves_outside_static_root(tmp_path, monkeypatch):
    static_dir = tmp_path / "site"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><title>DubSync</title></html>", encoding="utf-8")
    (static_dir / "robots.txt").write_text("outside target", encoding="utf-8")
    monkeypatch.setattr("dubsync.web.app._inside", lambda _path, _directory: False)
    app = create_app(settings=replace(_settings(tmp_path), static_dir=static_dir), processor=_fake_processor)

    with TestClient(app) as client:
        assert client.get("/robots.txt").status_code == 404


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
        service.store.create(
            replace(
                job,
                status="complete",
                progress=100,
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )

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
    assert calls["audio_limits"].job_directory == directory
    assert calls["audio_limits"].max_duration_seconds == settings.max_audio_duration_seconds
    assert calls["audio_limits"].max_output_bytes == settings.max_normalized_audio_bytes
    assert calls["audio_limits"].max_job_storage_bytes == settings.max_job_storage_bytes


def test_default_processor_applies_the_resolved_generation_style(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    directory = settings.data_dir / "job-custom-style"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"fixture audio")
    resolved_style = {
        "source": "custom",
        "preset": None,
        "profile": {
            "fps": 30.0,
            "max_lines_per_cue": 1,
            "max_chars_per_line": 32,
            "min_cue_dur": 0.7,
            "allow_zero_gap": True,
            "lead_in_ms": 60,
            "tail_ms": 100,
            "overlap_policy": "stack",
            "drop_policy": "keep_flagged",
            "notes": [],
        },
        "constraints": {
            "max_gap_seconds": 0.6,
            "max_cue_duration_seconds": 4.0,
            "min_cps": 3.0,
            "max_cps": 21.0,
        },
    }
    job = new_job_record(
        job_id="custom-style",
        token_hash=hash_job_token("token"),
        mode="generate",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30,
        language="auto",
        style=json.dumps(resolved_style),
        retention_hours=24,
    )
    calls = {}

    def fake_generate(_audio, output, _workdir, **kwargs):
        calls.update(kwargs)
        output.write_text("1\n00:00:00,000 --> 00:00:00,700\nReady.\n", encoding="utf-8")
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

    assert calls["style_profile"].max_chars_per_line == 32
    assert calls["style_profile"].lead_in_ms == 60
    assert calls["generation_constraints"].max_cue_duration_seconds == 4.0
    assert calls["generation_constraints"].max_cps == 21.0


def test_default_processor_preserves_the_configured_profile_for_the_standard_preset(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    configured_style = tmp_path / "configured-style.yaml"
    configured_style.write_text("max_chars_per_line: 35\nmax_lines_per_cue: 2\n", encoding="utf-8")
    settings = replace(settings, style_path=configured_style)
    settings.ensure_directories()
    directory = settings.data_dir / "job-standard-style"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"fixture audio")
    resolved_style = {
        "source": "preset",
        "preset": "standard",
        "profile": {"fps": 30.0, "max_lines_per_cue": 2, "max_chars_per_line": 26, "min_cue_dur": 0.5},
        "constraints": {
            "max_gap_seconds": 0.8,
            "max_cue_duration_seconds": 5.0,
            "min_cps": 2.0,
            "max_cps": 30.0,
        },
    }
    job = new_job_record(
        job_id="standard-style",
        token_hash=hash_job_token("token"),
        mode="generate",
        directory=directory,
        audio_path=audio,
        srt_path=None,
        fps=30,
        language="auto",
        style=json.dumps(resolved_style),
        retention_hours=24,
    )
    calls = {}

    def fake_generate(_audio, output, _workdir, **kwargs):
        calls.update(kwargs)
        output.write_text("1\n00:00:00,000 --> 00:00:00,700\nReady.\n", encoding="utf-8")
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

    assert calls["style_path"] == configured_style
    assert "style_profile" not in calls
    assert "generation_constraints" not in calls


def test_default_processor_derives_sync_style_from_the_uploaded_srt(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    global_style = tmp_path / "global-style.yaml"
    global_style.write_text("max_chars_per_line: 26\n", encoding="utf-8")
    settings = replace(settings, style_path=global_style)
    settings.ensure_directories()
    directory = settings.data_dir / "job-source-style"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"fixture audio")
    source = directory / "original.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nSource style.\n", encoding="utf-8")
    job = new_job_record(
        job_id="source-style",
        token_hash=hash_job_token("token"),
        mode="sync",
        directory=directory,
        audio_path=audio,
        srt_path=source,
        fps=25,
        language="auto",
        style="source",
        retention_hours=24,
    )
    calls = {}

    def fake_sync(_srt, _audio, output, _workdir, **kwargs):
        calls.update(kwargs)
        output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
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

    monkeypatch.setattr("dubsync.web.jobs.sync_episode", fake_sync)

    default_processor(job, settings)

    assert calls["style_path"] is None
    assert calls["fps"] == 25


def test_web_settings_loads_dotenv_from_current_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DUBSYNC_RETENTION_HOURS=12\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUBSYNC_RETENTION_HOURS", raising=False)

    settings = WebSettings.from_env()

    assert settings.retention_hours == 12
