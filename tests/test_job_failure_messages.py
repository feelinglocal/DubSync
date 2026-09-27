from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubsync.providers import ProviderError
from dubsync.web.app import create_app
from dubsync.web.jobs import STORAGE_RESERVATION_FILENAME
from dubsync.web.settings import WebSettings


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/batches"])
@pytest.mark.parametrize(
    "error_type,code,expected_terms",
    [
        (ProviderError, "authentication", ("authentication", "API key")),
        (ProviderError, "configuration", ("not configured", "server API key")),
        (ProviderError, "credits", ("credits", "add")),
        (ProviderError, "rate_limit", ("limit", "later")),
        (ProviderError, None, ("Processing failed",)),
        (ProviderError, "unknown-secret-code", ("Processing failed",)),
        (RuntimeError, "authentication", ("Processing failed",)),
    ],
)
def test_failed_job_status_uses_safe_actionable_provider_codes(
    tmp_path: Path, endpoint: str, error_type, code, expected_terms: tuple[str, ...],
):
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    settings = WebSettings(
        data_dir=tmp_path / "data", providers_path=providers, style_path=None,
        processing_inline=True, max_submissions_per_hour=100, retention_hours=2,
    )
    exception_secret = "private-upstream-body-and-api-key"
    processed = []

    def processor(job, _settings):
        processed.append(job)
        (job.directory / "partial.srt").write_text("partial", encoding="utf-8")
        work = job.directory / "work"
        work.mkdir()
        (work / "partial.wav").write_bytes(b"partial")
        (job.directory / STORAGE_RESERVATION_FILENAME).write_text("1024", encoding="utf-8")
        if error_type is ProviderError:
            failure = ProviderError(exception_secret, code=code)
        else:
            failure = error_type(exception_secret)
            failure.code = code
        raise failure

    app = create_app(settings=settings, processor=processor)
    before_submit = datetime.now(UTC)
    stems = ["one", "two"] if endpoint.endswith("batches") else ["one"]
    with TestClient(app) as client:
        submitted = client.post(
            endpoint,
            data={"mode": "generate", "transcription_provider": "microsoft/mai-transcribe-2"},
            files=[("audio", (f"{stem}.wav", b"audio", "audio/wav")) for stem in stems],
        )
        assert submitted.status_code == 202, submitted.text
        payload = submitted.json()
        children = payload["jobs"] if endpoint.endswith("batches") else [payload]
        assert len(children) == len(stems)
        for child in children:
            response = client.get(
                f"/api/jobs/{child['id']}",
                headers={"Authorization": f"Bearer {child['token']}"},
            )
            assert response.status_code == 200
            status = response.json()
            assert status["status"] == "failed"
            assert status["progress"] == 100
            assert all(term.lower() in status["error"].lower() for term in expected_terms)
            assert exception_secret not in response.text
            assert "unknown-secret-code" not in response.text
            stored = app.state.jobs.store.get(child["id"])
            assert stored.error == status["error"]
            assert stored.expires_at >= before_submit + timedelta(hours=settings.retention_hours)

    assert len(processed) == len(stems)
    for job in processed:
        assert job.audio_path.read_bytes() == b"audio"
        assert not (job.directory / "partial.srt").exists()
        assert not (job.directory / "work").exists()
        assert not (job.directory / STORAGE_RESERVATION_FILENAME).exists()
