import json

import pytest
import yaml
from fastapi.testclient import TestClient

from dubsync.srt_io import parse_srt_text
from dubsync.web.app import create_app
from dubsync.web.settings import WebSettings


@pytest.mark.parametrize("mode", ["sync", "generate"])
@pytest.mark.parametrize("language", ["ja", "jpn", "ja-JP", "auto"])
def test_japanese_jobs_complete_and_deliver_unicode_srt(tmp_path, mode, language):
    words = [
        {"text": "今日は", "start": 0.5, "end": 0.9},
        {"text": "いい", "start": 1.0, "end": 1.3},
        {"text": "天気", "start": 1.4, "end": 1.8},
        {"text": "です。", "start": 1.9, "end": 2.3},
    ]
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"provider": "fixture", "fixture_path": str(fixture)},
        "llm": {"provider": "fixture"},
    }), encoding="utf-8")
    settings = WebSettings(
        data_dir=tmp_path / "data", providers_path=providers, style_path=None,
        processing_inline=True, require_job_access_code=False, max_jobs_per_hour=20,
    )
    source = "1\n00:00:05,000 --> 00:00:07,000\n今日はいい天気です。\n"
    files = {"audio": ("dialogue.wav", b"fixture audio", "audio/wav")}
    if mode == "sync":
        files["subtitle"] = ("source.srt", source.encode("utf-8"), "application/x-subrip")

    with TestClient(create_app(settings=settings)) as client:
        response = client.post("/api/jobs", data={"mode": mode, "language": language}, files=files)
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["status"] == "complete", job
        headers = {"Authorization": f"Bearer {job['token']}"}
        download = client.get(f"/api/jobs/{job['id']}/downloads/srt", headers=headers)
        assert download.status_code == 200
        cues = parse_srt_text(download.content.decode("utf-8-sig"))

    assert "".join("".join(cue.lines) for cue in cues) == "今日はいい天気です。"
    assert all(cue.end_ms > cue.start_ms >= 0 for cue in cues)
    if mode == "sync":
        assert cues[0].start_ms == 500
        alignment_paths = list(settings.data_dir.rglob("align.json"))
        alignment = json.loads(alignment_paths[0].read_text(encoding="utf-8"))
        assert alignment["anchor_coverage"] == 1.0
        assert alignment["divergence_spans"] == []


@pytest.mark.parametrize("mode", ["sync", "generate"])
def test_web_auto_detection_is_forwarded_explicitly(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    from dubsync.web.jobs import default_processor, new_job_record

    directory = tmp_path / "job"
    directory.mkdir()
    audio = directory / "audio.wav"
    source = directory / "source.srt"
    audio.write_bytes(b"audio")
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\n日本語\n", encoding="utf-8")
    settings = WebSettings(data_dir=tmp_path / "data", providers_path=tmp_path / "provider.yaml", style_path=None)
    captured = {}

    def pipeline(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_srt=directory / "result.srt", episode_workdir=directory,
            report={"summary": {"cue_count": 1}}, cost_meter=SimpleNamespace(total_usd=0),
        )

    monkeypatch.setattr("dubsync.web.jobs.sync_episode", pipeline)
    monkeypatch.setattr("dubsync.web.jobs.generate_srt_from_audio", pipeline)
    job = new_job_record(
        job_id="japanese", token_hash="hash", mode=mode, directory=directory,
        audio_path=audio, srt_path=source if mode == "sync" else None,
        fps=30, language="auto", style="standard", retention_hours=24,
    )
    default_processor(job, settings)
    assert captured["language"] == "auto"


def test_japanese_narrow_style_reports_overflow_without_blocking_export(tmp_path):
    from dubsync.style_profile import StyleProfile
    from dubsync.transcription import generate_srt_from_audio

    audio = tmp_path / "dialogue.wav"
    audio.write_bytes(b"fixture audio")
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"words": [
        {"text": "「あぁぁぁぁぁぁぁ。」", "start": 0.5, "end": 3.0},
    ]}, ensure_ascii=False), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(words)}, "llm": {"provider": "fixture"},
    }), encoding="utf-8")

    result = generate_srt_from_audio(
        audio, tmp_path / "generated.srt", tmp_path / "work", providers_path=config,
        language="auto", style_profile=StyleProfile(max_chars_per_line=10, max_lines_per_cue=1),
    )

    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert "".join("".join(cue.lines) for cue in cues) == "「あぁぁぁぁぁぁぁ。」"
    assert result.report["summary"]["style_violations"] > 0
