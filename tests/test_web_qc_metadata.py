from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubsync.web.app import _public_job, _qc_result_metadata
from dubsync.web.jobs import default_processor, new_job_record
from dubsync.web.security import hash_job_token
from dubsync.web.settings import WebSettings


def test_completed_job_exposes_qc_counts_with_one_existing_artifact_read(tmp_path, monkeypatch):
    qc_json = tmp_path / "qc_report.json"
    summary = {
        "flags": 4, "style_violations": 2,
        "error_count": 1, "warning_count": 3, "info_count": 2,
    }
    qc_json.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    job = new_job_record(
        job_id="qc-job", token_hash="hash", mode="sync", directory=tmp_path,
        audio_path=tmp_path / "audio.wav", srt_path=tmp_path / "source.srt",
        fps=30, language="en", style="source", retention_hours=24,
    )
    job = replace(job, status="complete", cue_count=5, qc_json=qc_json)
    reads = []
    read_text = Path.read_text

    def tracked_read(path, *args, **kwargs):
        reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    payload = _public_job(job)
    assert payload["result"]["qc_summary"] == summary
    assert reads == [qc_json]
    reads.clear()
    assert _public_job(replace(job, status="processing"))["result"] is None
    assert reads == []


@pytest.mark.parametrize("bad_count", [True, -1, 1.5, "2", None, 2**53])
def test_qc_metadata_rejects_invalid_totals_without_losing_valid_fps(tmp_path, bad_count):
    qc_json = tmp_path / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {
        "flags": bad_count, "style_violations": 0,
        "fps": 24, "fps_source": "detected", "fps_detection_confident": True,
    }}), encoding="utf-8")
    assert _qc_result_metadata(qc_json) == {
        "fps": 24.0, "fps_source": "detected", "fps_detection_confident": True,
    }


def test_legacy_qc_counts_do_not_invent_missing_severities(tmp_path):
    qc_json = tmp_path / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {"flags": 2, "style_violations": 0}}), encoding="utf-8")
    assert _qc_result_metadata(qc_json) == {"qc_summary": {"flags": 2, "style_violations": 0}}


def test_qc_metadata_exposes_review_tiers_and_change_counts(tmp_path):
    qc_json = tmp_path / "qc_report.json"
    summary = {
        "flags": 854, "style_violations": 176, "error_count": 117, "warning_count": 911, "info_count": 2,
        "verdict": "check", "review_item_count": 3, "review_error_count": 0, "review_warning_count": 3,
        "review_cue_count": 5, "review_cue_ratio": 0.005, "change_count": 280, "text_change_count": 275,
        "timing_change_count": 5, "note_count": 4, "diagnostic_count": 121,
    }
    qc_json.write_text(json.dumps({"summary": summary}), encoding="utf-8")

    qc_summary = _qc_result_metadata(qc_json)["qc_summary"]

    assert qc_summary["verdict"] == "check"
    assert qc_summary["review_item_count"] == 3
    assert qc_summary["review_error_count"] == 0
    assert qc_summary["review_warning_count"] == 3
    assert qc_summary["review_cue_count"] == 5
    assert qc_summary["change_count"] == 280
    assert qc_summary["text_change_count"] == 275
    assert qc_summary["note_count"] == 4
    assert qc_summary["error_count"] == 117


@pytest.mark.parametrize("verdict", ["green", "", None, 1, True])
def test_qc_metadata_drops_unknown_verdicts_and_invalid_review_counts(tmp_path, verdict):
    qc_json = tmp_path / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {
        "flags": 2, "style_violations": 0, "verdict": verdict,
        "review_item_count": True, "review_error_count": -1, "change_count": "4",
    }}), encoding="utf-8")

    assert _qc_result_metadata(qc_json) == {"qc_summary": {"flags": 2, "style_violations": 0}}


@pytest.mark.parametrize("change_log,offered", [("", False), ("1\n00:00:01,000 --> 00:00:02,000\n# SRT #1 edited\n", True)])
def test_change_log_download_is_offered_only_when_it_lists_changes(tmp_path, monkeypatch, change_log, offered):
    providers = tmp_path / "providers.yaml"
    providers.write_text("asr:\n  provider: fixture\n", encoding="utf-8")
    settings = WebSettings(
        data_dir=tmp_path / "data", providers_path=providers, style_path=None,
        max_upload_bytes=1024 * 1024, retention_hours=24, processing_inline=True, max_submissions_per_hour=20,
    )
    settings.ensure_directories()
    directory = settings.data_dir / "job-changes"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"fixture audio")
    job = new_job_record(
        job_id="changes", token_hash=hash_job_token("token"), mode="generate", directory=directory,
        audio_path=audio, srt_path=None, fps=30, language="en", style="standard", retention_hours=24,
    )

    def fake_generate(_audio, output, _workdir, **_kwargs):
        output.write_text("1\n00:00:00,000 --> 00:00:00,500\nReady.\n", encoding="utf-8")
        artifacts = directory / "artifacts"
        artifacts.mkdir()
        (artifacts / "qc_report.json").write_text("{}", encoding="utf-8")
        (artifacts / "qc_report.html").write_text("<h1>QC</h1>", encoding="utf-8")
        (artifacts / "changes.diff.srt").write_text(change_log, encoding="utf-8")
        return SimpleNamespace(
            output_srt=output, episode_workdir=artifacts, report={"summary": {"cue_count": 1}},
            cost_meter=SimpleNamespace(total_usd=0.0),
        )

    monkeypatch.setattr("dubsync.web.jobs.generate_srt_from_audio", fake_generate)

    artifacts = default_processor(job, settings)

    assert (artifacts.changes_srt is not None) is offered


def test_invalid_severity_counts_remain_unavailable(tmp_path):
    qc_json = tmp_path / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {
        "flags": 2, "style_violations": 0, "error_count": False,
        "warning_count": "2", "info_count": -1,
    }}), encoding="utf-8")
    assert _qc_result_metadata(qc_json) == {"qc_summary": {"flags": 2, "style_violations": 0}}
