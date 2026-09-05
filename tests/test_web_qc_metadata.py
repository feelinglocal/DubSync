from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubsync.web.app import _public_job, _qc_result_metadata
from dubsync.web.jobs import new_job_record


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


def test_invalid_severity_counts_remain_unavailable(tmp_path):
    qc_json = tmp_path / "qc_report.json"
    qc_json.write_text(json.dumps({"summary": {
        "flags": 2, "style_violations": 0, "error_count": False,
        "warning_count": "2", "info_count": -1,
    }}), encoding="utf-8")
    assert _qc_result_metadata(qc_json) == {"qc_summary": {"flags": 2, "style_violations": 0}}
