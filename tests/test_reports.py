from __future__ import annotations

import json

from dubsync.models import Cue, QCFlag, StyleIssue
from dubsync.reports import write_qc_report


def test_qc_report_groups_counts_and_sorts_by_severity(tmp_path):
    json_path = tmp_path / "qc.json"
    html_path = tmp_path / "qc.html"
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello"])]
    flags = [
        QCFlag(kind="punctuation_changed", cue_ids=[2], message="low priority", severity="info", start=2.0),
        QCFlag(kind="min_duration_unattainable", cue_ids=[1], message="fix first", severity="error"),
        QCFlag(kind="fps_override_mismatch", cue_ids=[], message="check grid", severity="warning", start=1.0),
    ]
    issues = [StyleIssue(kind="high_cps", cue_id=1, message="too fast", severity="error")]

    payload = write_qc_report(
        json_path,
        html_path,
        cues,
        flags,
        issues,
        summary_metadata={"fps": 24.0, "fps_source": "detected", "fps_detection_confident": True},
    )

    assert payload["summary"]["flags_by_severity"] == {"error": 1, "warning": 1, "info": 1}
    assert payload["summary"]["style_issues_by_severity"] == {"error": 1, "warning": 0, "info": 0}
    assert [flag["severity"] for flag in payload["flags"]] == ["error", "warning", "info"]
    assert [issue["severity"] for issue in payload["style_issues"]] == ["error"]
    html = html_path.read_text(encoding="utf-8")
    assert "<th>Severity</th>" in html
    assert html.index("min_duration_unattainable") < html.index("fps_override_mismatch") < html.index("punctuation_changed")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["error_count"] == 2
    assert payload["summary"]["fps"] == 24.0
    assert payload["summary"]["fps_source"] == "detected"


def test_qc_report_sorts_same_severity_flags_by_start_then_cue_id(tmp_path):
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["one"]),
        Cue(index=2, start_ms=2_000, end_ms=3_000, lines=["two"]),
        Cue(index=9, start_ms=9_000, end_ms=10_000, lines=["nine"]),
    ]
    payload = write_qc_report(
        tmp_path / "qc.json",
        tmp_path / "qc.html",
        cues,
        [
            QCFlag(kind="late_warning", cue_ids=[9], message="later", severity="warning", start=9.0),
            QCFlag(kind="early_warning", cue_ids=[2], message="earlier", severity="warning", start=2.0),
            QCFlag(kind="cue_only_warning", cue_ids=[1], message="cue", severity="warning"),
        ],
        [],
    )

    assert [flag["kind"] for flag in payload["flags"]] == [
        "cue_only_warning",
        "early_warning",
        "late_warning",
    ]
