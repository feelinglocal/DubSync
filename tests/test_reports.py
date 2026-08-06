from __future__ import annotations

import json

from dubsync.models import Cue, QCFlag, StyleIssue
from dubsync.reports import write_qc_report


def test_qc_report_groups_counts_and_sorts_by_severity(tmp_path):
    json_path = tmp_path / "qc.json"
    html_path = tmp_path / "qc.html"
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello"])]
    flags = [
        QCFlag(kind="punctuation_changed", cue_ids=[1], message="low priority", severity="info"),
        QCFlag(kind="min_duration_unattainable", cue_ids=[1], message="fix first", severity="error"),
        QCFlag(kind="fps_override_mismatch", cue_ids=[], message="check grid", severity="warning"),
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
