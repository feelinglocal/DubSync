from __future__ import annotations

import json

import pytest

from dubsync.models import Cue, QCFlag, StyleIssue
from dubsync.reports import write_changes_diff, write_qc_report
from dubsync.srt_io import parse_srt_text, write_srt


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


def test_qc_report_keeps_raw_findings_and_adds_customer_sections(tmp_path):
    source = [
        Cue(index=33, start_ms=10_000, end_ms=11_000, lines=["Primeiro."]),
        Cue(index=34, start_ms=12_000, end_ms=13_000, lines=["Segundo."]),
    ]
    delivered = [
        source[0],
        Cue(index=940, start_ms=11_200, end_ms=11_600, lines=["Ei."]),
        Cue(index=34, start_ms=12_000, end_ms=13_000, lines=["Segundo!"]),
    ]
    flags = [
        QCFlag(kind="timing_evidence_held", cue_ids=[34], severity="error", message="Sparse timing.",
               start=12.0, end=13.0),
        QCFlag(kind="adlib_inserted", cue_ids=[940], message="Adjudication verdict use_audio: [hybrid:primary] Ei.",
               new_text="Ei.", start=11.2, end=11.6),
        QCFlag(kind="text_changed", cue_ids=[34], message="Adjudication verdict use_audio: [hybrid:fallback] Shout.",
               old_text="Segundo.", new_text="Segundo!", start=12.0, end=12.5),
        QCFlag(kind="asr_word_clamped", message="ASR word endpoint clamped."),
    ]

    payload = write_qc_report(
        tmp_path / "qc.json", tmp_path / "qc.html", delivered, flags, [],
        summary_metadata={"fps_detection_confident": False}, source_cues=source,
    )

    stored = json.loads((tmp_path / "qc.json").read_text(encoding="utf-8"))
    assert stored == payload
    assert len(payload["flags"]) == 4
    assert [item["kind"] for item in payload["review"]] == ["timing_evidence_held"]
    # Delivered numbering: internal cue 34 is SRT #3 because the ad-lib was inserted before it.
    delivered_srt = parse_srt_text(write_srt(delivered, renumber=True))
    review_item = payload["review"][0]
    assert review_item["srt_numbers"] == [3]
    assert review_item["cue_ids"] == [34]
    assert delivered_srt[2].plain_text == review_item["text"] == "Segundo!"
    assert [(change["srt_number"], change["change"]) for change in payload["changes"]] == [(2, "added"), (3, "edited")]
    assert [item["kind"] for item in payload["diagnostics"]] == ["asr_word_clamped"]
    summary = payload["summary"]
    assert summary["verdict"] == "attention"
    assert (summary["review_item_count"], summary["review_error_count"], summary["change_count"]) == (1, 1, 2)
    assert summary["error_count"] == 1 and summary["flags"] == 4
    covered = {index for section in ("review", "changes", "notes", "diagnostics")
               for item in payload[section] for index in item["raw_flags"]}
    assert covered == {0, 1, 2, 3}


@pytest.mark.parametrize("start,end", [(None, None), (2.0, 2.0), (1491.8, 1490.666)])
def test_changes_diff_uses_explicit_valid_markers_for_untimed_or_reversed_findings(tmp_path, start, end):
    destination = tmp_path / "changes.diff.srt"
    flag = QCFlag(
        kind="invalid_cue_duration", cue_ids=[951], message="Review timing.",
        old_text="Eu", new_text=None, start=start, end=end,
    )
    original = flag.model_copy(deep=True)

    write_changes_diff(destination, [flag])

    cues = parse_srt_text(destination.read_text(encoding="utf-8"))
    assert len(cues) == 1
    assert cues[0].end_ms == cues[0].start_ms + 1
    assert "diagnostic marker" in cues[0].text
    assert "- Eu" in cues[0].text
    assert flag == original


def test_changes_diff_preserves_supported_range(tmp_path):
    destination = tmp_path / "changes.diff.srt"
    flag = QCFlag(
        kind="text_changed", cue_ids=[1], message="Adjusted dialogue.",
        old_text="Hello.", new_text="Hi.", start=1.5, end=2.3,
    )

    write_changes_diff(destination, [flag])

    cue, = parse_srt_text(destination.read_text(encoding="utf-8"))
    assert (cue.start_ms, cue.end_ms) == (1500, 2300)
    assert "diagnostic marker" not in cue.text
    assert "- Hello.\n+ Hi." in cue.text
