from __future__ import annotations

import json

from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.evaluation import evaluate_against_golden
from dubsync.models import QCFlag
from dubsync.srt_io import parse_srt_text


def test_evaluate_against_golden_computes_timing_and_review_metrics():
    predicted = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:01,020 --> 00:00:01,500\nthere\n\n"
        "3\n00:00:02,110 --> 00:00:02,500\ngeneral\n\n"
    )
    golden = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nthere\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\ngeneral\n\n"
    )

    metrics = evaluate_against_golden(
        predicted,
        golden,
        fps=30.0,
        flags=[QCFlag(kind="text_changed", cue_ids=[2], message="changed")],
        style_violations=0,
    )

    assert metrics["cue_count_predicted"] == 3
    assert metrics["cue_count_golden"] == 3
    assert metrics["matched_cues"] == 3
    assert metrics["start_mae_ms"] == 43.333
    assert metrics["starts_within_1_frame_ratio"] == 2 / 3
    assert metrics["starts_within_3_frames_ratio"] == 2 / 3
    assert metrics["review_burden_ratio"] == 1 / 3
    assert metrics["meets_timing_target"] is False
    assert metrics["meets_structure_target"] is True


def test_evaluate_against_golden_requires_mae_under_plan_target():
    predicted_blocks = []
    golden_blocks = []
    for cue_id in range(1, 101):
        predicted_start = "00:00:03,000" if cue_id > 98 else "00:00:00,000"
        predicted_blocks.append(f"{cue_id}\n{predicted_start} --> 00:00:04,000\nline {cue_id}\n")
        golden_blocks.append(f"{cue_id}\n00:00:00,000 --> 00:00:04,000\nline {cue_id}\n")
    predicted = parse_srt_text("\n".join(predicted_blocks))
    golden = parse_srt_text("\n".join(golden_blocks))

    metrics = evaluate_against_golden(predicted, golden, fps=30.0)

    assert metrics["starts_within_1_frame_ratio"] == 0.98
    assert metrics["starts_within_3_frames_ratio"] == 0.98
    assert metrics["start_mae_ms"] == 60.0
    assert metrics["meets_timing_target"] is False


def test_evaluate_against_golden_computes_improv_precision_and_recall():
    predicted = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nunchanged\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\ncorrect improvised line\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\nwrong flagged line\n\n"
        "4\n00:00:03,000 --> 00:00:03,500\nmissed source line\n\n"
    )
    golden = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nunchanged\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\ncorrect improvised line\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\nsource line should stay\n\n"
        "4\n00:00:03,000 --> 00:00:03,500\nmissed improvised line\n\n"
    )

    metrics = evaluate_against_golden(
        predicted,
        golden,
        fps=30.0,
        flags=[
            QCFlag(kind="text_changed", cue_ids=[2], message="actor improvised"),
            QCFlag(kind="text_changed", cue_ids=[3], message="actor improvised"),
        ],
    )

    assert metrics["improv_true_positives"] == 1
    assert metrics["improv_false_positives"] == 1
    assert metrics["improv_false_negatives"] == 1
    assert metrics["improv_precision"] == 0.5
    assert metrics["improv_recall"] == 0.5
    assert metrics["meets_improv_target"] is False


def test_report_command_can_emit_golden_evaluation_metrics(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    golden_path = tmp_path / "golden.srt"
    report_path = workdir / "qc_report.json"

    predicted_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    golden_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "summary": {"cue_count": 1, "flags": 0, "style_violations": 0},
                "flags": [],
                "style_issues": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "report",
            str(workdir),
            "--synced",
            str(predicted_path),
            "--golden",
            str(golden_path),
            "--fps",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["evaluation"]["meets_timing_target"] is True
    assert payload["evaluation"]["review_burden_ratio"] == 0.0


def test_report_command_uses_ingest_source_for_improv_recall_metrics(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    golden_path = tmp_path / "golden.srt"

    predicted_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nnew line\n\n", encoding="utf-8")
    golden_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nnew line\n\n", encoding="utf-8")
    (workdir / "ingest.json").write_text(
        json.dumps(
            {
                "cues": [
                    {
                        "index": 1,
                        "start_ms": 0,
                        "end_ms": 500,
                        "lines": ["old line"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (workdir / "qc_report.json").write_text(
        json.dumps({"summary": {"style_violations": 0}, "flags": [], "style_issues": []}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "report",
            str(workdir),
            "--synced",
            str(predicted_path),
            "--golden",
            str(golden_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["evaluation"]["improv_true_positives"] == 0
    assert payload["evaluation"]["improv_false_negatives"] == 1
    assert payload["evaluation"]["improv_recall"] == 0.0
    assert payload["evaluation"]["meets_improv_target"] is False


def test_report_command_rejects_ambiguous_parent_workdir(tmp_path):
    workdir = tmp_path / "work"
    for episode_name in ("episode-a", "episode-b"):
        episode_dir = workdir / episode_name
        episode_dir.mkdir(parents=True)
        (episode_dir / "qc_report.json").write_text(
            json.dumps({"summary": {"cue_count": 1}, "flags": [], "style_issues": []}),
            encoding="utf-8",
        )

    result = CliRunner().invoke(app, ["report", str(workdir)])

    assert result.exit_code != 0
    assert "multiple qc_report.json files found" in result.output
    assert "Traceback" not in result.output


def test_report_command_rejects_malformed_qc_report_with_clear_error(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    (workdir / "qc_report.json").write_text("{not json", encoding="utf-8")

    result = CliRunner().invoke(app, ["report", str(workdir)])

    assert result.exit_code != 0
    assert "invalid qc_report.json" in result.output
    assert "Traceback" not in result.output


def test_report_command_rejects_malformed_ingest_artifact_with_clear_error(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    golden_path = tmp_path / "golden.srt"
    predicted_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    golden_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    (workdir / "qc_report.json").write_text(
        json.dumps({"summary": {"style_violations": 0}, "flags": [], "style_issues": []}),
        encoding="utf-8",
    )
    (workdir / "ingest.json").write_text("{not json", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["report", str(workdir), "--synced", str(predicted_path), "--golden", str(golden_path)],
    )

    assert result.exit_code != 0
    assert "invalid ingest.json" in result.output
    assert "Traceback" not in result.output


def test_report_command_rejects_invalid_ingest_cue_with_clear_error(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    golden_path = tmp_path / "golden.srt"
    predicted_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    golden_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    (workdir / "qc_report.json").write_text(
        json.dumps({"summary": {"style_violations": 0}, "flags": [], "style_issues": []}),
        encoding="utf-8",
    )
    (workdir / "ingest.json").write_text(json.dumps({"cues": [{}]}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["report", str(workdir), "--synced", str(predicted_path), "--golden", str(golden_path)],
    )

    assert result.exit_code != 0
    assert "invalid ingest.json" in result.output
    assert "Traceback" not in result.output


def test_report_command_rejects_malformed_comparison_srt_with_clear_error(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    golden_path = tmp_path / "golden.srt"
    predicted_path.write_text("not an srt", encoding="utf-8")
    golden_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    (workdir / "qc_report.json").write_text(
        json.dumps({"summary": {"style_violations": 0}, "flags": [], "style_issues": []}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["report", str(workdir), "--synced", str(predicted_path), "--golden", str(golden_path)],
    )

    assert result.exit_code != 0
    assert "invalid --synced SRT" in result.output
    assert "Traceback" not in result.output
