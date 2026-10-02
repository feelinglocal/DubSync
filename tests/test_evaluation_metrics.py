from __future__ import annotations

import json

from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.evaluation import evaluate_against_golden
from dubsync.models import QCFlag
from dubsync.qc_review import build_review
from dubsync.reports import write_qc_report
from dubsync.srt_io import parse_srt_text, write_srt


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
        flags=[
            # A logged wording change is not a review item; a held timing is.
            QCFlag(kind="text_changed", cue_ids=[2], message="changed", old_text="their", new_text="there"),
            QCFlag(kind="timing_evidence_held", cue_ids=[3], message="held", severity="error"),
        ],
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


def test_review_burden_counts_review_cues_not_raw_flags():
    predicted = parse_srt_text("".join(
        f"{index}\n00:00:{index:02d},000 --> 00:00:{index:02d},500\nline {index}\n\n" for index in range(1, 11)
    ))
    flags = [
        # Song-caption cascade, change log and diagnostics: no human look needed.
        QCFlag(kind="missing_audio_timing_held", cue_ids=[1], message="held", severity="error"),
        QCFlag(kind="cue_without_speech_activity", cue_ids=[1], message="no speech"),
        QCFlag(kind="timing_refined", cue_ids=[4], message="moved", old_text="4.000 --> 4.500",
               new_text="4.000 --> 4.566"),
        QCFlag(kind="asr_word_clamped", message="clamped"),
        QCFlag(kind="source_error", cue_ids=[6, 7], message="repeated phrase"),
        # One overlap reported twice is one review item covering two cues.
        QCFlag(kind="overlap_stacked", cue_ids=[8, 9], message="overlap", start=8.4, end=8.5),
        QCFlag(kind="output_overlap_unresolved", cue_ids=[8, 9], message="overlap", severity="error"),
    ]
    predicted[0] = predicted[0].with_lines(["♪Song line♪"])
    predicted[7] = predicted[7].with_timing(8_000, 9_200)

    metrics = evaluate_against_golden(predicted, predicted, fps=30.0, flags=flags)

    assert metrics["review_burden_ratio"] == 2 / 10


def test_review_burden_and_improv_metrics_use_delivered_numbers_from_the_qc_report():
    # The ad-lib inserted before "goodbye" has internal id 3 but is SRT #2.
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:02,000 --> 00:00:02,500\nbye\n\n"
    )
    predicted = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nnew adlib\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\nbye\n\n"
    )
    flags = [
        QCFlag(kind="adlib_inserted", cue_ids=[3], message="actor improvised", new_text="new adlib"),
        QCFlag(kind="timing_evidence_held", cue_ids=[2], message="held", severity="error"),
    ]
    review_items = [{"kind": "timing_evidence_held", "srt_numbers": [3], "cue_ids": [2]}]
    change_items = [{"change": "added", "srt_number": 2, "cue_id": 3}]

    metrics = evaluate_against_golden(
        predicted, predicted, fps=30.0, flags=flags, source=source,
        review_items=review_items, change_items=change_items,
    )

    assert metrics["review_burden_ratio"] == 1 / 3
    assert metrics["improv_true_positives"] == 1
    assert metrics["improv_false_positives"] == 0
    assert metrics["improv_recall"] == 1.0


def _improv_case_with_a_reflow():
    """Cue 1 is only re-broken, cue 2 is a real improvisation the golden also has."""
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,500\nHello there,\nmy friend.\n\n"
        "2\n00:00:02,000 --> 00:00:02,500\nold line\n\n"
    )
    predicted = [source[0].with_lines(["Hello there, my friend."]), source[1].with_lines(["new line"])]
    flags = [
        QCFlag(kind="output_line_limit_reflow", cue_ids=[1], severity="info",
               message="The complete spoken phrase fits the available display lines.",
               old_text=source[0].text, new_text=predicted[0].text, start=0.0, end=1.5),
        QCFlag(kind="text_changed", cue_ids=[2], message="actor improvised", old_text="old line", new_text="new line"),
    ]
    return source, predicted, flags


def test_line_break_only_change_items_are_not_counted_as_flagged_improvisations():
    source, predicted, flags = _improv_case_with_a_reflow()
    review = build_review(flags, [], predicted, source_cues=source)
    change_items = [item.model_dump() for item in review.changes]
    assert [(item["srt_number"], item["change"], item["kind"]) for item in change_items] == [
        (1, "edited", "output_line_limit_reflow"), (2, "edited", "text_changed")]

    metrics = evaluate_against_golden(
        predicted, predicted, fps=30.0, flags=flags, source=source, change_items=change_items,
    )
    wording_only = evaluate_against_golden(
        predicted, predicted, fps=30.0, flags=flags, source=source, change_items=change_items[1:],
    )

    # The golden kept cue 1's wording: its reflow is neither a hit nor a false alarm.
    assert (metrics["improv_true_positives"], metrics["improv_false_positives"]) == (1, 0)
    assert metrics["improv_precision"] == 1.0
    assert {key: value for key, value in metrics.items() if key.startswith("improv_")} == {
        key: value for key, value in wording_only.items() if key.startswith("improv_")}


def test_caption_page_change_items_are_not_counted_as_flagged_improvisations():
    source, predicted, flags = _improv_case_with_a_reflow()
    change_items = [
        {"change": "edited", "kind": "annotation_line_limit_pagination", "srt_number": 1,
         "old_text": "[Luan Nian: todo mundo pode sair.]", "new_text": "[Luan Nian:]\n\n[todo mundo pode sair.]"},
        {"change": "edited", "kind": "annotation_line_limit_reflow", "srt_number": 1,
         "old_text": "[Aviso: entrada\nproibida.]", "new_text": "[Aviso: entrada proibida.]"},
        {"change": "edited", "kind": "text_changed", "srt_number": 2, "old_text": "old line", "new_text": "new line"},
    ]

    metrics = evaluate_against_golden(
        predicted, predicted, fps=30.0, flags=flags, source=source, change_items=change_items,
    )

    assert (metrics["improv_true_positives"], metrics["improv_false_positives"]) == (1, 0)


def test_wording_change_logged_under_a_reflow_kind_still_counts_as_flagged():
    source, predicted, flags = _improv_case_with_a_reflow()
    # The item's lines differ in more than their breaks: it is a wording change.
    change_items = [
        {"change": "edited", "kind": "output_line_limit_reflow", "srt_number": 2,
         "old_text": "old\nline", "new_text": "new line"},
    ]

    metrics = evaluate_against_golden(
        predicted, predicted, fps=30.0, flags=flags, source=source, change_items=change_items,
    )

    assert (metrics["improv_true_positives"], metrics["improv_false_positives"]) == (1, 0)
    assert metrics["improv_recall"] == 1.0


def test_report_command_does_not_count_reflow_entries_as_improvisations(tmp_path):
    source, predicted, flags = _improv_case_with_a_reflow()
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    predicted_path.write_text(write_srt(predicted), encoding="utf-8")
    (workdir / "ingest.json").write_text(
        json.dumps({"cues": [cue.model_dump() for cue in source]}), encoding="utf-8")
    write_qc_report(workdir / "qc_report.json", workdir / "qc_report.html", predicted, flags, [], source_cues=source)

    result = CliRunner().invoke(
        app, ["report", str(workdir), "--synced", str(predicted_path), "--golden", str(predicted_path)],
        env={"COLUMNS": "1000"},  # keep the printed JSON strings unwrapped
    )

    assert result.exit_code == 0, result.output
    evaluation = json.loads(result.output)["evaluation"]
    assert (evaluation["improv_true_positives"], evaluation["improv_false_positives"]) == (1, 0)
    assert evaluation["improv_precision"] == 1.0


def test_source_aware_improv_metrics_count_inserted_golden_cue_as_change():
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:02,000 --> 00:00:02,500\nbye\n\n"
    )
    predicted = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nnew adlib\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\nbye\n\n"
    )
    golden = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nnew adlib\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\nbye\n\n"
    )

    metrics = evaluate_against_golden(
        predicted,
        golden,
        fps=30.0,
        flags=[QCFlag(kind="adlib_inserted", cue_ids=[2], message="actor improvised")],
        source=source,
    )

    assert metrics["improv_true_positives"] == 1
    assert metrics["improv_false_positives"] == 0
    assert metrics["improv_false_negatives"] == 0
    assert metrics["improv_precision"] == 1.0
    assert metrics["improv_recall"] == 1.0


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


def test_report_command_measures_review_burden_from_the_report_review_list(tmp_path):
    workdir = tmp_path / "work" / "episode"
    workdir.mkdir(parents=True)
    predicted_path = tmp_path / "predicted.srt"
    srt = "".join(f"{index}\n00:00:0{index},000 --> 00:00:0{index},500\nline {index}\n\n" for index in range(1, 5))
    predicted_path.write_text(srt, encoding="utf-8")
    (workdir / "qc_report.json").write_text(json.dumps({
        "summary": {"cue_count": 4, "flags": 1, "style_violations": 0},
        # Raw flag on an ad-lib's internal cue id 5; the delivered SRT numbers it #4.
        "flags": [{"kind": "timing_evidence_held", "cue_ids": [5], "message": "held", "severity": "error"}],
        "style_issues": [],
        "review": [{"kind": "timing_evidence_held", "srt_numbers": [4], "cue_ids": [5]}],
        "changes": [],
    }), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["report", str(workdir), "--synced", str(predicted_path), "--golden", str(predicted_path)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["evaluation"]["review_burden_ratio"] == 0.25


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
