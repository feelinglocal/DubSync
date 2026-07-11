from __future__ import annotations

from dubsync.changes import apply_adjudication_decisions, flow_text_to_lines
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag
from dubsync.overlap import apply_overlap_policy
from dubsync.reports import write_changes_diff, write_qc_report
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def test_apply_adjudication_replaces_text_for_use_audio_and_flags_change():
    cues = [Cue(index=1, start_ms=1000, end_ms=1500, lines=["old line"])]
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="old line", asr_text="new spoken line")
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="new spoken line",
        confidence=0.91,
        reason="actor improvised",
    )

    changed, flags = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())

    assert changed[0].lines == ["new spoken line"]
    assert flags[0].kind == "text_changed"
    assert flags[0].old_text == "old line"
    assert flags[0].new_text == "new spoken line"


def test_apply_adjudication_replaces_only_divergent_phrase_inside_single_cue():
    cues = [
        Cue(index=2, start_ms=2800, end_ms=5366, lines=["Ich habe eine", "SSS-Rang-Klasse"]),
        Cue(index=11, start_ms=24166, end_ms=26333, lines=["Drachen-", "Evolutionssystem besitze."]),
    ]
    spans = [
        DivergenceSpan(
            case_id="case-1",
            cue_ids=[2],
            srt_text="SSS Rang Klasse",
            asr_text="SSS-Rangklasse",
        ),
        DivergenceSpan(
            case_id="case-2",
            cue_ids=[11],
            srt_text="Drachen Evolutionssystem",
            asr_text="Drachenevolutionssystem",
        ),
    ]
    decisions = [
        AdjudicationDecision(
            case_id="case-1",
            verdict="use_audio",
            final_text="SSS-Rangklasse",
            confidence=0.95,
            reason="compound spelling from audio",
        ),
        AdjudicationDecision(
            case_id="case-2",
            verdict="use_audio",
            final_text="Drachenevolutionssystem",
            confidence=0.95,
            reason="compound spelling from audio",
        ),
    ]

    changed, flags = apply_adjudication_decisions(cues, spans, decisions, StyleProfile())

    assert changed[0].lines == ["Ich habe eine", "SSS-Rangklasse"]
    assert changed[1].plain_text == "Drachenevolutionssystem besitze."
    assert changed[1].plain_text.endswith("besitze.")
    assert flags[0].old_text == "Ich habe eine\nSSS-Rang-Klasse"
    assert flags[0].new_text == "Ich habe eine\nSSS-Rangklasse"
    assert flags[1].old_text == "Drachen-\nEvolutionssystem besitze."
    assert flags[1].new_text == "Drachenevolutionssystem\nbesitze."


def test_changed_text_reflow_hyphen_splits_long_compound():
    text = "DrachenEvolutionssystemkontrolle"

    lines = flow_text_to_lines(text, max_chars=26, max_lines=2)

    assert len(lines) == 2
    assert lines[0].endswith("-")
    assert "".join(line.removesuffix("-") for line in lines) == text
    assert all(len(line) <= 26 for line in lines)


def test_apply_adjudication_distributes_multi_cue_replacement_without_duplication():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["old first"]),
        Cue(index=2, start_ms=1500, end_ms=2000, lines=["old second"]),
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2],
        srt_text="old first old second",
        asr_text="new spoken first second",
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="new spoken first second",
        confidence=0.91,
        reason="actor improvised",
    )

    changed, flags = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(max_chars_per_line=20),
    )

    assert [cue.plain_text for cue in changed] == ["new spoken", "first second"]
    assert len(flags) == 1
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].old_text == "old first\nold second"
    assert flags[0].new_text == "new spoken\nfirst second"


def test_overlap_dash_policy_merges_two_overlapping_cues():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["first"], speaker_id="A"),
        Cue(index=2, start_ms=1500, end_ms=2300, lines=["second"], speaker_id="B"),
    ]

    merged, flags = apply_overlap_policy(cues, policy="dash")

    assert len(merged) == 1
    assert merged[0].start_ms == 1000
    assert merged[0].end_ms == 2300
    assert merged[0].lines == ["- first", "- second"]
    assert flags[0].kind == "overlap_dash_merge"


def test_overlap_dash_policy_does_not_merge_unknown_speaker_overlap():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["first"]),
        Cue(index=2, start_ms=1500, end_ms=2300, lines=["second"], speaker_id="B"),
    ]

    unchanged, flags = apply_overlap_policy(cues, policy="dash")

    assert unchanged == cues
    assert flags[0].kind == "overlap_flag_only"
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].start == 1.5
    assert flags[0].end == 2.0


def test_overlap_stack_policy_flags_unknown_speaker_overlap():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["first"]),
        Cue(index=2, start_ms=1500, end_ms=2300, lines=["second"]),
    ]

    unchanged, flags = apply_overlap_policy(cues, policy="stack")

    assert unchanged == cues
    assert flags[0].kind == "overlap_stacked"
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].start == 1.5
    assert flags[0].end == 2.0


def test_overlap_flag_only_records_overlap_timestamp_window():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["first"], speaker_id="A"),
        Cue(index=2, start_ms=1500, end_ms=2300, lines=["second"], speaker_id="B"),
    ]

    unchanged, flags = apply_overlap_policy(cues, policy="flag_only")

    assert unchanged == cues
    assert flags[0].kind == "overlap_flag_only"
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].start == 1.5
    assert flags[0].end == 2.0


def test_changes_diff_lists_old_and_new_text(tmp_path):
    path = tmp_path / "changes.diff.srt"
    write_changes_diff(
        path,
        [
            QCFlag(
                kind="text_changed",
                cue_ids=[1],
                message="changed",
                old_text="old",
                new_text="new",
                start=1.0,
                end=2.0,
            )
        ],
    )

    assert "- old" in path.read_text(encoding="utf-8")
    assert "+ new" in path.read_text(encoding="utf-8")


def test_changes_diff_is_valid_srt_review_file(tmp_path):
    path = tmp_path / "changes.diff.srt"
    write_changes_diff(
        path,
        [
            QCFlag(
                kind="text_changed",
                cue_ids=[1],
                message="changed",
                old_text="old",
                new_text="new",
                start=1.0,
                end=2.0,
            )
        ],
    )

    diff_cues = parse_srt_text(path.read_text(encoding="utf-8"))
    assert len(diff_cues) == 1
    assert diff_cues[0].start_ms == 1000
    assert diff_cues[0].end_ms == 2000
    assert diff_cues[0].lines == ["# text_changed cue=1", "- old", "+ new"]


def test_qc_html_report_lists_flag_timestamps_and_text_changes(tmp_path):
    json_path = tmp_path / "qc_report.json"
    html_path = tmp_path / "qc_report.html"

    write_qc_report(
        json_path,
        html_path,
        [Cue(index=1, start_ms=1000, end_ms=2000, lines=["new line"])],
        [
            QCFlag(
                kind="text_changed",
                cue_ids=[1],
                message="actor improvised",
                confidence=0.91,
                old_text="old line",
                new_text="new line",
                start=1.0,
                end=2.0,
            )
        ],
        [],
    )

    html = html_path.read_text(encoding="utf-8")
    assert "1.000" in html
    assert "2.000" in html
    assert "old line" in html
    assert "new line" in html
