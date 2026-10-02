"""Changes that keep the customer's wording are judged by delivered timing and lines.

A timing recovery or a display reflow leaves the delivered words equal to the
source cue. QC logs such a change against the delivered SRT; only a flag whose
cue really kept its pre-change state is reported as later undone.
"""
from __future__ import annotations

import html
import json

import pytest

from dubsync import pipeline
from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import AlignmentResult, Cue, QCFlag, SpeechRegion, TokenMatch, Word
from dubsync.qc_review import build_review
from dubsync.reports import write_change_log, write_qc_report
from dubsync.srt_io import format_timestamp
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import tokenize_cues
from test_collapsed_singleton_timing import _ask, _case, _hear, _resolve
from test_missing_dialogue_reconciliation import _pipeline_case
from test_pipeline_annotation_composition import _run_case


def _undone(diagnostics) -> list[str]:
    kinds = [item["kind"] if isinstance(item, dict) else item.kind for item in diagnostics]
    return [kind for kind in kinds if kind.endswith(":not_delivered")]


def _timing(cue: dict | Cue) -> str:
    start, end = (cue["start_ms"], cue["end_ms"]) if isinstance(cue, dict) else (cue.start_ms, cue.end_ms)
    return f"{format_timestamp(start)} --> {format_timestamp(end)}"


def _delivered(result, cue_id: int) -> dict:
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    return next(cue for cue in payload["cues"] if cue["index"] == cue_id)


def test_whole_utterance_timing_recovery_is_logged_as_a_timing_change(tmp_path, monkeypatch):
    cues = [Cue(index=1, start_ms=900, end_ms=1300, lines=["Wait here."]),
            Cue(index=2, start_ms=2000, end_ms=2300, lines=["Tao."]),
            Cue(index=3, start_ms=2300, end_ms=2800, lines=["Come."])]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Wait", 1.0, 1.15), ("here", 1.16, 1.3), ("Tao", 1.701, 1.702), ("Come", 2.1, 2.4),
    ]]
    alignment = AlignmentResult(
        cue_word_indices={1: [0, 1], 2: [2], 3: [3]},
        token_matches=[TokenMatch(cue_id=token.cue_id, srt_token_index=token.token_index,
                                  asr_word_index=token.token_index, score=1) for token in tokenize_cues(cues)],
        divergence_spans=[], unmatched_cue_ids=[],
        diagnostics={"missing_audio_cue_ids": [], "missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION},
    )
    regions = [SpeechRegion(start=1, end=1.3), SpeechRegion(start=1.6, end=1.9), SpeechRegion(start=2.1, end=2.4)]
    _, _, run = _pipeline_case(tmp_path, monkeypatch, case_override=(cues, words, alignment, regions), heard="Tao.")

    result = run()

    delivered = _delivered(result, 2)
    assert delivered["lines"] == ["Tao."] and _timing(delivered) != _timing(cues[1])
    report = result.report
    change, = [item for item in report["changes"] if item["cue_id"] == 2]
    assert (change["change"], change["kind"], change["srt_number"]) == (
        "timing", "missing_dialogue_audio_reconciled", 2)
    assert (change["old_timing"], change["new_timing"]) == (_timing(cues[1]), _timing(delivered))
    assert report["summary"]["timing_change_count"] == 1
    assert _undone(report["diagnostics"]) == []
    page = (result.episode_workdir / "qc_report.html").read_text(encoding="utf-8")
    assert html.escape(change["new_timing"]) in page and "later undone" not in page


def test_collapsed_singleton_recovery_is_logged_as_a_timing_change():
    case = _case()
    questions = _ask(case)
    result = _resolve(case, questions, [_hear(questions[0])])
    cues = sorted(result.cues, key=lambda cue: (cue.start_ms, cue.index))
    target = next(cue for cue in cues if cue.index == 601)
    source = next(cue for cue in case["source_cues"] if cue.index == 601)
    assert target.lines == source.lines and _timing(target) != _timing(source)

    review = build_review(result.flags, [], cues, source_cues=case["source_cues"])

    change, = [item for item in review.changes if item.cue_id == 601]
    assert (change.change, change.kind) == ("timing", "collapsed_singleton_audio_reconciled")
    assert (change.old_timing, change.new_timing) == (_timing(source), _timing(target))
    assert change.srt_number == cues.index(target) + 1
    assert _undone(review.diagnostics) == []


def test_latin_reflow_is_logged_with_old_and_new_lines(tmp_path, monkeypatch):
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hello", "there", "my friend."])]
    words = [Word(text="Hello", start=1.1, end=1.4), Word(text="there", start=1.5, end=1.8),
             Word(text="my", start=1.9, end=2.0), Word(text="friend.", start=2.05, end=2.6)]
    _, run = _run_case(tmp_path, monkeypatch, case_override=(cues, words))

    result = run()

    report = result.report
    change, = report["changes"]
    assert (change["change"], change["kind"], change["srt_number"]) == ("edited", "output_line_limit_reflow", 1)
    assert (change["old_text"], change["new_text"]) == ("Hello\nthere\nmy friend.", "Hello there my friend.")
    assert _undone(report["diagnostics"]) == []
    diff = (result.episode_workdir / "changes.diff.srt").read_text(encoding="utf-8")
    assert "# SRT #1 edited (cue 1)" in diff and "- Hello\n- there\n- my friend.\n+ Hello there my friend." in diff
    page = (result.episode_workdir / "qc_report.html").read_text(encoding="utf-8")
    assert "Changes (1)" in page and "later undone" not in page


def test_caption_pages_are_logged_against_the_delivered_display_cues(tmp_path, monkeypatch):
    cues = [Cue(index=1, start_ms=0, end_ms=1320, lines=["[Luan Nian: todo mundo da empresa pode sair mais cedo.]"]),
            Cue(index=2, start_ms=620, end_ms=1120, lines=["Hum."])]
    _, run = _run_case(tmp_path, monkeypatch, case_override=(cues, [Word(text="Hum.", start=.62, end=1.1)]))

    result = run(style_profile=StyleProfile(max_chars_per_line=47))

    report = result.report
    pages, = [item for item in report["changes"] if item["kind"] == "annotation_line_limit_pagination"]
    assert (pages["change"], pages["cue_id"], pages["srt_number"]) == ("edited", 1, 1)
    assert pages["old_text"] == cues[0].text
    assert pages["new_text"] == "[Luan Nian:]\n\n[todo mundo da empresa pode sair mais cedo.]"
    # The earlier two-line reflow of the same caption was replaced by these pages.
    assert {report["flags"][index]["kind"] for index in pages["raw_flags"]} == {
        "output_line_limit_reflow", "annotation_line_limit_pagination"}
    assert _undone(report["diagnostics"]) == []
    diff = (result.episode_workdir / "changes.diff.srt").read_text(encoding="utf-8")
    assert "+ [Luan Nian:]" in diff


def test_delayed_caption_page_is_logged_as_a_timing_change(tmp_path, monkeypatch):
    cues = [Cue(index=30, start_ms=567, end_ms=2700, lines=["mas também por causa de um escândalo sexual."]),
            Cue(index=31, start_ms=620, end_ms=1840, lines=["[Em junho de 2011, após um jantar da empresa,]"]),
            Cue(index=32, start_ms=1840, end_ms=3740,
                lines=["[uma funcionária denunciou um caso de conduta sexual inadequada.]"])]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("mas", .6, .72), ("também", .8, .999), ("por", 1.08, 1.159), ("causa", 1.2, 1.4), ("de", 1.44, 1.5),
        ("um", 1.54, 1.6), ("escândalo", 1.68, 2.059), ("sexual.", 2.16, 2.639),
    ]]
    _, run = _run_case(tmp_path, monkeypatch, case_override=(cues, words))

    result = run(style_profile=StyleProfile(max_chars_per_line=53))

    caption = _delivered(result, 32)
    assert caption["start_ms"] > cues[2].start_ms
    report = result.report
    delay, = [item for item in report["changes"] if item["kind"] == "annotation_line_limit_pagination"]
    assert (delay["change"], delay["cue_id"]) == ("timing", 32)
    assert (delay["old_timing"], delay["new_timing"]) == (_timing(cues[2]), _timing(caption))
    reflow, = [item for item in report["changes"] if item["kind"] == "output_line_limit_reflow"]
    assert (reflow["change"], reflow["cue_id"], reflow["old_text"]) == ("edited", 32, cues[2].text)
    assert reflow["new_text"] == "\n".join(caption["lines"])
    assert _undone(report["diagnostics"]) == []


def test_caption_reflowed_into_a_speech_cue_is_logged_on_that_display_cue(tmp_path):
    source = [Cue(index=1, start_ms=1000, end_ms=2000, lines=["[Aviso: entrada proibida", "para estranhos.]"]),
              Cue(index=2, start_ms=1000, end_ms=2000, lines=["Ei!"])]
    composed = compose_bracketed_annotations(source, {2: [0]}, words=[Word(text="Ei!", start=1.0, end=1.9)],
                                             profile=StyleProfile(max_chars_per_line=26))
    assert [cue.lines for cue in composed.cues] == [["Ei!", "[Aviso: entrada proibida para estranhos.]"]]

    report = write_qc_report(tmp_path / "qc.json", tmp_path / "qc.html", composed.cues, composed.flags, [],
                             source_cues=source, pre_annotation_cues=source)
    write_change_log(tmp_path / "changes.diff.srt", report["changes"])

    change, = report["changes"]
    assert (change["change"], change["kind"], change["srt_number"]) == ("edited", "annotation_line_limit_reflow", 1)
    assert (change["old_text"], change["new_text"]) == (source[0].text, "[Aviso: entrada proibida para estranhos.]")
    # The speech line-limit pass on the same display cue changed nothing by itself.
    assert {report["flags"][index]["kind"] for index in change["raw_flags"]} == {
        "annotation_line_limit_reflow", "output_line_limit_reflow"}
    assert _undone(report["diagnostics"]) == []
    assert "+ [Aviso: entrada proibida para estranhos.]" in (tmp_path / "changes.diff.srt").read_text(encoding="utf-8")


def test_second_reflow_pass_that_changed_nothing_does_not_log_the_reflow_twice(tmp_path):
    # testing-002 cue 40: the reflowed cue still has an over-wide line, so the
    # line-limit pass runs again and reports the same lines as old and new.
    source = [Cue(index=39, start_ms=100_000, end_ms=101_000, lines=["Was ist los?"]),
              Cue(index=40, start_ms=101_500, end_ms=105_000,
                  lines=["Äh, sie haben mich seit", "drei Monaten nicht bezahlt."])]
    reflowed = source[1].with_lines(["Äh,", "sie haben mich seit drei Monaten nicht bezahlt."])
    delivered = [source[0], reflowed]
    flags = [
        QCFlag(kind="output_line_limit_reflow", cue_ids=[40], severity="info",
               message="The complete spoken phrase fits the available display lines.",
               old_text=source[1].text, new_text=reflowed.text, start=101.5, end=105.0),
        QCFlag(kind="output_line_limit_reflow", cue_ids=[40], severity="info",
               message="A safe spoken split was unavailable. Text was reflowed within its existing interval.",
               old_text=reflowed.text, new_text=reflowed.text, start=101.5, end=105.0),
    ]

    report = write_qc_report(tmp_path / "qc.json", tmp_path / "qc.html", delivered, flags, [], source_cues=source)
    write_change_log(tmp_path / "changes.diff.srt", report["changes"])

    change, = report["changes"]
    assert (change["change"], change["kind"], change["srt_number"]) == ("edited", "output_line_limit_reflow", 2)
    assert (change["old_text"], change["new_text"]) == (source[1].text, reflowed.text)
    assert change["raw_flags"] == [0, 1]
    assert report["summary"]["change_count"] == 1
    assert _undone(report["diagnostics"]) == []
    diff = (tmp_path / "changes.diff.srt").read_text(encoding="utf-8")
    assert diff.count("# SRT #2 edited (cue 40)") == 1


def test_recovery_and_reflow_that_were_reverted_stay_not_delivered():
    source = [Cue(index=1, start_ms=1000, end_ms=2000, lines=["Hello there,", "my friend."]),
              Cue(index=2, start_ms=64666, end_ms=65466, lines=["三分？"])]
    flags = [
        QCFlag(kind="output_line_limit_reflow", cue_ids=[1], severity="info",
               message="The complete spoken phrase fits the available display lines.",
               old_text="Hello there,\nmy friend.", new_text="Hello there, my friend.", start=1.0, end=2.0),
        QCFlag(kind="missing_dialogue_audio_reconciled", cue_ids=[2], severity="info", confidence=1.0,
               message="Native audio confirmed this whole cue.", old_text="三分？", new_text="三分？",
               start=64.375, end=65.135),
    ]

    # A later hold put both cues back to the customer's exact lines and timing.
    review = build_review(flags, [], source, source_cues=source)

    assert review.changes == []
    assert sorted(_undone(review.diagnostics)) == [
        "missing_dialogue_audio_reconciled:not_delivered", "output_line_limit_reflow:not_delivered"]
    assert all("later undone" in item.title for item in review.diagnostics)


@pytest.mark.parametrize("heard", ["", "三分だ"])
def test_recovery_whose_heard_wording_is_not_delivered_is_not_logged_as_a_retime(heard):
    source = [Cue(index=1, start_ms=1000, end_ms=2000, lines=["Hello."]),
              Cue(index=2, start_ms=64666, end_ms=65466, lines=["三分？"])]
    # The confirmed omission ("") or the heard wording was undone: the cue kept the
    # customer's words, and its timing moved for another reason.
    delivered = [source[0], source[1].with_timing(64366, 65133)]
    flags = [QCFlag(kind="missing_dialogue_audio_reconciled", cue_ids=[2], severity="info", confidence=1.0,
                    message="Native audio confirmed this whole cue.", old_text="三分？", new_text=heard,
                    start=64.375, end=65.135)]

    review = build_review(flags, [], delivered, source_cues=source)

    assert review.changes == []
    assert _undone(review.diagnostics) == ["missing_dialogue_audio_reconciled:not_delivered"]


def test_layout_change_does_not_approve_a_spelling_finding():
    source = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hello there", "Lukas, my friend."])]
    delivered = [source[0].with_lines(["Hello there Lukas, my friend."])]
    flags = [
        QCFlag(kind="output_line_limit_reflow", cue_ids=[1], severity="info",
               message="The complete spoken phrase fits the available display lines.",
               old_text=source[0].text, new_text=delivered[0].text, start=1.0, end=3.0),
        QCFlag(kind="name_spelling_inconsistency", cue_ids=[1], old_text="Lucas", new_text="Lukas",
               message="Output spelling 'Lukas' differs from 'Lucas' elsewhere."),
    ]

    review = build_review(flags, [], delivered, source_cues=source)

    assert [(change.change, change.kind) for change in review.changes] == [("edited", "output_line_limit_reflow")]
    assert [item.kind for item in review.review] == ["name_spelling_inconsistency"]
