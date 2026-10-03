"""Screen text is a display track, not additional spoken-word ownership."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import Cue, SpeechRegion, Word
from dubsync.qc_review import build_review
from dubsync.srt_io import format_timestamp, parse_srt_text, write_srt
from dubsync.timing_refinement import SpeechEvidence


def _run_case(tmp_path, monkeypatch, *, speech_lines=None, no_overlaps=True, case_override=None):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2400, lines=["[Station]"]),
        Cue(index=2, start_ms=1600, end_ms=2250, lines=speech_lines or ["Hello there."]),
    ]
    words = [Word(text="Hello", start=1.6, end=1.85), Word(text="there.", start=1.9, end=2.2)]
    if case_override is not None:
        cues, words = case_override
    source, audio, fixture, config = (tmp_path / name for name in (
        "episode.srt", "episode.wav", "words.json", "providers.yaml",
    ))
    source.write_text(write_srt(cues), encoding="utf-8")
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 64000)
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)},
                                    "output": {"no_overlaps": no_overlaps}}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "speech_evidence_for_words", lambda *_a, **_k: SpeechEvidence(
        words=words, regions=[SpeechRegion(start=words[0].start, end=words[-1].end)], detected=True,
    ))
    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, no_llm=True, **kwargs)
    return cues, run


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_short_first_caption_page_keeps_its_own_duration_warning(tmp_path, monkeypatch, mode):
    from dubsync.style_profile import StyleProfile

    cues = [Cue(index=1, start_ms=0, end_ms=1320,
                lines=["[Luan Nian: todo mundo da empresa pode sair mais cedo.]"]),
            Cue(index=2, start_ms=100, end_ms=1120, lines=["Hum."])]
    _, run = _run_case(tmp_path, monkeypatch,
                       case_override=(cues, [Word(text="Hum.", start=.1, end=1.1)]))
    profile = StyleProfile(max_chars_per_line=47, lead_in_ms=0, tail_ms=0)
    result = run(style_profile=profile)
    if mode != "fresh":
        result = run(style_profile=profile, **({} if mode == "cache" else {"resume": mode}))
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    delivered = [Cue.model_validate(cue) for cue in payload["cues"]]
    assert delivered[0].lines == ["[Luan Nian:]"]
    assert delivered[0].duration_ms == 100
    assert any(issue["kind"] == "min_duration" and issue["cue_id"] == delivered[0].index
               for issue in result.report["style_issues"])
    assert not any(issue["kind"] == "min_duration" and issue["cue_id"] == delivered[-1].index
                   for issue in result.report["style_issues"])


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_pipeline_composes_screen_text_without_changing_speech_or_inventing_wording(tmp_path, monkeypatch, mode):
    source, run = _run_case(tmp_path, monkeypatch)
    result = run()
    first_bytes = result.output_srt.read_bytes()
    if mode != "fresh":
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert result.output_srt.read_bytes() == first_bytes
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    delivered = [Cue.model_validate(item) for item in payload["cues"]]
    assert all(left.end_ms <= right.start_ms for left, right in zip(delivered, delivered[1:]))
    base = {cue["index"]: Cue.model_validate(cue) for cue in payload["pre_annotation_cues"]}
    spoken = next(cue for cue in delivered if cue.index == 2)
    assert (spoken.start_ms, spoken.end_ms) == (base[2].start_ms, base[2].end_ms)
    assert spoken.lines == [*source[1].lines, "[Station]"]
    assert payload["alignment"]["cue_word_indices"]["2"] == [0, 1]
    assert all(not indices for cue_id, indices in payload["alignment"]["cue_word_indices"].items() if cue_id != "2")
    for instant in range(1000, 2400):
        assert any(cue.start_ms <= instant < cue.end_ms and "[Station]" in cue.lines for cue in delivered)
    assert not [item for item in result.report["changes"] if item["change"] in {"added", "edited", "removed"}]
    assert not [issue for issue in result.report["style_issues"] if issue["kind"] == "min_duration" and issue["cue_id"] != 2]
    assert not [issue for issue in result.report["style_issues"] if issue["kind"] == "overlap"]
    assert len(parse_srt_text(result.output_srt.read_text(encoding="utf-8"))) == len(delivered)


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_composed_display_reflows_crowded_speech_to_two_lines(tmp_path, monkeypatch, mode):
    _, run = _run_case(tmp_path, monkeypatch, speech_lines=["Hello", "there."])
    result = run()
    first_bytes = result.output_srt.read_bytes()
    if mode != "fresh":
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert result.output_srt.read_bytes() == first_bytes
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered)
    spoken = next(cue for cue in delivered if "Hello" in cue.text)
    assert spoken.lines == ["Hello there.", "[Station]"]
    assert not [issue for issue in result.report["style_issues"] if issue["kind"] == "line_count"]
    # The delivered reflow is logged against the customer's lines, not reported as undone.
    reflow, = [item for item in result.report["changes"] if item["change"] in {"added", "edited", "removed"}]
    assert (reflow["kind"], reflow["old_text"], reflow["new_text"]) == (
        "output_line_limit_reflow", "Hello\nthere.", "Hello there.")
    assert not [item for item in result.report["diagnostics"] if item["kind"].endswith(":not_delivered")]


def test_annotation_trace_keeps_real_spoken_edit_and_final_display_number():
    caption = Cue(index=1, start_ms=1000, end_ms=3000, lines=["[Station]"])
    source = Cue(index=2, start_ms=1000, end_ms=2000, lines=["Hello there."])
    changed = source.with_lines(["Hey there."])
    delivered = [changed.with_lines([*changed.lines, *caption.lines]), caption.with_timing(2000, 3000)]
    review = build_review([], [], delivered, source_cues=[caption, source], pre_annotation_cues=[caption, changed])
    assert len(review.changes) == 1
    edit = review.changes[0]
    assert edit.old_text == "Hello there." and edit.new_text == "Hey there."
    assert edit.cue_id == 2 and edit.srt_number == 1


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_explicit_allow_overlap_preserves_authored_annotation_segmentation(tmp_path, monkeypatch, mode):
    source, run = _run_case(tmp_path, monkeypatch, no_overlaps=False)
    result = run()
    if mode == "verify":
        result = run(resume="verify")
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert len(payload["cues"]) == 2
    assert payload["cues"][0] == source[0].model_dump()
    assert payload["cues"][1]["lines"] == source[1].lines
    assert "annotation_composition" not in payload


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_held_dialogue_around_a_caption_keeps_two_lines_and_moves_the_caption_as_a_logged_change(
        tmp_path, monkeypatch, mode):
    # W4R-1: a held two-turn cue fills the display for the caption's whole
    # time. The delivery never shows three lines; the caption is shown on its
    # own beside the speech, and the change log says where it went (W4C-4).
    turns = ["- Você vem com a gente?", "- Não, fico aqui."]
    cues = [Cue(index=1, start_ms=500, end_ms=2500, lines=turns),
            Cue(index=2, start_ms=1000, end_ms=2000, lines=["[Station]"]),
            Cue(index=3, start_ms=3000, end_ms=3800, lines=["Hello there."])]
    words = [Word(text="Hello", start=3.1, end=3.4), Word(text="there.", start=3.5, end=3.7)]
    _, run = _run_case(tmp_path, monkeypatch, case_override=(cues, words))
    result = run()
    first_bytes = result.output_srt.read_bytes()
    if mode == "verify":
        result = run(resume="verify")
        assert result.output_srt.read_bytes() == first_bytes
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered), [cue.lines for cue in delivered]
    assert all(left.end_ms <= right.start_ms for left, right in zip(delivered, delivered[1:]))
    speech = next(cue for cue in delivered if cue.lines[0] == turns[0])
    assert speech.lines == turns
    caption, = [cue for cue in delivered if "[Station]" in cue.lines]
    assert caption.lines == ["[Station]"]
    assert caption.end_ms <= speech.start_ms or caption.start_ms >= speech.end_ms
    assert not [issue for issue in result.report["style_issues"] if issue["kind"] == "line_count"]
    assert not [item for item in result.report["review"] if "annotation_display_full" in item["reasons"]]
    _assert_caption_move_logged(result, delivered, caption, cues[1])


def _assert_caption_move_logged(result, delivered, caption, source_caption):
    number = delivered.index(caption) + 1
    moved, = [item for item in result.report["changes"] if item["kind"] == "annotation_line_limit_pagination"]
    assert (moved["change"], moved["srt_number"], moved["cue_id"]) == ("timing", number, source_caption.index)
    assert moved["old_timing"] == f"{format_timestamp(source_caption.start_ms)} --> {format_timestamp(source_caption.end_ms)}"
    assert moved["new_timing"] == f"{format_timestamp(caption.start_ms)} --> {format_timestamp(caption.end_ms)}"
    assert "that speech" in moved["reason"]


_TURNS = ["- Você vem com a gente?", "- Não, fico aqui."]


def _boxed_turns_case(tmp_path, monkeypatch, *, after=True):
    cues = [Cue(index=1, start_ms=200, end_ms=1000, lines=["Hello there."]),
            Cue(index=2, start_ms=1000, end_ms=3000, lines=_TURNS),
            Cue(index=3, start_ms=1500, end_ms=2500, lines=["[Station]"])]
    words = [Word(text="Hello", start=.25, end=.5), Word(text="there.", start=.55, end=.95)]
    if after:
        cues.append(Cue(index=4, start_ms=3000, end_ms=3900, lines=["Okay then."]))
        words += [Word(text="Okay", start=3.0, end=3.3), Word(text="then.", start=3.35, end=3.85)]
    return _run_case(tmp_path, monkeypatch, case_override=(cues, words))


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_caption_beside_the_last_full_speech_is_shown_before_the_end_of_the_media(tmp_path, monkeypatch, mode):
    # W4C-4: nothing follows the speech, but the media does (the audio is 4 s long).
    cues, run = _boxed_turns_case(tmp_path, monkeypatch, after=False)
    result = run()
    first_bytes = result.output_srt.read_bytes()
    if mode == "verify":
        result = run(resume="verify")
        assert result.output_srt.read_bytes() == first_bytes
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered) and delivered[-1].end_ms <= 4000
    assert all(left.end_ms <= right.start_ms for left, right in zip(delivered, delivered[1:]))
    speech = next(cue for cue in delivered if cue.lines == _TURNS)
    assert (speech.start_ms, speech.end_ms) == (1000, 3000)
    caption, = [cue for cue in delivered if "[Station]" in cue.lines]
    assert (caption.start_ms, caption.end_ms, caption.lines) == (3000, 4000, ["[Station]"])
    assert not [item for item in result.report["review"] if "annotation_display_full" in item["reasons"]]
    _assert_caption_move_logged(result, delivered, caption, cues[2])


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_caption_with_no_room_beside_full_speech_is_an_error_and_a_removed_line_in_the_change_log(
        tmp_path, monkeypatch, mode):
    # W4C-4: never silently lose the customer's caption.
    cues, run = _boxed_turns_case(tmp_path, monkeypatch)
    result = run()
    first_bytes = result.output_srt.read_bytes()
    if mode == "verify":
        result = run(resume="verify")
        assert result.output_srt.read_bytes() == first_bytes
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered)
    assert all(left.end_ms <= right.start_ms for left, right in zip(delivered, delivered[1:]))
    assert not [cue for cue in delivered if "[Station]" in cue.lines]
    speech = next(cue for cue in delivered if cue.lines == _TURNS)
    assert (speech.start_ms, speech.end_ms) == (1000, 3000)
    report = result.report
    item, = [item for item in report["review"] if "annotation_display_full" in item["reasons"]]
    assert (item["kind"], item["severity"], item["srt_numbers"], item["after_srt_number"]) == (
        "annotation_display_full", "error", [], delivered.index(speech) + 1)
    assert "[Station]" in item["detail"] and "could not be displayed" in item["detail"]
    assert report["summary"]["verdict"] == "attention"
    removed, = [change for change in report["changes"] if change["change"] == "removed"]
    assert (removed["kind"], removed["cue_id"], removed["old_text"], removed["new_text"]) == (
        "annotation_display_full", 3, "[Station]", None)
    diff = (result.episode_workdir / "changes.diff.srt").read_text(encoding="utf-8")
    assert "removed after SRT #" in diff and "(cue 3)" in diff and "- [Station]" in diff
