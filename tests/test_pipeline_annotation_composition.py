"""Screen text is a display track, not additional spoken-word ownership."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import Cue, SpeechRegion, Word
from dubsync.qc_review import build_review
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.timing_refinement import SpeechEvidence


def _run_case(tmp_path, monkeypatch, *, speech_lines=None, no_overlaps=True):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2400, lines=["[Station]"]),
        Cue(index=2, start_ms=1600, end_ms=2250, lines=speech_lines or ["Hello there."]),
    ]
    words = [Word(text="Hello", start=1.6, end=1.85), Word(text="there.", start=1.9, end=2.2)]
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
        words=words, regions=[SpeechRegion(start=1.6, end=2.2)], detected=True,
    ))
    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, no_llm=True, **kwargs)
    return cues, run


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


def test_composed_display_still_reports_real_three_line_crowding(tmp_path, monkeypatch):
    _, run = _run_case(tmp_path, monkeypatch, speech_lines=["Hello", "there."])
    result = run()
    assert any(issue["kind"] == "line_count" and issue["cue_id"] == 2 for issue in result.report["style_issues"])
    assert any("line" in item["title"].lower() and 2 in item["cue_ids"] for item in result.report["review"])


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
