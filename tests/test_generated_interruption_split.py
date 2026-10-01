from __future__ import annotations

import json
import socket
import wave

import pytest
import yaml

from dubsync import cue_segmentation, pipeline
from dubsync.models import AlignmentResult, Cue, Word
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile


def _episode_11_interruption():
    """Captured MAI words 926..934, shifted 664 seconds earlier."""
    words = [Word(text=text, start=start, end=end, speaker_id=speaker) for text, start, end, speaker in [
        ("se", 1.600, 1.680, "chunk_1:0"),
        ("uma", 1.720, 1.819, "chunk_1:0"),
        ("garota", 1.840, 2.099, "chunk_1:0"),
        ("Hã?", 2.120, 2.460, "chunk_1:chunk_3.2"),
        ("não", 2.480, 2.579, "chunk_1:0"),
        ("recebe", 2.600, 2.880, "chunk_1:0"),
        ("flores", 2.960, 3.199, "chunk_1:0"),
        ("de", 3.240, 3.339, "chunk_1:0"),
        ("ninguém,", 3.400, 3.759, "chunk_1:0"),
    ]]
    cues = [
        Cue(index=217, start_ms=1600, end_ms=3800,
            lines=["se uma garota não recebe flores de ninguém,"], speaker_id="chunk_1:0", character="Narrator"),
        Cue(index=945, start_ms=2100, end_ms=2500, lines=["Hã?"], speaker_id="chunk_1:chunk_3.2"),
    ]
    alignment = AlignmentResult(cue_word_indices={217: [0, 1, 2, 4, 5, 6, 7, 8], 945: [3]})
    return cues, words, alignment


def _split(cues, words, alignment, **kwargs):
    return cue_segmentation.split_at_generated_interruptions(
        cues, words, alignment, {945}, StyleProfile(fps=30, min_cue_dur=.2), **kwargs,
    )


def test_captured_interruption_splits_only_the_enclosing_source_without_changing_word_ownership():
    cues, words, alignment = _episode_11_interruption()
    original_cues = [cue.model_dump() for cue in cues]
    original_alignment = alignment.model_dump()
    original_words = [word.model_dump() for word in words]

    output, updated, flags, expansions = _split(cues, words, alignment)

    assert expansions == {217: [217, 946]}
    assert {cue.index: cue.lines for cue in output} == {
        217: ["se uma garota"], 946: ["não recebe flores de ninguém,"], 945: ["Hã?"],
    }
    assert updated.cue_word_indices == {217: [0, 1, 2], 946: [4, 5, 6, 7, 8], 945: [3]}
    assert next(cue for cue in output if cue.index == 945) == cues[1]
    assert all(cue.character == "Narrator" for cue in output if cue.index in {217, 946})
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]
    assert flags[0].cue_ids == [217, 946]
    assert flags[0].old_text == cues[0].text
    assert flags[0].new_text == "se uma garota\n\nnão recebe flores de ninguém,"
    assert [cue.model_dump() for cue in cues] == original_cues
    assert alignment.model_dump() == original_alignment
    assert [word.model_dump() for word in words] == original_words


@pytest.mark.parametrize("fault", [
    "multiline", "markup", "bracket", "song", "nonexact", "spacing", "protected_parent", "protected_insert",
    "missing_parent", "missing_insert", "shared_parent", "shared_insert", "unknown_parent", "unknown_insert",
    "same_speaker", "different_scope", "mixed_parent", "prefix_overlap", "suffix_overlap", "internal_overlap",
    "zero_duration", "collapsed_insert", "nonfinite", "invalid_index", "duplicate_index", "foreign_word", "unaccepted_insert",
])
def test_unsafe_or_inexact_interruption_preserves_the_complete_parent(fault):
    cues, words, alignment = _episode_11_interruption()
    protected = set()
    if fault == "multiline":
        cues[0] = cues[0].with_lines(["se uma garota", "não recebe flores de ninguém,"])
    elif fault == "markup":
        cues[0] = cues[0].with_lines(["<i>se uma garota não recebe flores de ninguém,</i>"])
    elif fault == "bracket":
        cues[0] = cues[0].with_lines(["[se uma garota não recebe flores de ninguém,]"])
    elif fault == "song":
        cues[0] = cues[0].with_lines(["♪ se uma garota não recebe flores de ninguém, ♪"])
    elif fault == "nonexact":
        cues[0] = cues[0].with_lines(["se uma garota nunca recebe flores de ninguém,"])
    elif fault == "spacing":
        cues[0] = cues[0].with_lines(["se  uma garota não recebe flores de ninguém,"])
    elif fault == "protected_parent":
        protected = {217}
    elif fault == "protected_insert":
        protected = {945}
    elif fault == "missing_parent":
        alignment.diagnostics.missing_audio_cue_ids = [217]
    elif fault == "missing_insert":
        alignment.diagnostics.missing_audio_cue_ids = [945]
    elif fault == "shared_parent":
        alignment.cue_word_indices[999] = [2]
    elif fault == "shared_insert":
        alignment.cue_word_indices[999] = [3]
    elif fault == "unknown_parent":
        words[2] = words[2].model_copy(update={"speaker_id": None})
    elif fault == "unknown_insert":
        words[3] = words[3].model_copy(update={"speaker_id": None})
    elif fault == "same_speaker":
        words[3] = words[3].model_copy(update={"speaker_id": "chunk_1:0"})
    elif fault == "different_scope":
        words[3] = words[3].model_copy(update={"speaker_id": "chunk_2:1"})
    elif fault == "mixed_parent":
        words[-1] = words[-1].model_copy(update={"speaker_id": "chunk_1:1"})
    elif fault == "prefix_overlap":
        words[2] = words[2].model_copy(update={"end": 2.140})
    elif fault == "suffix_overlap":
        words[4] = words[4].model_copy(update={"start": 2.440})
    elif fault == "internal_overlap":
        words[0] = words[0].model_copy(update={"end": 1.750})
    elif fault == "zero_duration":
        words[1] = words[1].model_copy(update={"end": words[1].start})
    elif fault == "collapsed_insert":
        words[3] = words[3].model_copy(update={"end": words[3].start + .001})
    elif fault == "nonfinite":
        words[1] = words[1].model_copy(update={"end": float("inf")})
    elif fault == "invalid_index":
        alignment.cue_word_indices[217].append(99)
    elif fault == "duplicate_index":
        alignment.cue_word_indices[217].append(2)
    elif fault == "foreign_word":
        words.append(Word(text="outro", start=2.105, end=2.115, speaker_id="chunk_1:9"))

    before = alignment.model_dump()
    generated = set() if fault == "unaccepted_insert" else {945}
    output, updated, flags, expansions = cue_segmentation.split_at_generated_interruptions(
        cues, words, alignment, generated, StyleProfile(fps=30, min_cue_dur=.2), protected_cue_ids=protected,
    )
    assert output == cues
    assert updated.model_dump() == before
    assert flags == []
    assert expansions == {}


def test_new_child_id_cannot_overwrite_an_alignment_owner_outside_the_cue_list():
    cues, words, alignment = _episode_11_interruption()
    alignment.cue_word_indices[946] = []
    _, updated, _, expansions = _split(cues, words, alignment)
    assert expansions == {217: [217, 947]}
    assert updated.cue_word_indices[946] == []


@pytest.mark.parametrize("mode", ["fresh", "cached", "rebuild", "verify"])
def test_interruption_reaches_fresh_cached_and_resumed_output(tmp_path, monkeypatch, mode):
    def no_network(*_args, **_kwargs):
        raise AssertionError("The captured interruption regression must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    cues, words, _ = _episode_11_interruption()
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 5)
    source = tmp_path / "source.srt"
    source.write_text(write_srt(cues[:1]), encoding="utf-8")
    word_path = tmp_path / "words.json"
    word_path.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": [
        {"start": 1.600, "end": 2.099}, {"start": 2.120, "end": 2.460}, {"start": 2.480, "end": 3.759},
    ]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(word_path)},
        "vad": {"fixture_path": str(regions), "boundary_refinement": {"start_pad_ms": 0, "end_pad_ms": 0}},
        "llm": {"provider": "fixture", "responses": {"case-1": {
            "case_id": "case-1", "verdict": "use_audio", "final_text": "Hã?", "confidence": 1,
            "evidence": "heard_clearly", "heard_text": "Hã?", "reason": "Captured native interjection.",
        }}},
    }, allow_unicode=True), encoding="utf-8")
    options = dict(
        srt_path=source, audio_path=audio, output_path=tmp_path / "output.srt",
        workdir=tmp_path / "work", providers_path=config,
        style_profile=StyleProfile(fps=30, min_cue_dur=.2, max_chars_per_line=70, tail_ms=0),
    )
    result = pipeline.sync_episode(**options)
    if mode != "fresh":
        result = pipeline.sync_episode(**options, **({"resume": mode} if mode != "cached" else {}))
    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.lines for cue in output] == [["se uma garota"], ["Hã?"], ["não recebe flores de ninguém,"]]
    assert all(left.end_ms <= right.start_ms for left, right in zip(output, output[1:]))
    assert [cue.index for cue in output] == [1, 2, 3]
    for cue, (start, end) in zip(output, [(1.600, 2.099), (2.120, 2.460), (2.480, 3.759)], strict=True):
        assert abs(cue.start_ms - start * 1000) <= 1000 / 30 + 1
        assert abs(cue.end_ms - end * 1000) <= 1000 / 30 + 1
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    ownership = rebuilt["alignment"]["cue_word_indices"]
    assert [ownership[str(cue["index"])] for cue in rebuilt["cues"]] == [[0, 1, 2], [3], [4, 5, 6, 7, 8]]
    assert sorted(index for indices in ownership.values() for index in indices) == list(range(9))
    flags = result.report["flags"]
    assert any(flag["kind"] == "speaker_turn_split" for flag in flags)
    assert not any(flag["kind"] in {"source_order_inversion", "output_order_inversion", "timing_evidence_held"} for flag in flags)
    split_changes = [change for change in result.report["changes"] if change["kind"] == "speaker_turn_split"]
    assert [(change["change"], change["srt_number"], change["new_text"]) for change in split_changes] == [
        ("edited", 1, "se uma garota"), ("added", 3, "não recebe flores de ninguém,"),
    ]
    assert all("interruption" in change["reason"] for change in split_changes)
