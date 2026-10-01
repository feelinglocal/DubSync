from __future__ import annotations

import json
import socket
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
from dubsync.recue import rebuild_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile


def _japanese_inputs(tmp_path, monkeypatch, *, evidence="heard_clearly", heard_text="あっ放"):
    """The native Scribe 1A cue 59/60 seam, shifted 112 seconds earlier."""
    def no_network(*_args, **_kwargs):
        raise AssertionError("The captured live-case regression must not use a network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 6)
    source = tmp_path / "source.srt"
    source.write_text(write_srt([
        Cue(index=1, start_ms=1360, end_ms=3330, lines=["うあっ！"]),
        Cue(index=2, start_ms=4130, end_ms=4930, lines=["放したぞ"]),
    ]), encoding="utf-8")
    words = [
        {"text": text, "start": start, "end": end, "speaker_id": speaker}
        for text, start, end, speaker in [
            ("う", 1.040, 1.160, "speaker_1"),
            ("わ", 1.240, 1.260, "speaker_1"),
            ("ぁ", 1.400, 1.460, "speaker_1"),
            ("。", 1.460, 1.461, "speaker_1"),
            ("話", 3.620, 3.680, "speaker_2"),
            ("し", 3.680, 3.760, "speaker_2"),
            ("た", 3.760, 3.840, "speaker_2"),
            ("ぞ", 3.860, 3.940, "speaker_2"),
            ("。", 3.940, 3.941, "speaker_2"),
        ]
    ]
    word_path = tmp_path / "words.json"
    word_path.write_text(json.dumps({"words": words}), encoding="utf-8")
    region_path = tmp_path / "regions.json"
    region_path.write_text(json.dumps({"regions": [
        {"start": .995, "end": 1.465}, {"start": 3.365, "end": 4.095},
    ]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    decision = {
        "case_id": "case-1", "verdict": "keep_srt", "final_text": "あっ放",
        "confidence": 1.0,
        "reason": "Spoken はなしたぞ matches source 放したぞ; ASR transcribed the homophone 話.",
    }
    if evidence is not None:
        decision.update(evidence=evidence, heard_text=heard_text)
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(word_path)},
        "vad": {"fixture_path": str(region_path), "boundary_refinement": True},
        "llm": {"provider": "fixture", "responses": {"case-1": decision}},
    }, allow_unicode=True), encoding="utf-8")
    return dict(
        srt_path=source, audio_path=audio, output_path=tmp_path / "output.srt",
        workdir=tmp_path / "work", providers_path=config,
        style_profile=StyleProfile(fps=30, min_cue_dur=.5, tail_ms=40),
    )


@pytest.mark.parametrize("resume", [None, "rebuild", "verify"])
def test_explicit_heard_fragment_and_unique_burst_retime_unchanged_source(tmp_path, monkeypatch, resume):
    options = _japanese_inputs(tmp_path, monkeypatch)
    result = pipeline.sync_episode(**options)
    if resume:
        result = pipeline.sync_episode(**options, resume=resume)

    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.lines for cue in cues] == [["うあっ！"], ["放したぞ"]]
    assert cues[0].start_ms <= 995
    assert 1465 <= cues[0].end_ms <= 1534
    assert not any(flag["kind"] == "timing_evidence_held" and 1 in flag["cue_ids"]
                   for flag in result.report["flags"])


@pytest.mark.parametrize("evidence, heard", [
    (None, None), ("heard_unclear", "あっ放"), ("heard_clearly", "放"),
])
def test_unconfirmed_fragment_still_preserves_source_timing(tmp_path, monkeypatch, evidence, heard):
    result = pipeline.sync_episode(**_japanese_inputs(
        tmp_path, monkeypatch, evidence=evidence, heard_text=heard,
    ))
    cue = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))[0]
    assert (cue.start_ms, cue.end_ms, cue.text) == (1360, 3330, "うあっ！")


def _owned_heard_case():
    sources = [
        Cue(index=1, start_ms=1360, end_ms=3330, lines=["うあっ！"]),
        Cue(index=2, start_ms=4130, end_ms=4930, lines=["放したぞ"]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("う", .995, 1.160), ("わ", 1.240, 1.260), ("ぁ", 1.400, 1.465),
        ("話", 3.620, 3.680), ("し", 3.680, 3.760), ("た", 3.760, 3.840), ("ぞ", 3.860, 3.940),
    ]]
    span = DivergenceSpan(
        case_id="case-29", cue_ids=[1, 2], srt_text="あっ放", asr_text="わぁ話",
        srt_token_indices=[1, 2, 3], asr_word_indices=[1, 2, 3],
    )
    alignment = AlignmentResult(
        cue_word_indices={1: [0, 1, 2], 2: [3, 4, 5, 6]}, divergence_spans=[span],
        token_matches=[TokenMatch(cue_id=cue_id, srt_token_index=index, asr_word_index=index, score=1)
                       for cue_id, index in [(1, 0), (2, 4), (2, 5), (2, 6)]],
    )
    decision = AdjudicationDecision(
        case_id="case-29", verdict="keep_srt", final_text="あっ放", confidence=1,
        evidence="heard_clearly", heard_text="あっ放", reason="Native heard source fragment.",
    )
    return dict(
        cues=list(sources), source_cues=sources, words=words, alignment=alignment,
        decisions=[decision], regions=[SpeechRegion(start=.995, end=1.465), SpeechRegion(start=3.365, end=4.095)],
        protected_cue_ids=set(),
    )


def test_semantic_confirmation_is_bound_to_owned_source_tokens_without_changing_inputs():
    case = _owned_heard_case()
    original_words = [word.model_dump() for word in case["words"]]
    original_cues = [cue.model_dump() for cue in case["cues"]]
    assert pipeline._confirmed_source_wording_cue_ids(**case) == {1, 2}
    assert [word.model_dump() for word in case["words"]] == original_words
    assert [cue.model_dump() for cue in case["cues"]] == original_cues


@pytest.mark.parametrize("fault", [
    "no_regions", "multiple_bursts", "foreign_word", "shared_word", "no_anchor",
    "partial_source", "wrong_source_indices", "changed_wording", "protected",
    "unresolved", "word_outside_burst", "missing_word_ownership",
])
def test_semantic_wording_cannot_supply_missing_acoustic_ownership(fault):
    case = _owned_heard_case()
    alignment = case["alignment"]
    if fault == "no_regions":
        case["regions"] = []
    elif fault == "multiple_bursts":
        case["regions"][:1] = [SpeechRegion(start=.995, end=1.170), SpeechRegion(start=1.220, end=1.465)]
    elif fault == "foreign_word":
        case["words"].append(Word(text="Hey", start=1.300, end=1.350))
    elif fault == "shared_word":
        alignment.cue_word_indices[9] = [1]
    elif fault == "no_anchor":
        alignment.token_matches = [match for match in alignment.token_matches if match.cue_id != 1]
    elif fault == "partial_source":
        alignment.divergence_spans[0] = alignment.divergence_spans[0].model_copy(update={
            "srt_token_indices": [2, 3], "srt_text": "っ放",
        })
        case["decisions"][0] = case["decisions"][0].model_copy(update={"heard_text": "っ放", "final_text": "っ放"})
    elif fault == "wrong_source_indices":
        alignment.divergence_spans[0].srt_token_indices = [1, 2, 4]
    elif fault == "changed_wording":
        case["cues"][0] = case["cues"][0].with_lines(["うわぁ！"])
    elif fault == "protected":
        case["protected_cue_ids"] = {1}
    elif fault == "unresolved":
        alignment.diagnostics.unresolved = True
    elif fault == "word_outside_burst":
        case["words"][0].start = .900
    elif fault == "missing_word_ownership":
        alignment.cue_word_indices[1] = [0, 1]
    assert 1 not in pipeline._confirmed_source_wording_cue_ids(**case)


@pytest.mark.parametrize("fault", ["collapsed", "invalid", "ambiguous", "shared"])
def test_confirmed_wording_never_disables_timestamp_safety(fault):
    case = _owned_heard_case()
    words = case["words"]
    if fault == "collapsed":
        for index in range(3):
            words[index] = words[index].model_copy(update={"start": 1.0 + index * .01, "end": 1.001 + index * .01})
    elif fault == "invalid":
        words[1] = words[1].model_copy(update={"end": float("nan")})
    elif fault == "shared":
        case["alignment"].cue_word_indices[2].append(1)
    rebuilt, flags = rebuild_cues(
        case["cues"], words, case["alignment"], StyleProfile(fps=30),
        confirmed_wording_cue_ids={1}, ambiguous_word_indices={1} if fault == "ambiguous" else set(),
    )
    assert rebuilt[0] == case["cues"][0]
    assert any(flag.kind in {"timing_evidence_held", "shared_word_timing_preserved"} and 1 in flag.cue_ids for flag in flags)
