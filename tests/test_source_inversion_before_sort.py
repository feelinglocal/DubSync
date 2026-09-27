from __future__ import annotations

import json
import wave

import yaml

import dubsync.pipeline as pipeline
from dubsync.models import AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.output_order import source_order_inversion_flags
from dubsync.srt_io import parse_srt_text, write_srt


def _sync_fixture(tmp_path, monkeypatch, cues, words, alignment, *, responses=None):
    source = tmp_path / "episode.srt"
    audio = tmp_path / "episode.wav"
    wordstream = tmp_path / "words.json"
    config = tmp_path / "providers.yaml"
    output = tmp_path / "result.srt"
    source.write_text(write_srt(cues), encoding="utf-8")
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(6 * 32000))
    wordstream.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"provider": "fixture", "responses": responses or {}},
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "align_cues_to_words", lambda _cues, _words: alignment)

    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=config,
        no_llm=not responses,
    )
    return result, parse_srt_text(output.read_text(encoding="utf-8"))


def test_pipeline_retains_genuine_source_inversion_qc_before_acoustic_sort(tmp_path, monkeypatch):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["First"]),
        Cue(index=2, start_ms=2000, end_ms=2500, lines=["Second"]),
    ]
    words = [
        Word(text="Second", start=1.0, end=1.5, speaker_id="B"),
        Word(text="First", start=2.0, end=2.5, speaker_id="A"),
    ]
    result, exported = _sync_fixture(
        tmp_path, monkeypatch, cues, words,
        AlignmentResult(cue_word_indices={1: [1], 2: [0]}, anchor_coverage=1.0),
    )

    assert [cue.plain_text for cue in exported] == ["Second", "First"]
    inversion, = [flag for flag in result.report["flags"] if flag["kind"] == "output_order_inversion"]
    assert inversion["cue_ids"] == [1, 2]
    assert inversion["severity"] == "error"
    assert inversion["old_text"] == "1: First\n2: Second"


def test_accepted_earlier_eu_adlib_is_sorted_without_false_source_inversion(tmp_path, monkeypatch):
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["First"]),
        Cue(index=2, start_ms=3000, end_ms=4000, lines=["Second"]),
    ]
    words = [
        Word(text="Eu", start=1.5, end=1.7, speaker_id="A"),
        Word(text="First", start=2.0, end=2.5, speaker_id="A"),
        Word(text="Second", start=4.0, end=4.5, speaker_id="B"),
    ]
    addition = DivergenceSpan(
        case_id="eu", cue_ids=[], srt_text="", asr_text="Eu", start=1.5, end=1.7,
        asr_word_indices=[0], speaker_ids=["A"], confidence=1.0,
    )
    result, exported = _sync_fixture(
        tmp_path, monkeypatch, cues, words,
        AlignmentResult(cue_word_indices={1: [1], 2: [2]}, divergence_spans=[addition], anchor_coverage=1.0),
        responses={"eu": {
            "case_id": "eu", "verdict": "use_audio", "final_text": "Eu",
            "confidence": 1.0, "speaker": "A", "reason": "Confirmed local spoken addition",
        }},
    )

    assert [cue.plain_text for cue in exported] == ["Eu", "First", "Second"]
    assert all(cue.duration_ms > 0 for cue in exported)
    assert any(flag["kind"] == "adlib_inserted" for flag in result.report["flags"])
    assert not any(flag["kind"] == "output_order_inversion" for flag in result.report["flags"])


def test_source_order_check_excludes_speaker_children_and_generated_insertions():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["First actor"]),
        Cue(index=3, start_ms=3000, end_ms=3500, lines=["Speaker child"]),
        Cue(index=4, start_ms=500, end_ms=700, lines=["Eu"]),
        Cue(index=2, start_ms=2000, end_ms=2500, lines=["Second source cue"]),
    ]

    assert source_order_inversion_flags(cues, source_cue_ids={1, 2}) == []


def test_source_order_check_matches_protected_and_screen_text_policy():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["First actor"]),
        Cue(index=2, start_ms=5000, end_ms=5500, lines=["Source hold"]),
        Cue(index=3, start_ms=0, end_ms=500, lines=["[Screen text]"]),
        Cue(index=4, start_ms=2000, end_ms=2500, lines=["Second actor"]),
    ]

    assert source_order_inversion_flags(
        cues, source_cue_ids={1, 2, 3, 4}, protected_cue_ids={2},
    ) == []
