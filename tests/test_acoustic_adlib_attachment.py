from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, SpeechRegion
from dubsync.pipeline import _adlib_cue_ids_by_case, sync_episode
from dubsync.srt_io import parse_srt_text


def _attachment(regions, *, gap=0.201, speaker=None, anchor_speaker=None, left=False):
    cue = Cue(index=72, start_ms=234400, end_ms=235800, lines=["Qual é o seu plano?"])
    span = DivergenceSpan(
        case_id="case-20", cue_ids=[], srt_text="", asr_text="E",
        start=234.56, end=234.639, asr_word_indices=[219],
        speaker_ids=[] if speaker is None else [speaker],
        left_anchor_cue_id=72 if left else None,
        left_anchor_end=234.56 - gap if left else None,
        left_anchor_speaker_id=anchor_speaker if left else None,
        right_anchor_cue_id=None if left else 72,
        right_anchor_start=None if left else 234.639 + gap,
        right_anchor_speaker_id=None if left else anchor_speaker,
    )
    if left:
        cue = cue.with_lines(["Eu disse"])
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="E", confidence=0.95,
        reason="Fixture: the conjunction is audible.",
    )
    return _adlib_cue_ids_by_case([cue], [span], [decision], [], speech_regions=regions)[0]


def test_episode_17_tight_mai_word_ends_attach_inside_one_speech_burst():
    assert _attachment([SpeechRegion(start=234.475, end=235.215)]) == {"case-20": 72}


@pytest.mark.parametrize("speaker,anchor", [(None, None), ("actor", "actor"), ("chunk_1:A", "chunk_2:A")])
def test_same_burst_attachment_accepts_same_or_unknown_speaker_relation(speaker, anchor):
    assert _attachment([SpeechRegion(start=234.475, end=235.215)], speaker=speaker, anchor_speaker=anchor) == {"case-20": 72}


def test_same_burst_cannot_attach_to_a_known_different_actor():
    assert _attachment([SpeechRegion(start=234.475, end=235.215)], speaker="A", anchor_speaker="B") == {"case-20": 73}


@pytest.mark.parametrize("regions", [[], [SpeechRegion(start=234.475, end=234.65), SpeechRegion(start=234.72, end=235.215)]])
def test_detected_silence_does_not_attach_even_across_a_short_word_gap(regions):
    assert _attachment(regions, gap=0.1) == {"case-20": 73}


def test_contiguous_regions_covering_the_gap_are_one_acoustic_continuation():
    assert _attachment([SpeechRegion(start=234.475, end=234.7), SpeechRegion(start=234.7, end=235.215)]) == {"case-20": 72}


def test_acoustic_attachment_also_extends_an_unfinished_previous_phrase():
    assert _attachment([SpeechRegion(start=234.1, end=235.215)], gap=0.3, left=True) == {"case-20": 72}


def test_no_detector_retains_the_existing_word_gap_fallback():
    assert _attachment(None, gap=0.2) == {"case-20": 72}
    assert _attachment(None, gap=0.201) == {"case-20": 73}


def test_a_saturated_region_cannot_join_an_interjection_across_a_long_pause():
    assert _attachment([SpeechRegion(start=200, end=280)], gap=12) == {"case-20": 73}


def test_sync_shares_detected_bursts_with_the_adlib_attachment_stage(tmp_path):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:01,300\nAntes.\n\n2\n00:00:02,400 --> 00:00:03,800\nQual é o plano?\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 4)
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"words": [
        {"text": "Antes.", "start": 1, "end": 1.2},
        {"text": "E", "start": 2.56, "end": 2.639},
        {"text": "Qual", "start": 2.84, "end": 3},
        {"text": "é", "start": 3.01, "end": 3.1},
        {"text": "o", "start": 3.12, "end": 3.2},
        {"text": "plano?", "start": 3.22, "end": 3.5},
    ]}), encoding="utf-8")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": [{"start": 1, "end": 1.2}, {"start": 2.475, "end": 3.55}]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(words)},
        "vad": {"fixture_path": str(regions)},
        "llm": {"provider": "fixture", "responses": {"case-1": {
            "case_id": "case-1", "verdict": "use_audio", "final_text": "E", "confidence": 0.95,
            "reason": "The conjunction is audible.",
        }}},
    }), encoding="utf-8")
    result = sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers)
    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))

    assert len(output) == 2
    assert output[1].plain_text.casefold() == "e qual é o plano?"
    assert output[1].start_ms <= 2560
    assert output[1].end_ms >= 3500
