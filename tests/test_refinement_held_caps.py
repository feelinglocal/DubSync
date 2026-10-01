from __future__ import annotations

import json
import wave
from array import array

import pytest
import yaml

from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.pipeline import sync_episode
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import refine_cues_to_speech_activity


@pytest.mark.parametrize("hold_kind", ["protected", "missing_audio", "fixed"])
@pytest.mark.parametrize("with_words", [True, False])
def test_minimum_duration_padding_stops_before_held_dialogue(hold_kind, with_words):
    first = Cue(index=1, start_ms=1000, end_ms=1267, lines=["Oi."])
    held = Cue(index=2, start_ms=1300, end_ms=2600, lines=["Hahaha!"])
    later = Cue(index=3, start_ms=3000, end_ms=4000, lines=["Vamos embora."])
    alignment = AlignmentResult(
        cue_word_indices={1: [0]},
        diagnostics={"missing_audio_cue_ids": [2] if hold_kind == "missing_audio" else []},
    )

    refined, flags = refine_cues_to_speech_activity(
        [first, held, later],
        [SpeechRegion(start=1.0, end=1.2), SpeechRegion(start=3.0, end=3.9)],
        StyleProfile(fps=30, min_cue_dur=0.5),
        words=[Word(text="Oi.", start=1.0, end=1.2)] if with_words else None,
        alignment=alignment,
        protected_cue_ids={2} if hold_kind == "protected" else set(),
        fixed_cue_ids={2} if hold_kind == "fixed" else set(),
    )

    assert refined[0].start_ms == first.start_ms
    assert 1200 <= refined[0].end_ms <= held.start_ms
    assert refined[1] == held
    assert any(flag.kind == "min_duration_unattainable" and flag.cue_ids == [1] for flag in flags)
    assert not any(flag.cue_ids == [2] for flag in flags)


def test_held_dialogue_cap_does_not_clip_a_real_spoken_overlap():
    first = Cue(index=1, start_ms=1000, end_ms=1467, lines=["Oi."])
    held = Cue(index=2, start_ms=1300, end_ms=2600, lines=["Hahaha!"])

    refined, _ = refine_cues_to_speech_activity(
        [first, held],
        [SpeechRegion(start=1.0, end=1.4)],
        StyleProfile(fps=30, min_cue_dur=0.5),
        words=[Word(text="Oi.", start=1.0, end=1.4)],
        alignment=AlignmentResult(cue_word_indices={1: [0]}),
        protected_cue_ids={2},
    )

    assert refined[0].start_ms == first.start_ms
    assert refined[0].end_ms == 1400
    assert refined[1] == held


def test_held_screen_text_does_not_block_padding_before_held_dialogue():
    first = Cue(index=1, start_ms=1000, end_ms=1267, lines=["Oi."])
    screen = Cue(index=2, start_ms=1280, end_ms=2600, lines=["[Later]"])
    held = Cue(index=3, start_ms=1400, end_ms=2600, lines=["Hahaha!"])

    refined, _ = refine_cues_to_speech_activity(
        [first, screen, held],
        [SpeechRegion(start=1.0, end=1.2)],
        StyleProfile(fps=30, min_cue_dur=0.5),
        words=[Word(text="Oi.", start=1.0, end=1.2)],
        alignment=AlignmentResult(cue_word_indices={1: [0]}),
        protected_cue_ids={2, 3},
    )

    assert refined[0].end_ms == held.start_ms
    assert refined[1:] == [screen, held]


@pytest.mark.parametrize("policy", ["extend_into_silence", "acoustic"])
def test_full_sync_keeps_following_missing_audio_cue_start(tmp_path, policy):
    source = tmp_path / "episode.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:01,400\nOi.\n\n"
        "2\n00:00:01,300 --> 00:00:02,600\nHahaha!\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nVamos embora.\n",
        encoding="utf-8",
    )
    audio = tmp_path / "episode.wav"
    samples = array("h", [0]) * (5 * 16000)
    for start, end in [(1.0, 1.2), (3.0, 3.9)]:
        for index in range(int(start * 16000), int(end * 16000)):
            samples[index] = 9000 if index % 2 else -9000
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": "Oi.", "start": 1.0, "end": 1.2},
        {"text": "Vamos", "start": 3.0, "end": 3.3},
        {"text": "embora.", "start": 3.35, "end": 3.9},
    ]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)},
        "vad": {"provider": "energy", "boundary_refinement": True},
        "timing": {"min_duration_policy": policy},
        "output": {"no_overlaps": True},
    }), encoding="utf-8")
    output = tmp_path / "output.srt"

    result = sync_episode(
        source, audio, output, tmp_path / "work", providers_path=config, no_llm=True,
    )

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["Oi.", "Hahaha!", "Vamos embora."]
    assert cues[1].start_ms == 1300
    assert cues[1].end_ms == 2600
    assert cues[0].start_ms <= 1000
    assert 1200 <= cues[0].end_ms <= cues[1].start_ms
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues, cues[1:]))
    assert any(flag["kind"] == "missing_audio_timing_held" for flag in result.report["flags"])
