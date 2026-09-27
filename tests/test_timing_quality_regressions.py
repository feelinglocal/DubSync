from __future__ import annotations

import pytest

from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import refine_cues_to_speech_activity


def test_inconsistent_adlib_word_window_cannot_reverse_a_valid_cue():
    # Episode 11's inserted "Eu" was moved after its actual word by an earlier
    # rebuild stage. Refinement must hold and report that conflict, not publish
    # the 24:51.800 -> 24:50.666 reversal seen in the supplied output.
    cue = Cue(index=951, start_ms=1_491_800, end_ms=1_492_300, lines=["Eu"])
    words = [Word(text="Eu", start=1_490.19, end=1_490.55, speaker_id="speaker_1")]
    refined, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1_490.1, end=1_490.62)],
        StyleProfile(fps=30, min_cue_dur=0.5), words=words,
        alignment=AlignmentResult(cue_word_indices={951: [0]}),
    )
    assert refined == [cue]
    assert cue.end_ms > cue.start_ms
    held = [flag for flag in flags if flag.kind == "timing_refinement_held"]
    assert len(held) == 1 and held[0].cue_ids == [951]
    assert held[0].severity == "error"
    assert held[0].start == 1_491.8 and held[0].end == 1_492.3


@pytest.mark.parametrize("speech_end", [2.2, 6.0])
def test_other_speech_in_vad_region_cannot_extend_last_owned_word(speech_end):
    cue = Cue(index=1, start_ms=1000, end_ms=3500, lines=["Uau, que lindo!"])
    words = [
        Word(text="Uau,", start=1.05, end=1.3, speaker_id="speaker_1"),
        Word(text="que", start=1.35, end=1.5, speaker_id="speaker_1"),
        Word(text="lindo!", start=1.55, end=1.8, speaker_id="speaker_1"),
        Word(text="Outra", start=2.0, end=2.2, speaker_id="speaker_2"),
    ]
    refined, _ = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=speech_end)],
        StyleProfile(fps=30, min_cue_dur=0.5), words=words,
        alignment=AlignmentResult(cue_word_indices={1: [0, 1, 2]}),
    )
    assert 1800 <= refined[0].end_ms <= 1867
    assert refined[0].lines == cue.lines


def test_minimum_display_duration_cannot_restore_another_speakers_audio_tail():
    cue = Cue(index=1, start_ms=1000, end_ms=1600, lines=["Hã?"])
    words = [Word(text="Hã?", start=1.04, end=1.2, speaker_id="speaker_1")]
    refined, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=3.0)],
        StyleProfile(fps=30, min_cue_dur=0.5), words=words,
        alignment=AlignmentResult(cue_word_indices={1: [0]}),
    )
    assert 1200 <= refined[0].end_ms <= 1267
    assert any(flag.kind == "min_duration_unattainable" for flag in flags)


@pytest.mark.parametrize("noise_start", [2.0, 2.1, 2.2, 2.1004, 2.1006])
def test_float_roundoff_cannot_turn_duration_only_clamp_into_speech_evidence(noise_start):
    cue = Cue(index=1, start_ms=900, end_ms=1400, lines=["Drache!"])
    words = [Word(text="Drache!", start=.92, end=1.3),
             Word(text="noise", start=noise_start, end=noise_start + 10)]
    refined, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=.9, end=noise_start + 10)],
        StyleProfile(fps=30), words=words,
        alignment=AlignmentResult(cue_word_indices={1: [0, 1]}),
    )
    assert refined[0].end_ms <= 1400
    assert not any(flag.kind == "asr_word_clamped" for flag in flags)
