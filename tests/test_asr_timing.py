from __future__ import annotations

from dubsync.asr_timing import clamp_asr_word_durations
from dubsync.models import SpeechRegion, Word


def test_clamp_asr_word_durations_enforces_max_even_inside_long_region():
    words = [Word(text="stretched", start=0.0, end=18.0, confidence=0.9)]
    regions = [SpeechRegion(start=0.0, end=20.0)]

    clamped, flags = clamp_asr_word_durations(words, regions, max_word_duration=2.0)

    assert clamped[0].end == 2.0
    assert flags[0].kind == "asr_word_clamped"
    assert flags[0].old_text == "stretched 0.000 --> 18.000"


def test_clamp_asr_word_endpoint_that_overruns_speech_region_limit():
    words = [Word(text="Drachenei", start=1.94, end=3.12, confidence=0.9)]
    regions = [SpeechRegion(start=1.6, end=2.6)]

    clamped, flags = clamp_asr_word_durations(
        words,
        regions,
        max_word_duration=2.0,
    )

    assert clamped[0].end == 2.6
    assert flags[0].kind == "asr_word_clamped"
    assert "speech region" in flags[0].message
