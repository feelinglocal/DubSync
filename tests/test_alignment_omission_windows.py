"""A mostly matched cue with one unspoken source word is not locked as missing audio."""

from __future__ import annotations

from dubsync import aligner
from dubsync.models import Cue, Word


def _words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=text, start=start, end=end, confidence=None, speaker_id="A") for text, start, end in spec]


SPOKEN = [("Hoje", 1.05, 1.30), ("eu", 1.30, 1.50), ("preciso", 1.50, 1.80), ("ir", 1.80, 2.00), ("embora", 2.00, 2.35)]


def test_unspoken_first_word_of_the_file_keeps_the_cue_acoustically_timed():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Olha, hoje eu preciso ir embora."])]

    alignment = aligner.align_cues_to_words(cues, _words(SPOKEN))

    assert alignment.diagnostics.missing_audio_cue_ids == []
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4]}
    omission, = alignment.divergence_spans
    assert (omission.srt_text, omission.asr_text) == ("Olha", "")
    assert (omission.start, omission.end) == (1.05, 2.35)
    assert not any(flag.kind == "missing_audio_timing_held" for flag in alignment.flags)


def test_unspoken_last_word_of_the_file_keeps_the_cue_acoustically_timed():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hoje eu preciso ir embora, viu?"])]

    alignment = aligner.align_cues_to_words(cues, _words(SPOKEN))

    assert alignment.diagnostics.missing_audio_cue_ids == []
    omission, = alignment.divergence_spans
    assert omission.srt_text == "viu"
    assert omission.end > omission.start


def test_omission_between_slightly_overlapping_words_is_not_locked():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hoje eu realmente preciso ir embora."])]
    spoken = [("Hoje", 1.05, 1.30), ("eu", 1.30, 1.52), ("preciso", 1.50, 1.80), ("ir", 1.80, 2.00), ("embora", 2.00, 2.35)]

    alignment = aligner.align_cues_to_words(cues, _words(spoken))

    assert alignment.diagnostics.missing_audio_cue_ids == []
    omission, = alignment.divergence_spans
    assert omission.srt_text == "realmente"
    assert omission.end > omission.start


def test_mostly_unmatched_cue_with_an_edge_omission_stays_locked():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Olha só, menina, hoje eu vou"])]

    alignment = aligner.align_cues_to_words(cues, _words(SPOKEN[:3]))

    assert alignment.diagnostics.missing_audio_cue_ids == [1]


def test_song_lyric_edge_omission_stays_locked():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["♪Olha, hoje eu preciso ir embora♪"])]

    alignment = aligner.align_cues_to_words(cues, _words(SPOKEN))

    assert alignment.diagnostics.missing_audio_cue_ids == [1]
