"""Open, hyphenated and closed spellings of the same words align without review."""

from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, Word


def _align(source: str, spoken: list[str]):
    words = [
        Word(text=text, start=round(1.0 + index * 0.35, 3), end=round(1.3 + index * 0.35, 3), confidence=None)
        for index, text in enumerate(spoken)
    ]
    return align_cues_to_words([Cue(index=1, start_ms=1_000, end_ms=5_000, lines=[source])], words), words


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("Meine Ehefrau kommt gleich.", ["Meine", "Ehe", "Frau", "kommt", "gleich."]),
        ("Das ist so weit weg.", ["Das", "ist", "soweit", "weg."]),
        ("Ich bin zu Hause, komm.", ["Ich", "bin", "Zuhause,", "komm."]),
        ("Das Verlobungs kleid ist da.", ["Das", "Verlobungskleid", "ist", "da."]),
        ("Se não vier, eu vou.", ["Senão", "vier,", "eu", "vou."]),
        ("Ele é o Yuanzhu, meu amigo.", ["Ele", "é", "o", "Yuan", "Zhu,", "meu", "amigo."]),
    ],
)
def test_closed_and_open_compounds_match_their_other_spelling(source, spoken):
    result, words = _align(source, spoken)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: list(range(len(words)))}


def test_short_accidental_concatenation_stays_reviewable():
    result, _words = _align("Vou a o mercado.", ["Vou", "ao", "mercado."])

    span, = result.divergence_spans
    assert (span.srt_text, span.asr_text) == ("a o", "ao")
