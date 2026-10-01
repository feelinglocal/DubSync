"""Digits and scripted number words align across languages; distinct words stay distinct."""

from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, Word
from dubsync.tokenize import normalize_token


def _words(texts: list[str], *, start: float = 1.0, step: float = 0.35) -> list[Word]:
    return [
        Word(text=text, start=round(start + index * step, 3), end=round(start + index * step + 0.3, 3), confidence=None)
        for index, text in enumerate(texts)
    ]


def _align(source: str, spoken: list[str]):
    return align_cues_to_words([Cue(index=1, start_ms=1_000, end_ms=6_000, lines=[source])], _words(spoken))


@pytest.mark.parametrize(
    ("word", "digits"),
    [
        # Portuguese
        ("dois", "2"), ("duas", "2"), ("três", "3"), ("quatro", "4"), ("cinco", "5"), ("sete", "7"), ("oito", "8"),
        ("nove", "9"), ("dez", "10"), ("onze", "11"), ("treze", "13"), ("dezesseis", "16"), ("dezanove", "19"),
        ("vinte", "20"), ("trinta", "30"), ("cinquenta", "50"), ("cem", "100"), ("quinhentos", "500"), ("mil", "1000"),
        # Spanish
        ("cuatro", "4"), ("siete", "7"), ("diez", "10"), ("quince", "15"), ("dieciséis", "16"), ("veinticinco", "25"),
        ("cuarenta", "40"), ("cien", "100"), ("quinientos", "500"),
        # French
        ("deux", "2"), ("trois", "3"), ("cinq", "5"), ("huit", "8"), ("douze", "12"), ("dix-sept", "17"),
        ("vingt-deux", "22"), ("soixante-dix", "70"), ("quatre-vingts", "80"), ("quatre-vingt-dix", "90"), ("mille", "1000"),
        # English and German beyond the old tables
        ("fifteen", "15"), ("twenty-one", "21"), ("ninety", "90"), ("hundred", "100"), ("thousand", "1000"),
        ("hundertfünfzig", "150"), ("zweihundert", "200"), ("tausend", "1000"),
        ("neunzehnhundertneunundneunzig", "1999"), ("zweitausendzwanzig", "2020"),
    ],
)
def test_number_words_share_the_digit_key(word, digits):
    assert normalize_token(word) == digits


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("Um, dois, três, vamos!", ["1,", "2,", "3,", "vamos!"]),
        ("Tenho vinte e seis anos.", ["Tenho", "26", "anos."]),
        ("Custou mil quinhentos e vinte reais.", ["Custou", "1520", "reais."]),
        ("Subiu trinta por cento hoje.", ["Subiu", "30%", "hoje."]),
        ("Subiu 30% hoje.", ["Subiu", "trinta", "por", "cento", "hoje."]),
        ("O voo um nove zero saiu.", ["O", "voo", "190", "saiu."]),
        ("Son las once en punto.", ["Son", "las", "11", "en", "punto."]),
        ("J'ai vingt-deux ans.", ["J'ai", "22", "ans."]),
        ("Ich habe eine Frage.", ["Ich", "habe", "1", "Frage."]),
        ("Er wurde erster heute.", ["Er", "wurde", "1.", "heute."]),
        ("Das kostet 150 Euro.", ["Das", "kostet", "hundertfünfzig", "Euro."]),
        ("It costs two hundred dollars.", ["It", "costs", "200", "dollars."]),
    ],
)
def test_digits_align_with_scripted_number_words(source, spoken):
    result = _align(source, spoken)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: list(range(len(spoken)))}


@pytest.mark.parametrize(
    ("source", "spoken", "srt_text", "asr_text"),
    [
        ("Der erste Mann geht.", ["Der", "eine", "Mann", "geht."], "erste", "eine"),
        ("Es kostet eins mehr.", ["Es", "kostet", "einen", "mehr."], "eins", "einen"),
        ("Der zweite Versuch.", ["Der", "zwei", "Versuch."], "zweite", "zwei"),
        ("A avó dele chegou.", ["A", "avô", "dele", "chegou."], "avó", "avô"),
        ("Ela está aqui agora.", ["Ela", "esta", "aqui", "agora."], "está", "esta"),
        ("Es ist eins.", ["Es", "ist", "ein."], "eins", "ein."),
    ],
)
def test_words_the_language_keeps_distinct_stay_reviewable(source, spoken, srt_text, asr_text):
    result = _align(source, spoken)

    span, = result.divergence_spans
    assert (span.srt_text, span.asr_text) == (srt_text, asr_text)


def test_countdown_with_re_decoded_digit_copies_keeps_each_cue_whole():
    # MAI ep11: "Dez. / Nove. / Oito." heard as "10, 10, 9, 9, 8, 8," with touching copies.
    cues = [
        Cue(index=1, start_ms=1_000, end_ms=1_900, lines=["Dez."]),
        Cue(index=2, start_ms=2_000, end_ms=2_900, lines=["Nove."]),
        Cue(index=3, start_ms=3_000, end_ms=3_900, lines=["Oito."]),
    ]
    words = [
        Word(text="10,", start=1.00, end=1.18), Word(text="10,", start=1.20, end=1.70),
        Word(text="9,", start=2.00, end=2.18), Word(text="9,", start=2.20, end=2.70),
        Word(text="8,", start=3.00, end=3.18), Word(text="8.", start=3.20, end=3.70),
    ]

    result = align_cues_to_words(cues, words)

    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1], 2: [2, 3], 3: [4, 5]}


def test_repetition_after_a_pause_stays_reviewable():
    cues = [Cue(index=1, start_ms=1_000, end_ms=3_000, lines=["Não, eu vou."])]
    words = [
        Word(text="Não,", start=1.0, end=1.2), Word(text="não,", start=1.6, end=1.8),
        Word(text="eu", start=1.9, end=2.0), Word(text="vou.", start=2.05, end=2.4),
    ]

    result = align_cues_to_words(cues, words)

    span, = result.divergence_spans
    assert span.srt_text == "" and normalize_token(span.asr_text) == normalize_token("não")
    assert len(span.asr_word_indices) == 1


def test_article_inflections_share_a_key_that_is_not_a_number():
    keys = {normalize_token(article) for article in ("ein", "eine", "einen", "einem", "einer", "eines")}

    assert len(keys) == 1
    assert keys.isdisjoint({normalize_token("eins"), normalize_token("1"), normalize_token("erste")})


def test_orthographic_accent_variants_still_match():
    assert normalize_token("júnior") == normalize_token("junior")
    assert normalize_token("à") == normalize_token("a")
    assert normalize_token("Où") == normalize_token("ou")
