"""SRT text and ASR provider words are compared with the same tokenisation."""

from __future__ import annotations

import pytest

from dubsync import aligner
from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, Word
from dubsync.tokenize import normalize_token


def _words(texts: list[str], *, start: float = 1.0, step: float = 0.35) -> list[Word]:
    return [
        Word(text=text, start=round(start + index * step, 3), end=round(start + index * step + 0.3, 3), confidence=None)
        for index, text in enumerate(texts)
    ]


def _cue(text: str, index: int = 1) -> Cue:
    return Cue(index=index, start_ms=1_000, end_ms=4_000, lines=[text])


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("Wie geht's dir heute?", ["Wie", "geht's", "dir", "heute?"]),
        ("Ich hab’s dir gesagt.", ["Ich", "hab's", "dir", "gesagt."]),
        ("Wär's nicht schön?", ["Wär's", "nicht", "schön?"]),
        ("J'ai vu l'homme d'ici.", ["J'ai", "vu", "l'homme", "d'ici."]),
        ("C'est aujourd'hui, qu'on part.", ["C'est", "aujourd'hui,", "qu'on", "part."]),
        ("I don't know what it's about.", ["I", "don't", "know", "what", "it's", "about."]),
        ("Das sind 10% mehr Leute.", ["Das", "sind", "10%", "mehr", "Leute."]),
        ("Wir treffen uns um 8:30 Uhr.", ["Wir", "treffen", "uns", "um", "8:30", "Uhr."]),
        ("Das sind 2,5 Liter Wasser.", ["Das", "sind", "2,5", "Liter", "Wasser."]),
        ("Es kostet 1.000 Euro genau.", ["Es", "kostet", "1.000", "Euro", "genau."]),
        ("Er fuhr 120 km/h schnell.", ["Er", "fuhr", "120", "km/h", "schnell."]),
        ("Zum Beispiel z.B. heute.", ["Zum", "Beispiel", "z.B.", "heute."]),
        ("Guten Morgen zusammen.", ["Guten Morgen", "zusammen."]),
    ],
)
def test_provider_words_split_like_source_tokens(source, spoken):
    words = _words(spoken)

    result = align_cues_to_words([_cue(source)], words)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: list(range(len(words)))}
    assert sorted({match.asr_word_index for match in result.token_matches}) == list(range(len(words)))


def test_untimed_spaced_punctuation_words_open_no_insertion_case():
    words = [
        Word(text="Tu", start=1.0, end=1.2),
        Word(text="viens", start=1.25, end=1.6),
        Word(text="?", start=1.6, end=1.601),
        Word(text="Oui", start=2.0, end=2.2),
        Word(text="!", start=2.2, end=2.201),
    ]

    result = align_cues_to_words([_cue("Tu viens ? Oui !")], words)

    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1, 3]}


def test_partially_matched_provider_word_stays_atomic_in_its_span():
    words = _words(["Wie", "geht's", "dir", "heute?"])

    result = align_cues_to_words([_cue("Wie geht es dir heute?")], words)

    assert len(result.divergence_spans) == 1
    span = result.divergence_spans[0]
    assert span.srt_text == "geht es"
    # The span carries the provider word, not its comparison units.
    assert span.asr_text == "geht's"
    assert span.asr_word_indices == [1]
    assert result.cue_word_indices == {1: [0, 2, 3]}
    assert all(match.asr_word_index != 1 for match in result.token_matches)


def test_split_word_spans_keep_provider_text_and_envelope():
    words = _words(["Sie", "sagte", "c'est", "fini."])

    result = align_cues_to_words([_cue("Sie sagte das ist fini.")], words)

    span, = result.divergence_spans
    assert span.srt_text == "das ist"
    assert span.asr_text == "c'est"
    assert span.asr_word_indices == [2]
    assert (span.start, span.end) == (words[2].start, words[2].end)


@pytest.mark.parametrize(
    ("stutter", "plain"),
    [("N-não", "não"), ("De-deixa", "deixa"), ("eu-eu", "eu"), ("I-I", "I"), ("W-Warte", "Warte")],
)
def test_glued_stutter_has_the_same_alignment_key_as_the_word(stutter, plain):
    assert normalize_token(stutter) == normalize_token(plain)


@pytest.mark.parametrize("compound", ["E-Mail", "Level-1-Versager", "Ano-Novo", "Bio-Laden", "so-so-Typ"])
def test_hyphen_compounds_are_not_mistaken_for_stutters(compound):
    assert normalize_token(compound) == normalize_token(compound.replace("-", ""))


def test_hyphen_compound_in_source_owns_every_split_provider_word():
    words = _words(["Feliz", "Ano", "Novo!", "Vamos."])

    result = align_cues_to_words([_cue("Feliz Ano-Novo! Vamos.")], words)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: [0, 1, 2, 3]}


def test_split_source_words_share_one_hyphenated_provider_word():
    words = _words(["Das", "Drachen-Evolutionssystem", "ist", "da."])

    result = align_cues_to_words([_cue("Das Drachen Evolutionssystem ist da.")], words)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: [0, 1, 2, 3]}
    assert [match.asr_word_index for match in result.token_matches] == [0, 1, 1, 2, 3]


def test_hyphen_compound_never_gives_one_provider_word_to_two_cues():
    cues = [
        Cue(index=1, start_ms=1_000, end_ms=2_000, lines=["Feliz Ano"]),
        Cue(index=2, start_ms=2_000, end_ms=3_000, lines=["Novo para todos."]),
    ]
    words = _words(["Feliz", "Ano-Novo", "para", "todos."])

    result = align_cues_to_words(cues, words)

    span, = result.divergence_spans
    assert span.cue_ids == [1, 2]
    assert span.asr_word_indices == [1]
    assert all(1 not in indices for indices in result.cue_word_indices.values())


def test_rejected_cue_reopens_each_compound_word_once():
    cues = [_cue("Das Drachen Evolutionssystem")]
    tokens = aligner.tokenize_cues(cues)
    ops = [
        aligner._Op("match", 0, 0, 1.0),
        aligner._Op("compound", 1, 1, aligner.COMPOUND_MATCH_SCORE),
        aligner._Op("compound", 2, 1, aligner.COMPOUND_MATCH_SCORE),
    ]

    reopened = aligner._without_cue_matches(ops, tokens, {1})

    assert [(op.kind, op.srt_index, op.asr_index) for op in reopened] == [
        ("delete", 0, None), ("insert", None, 0),
        ("delete", 1, None), ("insert", None, 1),
        ("delete", 2, None),
    ]


def test_split_provider_word_counts_once_as_timing_evidence():
    # A Scribe word can absorb many seconds of trailing silence. Splitting it
    # into comparison units must not let it outvote the cue's other words.
    cues = [
        Cue(index=index + 1, start_ms=start, end_ms=start + 1_200, lines=[text])
        for index, (start, text) in enumerate([
            (1_000, "Guten Morgen alle zusammen."), (5_000, "Wir fahren heute weit weg."),
            (9_000, "Das wird ein schöner Tag."), (13_000, "Bring bitte die Karten mit."),
            (48_266, "Los geht’s."), (69_733, "Das Spiel ist vorbei."),
        ])
    ]
    texts = [
        (1.0, "Guten Morgen alle zusammen."), (5.0, "Wir fahren heute weit weg."),
        (9.0, "Das wird ein schöner Tag."), (13.0, "Bring bitte die Karten mit."),
        (69.733, "Das Spiel ist vorbei."),
    ]
    words = []
    for start, text in texts[:4]:
        words.extend(_words(text.split(), start=start, step=0.25))
    words.extend([Word(text="Los", start=48.299, end=48.5), Word(text="geht's.", start=48.599, end=67.379)])
    words.extend(_words(texts[4][1].split(), start=texts[4][0], step=0.25))

    result = align_cues_to_words(cues, words)

    assert 5 not in result.diagnostics.missing_audio_cue_ids
    assert not any(flag.kind == "alignment_outlier" and flag.severity == "error" for flag in result.flags)


def test_glued_stutter_does_not_create_an_artefact_span():
    words = _words(["N-não,", "eu", "não", "sei."])

    result = align_cues_to_words([_cue("Não, eu não sei.")], words)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0
