"""Inline subtitle markup is never aligned as speech."""

from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, Word
from dubsync.style_profile import StyleProfile
from dubsync.text_metrics import token_character_spans, token_texts
from dubsync.tokenize import normalize_token


def _words(texts: list[str], *, start: float = 1.0, step: float = 0.35) -> list[Word]:
    return [
        Word(text=text, start=round(start + index * step, 3), end=round(start + index * step + 0.3, 3))
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<i>Hallo Welt, wie geht es?</i>", ["Hallo", "Welt", "wie", "geht", "es"]),
        ('<font color="#ffff00">Danke sehr</font>', ["Danke", "sehr"]),
        ("{\\an8}Mir geht es gut.", ["Mir", "geht", "es", "gut"]),
        ("<b>Nein</b>, <u>nie</u>!", ["Nein", "nie"]),
        ("<v Anna>Komm her.</v>", ["Komm", "her"]),
        ("a < b und c > d", ["a", "b", "und", "c", "d"]),
    ],
)
def test_inline_markup_produces_no_tokens(text, expected):
    assert token_texts(text) == expected


def test_token_spans_skip_markup_attributes():
    text = '<font color="red">red</font> car'

    spans = token_character_spans(text)

    assert [text[start:end] for start, end in spans] == ["red", "car"]


def test_tagged_source_aligns_like_plain_text():
    cues = [
        Cue(index=1, start_ms=1_000, end_ms=3_000, lines=["<i>Hallo Welt, wie geht es?</i>"]),
        Cue(index=2, start_ms=3_000, end_ms=4_000, lines=['<font color="#ffff00">Danke sehr</font>']),
        Cue(index=3, start_ms=4_000, end_ms=6_000, lines=["{\\an8}Mir geht es gut."]),
    ]
    words = _words(["Hallo", "Welt,", "wie", "geht", "es?", "Danke", "sehr.", "Mir", "geht", "es", "gut."])

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.diagnostics.missing_audio_cue_ids == []
    assert not any(flag.kind == "missing_audio_timing_held" for flag in result.flags)


def test_edit_inside_tagged_cue_keeps_the_markup():
    cues = [Cue(index=1, start_ms=1_000, end_ms=3_000, lines=["<i>Ich habe es gesagt</i>, er kommt."])]
    words = _words(["Ich", "habe", "es", "gesagt,", "sie", "kommt."])
    alignment = align_cues_to_words(cues, words)
    span, = alignment.divergence_spans
    assert (span.srt_text, span.asr_text) == ("er", "sie")
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="sie", confidence=0.95, reason="actor improvised",
    )

    # 31 visible characters: the tags take no display width.
    profile = StyleProfile(max_chars_per_line=32)
    changed, _flags = apply_adjudication_decisions(cues, [span], [decision], profile, words=words)

    assert changed[0].text == "<i>Ich habe es gesagt</i>, sie kommt."


@pytest.mark.parametrize("spoken", ["Scheiße,", "Scheiße!", "Scheiße.", "„Scheiße“"])
def test_profanity_alignment_key_ignores_attached_punctuation(spoken):
    assert normalize_token(spoken) == normalize_token("Scheiße") == normalize_token("Sch*iße")


def test_masked_source_profanity_matches_punctuated_provider_word():
    cues = [Cue(index=1, start_ms=1_000, end_ms=3_000, lines=["Was soll die Sch*iße?"])]
    words = _words(["Was", "soll", "die", "Scheiße?"])

    result = align_cues_to_words(cues, words)

    assert result.divergence_spans == []
    assert result.anchor_coverage == 1.0


def test_profanity_compound_key_ignores_trailing_punctuation():
    assert normalize_token("Scheißkerl!") == normalize_token("Scheißkerl")
