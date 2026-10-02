"""Crowded annotation tracks paginate instead of copying every caption."""

from collections import Counter

import pytest

from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue, Word
from dubsync.style_profile import StyleProfile
from dubsync.subtitle_annotations import speech_text_for_alignment
from dubsync.tokenize import alphanumeric_signature


@pytest.mark.parametrize("caption", ["[]", "[ ]", "[] [A real caption]"])
def test_empty_bracket_captions_preserve_the_visual_marker_without_crashing(caption):
    source = [Cue(index=1, start_ms=0, end_ms=1000, lines=[caption]),
              Cue(index=2, start_ms=100, end_ms=800, lines=["Hello"])]
    result = compose_bracketed_annotations(
        source, {2: [0]}, words=[Word(text="Hello", start=.1, end=.8)], profile=StyleProfile(),
    )
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    assert result.cue_word_indices[2] == [0]
    spoken = next(cue for cue in result.cues if cue.index == 2)
    assert (spoken.start_ms, spoken.end_ms, speech_text_for_alignment(spoken)) == (100, 800, "Hello")
    pages = result.tracks[1]["pages"]
    text = " ".join(" ".join(page["lines"]) for page in pages)
    assert "[]" in text
    assert alphanumeric_signature(text) == alphanumeric_signature(caption)


def test_ep17_two_line_display_preserves_real_words_and_complete_caption_clauses():
    source = [
        Cue(index=30, start_ms=134567, end_ms=136700,
            lines=["mas também por causa de um escândalo sexual."]),
        Cue(index=31, start_ms=134620, end_ms=135840,
            lines=["[Em junho de 2011, após um jantar da empresa,]"]),
        Cue(index=32, start_ms=135840, end_ms=137740,
            lines=["[uma funcionária denunciou um caso de conduta sexual inadequada.]" ]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("mas", 134.6, 134.72), ("também", 134.8, 134.999), ("por", 135.08, 135.159),
        ("causa", 135.2, 135.4), ("de", 135.44, 135.5), ("um", 135.54, 135.6),
        ("escândalo", 135.68, 136.059), ("sexual.", 136.16, 136.639),
    ]]
    result = compose_bracketed_annotations(source, {30: list(range(8))}, words=words,
                                          profile=StyleProfile(max_chars_per_line=53))
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    spoken = [cue for cue in result.cues if speech_text_for_alignment(cue)]
    assert [(cue.start_ms, cue.end_ms, speech_text_for_alignment(cue)) for cue in spoken] == [
        (134567, 136700, source[0].text)]
    assert Counter(index for cue in spoken for index in result.cue_word_indices[cue.index]) == Counter(range(8))
    assert result.cues[-1].start_ms == 136700 and result.cues[-1].end_ms == 137740
    for track in source[1:]:
        pages = result.tracks[track.index]["pages"]
        assert alphanumeric_signature(" ".join(" ".join(page["lines"]) for page in pages)) == alphanumeric_signature(track.text)
    assert result.tracks[32]["pages"][0]["display_intervals"] == [[136700, 137740]]
    assert result.tracks[32]["pages"][0]["delay_ms"] == 860
    assert all(left.end_ms <= right.start_ms for left, right in zip(result.cues, result.cues[1:]))


def test_long_named_message_paginates_at_colon_without_repeating_all_pages():
    source = [Cue(index=1, start_ms=0, end_ms=1320,
                  lines=["[Luan Nian: todo mundo da empresa pode sair mais cedo.]"]),
              Cue(index=2, start_ms=620, end_ms=1120, lines=["Hum."])]
    result = compose_bracketed_annotations(source, {2: [0]},
                                          words=[Word(text="Hum.", start=.62, end=1.1)],
                                          profile=StyleProfile(max_chars_per_line=47))
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    pages = result.tracks[1]["pages"]
    assert [page["lines"] for page in pages] == [["[Luan Nian:]"], ["[todo mundo da empresa pode sair mais cedo.]"]]
    assert alphanumeric_signature(" ".join(" ".join(page["lines"]) for page in pages)) == alphanumeric_signature(source[0].text)
    assert speech_text_for_alignment(next(cue for cue in result.cues if cue.index == 2)) == "Hum."
    assert result.cue_word_indices[2] == [0]
