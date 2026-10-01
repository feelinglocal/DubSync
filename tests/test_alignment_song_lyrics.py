"""Unspoken song-lyric cues are held as blocks and never borrow dialogue words."""

from __future__ import annotations

from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, Word


def _episode(lines: list[tuple[float, str]], spoken: list[tuple[float, str]]) -> tuple[list[Cue], list[Word]]:
    cues = [
        Cue(index=index + 1, start_ms=int(start * 1000), end_ms=int(start * 1000) + 1_800, lines=[text])
        for index, (start, text) in enumerate(lines)
    ]
    words: list[Word] = []
    for start, text in spoken:
        for offset, item in enumerate(text.split()):
            words.append(Word(text=item, start=round(start + offset * 0.3, 3), end=round(start + offset * 0.3 + 0.25, 3)))
    return cues, words


DIALOGUE = [
    (20.0, "Bom dia a todos aqui."),
    (23.0, "Hoje vamos trabalhar muito."),
    (26.0, "Onde está o relatório?"),
    (29.0, "Está na minha mesa agora."),
]


def test_unspoken_song_blocks_are_held_once_per_block_and_not_counted_as_missed_speech():
    lines = [
        (1.0, "♪Os olhos seguem sem querer♪"),
        (3.0, "♪O rosto do amor♪"),
        (5.0, "♪Só está certo quando dói♪"),
        *DIALOGUE,
        (40.0, "♪Se nunca te importou♪"),
        (42.0, "♪A maré após minha volta♪"),
    ]
    cues, words = _episode(lines, DIALOGUE)

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.diagnostics.missing_audio_cue_ids == [1, 2, 3, 8, 9]
    held = [flag for flag in result.flags if flag.kind == "missing_audio_timing_held"]
    assert [flag.cue_ids for flag in held] == [[1, 2, 3], [8, 9]]
    assert all(flag.severity == "error" for flag in held)
    assert (held[0].start, held[0].end) == (1.0, 6.8)


def test_dialogue_word_is_not_matched_into_an_unspoken_lyric_cue():
    lines = [*DIALOGUE[:2], (25.0, "♪Você é como a minha vida♪"), *DIALOGUE[2:]]
    spoken = [*DIALOGUE[:2], (25.4, "Minha"), *DIALOGUE[2:]]
    cues, words = _episode(lines, spoken)

    result = align_cues_to_words(cues, words)

    lyric_word = next(index for index, word in enumerate(words) if word.text == "Minha")
    assert 3 not in result.cue_word_indices
    assert 3 in result.diagnostics.missing_audio_cue_ids
    assert any(span.cue_ids == [] and lyric_word in span.asr_word_indices for span in result.divergence_spans)


def test_lyric_cues_pooled_with_a_dialogue_word_become_separate_cases():
    lines = [*DIALOGUE[:2], (25.0, "♪Se pudesse existir uma luz♪"), (27.0, "♪Que parasse por mim♪"), *DIALOGUE[2:]]
    lines = [(start + (3.0 if start >= 26.0 and "♪" not in text else 0.0), text) for start, text in lines]
    spoken = [*DIALOGUE[:2], (28.0, "Queria"), *[(start + 3.0, text) for start, text in DIALOGUE[2:]]]
    cues, words = _episode(lines, spoken)

    result = align_cues_to_words(cues, words)

    lyric = [span for span in result.divergence_spans if set(span.cue_ids) & {3, 4}]
    assert len(lyric) == 1
    assert lyric[0].cue_ids == [3, 4] and lyric[0].asr_word_indices == [] and lyric[0].asr_text == ""
    spoken_case = [span for span in result.divergence_spans if span.asr_text == "Queria"]
    assert len(spoken_case) == 1 and spoken_case[0].cue_ids == []
    assert {3, 4} <= set(result.diagnostics.missing_audio_cue_ids)


def test_lyric_cue_pooled_with_dialogue_tokens_is_split_from_them():
    lines = [*DIALOGUE[:2], (25.0, "♪Essa décima milésima luz acesa♪"), (27.0, "Tao, venha aqui."), *DIALOGUE[2:]]
    lines = [(start + (3.0 if start >= 26.0 and "Tao" not in text and "♪" not in text else 0.0), text) for start, text in lines]
    spoken = [*DIALOGUE[:2], (27.0, "Vem cá logo."), *[(start + 3.0, text) for start, text in DIALOGUE[2:]]]
    cues, words = _episode(lines, spoken)

    result = align_cues_to_words(cues, words)

    assert not any(set(span.cue_ids) == {3, 4} for span in result.divergence_spans)
    lyric = next(span for span in result.divergence_spans if 3 in span.cue_ids)
    assert lyric.cue_ids == [3] and not lyric.asr_word_indices
    dialogue = next(span for span in result.divergence_spans if 4 in span.cue_ids)
    assert dialogue.asr_text == "Vem cá logo."
    assert 4 not in result.diagnostics.missing_audio_cue_ids


def test_sung_lyrics_heard_in_the_audio_keep_their_words():
    lines = [*DIALOGUE[:2], (25.0, "♪Você é como a minha vida♪"), *DIALOGUE[2:]]
    spoken = [*DIALOGUE[:2], (25.0, "Você é como a minha vida"), *DIALOGUE[2:]]
    cues, words = _episode(lines, spoken)

    result = align_cues_to_words(cues, words)

    assert len(result.cue_word_indices[3]) == 6
    assert 3 not in result.diagnostics.missing_audio_cue_ids
    assert result.anchor_coverage == 1.0
