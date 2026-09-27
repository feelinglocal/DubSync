"""Speaker turns frozen from episode 11's prepared Scribe v2 words.

The small tuples retain actual timings/diarization from 2026-09-10. They are
regressions for lexical ownership and serialization, not independent acoustic
ground truth or an endorsement of every human-reference timestamp.
"""

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.cue_segmentation import split_speaker_turn_cues
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.pipeline import (
    _adlib_cue_ids_by_case, _alignment_with_decision_words,
    _anchored_adlib_cue_id, _validate_inline_adlib_ownership,
)
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


def _profile():
    return StyleProfile(max_chars_per_line=52, max_lines_per_cue=2, fps=30.0)


def _snow_words():
    # Full scribe-prepared.json words 1991..2002.
    return [
        Word(text=text, start=start, end=end, speaker_id=speaker)
        for text, start, end, speaker in [
            ("mas", 1400.418, 1400.598, "speaker_5"),
            ("hoje", 1400.638, 1400.768, "speaker_5"),
            ("eu", 1400.778, 1400.838, "speaker_5"),
            ("não", 1400.878, 1400.958, "speaker_5"),
            ("tô", 1401.018, 1401.138, "speaker_5"),
            ("muito", 1401.158, 1401.378, "speaker_5"),
            ("bem.", 1401.398, 1401.518, "speaker_5"),
            ("Estou", 1401.558, 1401.628, "speaker_9"),
            ("adorando", 1401.658, 1402.118, "speaker_9"),
            ("minha", 1402.198, 1402.298, "speaker_9"),
            ("prancha", 1402.318, 1402.558, "speaker_9"),
            ("nova.", 1402.598, 1402.798, "speaker_9"),
        ]
    ]


def _reaction_words():
    # Full scribe-prepared.json words 2513..2516.
    return [
        Word(text=text, start=start, end=end, speaker_id=speaker)
        for text, start, end, speaker in [
            ("Hã?", 1652.21, 1652.69, "speaker_7"),
            ("Uau,", 1653.11, 1653.41, "speaker_6"),
            ("que", 1653.67, 1653.75, "speaker_6"),
            ("lindo!", 1653.81, 1654.97, "speaker_6"),
        ]
    ]


def _split(cue, words):
    return split_speaker_turn_cues(
        [cue], words,
        AlignmentResult(cue_word_indices={cue.index: list(range(len(words)))}),
        _profile(),
    )


def test_episode_11_snow_clauses_split_by_real_speaker_below_line_limit():
    words = _snow_words()
    source = Cue(
        index=498, start_ms=1400400, end_ms=1402866,
        lines=["- mas hoje. - Eu não tô muito bem. estou adorando minha prancha nova."],
    )

    cues, updated, flags, expansions = _split(source, words)

    assert len(cues) == 2
    assert [cue.speaker_id for cue in cues] == ["speaker_5", "speaker_9"]
    assert [cue.plain_text for cue in cues] == [
        "mas hoje. Eu não tô muito bem.", "estou adorando minha prancha nova.",
    ]
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in cues)) == alphanumeric_signature(source.plain_text)
    assert [updated.cue_word_indices[cue_id] for cue_id in expansions[498]] == [list(range(7)), list(range(7, 12))]
    assert cues[0].end_ms <= cues[1].start_ms
    assert flags[0].kind == "speaker_turn_split"


def test_episode_11_short_ha_and_uau_keep_distinct_speakers():
    words = _reaction_words()
    source = Cue(
        index=617, start_ms=1652166, end_ms=1655500,
        lines=["Hã? Uau, que lindo!"],
        speaker_id="speaker_6", character="Speaker Six",
    )

    cues, updated, _flags, expansions = _split(source, words)

    assert len(cues) == 2
    assert [cue.plain_text for cue in cues] == ["Hã?", "Uau, que lindo!"]
    assert [cue.speaker_id for cue in cues] == ["speaker_7", "speaker_6"]
    assert [cue.character for cue in cues] == [None, "Speaker Six"]
    assert updated.cue_word_indices[expansions[617][0]] == [0]
    assert updated.cue_word_indices[expansions[617][1]] == [1, 2, 3]
    assert cues[0].end_ms <= 1652800
    assert cues[1].start_ms >= 1653000
    assert source.end_ms == 1655500


def test_episode_11_three_reactions_keep_comma_ended_edge_actor():
    # Frozen v10 case-254 retains Hã / Uau que lindo / Ah. The last
    # reaction is a distinct actor even though its punctuation is a comma.
    words = [
        *_reaction_words(),
        Word(text="Ah,", start=1655.11, end=1655.49, speaker_id="speaker_4"),
        Word(text="obrigada.", start=1655.95, end=1657.11, speaker_id="speaker_4"),
    ]
    source = [
        Cue(index=604, start_ms=1653440, end_ms=1654510, lines=["Hã? Uau, que lindo! Ah,"]),
        Cue(index=605, start_ms=1656750, end_ms=1657750, lines=["Obrigada."]),
    ]
    alignment = AlignmentResult(cue_word_indices={604: [0, 1, 2, 3, 4], 605: [5]})

    cues, updated, flags, expansions = split_speaker_turn_cues(source, words, alignment, _profile())

    assert [cue.plain_text for cue in cues] == ["Hã?", "Uau, que lindo!", "Ah,", "Obrigada."]
    assert [cue.speaker_id for cue in cues[:3]] == ["speaker_7", "speaker_6", "speaker_4"]
    assert expansions == {604: [604, 606, 607]}
    assert updated.cue_word_indices == {604: [0], 606: [1, 2, 3], 607: [4], 605: [5]}
    assert alignment.cue_word_indices == {604: [0, 1, 2, 3, 4], 605: [5]}
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues[:2], cues[1:3]))


def test_completed_multiword_run_supports_a_leading_singleton_without_a_word_allowlist():
    words = [
        Word(text=text, start=10 + i * 0.3, end=10.2 + i * 0.3, speaker_id=speaker)
        for i, (text, speaker) in enumerate([
            ("Mira,", "A"), ("tutto", "B"), ("pronto.", "B"), ("Sì!", "C"),
        ])
    ]
    source = Cue(index=1, start_ms=10000, end_ms=11200, lines=["Mira, tutto pronto. Sì!"])

    cues, _, flags, _ = _split(source, words)

    assert [cue.plain_text for cue in cues] == ["Mira,", "tutto pronto.", "Sì!"]
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]


@pytest.mark.parametrize("runs", [
    # Genuine A/B/A word-level label instability has no stable phrase.
    [("one", "A"), ("actor", "B"), ("speaks", "A")],
    # A complete phrase does not excuse an unfinished interior singleton.
    [("First", "A"), ("phrase.", "A"), ("short,", "B"), ("last", "C"), ("phrase.", "C")],
    # An unfinished middle run cannot establish a clean reaction boundary.
    [("First!", "A"), ("unfinished", "B"), ("phrase", "B"), ("last,", "C")],
])
def test_edge_reaction_exception_keeps_unstable_unfinished_runs_held(runs):
    words = [
        Word(text=text, start=10 + i * 0.3, end=10.2 + i * 0.3, speaker_id=speaker)
        for i, (text, speaker) in enumerate(runs)
    ]
    source = Cue(index=1, start_ms=10000, end_ms=12000, lines=[" ".join(text for text, _ in runs)])

    cues, updated, flags, expansions = _split(source, words)

    assert cues == [source]
    assert updated.cue_word_indices == {1: list(range(len(words)))}
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["speaker_turn_split_held"]
    assert "alternate" in flags[0].message


def test_speaker_split_does_not_break_same_actor_sentences_into_word_cues():
    words = [word.model_copy(update={"speaker_id": "actor"}) for word in _reaction_words()]
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=["Hã? Uau, que lindo!"])

    cues, _updated, flags, expansions = _split(source, words)

    assert cues == [source]
    assert flags == []
    assert expansions == {}


def test_speaker_split_preserves_ambiguous_same_length_editorial_words():
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=["Hã? Nossa, que lindo!"])

    cues, updated, flags, expansions = _split(source, _reaction_words())

    assert cues == [source]
    assert updated.cue_word_indices == {1: [0, 1, 2, 3]}
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["speaker_turn_split_held"]


@pytest.mark.parametrize("bad_word", [
    {"speaker_id": None}, {"start": 1653.11, "end": 1653.11},
    {"start": 1653.11, "end": 1656.5},
])
def test_speaker_split_holds_missing_identity_or_invalid_word_timing(bad_word):
    words = _reaction_words()
    words[1] = words[1].model_copy(update=bad_word)
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=["Hã? Uau, que lindo!"])

    cues, _updated, flags, expansions = _split(source, words)

    assert cues == [source]
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["speaker_turn_split_held"]


def test_speaker_split_does_not_borrow_exact_words_from_remote_scene():
    source = Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hã? Uau, que lindo!"])

    cues, _updated, flags, expansions = _split(source, _reaction_words())

    assert cues == [source]
    assert expansions == {}
    assert "local timing envelope" in flags[0].message


def test_episode_11_trailing_hum_splits_within_one_frame_of_local_envelope():
    # Actual source 682 and Scribe 2832..2834. The distinct actor's final word
    # ends only 10ms beyond the 1.5s envelope, within the active 30fps grid.
    words = [
        Word(text="devia", start=1900.34, end=1900.509, speaker_id="speaker_4"),
        Word(text="aproveitar.", start=1900.54, end=1902.52, speaker_id="speaker_4"),
        Word(text="Hum.", start=1902.56, end=1902.71, speaker_id="speaker_1"),
    ]
    source = Cue(index=682, start_ms=1900070, end_ms=1901200, lines=["devia aproveitar Hum."])
    alignment = AlignmentResult(cue_word_indices={682: [0, 1, 2]})

    cues, updated, flags, expansions = split_speaker_turn_cues([source], words, alignment, _profile())

    assert [cue.plain_text for cue in cues] == ["devia aproveitar", "Hum."]
    assert [cue.speaker_id for cue in cues] == ["speaker_4", "speaker_1"]
    assert updated.cue_word_indices == {682: [0, 1], 683: [2]}
    assert expansions == {682: [682, 683]}
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]
    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [
        (1900333, 1902533), (1902533, 1902766),
    ]
    assert alignment.cue_word_indices == {682: [0, 1, 2]}
    assert [(word.start, word.end) for word in words] == [
        (1900.34, 1900.509), (1900.54, 1902.52), (1902.56, 1902.71),
    ]


@pytest.mark.parametrize("fps,extra_ms,allowed", [
    (25.0, 40.0, True), (25.0, 40.1, False),
    (30.0, 33.0, True), (30.0, 34.0, False),
    (60.0, 16.0, True), (60.0, 17.0, False),
    # Low configured frame rates must not admit arbitrarily remote words.
    (1.0, 41.0, True), (1.0, 42.0, False), (1.0, 500.0, False),
])
@pytest.mark.parametrize("edge", ["start", "end"])
def test_speaker_locality_frame_tolerance_is_bounded_at_both_edges(fps, extra_ms, allowed, edge):
    source = Cue(index=1, start_ms=10000, end_ms=11000, lines=["Yes. Thanks."])
    words = [
        Word(text="Yes.", start=10.0, end=10.2, speaker_id="A"),
        Word(text="Thanks.", start=10.5, end=10.7, speaker_id="B"),
    ]
    if edge == "start":
        start = (8500.0 - extra_ms) / 1000
        words[0] = words[0].model_copy(update={"start": start, "end": start + 0.2})
    else:
        end = (12500.0 + extra_ms) / 1000
        words[1] = words[1].model_copy(update={"start": end - 0.2, "end": end})
    alignment = AlignmentResult(cue_word_indices={1: [0, 1]})

    cues, updated, flags, expansions = split_speaker_turn_cues(
        [source], words, alignment, _profile().model_copy(update={"fps": fps}),
    )

    if allowed:
        assert [cue.plain_text for cue in cues] == ["Yes.", "Thanks."]
        assert updated.cue_word_indices == {1: [0], 2: [1]}
        assert expansions == {1: [1, 2]}
        assert [flag.kind for flag in flags] == ["speaker_turn_split"]
    else:
        assert cues == [source]
        assert updated == alignment
        assert expansions == {}
        assert [flag.kind for flag in flags] == ["speaker_turn_split_held"]
        assert "local timing envelope" in flags[0].message


@pytest.mark.parametrize("text", ["[Foto] Hã? Uau, que lindo!", "♪Hã? Uau, que lindo!♪"])
def test_speaker_split_preserves_screen_text_and_lyrics(text):
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=[text])

    cues, _updated, flags, expansions = _split(source, _reaction_words())

    assert cues == [source]
    assert flags == []
    assert expansions == {}


def test_speaker_split_holds_markup_instead_of_unbalancing_tags():
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=["<i>Hã? Uau, que lindo!</i>"])

    cues, _updated, flags, expansions = _split(source, _reaction_words())

    assert cues == [source]
    assert expansions == {}
    assert "styling" in flags[0].message


def test_speaker_split_respects_protected_cues_and_does_not_mutate_alignment():
    source = Cue(index=7, start_ms=1652166, end_ms=1655500, lines=["Hã? Uau, que lindo!"])
    alignment = AlignmentResult(cue_word_indices={7: [0, 1, 2, 3]})

    cues, updated, flags, expansions = split_speaker_turn_cues(
        [source], _reaction_words(), alignment, _profile(), protected_cue_ids={7},
    )

    assert cues == [source]
    assert updated == alignment
    assert updated is not alignment
    assert flags == []
    assert expansions == {}
    assert alignment.cue_word_indices == {7: [0, 1, 2, 3]}


def test_speaker_split_does_not_trust_alternating_labels_on_every_word():
    words = [
        Word(text=text, start=10 + index * 0.3, end=10.2 + index * 0.3, speaker_id=speaker)
        for index, (text, speaker) in enumerate([
            ("one", "A"), ("actor", "B"), ("speaks", "A"), ("here.", "B"),
        ])
    ]
    source = Cue(index=1, start_ms=10000, end_ms=11200, lines=["one actor speaks here."])

    cues, _updated, flags, expansions = _split(source, words)

    assert cues == [source]
    assert expansions == {}
    assert "alternate" in flags[0].message


def test_speaker_split_does_not_shorten_real_overlapping_speech():
    words = [
        Word(text="Still", start=1.0, end=1.3, speaker_id="A"),
        Word(text="speaking.", start=1.4, end=2.0, speaker_id="A"),
        Word(text="Yes!", start=1.7, end=2.2, speaker_id="B"),
    ]
    source = Cue(index=1, start_ms=1000, end_ms=2250, lines=["Still speaking. Yes!"])

    cues, _updated, flags, _expansions = _split(source, words)

    assert len(cues) == 2
    assert cues[0].end_ms >= 2000
    assert cues[1].start_ms <= 1700
    assert cues[0].end_ms > cues[1].start_ms
    assert [cue.speaker_id for cue in cues] == ["A", "B"]
    assert flags[0].kind == "speaker_turn_split"


@pytest.mark.parametrize("approved_lines,held_lines", [
    (["- Rápido, rápido. - Vem cá."], ["- Rápido, rápido.", "- Vem cá."]),
    (["Rápido, rápido. Vem cá."], ["Rápido, rápido. Vem cá."]),
])
def test_episode_11_collapsed_photo_words_hold_whole_source_parent(approved_lines, held_lines):
    # Actual source439 and ASR1784..1787: both words of the second actor
    # occupy the same 1ms interval. There is no evidence for a new child cue.
    words = [
        Word(text="Rápido,", start=1300.648, end=1300.887, speaker_id="speaker_4"),
        Word(text="rápido.", start=1300.938, end=1301.558, speaker_id="speaker_4"),
        Word(text="Vem", start=1301.568, end=1301.569, speaker_id="speaker_5"),
        Word(text="cá,", start=1301.568, end=1301.569, speaker_id="speaker_5"),
    ]
    source = Cue(index=439, start_ms=1300510, end_ms=1301310, lines=approved_lines)
    alignment = AlignmentResult(cue_word_indices={439: [0, 1, 2, 3], 440: [9, 10]})

    cues, updated, flags, expansions = split_speaker_turn_cues([source], words, alignment, _profile())

    assert len(cues) == 1
    assert cues[0].index == source.index
    assert (cues[0].start_ms, cues[0].end_ms) == (source.start_ms, source.end_ms)
    assert cues[0].lines == held_lines
    assert alphanumeric_signature(cues[0].text) == alphanumeric_signature(source.text)
    assert updated.cue_word_indices == alignment.cue_word_indices
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["timing_evidence_held"]
    assert flags[0].cue_ids == [439]
    assert flags[0].severity == "error"
    assert "collapsed" in flags[0].message.lower()
    assert source.lines == approved_lines
    rebuilt, _ = rebuild_cues(cues, words, updated, _profile(), protected_cue_ids={439})
    assert rebuilt == cues


def test_speaker_split_allocates_unique_ids_for_multiple_original_cues():
    words = _reaction_words()
    sources = [
        Cue(index=index, start_ms=1652166, end_ms=1655500, lines=["Hã? Uau, que lindo!"])
        for index in [1, 10]
    ]
    alignment = AlignmentResult(cue_word_indices={cue.index: [0, 1, 2, 3] for cue in sources})

    cues, updated, _flags, expansions = split_speaker_turn_cues(sources, words, alignment, _profile())

    assert [cue.index for cue in cues] == [1, 11, 10, 12]
    assert expansions == {1: [1, 11], 10: [10, 12]}
    assert updated.cue_word_indices[11] == [1, 2, 3]
    assert updated.cue_word_indices[12] == [1, 2, 3]
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3], 10: [0, 1, 2, 3]}


def test_speaker_split_uses_only_retained_exact_word_window():
    words = [*_reaction_words(), Word(text="Afterward.", start=1660, end=1660.5, speaker_id="other")]
    source = Cue(index=1, start_ms=1652166, end_ms=1655500, lines=["Hã? Uau, que lindo!"])

    cues, updated, _flags, expansions = _split(source, words)

    assert len(cues) == 2
    assert [word for cue_id in expansions[1] for word in updated.cue_word_indices[cue_id]] == [0, 1, 2, 3]


def test_single_retained_actor_excludes_other_reactions_from_timing_and_preserves_other_owners():
    # Actual case-254 accepted only Uau/que/lindo from a larger Hã/Uau/Ah span.
    # Independently owned neighboring reactions must keep their own pointers.
    words = [
        *_reaction_words(),
        Word(text="Ah,", start=1655.11, end=1655.49, speaker_id="speaker_4"),
        Word(text="obrigada.", start=1655.95, end=1657.11, speaker_id="speaker_4"),
    ]
    source = [
        Cue(index=603, start_ms=1652200, end_ms=1652700, lines=["Hã?"]),
        Cue(index=604, start_ms=1652200, end_ms=1655533, lines=["Uau, que lindo."]),
        Cue(index=605, start_ms=1655400, end_ms=1657166, lines=["Ah, obrigada."]),
    ]
    alignment = AlignmentResult(cue_word_indices={603: [0], 604: [0, 1, 2, 3, 4], 605: [4, 5]})

    cues, updated, flags, expansions = split_speaker_turn_cues(source, words, alignment, _profile())

    assert cues == source
    assert expansions == {}
    assert updated.cue_word_indices == {603: [0], 604: [1, 2, 3], 605: [4, 5]}
    assert alignment.cue_word_indices == {603: [0], 604: [0, 1, 2, 3, 4], 605: [4, 5]}
    assert [flag.kind for flag in flags] == ["speaker_turn_word_window_refined"]
    rebuilt, _ = rebuild_cues(cues, words, updated, _profile())
    reaction = next(cue for cue in rebuilt if cue.index == 604)
    assert reaction.speaker_id == "speaker_6"
    assert 1653000 <= reaction.start_ms <= 1653110
    assert 1654970 <= reaction.end_ms < 1655450


def test_single_retained_actor_does_not_refine_protected_or_ambiguous_cue():
    words = [
        *_reaction_words(),
        Word(text="Uau,", start=1655.11, end=1655.41, speaker_id="speaker_6"),
        Word(text="que", start=1655.67, end=1655.75, speaker_id="speaker_6"),
        Word(text="lindo!", start=1655.81, end=1656.20, speaker_id="speaker_6"),
    ]
    cue = Cue(index=604, start_ms=1652200, end_ms=1656533, lines=["Uau, que lindo."])
    alignment = AlignmentResult(cue_word_indices={604: list(range(len(words)))})

    _, updated, flags, _ = split_speaker_turn_cues([cue], words, alignment, _profile())
    assert updated == alignment
    assert [flag.kind for flag in flags] == ["speaker_turn_split_held"]
    _, protected, protected_flags, _ = split_speaker_turn_cues(
        [cue], words[:4], AlignmentResult(cue_word_indices={604: [0, 1, 2, 3]}), _profile(),
        protected_cue_ids={604},
    )
    assert protected.cue_word_indices == {604: [0, 1, 2, 3]}
    assert protected_flags == []


@pytest.mark.parametrize("speaker_id,character,expected_character", [
    ("speaker_7", "Other Actor", None),
    ("speaker_6", "Retained Actor", "Retained Actor"),
])
def test_single_retained_actor_character_metadata_stays_with_its_actor(speaker_id, character, expected_character):
    source = Cue(
        index=604, start_ms=1652200, end_ms=1655533, lines=["Uau, que lindo."],
        speaker_id=speaker_id, character=character,
    )
    cues, updated, _, _ = _split(source, _reaction_words())

    rebuilt, _ = rebuild_cues(cues, _reaction_words(), updated, _profile())

    assert rebuilt[0].speaker_id == "speaker_6"
    assert rebuilt[0].character == expected_character
    assert source.character == character


def test_speaker_split_keeps_hyphenated_words_inside_the_correct_actor():
    words = [
        Word(text="Bem-vindo!", start=1.0, end=1.4, speaker_id="A"),
        Word(text="Obrigado!", start=1.7, end=2.1, speaker_id="B"),
    ]
    source = Cue(index=1, start_ms=1000, end_ms=2200, lines=["- Bem-vindo! - Obrigado!"])

    cues, _updated, _flags, _expansions = _split(source, words)

    assert [cue.plain_text for cue in cues] == ["Bem-vindo!", "Obrigado!"]


def test_episode_11_case_187_preserves_clause_before_the_other_actors_words():
    # Same actual source cue and Scribe words as full align.json case-186/187.
    # The insertion belongs after actor5's 'eu', before actor9's 'Estou'.
    source = [Cue(
        index=489, start_ms=1400790, end_ms=1402400,
        lines=["- A neve está ótima. - Eu estou adorando minha prancha nova."],
    )]
    words = _snow_words()
    alignment = align_cues_to_words(source, words)
    insertion = next(span for span in alignment.divergence_spans if not span.cue_ids)
    assert insertion.asr_text == "não tô muito bem."
    assert insertion.left_anchor_cue_id == insertion.right_anchor_cue_id == 489
    assert insertion.insertion_token_offset == 5
    assert insertion.left_anchor_speaker_id == "speaker_5"
    assert insertion.right_anchor_speaker_id == "speaker_9"
    decisions = [AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=0.99, reason="Frozen spoken words from the episode 11 Scribe window.",
    ) for span in alignment.divergence_spans]

    ids, _flags = _adlib_cue_ids_by_case(source, alignment.divergence_spans, decisions, [])
    assert ids[insertion.case_id] == 489
    ids, ownership_flags = _validate_inline_adlib_ownership(source, words, alignment, decisions, ids, _profile())
    assert ids[insertion.case_id] == 489
    assert ownership_flags == []
    changed, _flags = apply_adjudication_decisions(
        source, alignment.divergence_spans, decisions, _profile(), ids,
    )
    updated = _alignment_with_decision_words(
        alignment, decisions, alignment.divergence_spans, ids, source_cues=source,
    )
    cues, updated, _flags, expansions = split_speaker_turn_cues(changed, words, updated, _profile())

    assert [cue.plain_text for cue in cues] == [
        "mas hoje. Eu não tô muito bem.", "estou adorando minha prancha nova.",
    ]
    assert [cue.speaker_id for cue in cues] == ["speaker_5", "speaker_9"]
    assert [updated.cue_word_indices[index] for index in expansions[489]] == [list(range(7)), list(range(7, 12))]
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in cues)) == alphanumeric_signature(
        " ".join(word.text for word in words)
    )


def _actor_boundary_span():
    return DivergenceSpan(
        case_id="case-187", cue_ids=[], srt_text="", asr_text="não tô muito bem.",
        start=1400.878, end=1401.518, speaker_ids=["speaker_5"],
        asr_word_indices=[1994, 1995, 1996, 1997],
        left_anchor_cue_id=489, right_anchor_cue_id=489, insertion_token_offset=5,
        left_anchor_end=1400.838, right_anchor_start=1401.558,
        left_anchor_speaker_id="speaker_5", right_anchor_speaker_id="speaker_9",
    )


@pytest.mark.parametrize("update", [
    {"asr_word_indices": []}, {"asr_word_indices": [1994, 1996, 1997]},
    {"left_anchor_end": 1400.9}, {"right_anchor_start": 1401.4},
    {"start": None}, {"end": float("nan")},
    {"start": 1405.0, "end": 1405.2, "left_anchor_end": 1405.0, "right_anchor_start": 1405.3},
    {"speaker_ids": []}, {"speaker_ids": ["unrelated_actor"]}, {"speaker_ids": ["speaker_5", "speaker_9"]},
    {"insertion_token_offset": 0}, {"insertion_token_offset": 100},
    {"asr_text": "different spoken words here"},
])
def test_same_cue_actor_boundary_requires_exact_local_word_evidence(update):
    cue = Cue(
        index=489, start_ms=1400790, end_ms=1402400,
        lines=["- A neve está ótima. - Eu estou adorando minha prancha nova."],
    )
    span = _actor_boundary_span().model_copy(update=update)

    assert _anchored_adlib_cue_id({489: cue}, span, "não tô muito bem.") is None


@pytest.mark.parametrize("speaker", ["speaker_5", "speaker_9"])
def test_same_cue_actor_boundary_accepts_either_confirmed_adjacent_actor(speaker):
    cue = Cue(
        index=489, start_ms=1400790, end_ms=1402400,
        lines=["- A neve está ótima. - Eu estou adorando minha prancha nova."],
    )
    span = _actor_boundary_span().model_copy(update={"speaker_ids": [speaker]})

    assert _anchored_adlib_cue_id({489: cue}, span, "não tô muito bem.") == 489


def test_case_187_stays_separate_when_kept_source_words_prevent_speaker_split():
    source = [Cue(
        index=489, start_ms=1400790, end_ms=1402400,
        lines=["- A neve está ótima. - Eu estou adorando minha prancha nova."],
    )]
    words = _snow_words()
    alignment = align_cues_to_words(source, words)
    insertion = next(span for span in alignment.divergence_spans if not span.cue_ids)
    decisions = [AdjudicationDecision(
        case_id=span.case_id,
        verdict="keep_srt" if span.cue_ids else "use_audio",
        final_text=span.srt_text if span.cue_ids else span.asr_text,
        confidence=0.99, reason="Preserve uncertain source wording, accept the clear inserted clause.",
    ) for span in alignment.divergence_spans]

    ids, _flags = _adlib_cue_ids_by_case(source, alignment.divergence_spans, decisions, [])
    original_ids = dict(ids)
    ids, ownership_flags = _validate_inline_adlib_ownership(source, words, alignment, decisions, ids, _profile())
    assert original_ids[insertion.case_id] == 489
    assert ids[insertion.case_id] != 489
    assert [flag.kind for flag in ownership_flags] == ["adlib_speaker_ownership_held"]
    changed, _flags = apply_adjudication_decisions(source, alignment.divergence_spans, decisions, _profile(), ids)
    assert changed[0] == source[0]
    assert changed[1].plain_text == "não tô muito bem."


def test_inline_adlib_probe_preserves_protected_source_and_its_word_mapping():
    source = [Cue(
        index=489, start_ms=1400790, end_ms=1402400,
        lines=["- A neve está ótima. - Eu estou adorando minha prancha nova."],
    )]
    words = _snow_words()
    alignment = align_cues_to_words(source, words)
    before_source = source[0].model_dump()
    before_alignment = alignment.model_dump()
    insertion = next(span for span in alignment.divergence_spans if not span.cue_ids)
    decisions = [AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=0.99, reason="Clear local speech with separate source-review hold.",
    ) for span in alignment.divergence_spans]
    initial_map = {insertion.case_id: 489, "already-generated": 900}

    result, flags = _validate_inline_adlib_ownership(
        source, words, alignment, decisions, initial_map, _profile(), protected_cue_ids={489},
    )

    assert result == {insertion.case_id: 901, "already-generated": 900}
    assert initial_map == {insertion.case_id: 489, "already-generated": 900}
    assert source[0].model_dump() == before_source
    assert alignment.model_dump() == before_alignment
    assert [flag.kind for flag in flags] == ["adlib_speaker_ownership_held"]
    assert flags[0].cue_ids == [489, 901]


def test_inline_adlib_probe_skips_unrelated_cases(monkeypatch):
    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("No cross-actor inline target should invoke transformation probes")

    monkeypatch.setattr("dubsync.pipeline.apply_adjudication_decisions", unexpected_probe)
    initial_map = {"already-generated": 900}

    result, flags = _validate_inline_adlib_ownership([], [], AlignmentResult(), [], initial_map, _profile())

    assert result == initial_map
    assert result is not initial_map
    assert flags == []
