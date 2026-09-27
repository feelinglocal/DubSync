from __future__ import annotations

import pytest

from dubsync.models import AlignmentResult, Cue, Word
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile


def _rebuild(cue: Cue, words: list[Word], **kwargs):
    return rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={cue.index: list(range(len(words)))}),
        StyleProfile(min_cue_dur=0.1, tail_ms=0), **kwargs,
    )


def test_episode_11_collapsed_phrase_holds_approved_text_and_source_timing():
    # Original source 513 retained this approved phrase after deleting the
    # other actor's opening phrase. Scribe reports its four words in 91 ms.
    cue = Cue(index=513, start_ms=1432790, end_ms=1434680, lines=["- Não consegui pegar ele."])
    words = [
        Word(text="Não", start=1433.058, end=1433.118, speaker_id="speaker_4"),
        Word(text="consigo", start=1433.138, end=1433.148, speaker_id="speaker_4"),
        Word(text="pegar", start=1433.148, end=1433.149, speaker_id="speaker_4"),
        Word(text="ele.", start=1433.148, end=1433.149, speaker_id="speaker_4"),
    ]

    rebuilt, flags = _rebuild(cue, words)

    assert rebuilt == [cue]
    held = next(flag for flag in flags if flag.kind == "timing_evidence_held")
    assert held.cue_ids == [513]
    assert held.severity == "error"
    assert "collapsed" in held.message.lower()
    assert held.start == 1432.790 and held.end == 1434.680
    assert held.old_text == cue.text


def test_single_credible_word_cannot_supply_a_whole_sentence_timing():
    cue = Cue(index=8, start_ms=4000, end_ms=6000, lines=["Please bring the blue suitcase."])

    rebuilt, flags = _rebuild(cue, [Word(text="suitcase", start=4.8, end=5.0)])

    assert rebuilt == [cue]
    assert any(flag.kind == "timing_evidence_held" and "lexical" in flag.message for flag in flags)


def test_repeated_word_cannot_cover_multiple_source_tokens():
    cue = Cue(index=8, start_ms=4000, end_ms=6000, lines=["Go go go go now."])

    rebuilt, flags = _rebuild(cue, [Word(text="go", start=4.8, end=5.0)])

    assert rebuilt == [cue]
    assert any(flag.kind == "timing_evidence_held" for flag in flags)


def test_phrase_valued_word_preserves_legitimate_lexical_coverage():
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["We are coming now."])

    rebuilt, flags = _rebuild(cue, [Word(text="We are coming now.", start=2.0, end=2.8)])

    assert flags == []
    assert rebuilt[0].start_ms == 2000
    assert rebuilt[0].end_ms == 2800


def test_spelling_number_and_accent_aliases_still_retime():
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["Tristen has 2 cafés."])
    words = [Word(text=text, start=2 + i * 0.2, end=2.15 + i * 0.2)
             for i, text in enumerate(["Tristan", "has", "two", "cafes"])]

    rebuilt, flags = _rebuild(cue, words)

    assert flags == []
    assert rebuilt[0].start_ms == 2000
    assert rebuilt[0].end_ms >= 2750


def test_single_short_function_word_does_not_reject_healthy_phrase():
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["I can do it."])
    words = [
        Word(text="I", start=2.0, end=2.01),
        Word(text="can", start=2.02, end=2.2),
        Word(text="do", start=2.22, end=2.4),
        Word(text="it", start=2.42, end=2.6),
    ]

    rebuilt, flags = _rebuild(cue, words)

    assert flags == []
    assert rebuilt[0].start_ms == 2000


@pytest.mark.parametrize("end", [2.001, 2.0, float("nan")])
def test_placeholder_or_invalid_word_times_hold_source(end):
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["Yes."])

    rebuilt, flags = _rebuild(cue, [Word(text="Yes", start=2.0, end=end)])

    assert rebuilt == [cue]
    assert any(flag.kind == "timing_evidence_held" for flag in flags)


def test_already_protected_cue_keeps_timing_without_duplicate_hold_flag():
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["Yes."])

    rebuilt, flags = _rebuild(
        cue, [Word(text="Yes", start=2.0, end=2.001)], protected_cue_ids={1},
    )

    assert rebuilt == [cue]
    assert flags == []


def test_held_cue_is_not_moved_by_same_speaker_overlap_or_removed():
    held = Cue(index=2, start_ms=2000, end_ms=3000, lines=["You need to come here."], speaker_id="A")
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Hello"], speaker_id="A"), held]
    words = [Word(text="Hello", start=1.8, end=2.8, speaker_id="A"),
             Word(text="here", start=2.0, end=2.2, speaker_id="A")]

    rebuilt, flags = rebuild_cues(
        cues, words, AlignmentResult(cue_word_indices={1: [0], 2: [1]}),
        StyleProfile(drop_policy="remove"),
    )

    assert rebuilt[1] == held
    assert any(flag.kind == "timing_evidence_held" and flag.cue_ids == [2] for flag in flags)


def test_bracketed_screen_text_is_excluded_from_lexical_coverage():
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=["[Many words from an on-screen sign]", "Go."])

    rebuilt, flags = _rebuild(cue, [Word(text="Go", start=2.0, end=2.3)])

    assert flags == []
    assert rebuilt[0].start_ms == 2000
    assert rebuilt[0].text == cue.text


@pytest.mark.parametrize("hold_kind", ["protected", "collapsed", "unmatched"])
@pytest.mark.parametrize("allow_zero_gap", [True, False])
def test_padding_stops_at_preserved_source_start(hold_kind, allow_zero_gap):
    first = Cue(index=1, start_ms=0, end_ms=1000, lines=["Hi"], speaker_id="A")
    held = Cue(index=2, start_ms=1200, end_ms=2200, lines=["Please come over here."], speaker_id="B")
    later = Cue(index=3, start_ms=3000, end_ms=4000, lines=["Later"], speaker_id="A")
    words = [Word(text="Hi", start=1.0, end=1.1, speaker_id="A"),
             Word(text="here", start=1.2, end=1.201, speaker_id="B"),
             Word(text="Later", start=5.0, end=5.3, speaker_id="A")]
    mappings = {1: [0], 3: [2]}
    if hold_kind != "unmatched":
        mappings[2] = [1]
    alignment = AlignmentResult(
        cue_word_indices=mappings, unmatched_cue_ids=[2] if hold_kind == "unmatched" else [],
    )
    profile = StyleProfile(min_cue_dur=2.0, tail_ms=700, allow_zero_gap=allow_zero_gap)

    rebuilt, _ = rebuild_cues(
        [later, held, first], words, alignment, profile,
        protected_cue_ids={2} if hold_kind == "protected" else set(),
    )

    by_id = {cue.index: cue for cue in rebuilt}
    assert by_id[2] == held
    assert by_id[1].start_ms == 1000
    assert 1100 <= by_id[1].end_ms <= held.start_ms
    if not allow_zero_gap:
        assert by_id[1].end_ms < held.start_ms


def test_preserved_start_barrier_keeps_true_word_overlap():
    first = Cue(index=1, start_ms=0, end_ms=1000, lines=["Hi"], speaker_id="A")
    held = Cue(index=2, start_ms=1200, end_ms=2200, lines=["Held"], speaker_id="B")

    rebuilt, _ = rebuild_cues(
        [first, held], [Word(text="Hi", start=1.0, end=1.4, speaker_id="A")],
        AlignmentResult(cue_word_indices={1: [0]}), StyleProfile(min_cue_dur=2.0, tail_ms=700),
        protected_cue_ids={2},
    )

    assert rebuilt[0].end_ms == 1400
    assert rebuilt[1] == held
