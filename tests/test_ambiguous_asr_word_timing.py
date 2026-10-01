from __future__ import annotations

import pytest

from dubsync.asr_timing import PhraseEdgeSnap, ambiguous_word_indices, repair_asr_word_edges
from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.recue import cue_spoken_spans, rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import refine_cues_to_speech_activity


def test_two_spoken_names_inside_one_provider_word_do_not_choose_the_first_burst():
    # test-long word 2162: fresh ASR confirms Luan at the first burst and Luke
    # at the second. Energy alone cannot assign the stretched Luke word.
    words = [
        Word(text="Lan...", start=1573.452, end=1573.453, speaker_id="speaker_3"),
        Word(text="Luke.", start=1573.462, end=1579.422, speaker_id="speaker_0"),
        Word(text="Terminou", start=1581.362, end=1581.822, speaker_id="speaker_0"),
    ]
    regions = [
        SpeechRegion(start=1573.44, end=1573.625),
        SpeechRegion(start=1579.01, end=1579.43),
        SpeechRegion(start=1581.36, end=1582.35),
    ]
    original = [word.model_dump() for word in words]

    repaired, flags = repair_asr_word_edges(words, regions)

    assert repaired[1] is words[1]
    assert [word.model_dump() for word in words] == original
    ambiguous = [flag for flag in flags if flag.kind == "asr_word_timing_ambiguous"]
    assert len(ambiguous) == 1
    assert ambiguous[0].severity == "warning"
    assert ambiguous[0].old_text == "Luke. 1573.462 --> 1579.422"
    assert ambiguous[0].new_text is None
    assert (ambiguous[0].start, ambiguous[0].end) == (1573.462, 1579.422)
    assert "1573.440" in ambiguous[0].message and "1579.430" in ambiguous[0].message
    assert not any(flag.kind == "asr_word_clamped" and "Luke." in (flag.old_text or "") for flag in flags)


@pytest.mark.parametrize("text", ["Luke.", "Nan...", "Ah!"])
def test_repeated_names_or_interjections_do_not_disambiguate_separate_bursts(text):
    words = [
        Word(text=text, start=1.0, end=1.2, speaker_id="A"),
        Word(text=text, start=1.24, end=7.45, speaker_id="A"),
        Word(text=text, start=7.5, end=7.7, speaker_id="A"),
    ]
    regions = [SpeechRegion(start=1.0, end=1.7), SpeechRegion(start=7.0, end=7.7)]

    repaired, flags = repair_asr_word_edges(words, regions)

    assert repaired[1] is words[1]
    assert [flag.kind for flag in flags] == ["asr_word_timing_ambiguous"]


@pytest.mark.parametrize("first_end,last_start", [(1.08, 1.30), (1.08, 1.32), (1.06, 1.30)])
def test_two_short_bursts_do_not_choose_the_first_or_longest_without_whole_word_evidence(
    first_end, last_start,
):
    # Both overlaps are shorter than the whole-word heuristic. They can be
    # split phonemes or separate utterances; neither fact identifies a winner.
    word = Word(text="Luke.", start=1.0, end=1.38)
    regions = [SpeechRegion(start=1.0, end=first_end), SpeechRegion(start=last_start, end=1.38)]

    repaired, flags = repair_asr_word_edges([word], regions, max_word_duration=0.1)

    assert repaired[0] is word
    assert [flag.kind for flag in flags] == ["asr_word_timing_ambiguous"]
    assert ambiguous_word_indices(repaired, flags) == {0}


def test_ambiguity_includes_interior_bursts_and_skips_duration_cap_and_phrase_snap():
    word = Word(text="Yes.", start=1.05, end=7.45)
    regions = [
        SpeechRegion(start=1.0, end=1.4),
        SpeechRegion(start=3.0, end=3.4),
        SpeechRegion(start=7.0, end=7.5),
    ]

    repaired, flags = repair_asr_word_edges(
        [word], regions, max_word_duration=0.1, snap=PhraseEdgeSnap(1.0, 1.0),
    )

    assert repaired == [word]
    assert [flag.kind for flag in flags] == ["asr_word_timing_ambiguous"]


@pytest.mark.parametrize("start,end,expected", [
    (1.10, 7.05, (1.0, 1.60)),  # Only the first burst can contain the whole word.
    (1.59, 7.40, (7.0, 7.50)),  # The first burst is only the previous word's tail.
])
def test_unique_plausible_burst_remains_repairable(start, end, expected):
    word = Word(text="Luke.", start=start, end=end)
    regions = [SpeechRegion(start=1.0, end=1.6), SpeechRegion(start=7.0, end=7.5)]

    repaired, flags = repair_asr_word_edges([word], regions)

    assert (repaired[0].start, repaired[0].end) == expected
    assert [flag.kind for flag in flags] == ["asr_word_clamped"]


def test_ambiguous_repair_is_idempotent_and_keeps_word_order():
    words = [Word(text="Luke.", start=1.05, end=7.45), Word(text="Hello.", start=8.02, end=8.4)]
    regions = [
        SpeechRegion(start=1.0, end=1.4), SpeechRegion(start=7.0, end=7.5),
        SpeechRegion(start=8.0, end=8.5),
    ]

    once, first_flags = repair_asr_word_edges(words, regions)
    twice, second_flags = repair_asr_word_edges(once, regions)

    assert twice == once
    assert once[0] is words[0]
    assert [word.text for word in twice] == [word.text for word in words]
    assert [flag.model_dump() for flag in second_flags] == [flag.model_dump() for flag in first_flags]


@pytest.mark.parametrize("start,end,source_start,source_end", [
    (1.462, 7.422, 6800, 7600),
    (1.462, 7.422, 1462, 7422),
    (1.05, 2.4, 2000, 2500),
])
def test_direct_refinement_cannot_choose_a_burst_for_an_ambiguous_word(
    start, end, source_start, source_end,
):
    word = Word(text="Luke.", start=start, end=end)
    cue = Cue(index=1, start_ms=source_start, end_ms=source_end, lines=["Luke."])
    regions = [SpeechRegion(start=1.0, end=1.7), SpeechRegion(start=end - 0.4, end=end + 0.1)]
    words, _ = repair_asr_word_edges([word], regions)

    refined, flags = refine_cues_to_speech_activity(
        [cue], regions, StyleProfile(), words=words,
        alignment=AlignmentResult(cue_word_indices={1: [0]}),
    )

    assert refined == [cue]
    assert any(flag.kind == "timing_evidence_held" and flag.cue_ids == [1] for flag in flags)
    assert not any(flag.kind == "timing_refined" for flag in flags)


def test_rebuild_preserves_the_whole_cue_before_trimming_an_ambiguous_word():
    words = [
        Word(text="Hello", start=1.05, end=1.2),
        Word(text="Luke.", start=1.24, end=7.45),
        Word(text="Okay.", start=8.1, end=8.6),
    ]
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1800, lines=["Hello Luke."]),
        Cue(index=2, start_ms=8000, end_ms=8800, lines=["Okay."]),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0, 1], 2: [2]})

    rebuilt, flags = rebuild_cues(
        cues, words, alignment, StyleProfile(), ambiguous_word_indices={1},
    )

    assert rebuilt[0] == cues[0]
    assert rebuilt[1].start_ms > cues[1].start_ms
    assert any(flag.kind == "timing_evidence_held" and flag.cue_ids == [1] for flag in flags)
    assert not any(flag.kind == "timing_outlier_trimmed" for flag in flags)


def test_generated_cue_retains_the_full_ambiguous_word_for_review():
    word = Word(text="Luke.", start=1.462, end=7.422)
    cue = Cue(index=1, start_ms=1462, end_ms=7422, lines=["Luke."])
    regions = [SpeechRegion(start=1.4, end=1.8), SpeechRegion(start=7.0, end=7.5)]

    refined, flags = refine_cues_to_speech_activity(
        [cue], regions, StyleProfile(), words=[word],
        alignment=AlignmentResult(cue_word_indices={1: [0]}), ambiguous_word_indices={0},
    )

    assert refined == [cue]
    assert any(flag.kind == "timing_evidence_held" and flag.cue_ids == [1] for flag in flags)


def test_ambiguity_indices_require_exact_word_edges_and_text_after_reordering():
    word = Word(text="Luke.", start=1.05004, end=7.45004)
    regions = [SpeechRegion(start=1.0, end=1.4), SpeechRegion(start=7.0, end=7.5)]
    _, flags = repair_asr_word_edges([word], regions)
    rounded_same = word.model_copy(update={"start": 1.050049, "end": 7.450049})
    different_text = word.model_copy(update={"text": "Luan."})
    other = Word(text="Okay.", start=8.0, end=8.5)

    assert ambiguous_word_indices([rounded_same, other, word, different_text], flags) == {2}


def test_shared_ambiguity_evidence_is_not_repaired_or_detected_again(monkeypatch):
    from dubsync import timing_refinement

    def unexpected_call(*args, **kwargs):
        pytest.fail("The shared ambiguity evidence must be reused without another repair or detector.")

    monkeypatch.setattr(timing_refinement, "ambiguous_word_indices_from_regions", unexpected_call)
    monkeypatch.setattr(timing_refinement, "clamp_asr_word_durations", unexpected_call)
    word = Word(text="Luke.", start=1.462, end=7.422)
    cue = Cue(index=1, start_ms=6800, end_ms=7600, lines=["Luke."])

    refined, _ = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.4, end=1.8), SpeechRegion(start=7.0, end=7.5)],
        StyleProfile(), words=[word], alignment=AlignmentResult(cue_word_indices={1: [0]}),
        ambiguous_word_indices={0},
    )

    assert refined == [cue]


def test_ambiguous_cue_does_not_supply_a_guessed_spoken_span_to_overlap_resolution():
    cue = Cue(index=1, start_ms=6800, end_ms=7600, lines=["Luke."])
    words = [Word(text="Luke.", start=1.462, end=7.422)]

    spans = cue_spoken_spans(
        [cue], words, AlignmentResult(cue_word_indices={1: [0]}), ambiguous_word_indices={0},
    )

    assert spans == {}
