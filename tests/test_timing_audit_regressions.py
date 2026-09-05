from __future__ import annotations

import pytest

from dubsync.forced_alignment import (
    _cue_alignments_from_word_timestamps,
    apply_forced_alignment,
)
from dubsync.asr_timing import clamp_asr_word_durations
from dubsync.models import AlignmentResult, Cue, ForcedAlignmentCue, SpeechRegion, Word
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import (
    BoundaryRefinementConfig,
    boundary_refinement_config_from_config,
    refine_cues_to_speech_activity,
)


@pytest.mark.parametrize("text", ["don't go", "50% complete", "안녕하세요 여러분"])
def test_forced_alignment_maps_provider_words_without_stealing_next_cue(text):
    first_words = text.split()
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=[text]),
        Cue(index=2, start_ms=5000, end_ms=6000, lines=["next line"]),
    ]
    rows = [
        {"text": word, "start": 1.0 + index * 0.3, "end": 1.2 + index * 0.3}
        for index, word in enumerate(first_words)
    ] + [
        {"text": "next", "start": 5.0, "end": 5.2},
        {"text": "line", "start": 5.3, "end": 5.5},
    ]

    aligned = _cue_alignments_from_word_timestamps(cues, rows)

    assert [(row.cue_id, row.start, row.end) for row in aligned] == [
        (1, 1.0, 1.5),
        (2, 5.0, 5.5),
    ]


def test_forced_alignment_incomplete_lexical_stream_holds_source_for_review():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["hello there"]),
        Cue(index=2, start_ms=5000, end_ms=6000, lines=["next line"]),
    ]
    rows = [
        {"text": "hello", "start": 1.0, "end": 1.2},
        {"text": "next", "start": 5.0, "end": 5.2},
        {"text": "line", "start": 5.3, "end": 5.5},
    ]

    aligned = _cue_alignments_from_word_timestamps(cues, rows)
    updated, flags = apply_forced_alignment(cues, aligned, StyleProfile())

    assert updated == cues
    assert {cue_id for flag in flags if flag.kind == "forced_alignment_unresolved" for cue_id in flag.cue_ids} == {1, 2}


@pytest.mark.parametrize("start,end", [(2.0, 1.0), (1.0, 1.0), (float("nan"), 2.0), (1.0, float("inf"))])
def test_forced_alignment_invalid_timing_retains_source_instead_of_inventing_interval(start, end):
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["hello"])
    alignment = ForcedAlignmentCue.model_construct(cue_id=1, start=start, end=end, score=1.0)

    updated, flags = apply_forced_alignment([cue], [alignment], StyleProfile())

    assert updated == [cue]
    assert any(flag.kind == "forced_alignment_unresolved" and flag.cue_ids == [1] for flag in flags)


def test_short_final_utterance_minimum_duration_cannot_create_silence_tail():
    cue = Cue(index=1, start_ms=1000, end_ms=1050, lines=["yes"])
    profile = StyleProfile(fps=25.0, min_cue_dur=0.5)

    updated, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=1.05)], profile
    )

    assert updated[0].end_ms <= profile.snap_ceil(1050 + 40)
    assert any(flag.kind == "min_duration_unattainable" for flag in flags)


def test_boundary_refinement_preserves_simultaneous_speech_instead_of_zero_duration():
    cues = [
        Cue(index=1, start_ms=100, end_ms=200, lines=["yes"], speaker_id="A"),
        Cue(index=2, start_ms=100, end_ms=300, lines=["no"], speaker_id="B"),
    ]
    words = [Word(text="yes", start=0.10, end=0.20), Word(text="no", start=0.11, end=0.30)]

    updated, _flags = refine_cues_to_speech_activity(
        cues,
        [SpeechRegion(start=0.1, end=0.3)],
        StyleProfile(fps=30.0),
        words=words,
        alignment=AlignmentResult(cue_word_indices={1: [0], 2: [1]}),
    )

    assert updated[0].start_ms == 100
    assert updated[0].end_ms >= 200
    assert updated[1].start_ms == 100
    assert updated[1].end_ms >= 300
    assert all(cue.duration_ms > 0 for cue in updated)


def test_boundary_refinement_does_not_collapse_cues_in_source_order_inversion():
    cues = [
        Cue(index=1, start_ms=5000, end_ms=5500, lines=["first source cue"]),
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["second source cue"]),
    ]

    updated, _flags = refine_cues_to_speech_activity(
        cues,
        [SpeechRegion(start=1.0, end=1.5), SpeechRegion(start=5.0, end=5.5)],
        StyleProfile(fps=25.0),
    )

    assert updated[0] == cues[0]
    assert [cue.index for cue in updated] == [1, 2]
    assert all(cue.duration_ms > 0 for cue in updated)


def test_forced_alignment_maps_character_rows_and_ignores_non_speech_cues():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["[Title]", "你好！"]),
        Cue(index=2, start_ms=2000, end_ms=3000, lines=["[Screen title]"]),
        Cue(index=3, start_ms=5000, end_ms=6000, lines=["再见"]),
    ]
    rows = [
        {"text": char, "start": start, "end": start + 0.2}
        for char, start in [("你", 1.0), ("好", 1.3), ("！", 1.5), ("再", 5.0), ("见", 5.3)]
    ]

    aligned = _cue_alignments_from_word_timestamps(cues, rows)

    assert [(row.cue_id, row.start, row.end) for row in aligned] == [(1, 1.0, 1.5), (3, 5.0, 5.5)]
    updated, flags = apply_forced_alignment(cues, aligned, StyleProfile())
    assert updated[1] == cues[1]
    assert not any(2 in flag.cue_ids for flag in flags)


@pytest.mark.parametrize(
    "row",
    [
        {"text": "hello", "end": 1.2},
        {"text": "hello", "start": 1.0, "end": 1.2, "score": float("nan")},
        {"text": "hello", "start": 1.0, "end": 1.2, "score": 1.1},
    ],
)
def test_forced_alignment_missing_timestamps_or_invalid_scores_are_reviewable(row):
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["hello"])

    aligned = _cue_alignments_from_word_timestamps([cue], [row])
    updated, flags = apply_forced_alignment([cue], aligned, StyleProfile())

    assert updated == [cue]
    assert flags[0].kind == "forced_alignment_unresolved"


def test_forced_alignment_duplicate_cue_results_do_not_silently_choose_last():
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["hello"])
    alignments = [
        ForcedAlignmentCue(cue_id=1, start=3.0, end=4.0),
        ForcedAlignmentCue(cue_id=1, start=30.0, end=40.0),
    ]

    updated, flags = apply_forced_alignment([cue], alignments, StyleProfile())

    assert updated == [cue]
    assert flags[0].kind == "forced_alignment_unresolved"


def test_forced_minimum_duration_cannot_cross_next_trusted_speech_onset():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["yes"]),
        Cue(index=2, start_ms=3000, end_ms=4000, lines=["next"]),
    ]
    alignments = [
        ForcedAlignmentCue(cue_id=1, start=1.0, end=1.08),
        ForcedAlignmentCue(cue_id=2, start=1.12, end=1.8),
    ]

    updated, flags = apply_forced_alignment(cues, alignments, StyleProfile(fps=25.0))

    assert updated[0].end_ms <= updated[1].start_ms == 1120
    assert any(flag.kind == "min_duration_unattainable" and flag.cue_ids == [1] for flag in flags)


def test_region_duration_clamping_keeps_short_connected_regions_and_stops_at_silence():
    word = Word(text="hello", start=1.0, end=2.9)
    regions = [
        SpeechRegion(start=1.0, end=1.2),
        SpeechRegion(start=1.3, end=1.5),
        SpeechRegion(start=2.5, end=3.0),
    ]

    clamped, flags = clamp_asr_word_durations([word], regions)

    assert clamped[0].end == 1.5
    assert word.end == 2.9
    assert flags[0].kind == "asr_word_clamped"


def test_shared_boundary_policy_keeps_sync_defaults_and_provider_overrides():
    assert boundary_refinement_config_from_config({}) == BoundaryRefinementConfig(enabled=False)
    assert boundary_refinement_config_from_config({"vad": {"boundary_refinement": True}}) == BoundaryRefinementConfig()
    configured = boundary_refinement_config_from_config(
        {
            "timing": {"max_word_duration": 1.5},
            "vad": {"boundary_refinement": {"start_pad_ms": 20, "max_trailing_silence_ms": 100}},
        }
    )
    assert configured == BoundaryRefinementConfig(start_pad_ms=20, max_trailing_silence_ms=100, max_word_duration_ms=1500)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_shared_boundary_policy_rejects_nonfinite_or_nonpositive_word_limits(value):
    with pytest.raises(ValueError, match="timing.max_word_duration"):
        boundary_refinement_config_from_config({"vad": {"boundary_refinement": True}, "timing": {"max_word_duration": value}})
