from __future__ import annotations

from dubsync.models import Cue, SpeechRegion
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity


def test_refine_cues_to_speech_activity_tightens_obvious_silence_boundaries():
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5)
    cues = [
        Cue(index=13, start_ms=40466, end_ms=43600, lines=["Heute ist der Tag,"]),
        Cue(index=51, start_ms=118900, end_ms=121700, lines=["Diese Energie..."]),
        Cue(index=60, start_ms=130000, end_ms=132500, lines=["tail too long"]),
    ]
    regions = [
        SpeechRegion(start=42.4, end=43.3),
        SpeechRegion(start=120.6, end=121.8),
        SpeechRegion(start=130.1, end=131.0),
    ]

    refined, flags = refine_cues_to_speech_activity(
        cues,
        regions,
        profile,
        BoundaryRefinementConfig(
            start_pad_ms=40,
            end_pad_ms=40,
            max_end_extension_ms=300,
            max_leading_silence_ms=150,
            max_trailing_silence_ms=300,
        ),
    )

    assert refined[0].start_ms == 42333
    assert refined[0].end_ms == 43600
    assert refined[1].start_ms == 120533
    assert refined[1].end_ms == 121866
    assert refined[2].start_ms == 130000
    assert refined[2].end_ms == 131066
    assert [flag.cue_ids for flag in flags] == [[13], [51], [60]]
    assert all(flag.kind == "timing_refined" for flag in flags)


def test_refine_cues_to_speech_activity_preserves_timing_without_overlap():
    cue = Cue(index=3, start_ms=5133, end_ms=6333, lines=["als Drachenbändiger", "erwacht?"])

    refined, flags = refine_cues_to_speech_activity(
        [cue],
        [SpeechRegion(start=5.1, end=6.3)],
        StyleProfile(fps=30.0),
        BoundaryRefinementConfig(max_leading_silence_ms=150, max_trailing_silence_ms=300),
    )

    assert refined == [cue]
    assert flags == []


def test_refine_cues_to_speech_activity_does_not_extend_to_following_cues_in_merged_region():
    cue = Cue(index=5, start_ms=10233, end_ms=11133, lines=["brachte mir nur Verrat"])

    refined, flags = refine_cues_to_speech_activity(
        [cue],
        [SpeechRegion(start=10.2, end=14.7)],
        StyleProfile(fps=30.0),
        BoundaryRefinementConfig(max_end_extension_ms=300),
    )

    assert refined == [cue]
    assert flags == []


def test_refine_cues_to_speech_activity_caps_extension_at_next_cue_start():
    cues = [
        Cue(index=48, start_ms=109700, end_ms=111400, lines=["niemand ist besser"]),
        Cue(index=49, start_ms=111400, end_ms=112400, lines=["als du."]),
    ]

    refined, flags = refine_cues_to_speech_activity(
        cues,
        [SpeechRegion(start=109.7, end=111.52), SpeechRegion(start=111.4, end=112.5)],
        StyleProfile(fps=30.0),
        BoundaryRefinementConfig(max_end_extension_ms=300),
    )

    assert refined[0].end_ms == 111400
    assert refined[1].start_ms == 111400
    assert all(left.end_ms <= right.start_ms for left, right in zip(refined, refined[1:]))
    assert not any(flag.cue_ids == [48] for flag in flags)


def test_refine_cues_to_speech_activity_does_not_undo_next_cue_cap_for_min_duration():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1050, lines=["short"]),
        Cue(index=2, start_ms=1100, end_ms=1600, lines=["next"]),
    ]

    refined, flags = refine_cues_to_speech_activity(
        cues,
        [SpeechRegion(start=1.0, end=1.05), SpeechRegion(start=1.1, end=1.6)],
        StyleProfile(fps=30.0, min_cue_dur=0.5),
        BoundaryRefinementConfig(max_end_extension_ms=300),
    )

    assert refined[0].end_ms <= refined[1].start_ms
    assert refined[0].end_ms == 1100
    assert flags[0].kind == "timing_refined"
