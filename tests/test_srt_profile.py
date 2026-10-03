from __future__ import annotations

import pytest

from dubsync.models import Cue
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile, derive_style_profile, detect_fps_with_confidence


def test_example_srt_round_trips_with_only_trailing_space_normalization(sample_srt_path):
    original = sample_srt_path.read_text(encoding="utf-8-sig")

    cues = parse_srt_text(original)
    rendered = write_srt(cues)

    assert len(cues) == 68
    expected = "\n".join(
        line.rstrip()
        for line in original.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip() + "\n"
    assert rendered == expected


def test_profile_derivation_matches_plan_house_style(sample_srt_path):
    cues = parse_srt_text(sample_srt_path.read_text(encoding="utf-8-sig"))

    profile = derive_style_profile(cues)

    assert profile.fps == 30.0
    assert profile.frame_ms == 1000 / 30
    assert profile.max_lines_per_cue == 2
    assert profile.max_chars_per_line == 26
    assert profile.min_cue_dur == 0.5
    assert profile.allow_zero_gap is True
    assert profile.cue_count == 68


def test_profile_derivation_uses_robust_limits_instead_of_single_outliers():
    cues = [
        Cue(
            index=index + 1,
            start_ms=index * 1000,
            end_ms=index * 1000 + (100 if index == 0 else 500),
            lines=["x" * (48 if index == 0 else 26)],
        )
        for index in range(20)
    ]

    profile = derive_style_profile(cues)

    assert 26 <= profile.max_chars_per_line < 48
    assert 0.1 < profile.min_cue_dur <= 0.5
    assert profile.observed_min_duration == 0.1


@pytest.mark.parametrize("timing", ["00:00:03,500 --> 00:00:03,000", "00:00:03,500 --> 00:00:03,500"])
def test_profile_ignores_zero_and_reversed_source_durations(timing):
    # An unusable source duration is a timing error, not a house style: it must
    # neither fail the profile nor switch off the minimum display duration.
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n\n"
        f"2\n{timing}\nWie geht es dir?\n\n"
        "3\n00:00:05,000 --> 00:00:06,500\nGut, danke.\n"
    )

    profile = derive_style_profile(cues)

    assert profile.min_cue_dur == 0.5
    assert profile.observed_min_duration == 1.0
    assert profile.observed_max_duration == 1.5
    assert profile.cue_count == 3
    only_unusable = derive_style_profile(cues[1:2])
    assert only_unusable.min_cue_dur == 0.5
    assert only_unusable.observed_min_duration is None


def test_detect_fps_low_confidence_unsnapped_file_falls_back_to_default():
    cues = [
        Cue(index=1, start_ms=101, end_ms=923, lines=["alpha"]),
        Cue(index=2, start_ms=1277, end_ms=2089, lines=["beta"]),
        Cue(index=3, start_ms=2603, end_ms=3491, lines=["gamma"]),
    ]

    detection = detect_fps_with_confidence(cues)

    assert detection.fps == 30.0
    assert detection.confident is False
    assert detection.best_error_ms > 2.0


def test_detect_fps_reports_low_error_for_a_frame_snapped_grid():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["alpha"]),
        Cue(index=2, start_ms=2000, end_ms=3000, lines=["beta"]),
        Cue(index=3, start_ms=4000, end_ms=5000, lines=["gamma"]),
    ]

    detection = detect_fps_with_confidence(cues, default=25.0)

    assert detection.fps in {24.0, 25.0, 30.0}
    assert detection.best_error_ms <= 2.0


def test_snap_ceil_never_returns_before_fractional_millisecond_input():
    profile = StyleProfile(fps=30.0)

    snapped = profile.snap_ceil(1033.4)

    assert snapped >= 1033.4
    # Frame 31 (1033.33 ms) is written as 1034 ms, which already covers the input.
    assert snapped == 1034


def test_write_srt_rejects_blank_cue_instead_of_serializing_invalid_block():
    cues = [Cue(index=27, start_ms=59933, end_ms=61533, lines=[""])]

    with pytest.raises(ValueError, match="cue 27 has no subtitle text"):
        write_srt(cues)
