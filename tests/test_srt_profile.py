from __future__ import annotations

import pytest

from dubsync.models import Cue
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile, derive_style_profile


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


def test_snap_ceil_never_returns_before_fractional_millisecond_input():
    profile = StyleProfile(fps=30.0)

    snapped = profile.snap_ceil(1033.4)

    assert snapped >= 1033.4
    assert snapped == 1066


def test_write_srt_rejects_blank_cue_instead_of_serializing_invalid_block():
    cues = [Cue(index=27, start_ms=59933, end_ms=61533, lines=[""])]

    with pytest.raises(ValueError, match="cue 27 has no subtitle text"):
        write_srt(cues)
