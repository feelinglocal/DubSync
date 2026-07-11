from __future__ import annotations

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


def test_snap_ceil_never_returns_before_fractional_millisecond_input():
    profile = StyleProfile(fps=30.0)

    snapped = profile.snap_ceil(1033.4)

    assert snapped >= 1033.4
    assert snapped == 1066
