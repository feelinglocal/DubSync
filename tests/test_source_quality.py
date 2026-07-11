from __future__ import annotations

from dubsync.source_quality import detect_source_errors
from dubsync.srt_io import parse_srt_text


def test_example_scrambled_cues_are_flagged_as_source_error(sample_srt_path):
    cues = parse_srt_text(sample_srt_path.read_text(encoding="utf-8-sig"))

    flags = detect_source_errors(cues)

    source_error_cues = {cue_id for flag in flags if flag.kind == "source_error" for cue_id in flag.cue_ids}
    assert {33, 34, 35}.issubset(source_error_cues)


def test_source_error_flags_include_affected_timestamp_window():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nalpha beta gamma\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nalpha beta\n\n"
    )

    flags = detect_source_errors(cues)

    assert flags[0].kind == "source_error"
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].start == 0.0
    assert flags[0].end == 2.0
