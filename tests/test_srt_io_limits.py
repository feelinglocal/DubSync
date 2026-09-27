from __future__ import annotations

import pytest

from dubsync.srt_io import SRTParseError, SRTParseLimits, parse_srt_text


def _cue(index: int) -> str:
    return (
        f"{index}\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"Line {index}\n"
    )


def test_bounded_parser_rejects_excess_lines_before_parsing_all_input():
    text = "\n".join("x" for _ in range(100))

    with pytest.raises(SRTParseError, match="exceeds 8 lines"):
        parse_srt_text(
            text,
            limits=SRTParseLimits(max_lines=8, max_cues=10, max_line_chars=100),
        )


def test_bounded_parser_rejects_excess_cues():
    text = "\n".join(_cue(index) for index in range(1, 4))

    with pytest.raises(SRTParseError, match="exceeds 2 cues"):
        parse_srt_text(
            text,
            limits=SRTParseLimits(max_lines=20, max_cues=2, max_line_chars=100),
        )


def test_bounded_parser_rejects_an_overlong_line():
    text = f"1\n00:00:00,000 --> 00:00:00,500\n{'x' * 41}\n"

    with pytest.raises(SRTParseError, match="line 3 exceeds 40 characters"):
        parse_srt_text(
            text,
            limits=SRTParseLimits(max_lines=10, max_cues=2, max_line_chars=40),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lines": 0, "max_cues": 1, "max_line_chars": 1},
        {"max_lines": 1, "max_cues": 0, "max_line_chars": 1},
        {"max_lines": 1, "max_cues": 1, "max_line_chars": 0},
    ],
)
def test_parse_limits_must_be_positive(kwargs: dict[str, int]):
    with pytest.raises(ValueError, match="greater than zero"):
        SRTParseLimits(**kwargs)
