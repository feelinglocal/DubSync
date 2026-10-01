from __future__ import annotations

import pytest

from dubsync.source_quality import detect_source_errors
from dubsync.srt_io import parse_srt_text


def _srt(*cues: tuple[float, float, str]) -> str:
    def stamp(seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        return f"00:00:{milliseconds // 1000:02d},{milliseconds % 1000:03d}"

    return "".join(
        f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n"
        for index, (start, end, text) in enumerate(cues, start=1)
    )


def test_example_scrambled_cues_are_flagged_as_source_error(sample_srt_path):
    cues = parse_srt_text(sample_srt_path.read_text(encoding="utf-8-sig"))

    flags = detect_source_errors(cues)

    source_error_cues = {cue_id for flag in flags if flag.kind == "source_error" for cue_id in flag.cue_ids}
    assert {33, 34, 35}.issubset(source_error_cues)
    # The documented dirty block is the only scrambled passage in the example.
    assert len(flags) == 1


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


@pytest.mark.parametrize(
    "cues",
    [
        # echoed question
        [(1.0, 2.5, "mas é um amor secreto."), (2.5, 3.5, "Amor secreto?")],
        # the same greeting from two characters
        [(1.0, 2.0, "Bom dia, Sr. Luan."), (2.0, 3.0, "Bom dia, Sr. Luan.")],
        [(1.0, 2.0, "Es ist kaputt."), (2.1, 3.0, "Es ist kaputt.")],
        # shared opening words only
        [(1.0, 2.0, "Ich bin kein..."), (2.0, 3.0, "Ich bin die beste...")],
        # a repeated fragment after a real pause is a restart, not a scrambled file
        [(1.0, 2.0, "Du solltest, du solltest gehen"), (3.5, 4.5, "du solltest,")],
        # song lyrics repeat by design
        [(1.0, 2.0, "♪Não posso esquecer você♪"), (2.0, 3.0, "♪Não posso esquecer♪")],
        # bracketed screen text repeats by design
        [(1.0, 2.0, "[Luan Nian] Tudo bem"), (2.0, 3.0, "[Luan Nian]")],
    ],
)
def test_ordinary_repeated_dialogue_is_not_a_source_error(cues):
    assert detect_source_errors(parse_srt_text(_srt(*cues))) == []
