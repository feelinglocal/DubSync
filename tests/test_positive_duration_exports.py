from __future__ import annotations

import pytest

from dubsync.cache import write_text_atomic
from dubsync.models import Cue
from dubsync.output_order import finalize_cues_for_output
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile


# Exact malformed timing from the customer file. Parsing remains permissive so
# source files can be inspected/repaired, but this must never be downloadable.
REVERSED_EU = "548\n00:24:51,800 --> 00:24:50,666\nEu\n"


@pytest.mark.parametrize("renumber", [False, True])
def test_export_rejects_customer_reversed_eu_without_changing_its_text(renumber):
    cues = parse_srt_text(REVERSED_EU)
    original = cues[0].model_copy(deep=True)

    with pytest.raises(ValueError, match=r"cue 548.*non-positive duration"):
        write_srt(cues, renumber=renumber)

    assert cues == [original]
    assert cues[0].plain_text == "Eu"


def test_export_rejects_zero_duration_dialogue():
    cue = Cue(index=8, start_ms=2100, end_ms=2100, lines=["Wait!"])

    with pytest.raises(ValueError, match=r"cue 8.*non-positive duration"):
        write_srt([cue])


def test_invalid_export_keeps_previous_artifact_and_source_text(tmp_path):
    output_path = tmp_path / "final.srt"
    prior_output = "1\n00:00:01,000 --> 00:00:02,000\nPrevious accepted output.\n"
    output_path.write_text(prior_output, encoding="utf-8")
    cues = parse_srt_text(REVERSED_EU)

    with pytest.raises(ValueError, match=r"cue 548.*non-positive duration"):
        write_text_atomic(output_path, write_srt(cues, renumber=True))

    assert output_path.read_text(encoding="utf-8") == prior_output
    assert cues[0].plain_text == "Eu"
    assert list(tmp_path.iterdir()) == [output_path]


def test_export_rejects_negative_start_with_cue_identification():
    cue = Cue(index=8, start_ms=-100, end_ms=2100, lines=["Wait!"])

    with pytest.raises(ValueError, match=r"cue 8.*negative start"):
        write_srt([cue])


@pytest.mark.parametrize("preserve_timing", [False, True])
@pytest.mark.parametrize("cue_kind", ["dialogue", "protected", "screen_text"])
def test_final_order_cannot_export_or_mask_invalid_timing(cue_kind, preserve_timing):
    cue = parse_srt_text(REVERSED_EU)[0]
    if cue_kind == "screen_text":
        cue = cue.with_lines(["[Eu]"])
    before = cue.model_copy(deep=True)

    with pytest.raises(ValueError, match=r"cue 548.*non-positive duration"):
        finalize_cues_for_output(
            [cue],
            StyleProfile(),
            preserve_timing=preserve_timing,
            protected_cue_ids={548} if cue_kind == "protected" else set(),
            # Reading-speed extension must not fabricate a valid-looking fix.
            max_cps=30.0,
            media_duration_ms=3_000_000,
        )

    assert cue == before


def test_valid_simultaneous_speakers_and_short_utterances_remain_exportable():
    cues = [
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["Yes."], speaker_id="a"),
        Cue(index=3, start_ms=1300, end_ms=1301, lines=["No."], speaker_id="b"),
    ]

    finalized, flags = finalize_cues_for_output(
        cues, StyleProfile(), preserve_timing=True,
    )
    restored = parse_srt_text(write_srt(finalized))

    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in restored] == [
        (1000, 1500, "Yes."), (1300, 1301, "No."),
    ]
    assert any(flag.kind == "output_overlap_unresolved" for flag in flags)
