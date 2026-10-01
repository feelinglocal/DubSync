"""Source layout regressions from the fresh ep11 Scribe final-v3 artifact."""

import pytest

from dubsync.cue_segmentation import split_speaker_turn_cues
from dubsync.models import AlignmentResult, Cue, Word
from dubsync.style_profile import StyleProfile


def _photo_words(*, collapsed_second_actor=True):
    # Live source439 owns Scribe words1787..1790. Both second-actor words
    # have the same 1ms placeholder interval; this is not acoustic ground truth.
    return [
        Word(text="Rápido,", start=1300.738, end=1300.898, speaker_id="speaker_3"),
        Word(text="rápido.", start=1300.938, end=1301.058, speaker_id="speaker_3"),
        Word(
            text="Vem", start=1301.078,
            end=1301.079 if collapsed_second_actor else 1301.178,
            speaker_id="speaker_5",
        ),
        Word(
            text="cá,", start=1301.078 if collapsed_second_actor else 1301.198,
            end=1301.079 if collapsed_second_actor else 1301.358,
            speaker_id="speaker_5",
        ),
    ]


@pytest.mark.parametrize("source_lines", [
    ["- Rápido, rápido. - Vem cá."],
    ["- Rápido, rápido.", "- Vem cá."],
])
def test_live_ep11_held_actor_split_preserves_authored_source_lines(source_lines):
    source = Cue(
        index=439, start_ms=1300510, end_ms=1301310, lines=source_lines,
    )
    original = source.model_dump()
    alignment = AlignmentResult(cue_word_indices={439: [0, 1, 2, 3], 440: [4, 5]})

    cues, updated, flags, expansions = split_speaker_turn_cues(
        [source], _photo_words(), alignment, StyleProfile(fps=30, min_cue_dur=0.4),
    )

    assert [cue.model_dump() for cue in cues] == [original]
    assert source.model_dump() == original
    assert updated.cue_word_indices == alignment.cue_word_indices
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["timing_evidence_held"]
    assert flags[0].old_text == flags[0].new_text == source.text
    assert "collapsed" in flags[0].message.lower()


def test_confidently_timed_actors_still_split_authored_inline_turns():
    source = Cue(
        index=439, start_ms=1300510, end_ms=1301410,
        lines=["- Rápido, rápido. - Vem cá."],
    )
    alignment = AlignmentResult(cue_word_indices={439: [0, 1, 2, 3]})

    cues, updated, flags, expansions = split_speaker_turn_cues(
        [source], _photo_words(collapsed_second_actor=False), alignment,
        StyleProfile(fps=30, min_cue_dur=0.4),
    )

    assert [cue.lines for cue in cues] == [["Rápido, rápido."], ["Vem cá."]]
    assert [cue.speaker_id for cue in cues] == ["speaker_3", "speaker_5"]
    assert updated.cue_word_indices == {439: [0, 1], 440: [2, 3]}
    assert expansions == {439: [439, 440]}
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]
    assert cues[0].end_ms <= cues[1].start_ms
