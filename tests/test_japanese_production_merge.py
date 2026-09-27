from __future__ import annotations

import pytest

from dubsync.models import Cue, Word
from dubsync.punctuation import StaticPunctuationAdapter, apply_punctuation_pass
from dubsync.style_profile import StyleProfile
from dubsync.transcription import _build_cues_with_word_ownership


@pytest.mark.parametrize(
    ("source_lines", "proposal", "expected"),
    [
        (["今日は東京へ", "行きます"], "今日は、東京へ行きます。", ["今日は、東京へ", "行きます。"]),
        (["ｶﾞﾗｽと空は", "青い"], "ｶﾞﾗｽと、空は青い。", ["ｶﾞﾗｽと、空は", "青い。"]),
        (["か\u3099らすと空は", "青い"], "か\u3099らすと、空は青い。", ["か\u3099らすと、空は", "青い。"]),
    ],
)
def test_japanese_punctuation_retains_authored_source_line_boundaries(source_lines, proposal, expected):
    cue = Cue(index=1, start_ms=0, end_ms=3000, lines=source_lines)

    updated, flags = apply_punctuation_pass([cue], StaticPunctuationAdapter({1: proposal}))

    assert flags == []
    assert updated[0].lines == expected
    assert (updated[0].start_ms, updated[0].end_ms) == (0, 3000)


def test_japanese_punctuation_grouping_preserves_exact_acoustic_word_ownership():
    texts = ["「", "準備", "できました", "。", "」", "次", "です。"]
    words = [
        Word(text=text, start=index * 0.25, end=index * 0.25 + 0.2, speaker_id="A")
        for index, text in enumerate(texts)
    ]
    before = [word.model_dump() for word in words]
    profile = StyleProfile(fps=25, max_chars_per_line=80, tail_ms=0)

    cues, alignment = _build_cues_with_word_ownership(
        words, profile, max_gap_seconds=0.8, max_cue_duration_seconds=5.0, preserve_timing=True,
    )

    assert [cue.text for cue in cues] == ["「準備できました。」", "次です。"]
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4], 2: [5, 6]}
    assert cues[0].end_ms == profile.snap_ceil(words[4].end * 1000)
    assert cues[1].start_ms == profile.snap_floor(words[5].start * 1000)
    assert [word.model_dump() for word in words] == before
