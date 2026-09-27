from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions, flow_text_to_lines
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.punctuation import (
    PunctuationValidationError,
    StaticPunctuationAdapter,
    apply_punctuation_pass,
    validate_punctuation_only,
)
from dubsync.style_profile import StyleProfile


@pytest.mark.parametrize(
    ("source", "span_text", "replacement", "expected"),
    [
        ("昨日、「東京へ行く」と言った。", "東京", "京都", "昨日、「京都へ行く」と言った。"),
        ("今日は、雨です。", "雨", "雪", "今日は、雪です。"),
        ("「ガラスは青い。」", "青い", "赤い", "「ガラスは赤い。」"),
        ("「か\u3099らすは青い。」", "青い", "赤い", "「か\u3099らすは赤い。」"),
        ("「ｶﾞﾗｽは青い。」", "ガラス", "空", "「空は青い。」"),
        ("「ガラスは青い。」", "青い", "赤い。", "「ガラスは赤い。」"),
        ("「ガラスは青い。」", "青い", "「ガラスは赤い。」", "「ガラスは赤い。」"),
        ("He said, ‘old words’, then left.", "old words", "new words", "He said, ‘new words’, then left."),
        ("He said old words.", "old words", "He said new words.", "He said new words."),
    ],
)
def test_partial_replacement_preserves_surrounding_text(source, span_text, replacement, expected):
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=[source])
    span = DivergenceSpan(case_id="ja-1", cue_ids=[1], srt_text=span_text, asr_text=replacement)
    decision = AdjudicationDecision(
        case_id="ja-1", verdict="use_audio", final_text=replacement,
        confidence=0.95, reason="spoken correction",
    )

    changed, flags = apply_adjudication_decisions(
        [cue], [span], [decision], StyleProfile(max_chars_per_line=100),
    )

    assert changed[0].text == expected
    assert (changed[0].start_ms, changed[0].end_ms) == (1000, 3000)
    assert flags[0].new_text == expected


@pytest.mark.parametrize("text", ["今日は東京から京都へ行きます。", "東京OpenAIChatGPT会議へ行く。"])
def test_overflow_line_preserves_unspaced_japanese_text(text):
    lines = flow_text_to_lines(text, max_chars=8, max_lines=2)

    assert len(lines) == 2
    assert "".join(lines) == text


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("今日は晴れです", "今日は、晴れです。"),
        ("私はAIを使う", "「私は、AIを使う。」"),
        ("か\u3099らす", "がらす。"),
        ("セ\u309aとカ\u309a", "セ\u309a、とカ\u309a。"),
        ("don't re-enter", "Don't, re enter."),
    ],
)
def test_punctuation_accepts_japanese_punctuation_and_canonical_equivalence(before, after):
    assert validate_punctuation_only(before, after) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("か\u3099らす", "からす。"),
        ("がらす", "からす。"),
        ("は\u309aん", "はん。"),
        ("セ\u309a", "セ。"),
        ("カ\u309a", "カ。"),
    ],
)
def test_punctuation_does_not_remove_kana_voicing(before, after):
    with pytest.raises(PunctuationValidationError):
        validate_punctuation_only(before, after)


def test_punctuation_reflow_does_not_invent_spaces_between_japanese_lines():
    cue = Cue(index=1, start_ms=0, end_ms=3000, lines=["今日は東京へ行きます"])
    adapter = StaticPunctuationAdapter({1: "今日は、\n東京へ\n行きます。"})

    updated, flags = apply_punctuation_pass(
        [cue], adapter, max_chars_per_line=30, max_lines_per_cue=2,
    )

    assert flags == []
    assert updated[0].text == "今日は、東京へ行きます。"


def test_repeated_japanese_character_replacement_uses_global_source_index():
    cues = [
        Cue(index=4, start_ms=0, end_ms=900, lines=["前の字幕"]),
        Cue(index=9, start_ms=1000, end_ms=3000, lines=["ここはここです。"]),
    ]
    span = DivergenceSpan(
        case_id="repeat", cue_ids=[9], srt_text="こ", asr_text="そ", srt_token_indices=[7],
    )
    decision = AdjudicationDecision(
        case_id="repeat", verdict="use_audio", final_text="そ", confidence=0.95, reason="spoken correction",
    )

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())

    assert changed[0] == cues[0]
    assert changed[1].text == "ここはそこです。"


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("day", ["明", "明後"])
@pytest.mark.parametrize("full_context", [False, True])
def test_disjoint_japanese_corrections_accumulate_in_one_cue(reverse_order, day, full_context):
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=["「今日は雨です。」"])
    spans = [
        DivergenceSpan(case_id="day", cue_ids=[1], srt_text="今", asr_text=day, srt_token_indices=[0]),
        DivergenceSpan(case_id="weather", cue_ids=[1], srt_text="雨", asr_text="雪", srt_token_indices=[3]),
    ]
    decisions = [
        AdjudicationDecision(
            case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
            confidence=0.95, reason="spoken correction",
        )
        for span in spans
    ]
    if full_context:
        decisions[0].final_text = f"「{day}日は雨です。」"
        decisions[1].final_text = "「今日は雪です。」"
    if reverse_order:
        spans.reverse()

    changed, flags = apply_adjudication_decisions(
        [cue], spans, decisions, StyleProfile(max_chars_per_line=100),
    )

    expected = f"「{day}日は雪です。」"
    assert changed[0].text == expected
    assert all(flag.new_text == expected for flag in flags)


def test_disjoint_latin_corrections_preserve_spacing_and_repeated_word_position():
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=["Old birds, unlike old trees, fly."])
    spans = [
        DivergenceSpan(case_id="birds", cue_ids=[1], srt_text="birds", asr_text="owls", srt_token_indices=[1]),
        DivergenceSpan(case_id="trees", cue_ids=[1], srt_text="old", asr_text="young", srt_token_indices=[3]),
    ]
    decisions = [
        AdjudicationDecision(
            case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
            confidence=0.95, reason="spoken correction",
        )
        for span in spans
    ]

    changed, _ = apply_adjudication_decisions(
        [cue], spans, decisions, StyleProfile(max_chars_per_line=100),
    )

    assert changed[0].text == "Old owls, unlike young trees, fly."


@pytest.mark.parametrize("source,spoken", [("[画面]「ｶﾞﾗｽは青い。」", "ｶﾞﾗｽ"), ("[画面]「か\u3099らすは青い。」", "か\u3099らす")])
def test_japanese_normalized_tokens_preserve_screen_annotations(source, spoken):
    from dubsync.subtitle_annotations import alignment_token_character_spans
    from dubsync.tokenize import tokenize_cues

    cue = Cue(index=1, start_ms=0, end_ms=2000, lines=[source])
    boundaries = alignment_token_character_spans(cue)
    assert boundaries is not None
    assert source[boundaries[0][0]:boundaries[2][1]] == spoken
    tokens = tokenize_cues([cue])
    blue_index = next(token.token_index for token in tokens if token.text == "青")
    span = DivergenceSpan(case_id="screen-ja", cue_ids=[1], srt_text="青", asr_text="赤", srt_token_indices=[blue_index])
    decision = AdjudicationDecision(case_id="screen-ja", verdict="use_audio", final_text="赤", confidence=0.95, reason="spoken correction")
    changed, flags = apply_adjudication_decisions([cue], [span], [decision], StyleProfile(max_chars_per_line=100))
    assert changed[0].text == source.replace("青", "赤")
    assert not any(flag.kind == "screen_text_adjudication_held" for flag in flags)


def test_japanese_anchored_insertion_does_not_add_ascii_spaces():
    cue = Cue(index=1, start_ms=0, end_ms=2000, lines=["「今日はです。」"])
    span = DivergenceSpan(case_id="insert-ja", cue_ids=[], srt_text="", asr_text="晴れ", left_anchor_cue_id=1, right_anchor_cue_id=1, insertion_token_offset=3)
    decision = AdjudicationDecision(case_id="insert-ja", verdict="use_audio", final_text="晴れ", confidence=0.95, reason="spoken insertion")
    changed, _ = apply_adjudication_decisions([cue], [span], [decision], StyleProfile(), adlib_cue_ids_by_case={"insert-ja": 1})
    assert changed[0].text == "「今日は晴れです。」"
