from __future__ import annotations

import pytest

from dubsync import text_metrics
from dubsync.models import Cue
from dubsync.tokenize import alphanumeric_signature, normalize_token, tokenize_cues


@pytest.mark.parametrize("plain,voiced", [("か", "が"), ("は", "ば"), ("は", "ぱ"), ("ハ", "パ")])
def test_japanese_voicing_remains_distinct(plain: str, voiced: str):
    assert normalize_token(plain) != normalize_token(voiced)


def test_japanese_width_and_combining_forms_compare_equally_without_changing_source():
    source = "ｶﾞｯﾂﾎﾟｰｽﾞ！か\u3099んばる。"
    cue = Cue(index=1, start_ms=0, end_ms=1000, lines=[source])

    assert alphanumeric_signature(source) == alphanumeric_signature("ガッツポーズ！がんばる。")
    assert tokenize_cues([cue])
    assert cue.lines == [source]


def test_japanese_small_kana_remains_distinct():
    assert normalize_token("や") != normalize_token("ゃ")


def test_noncomposing_kana_voicing_remains_lexical():
    assert normalize_token("セ\u309a") != normalize_token("セ")
    assert alphanumeric_signature("セ\u309a") != alphanumeric_signature("セ")


def test_japanese_supplementary_kanji_and_iteration_marks_are_individual_tokens():
    assert text_metrics.token_texts("𠮷𠮟と人々々") == ["𠮷", "𠮟", "と", "人", "々", "々"]


def test_latin_comparison_behavior_is_preserved():
    assert normalize_token("FÜNF") == "5"
    assert normalize_token("café") == "cafe"
    assert normalize_token("Straße") == "strasse"


@pytest.mark.parametrize(
    "parts,expected",
    [
        (["今日", "は", "晴れ", "です", "。"], "今日は晴れです。"),
        (["「", "東京", "です", "。", "」", "と", "話した", "。"], "「東京です。」と話した。"),
        (["日本語", "と", "OpenAI", "API", "です。"], "日本語とOpenAI APIです。"),
        (["ｶﾞｯﾂ", "ﾎﾟｰｽﾞ", "！"], "ｶﾞｯﾂﾎﾟｰｽﾞ！"),
        (["｢", "A", "B", "｣", "｡"], "｢A B｣｡"),
        (["Hello", "world."], "Hello world."),
        (["Hello", ",", "world", "!"], "Hello, world!"),
        (["한국어", "문장"], "한국어 문장"),
        (["", " 東京 ", "", "です。"], "東京です。"),
    ],
)
def test_asr_word_joining_uses_japanese_boundaries(parts: list[str], expected: str):
    assert text_metrics.join_word_texts(parts) == expected


@pytest.mark.parametrize(
    "source,width,expected",
    [
        ("今日は、晴れです。", 6, ["今日", "は、晴", "れで", "す。"]),
        ("東京「晴天」です", 6, ["東京", "「晴", "天」で", "す"]),
        ("あいうゃえお", 6, ["あい", "うゃえ", "お"]),
        ("あいうーえお", 6, ["あい", "うーえ", "お"]),
    ],
)
def test_japanese_wrapping_prefers_natural_punctuation_boundaries(source: str, width: int, expected: list[str]):
    lines = text_metrics.wrap_visual_width(source, width)

    assert lines == expected
    assert "".join(lines) == source
    assert all(text_metrics.display_width(line) <= width for line in lines)


@pytest.mark.parametrize("width", [0, 1, 2, 3])
def test_narrow_japanese_wrapping_preserves_clusters_and_punctuation(width: int):
    source = "「か\u3099っ。」「ｶﾞ！」"

    lines = text_metrics.wrap_visual_width(source, width)

    assert lines
    assert "".join(lines) == source
    assert all(not line.startswith(("\u3099", "ﾞ", "。", "！", "」", "っ")) for line in lines)
    assert all(not line.endswith("「") for line in lines)


def test_japanese_wrapping_preserves_latin_word_wrapping_and_combining_width():
    assert text_metrics.wrap_visual_width("Hello world", 6) == ["Hello", "world"]
    assert text_metrics.display_width("か\u3099") == text_metrics.display_width("が") == 2
