from __future__ import annotations

import json

import pytest

from dubsync.models import Word
from dubsync.semantic_output import wrap_generated_lines
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.transcription import build_cues_from_words, generate_srt_from_audio


def _words(texts: list[str]) -> list[Word]:
    return [
        Word(text=text, start=0.2 + index * 0.2, end=0.35 + index * 0.2, confidence=0.99, speaker_id="A")
        for index, text in enumerate(texts)
    ]


def test_japanese_generation_joins_asr_tokens_without_inserting_spaces():
    words = _words(["今日", "は", "OpenAI", "の", "API", "を", "使い", "ます。"])
    before = [word.model_dump() for word in words]
    profile = StyleProfile(fps=25, max_chars_per_line=80, tail_ms=0)

    cues = build_cues_from_words(words, profile)

    assert [cue.plain_text for cue in cues] == ["今日はOpenAIのAPIを使います。"]
    assert cues[0].start_ms == profile.snap_floor(words[0].start * 1000)
    assert cues[0].end_ms == profile.snap_ceil(words[-1].end * 1000)
    assert cues[0].speaker_id == "A"
    assert [word.model_dump() for word in words] == before


@pytest.mark.parametrize("ending", ["。", "！", "？", "。』", "！）", "？\u201d", "｡", "｡｣"])
def test_japanese_sentence_endings_split_cues_including_closing_quotes(ending):
    words = _words(["準備", "でき", f"ました{ending}", "次", "に", "進みます。"])

    cues = build_cues_from_words(words, StyleProfile(max_chars_per_line=80))

    assert [cue.plain_text for cue in cues] == [f"準備できました{ending}", "次に進みます。"]
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues, cues[1:]))


@pytest.mark.parametrize(("opening", "closing"), [("「", "」"), ("｢", "｣")])
def test_japanese_separate_closing_quote_stays_with_its_sentence(opening, closing):
    words = _words([opening, "準備", "できました", "。", closing, "次", "に", "進みます。"])

    cues = build_cues_from_words(words, StyleProfile(fps=25, max_chars_per_line=80, tail_ms=0))

    assert [cue.plain_text for cue in cues] == [f"{opening}準備できました。{closing}", "次に進みます。"]
    assert cues[0].end_ms >= words[4].end * 1000
    assert cues[1].start_ms <= words[5].start * 1000


def test_japanese_joining_does_not_remove_english_word_spaces():
    words = _words(["OpenAI", "API", "を", "使います。", "English", "words", "stay", "spaced."])

    cues = build_cues_from_words(words, StyleProfile(max_chars_per_line=80))

    assert [cue.plain_text for cue in cues] == ["OpenAI APIを使います。", "English words stay spaced."]


def test_japanese_cue_budget_measures_joined_text_without_artificial_spaces():
    cues = build_cues_from_words(
        _words(["日", "本", "語", "字", "幕"]),
        StyleProfile(max_chars_per_line=10, max_lines_per_cue=1),
    )

    assert [cue.lines for cue in cues] == [["日本語字幕"]]


def test_japanese_narrow_cues_keep_standalone_punctuation_with_dialogue():
    words = _words(["今日", "は", "「", "東京", "です", "。", "」"])

    cues = build_cues_from_words(words, StyleProfile(max_chars_per_line=8, max_lines_per_cue=1))

    assert [cue.lines for cue in cues] == [["今日は"], ["「東京"], ["です。」"]]
    assert "".join(cue.plain_text for cue in cues) == "今日は「東京です。」"


@pytest.mark.parametrize(("sentence", "width", "expected"), [
    ("今日はとても良い天気ですね。", 26, ["今日はとても", "良い天気ですね。"]),
    ("明日の朝、駅で会いましょう。", 26, ["明日の朝、", "駅で会いましょう。"]),
    ("散歩に行きませんか？", 16, ["散歩に", "行きませんか？"]),
    ("ちょっと待ってくださいね。", 16, ["ちょっと待って", "くださいね。"]),
    ("俺はただ正当防衛をしたまでだ。", 16, ["俺はただ正当防衛", "をしたまでだ。"]),
    # Greedy wrapping broke inside these kanji and katakana runs (看|護師, スマート|フォン).
    ("彼は大学病院の看護師です。", 16, ["彼は大学病院の", "看護師です。"]),
    ("これは最新型スマートフォンだよ。", 20, ["これは最新型", "スマートフォンだよ。"]),
])
def test_japanese_generation_wraps_like_synchronized_output(tmp_path, sentence, width, expected):
    # MAI and Scribe return one Japanese character per word.
    words = [
        Word(text=character, start=1.0 + index * 0.1, end=1.08 + index * 0.1, speaker_id="A")
        for index, character in enumerate(sentence)
    ]
    words_path = tmp_path / "words.json"
    words_path.write_text(json.dumps({"words": [word.model_dump() for word in words]}, ensure_ascii=False), encoding="utf-8")
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    profile = StyleProfile(fps=25, max_chars_per_line=width, tail_ms=0)
    audio_path = tmp_path / "ja.wav"
    audio_path.write_bytes(b"fixture audio")

    generate_srt_from_audio(
        audio_path, tmp_path / "ja.srt", tmp_path / "work", providers_path=providers_path,
        no_llm=True, language="ja", style_profile=profile,
    )

    cues = parse_srt_text((tmp_path / "ja.srt").read_text(encoding="utf-8"))
    # Balanced lines at phrase boundaries: no orphaned final kana, no break
    # inside a kanji run, no line opening with closing punctuation.
    assert [cue.lines for cue in cues] == [expected]
    assert (cues[0].start_ms, cues[0].end_ms) == (1000, profile.snap_ceil(words[-1].end * 1000))


_MORPHEMES = ["テーブル", "を", "ひっくり返した", "の", "は", "誰", "だ？"]


@pytest.mark.parametrize("width", [24, 26])
def test_japanese_generation_never_breaks_inside_a_multi_character_asr_word(tmp_path, width):
    # 'ひっくり返した' is one ASR word: 'テーブルをひっく' / 'り返したのは誰だ？'
    # broke inside it although 'テーブルを' / 'ひっくり返したのは誰だ？' fits (review W4C-3).
    words = [Word(text=text, start=1.0 + index * 0.3, end=1.25 + index * 0.3, speaker_id="A")
             for index, text in enumerate(_MORPHEMES)]
    words_path = tmp_path / "words.json"
    words_path.write_text(json.dumps({"words": [word.model_dump() for word in words]}, ensure_ascii=False),
                          encoding="utf-8")
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    audio_path = tmp_path / "ja.wav"
    audio_path.write_bytes(b"fixture audio")

    generate_srt_from_audio(
        audio_path, tmp_path / "ja.srt", tmp_path / "work", providers_path=providers_path,
        no_llm=True, language="ja", style_profile=StyleProfile(fps=25, max_chars_per_line=width, tail_ms=0),
    )

    cues = parse_srt_text((tmp_path / "ja.srt").read_text(encoding="utf-8"))
    assert [cue.lines for cue in cues] == [["テーブルを", "ひっくり返したのは誰だ？"]]


def test_a_break_inside_an_asr_word_remains_only_when_no_other_layout_fits_the_lines():
    text = "".join(_MORPHEMES[:-1])
    # At width 16 every two-line layout divides a word; a third line is never added.
    assert wrap_generated_lines(text, 16, word_texts=_MORPHEMES[:-1]) == wrap_generated_lines(text, 16)
    assert len(wrap_generated_lines(text, 16)) == 2
    # One character per ASR word (MAI, Scribe) is unchanged: every boundary is a word edge.
    sentence = "テーブルをひっくり返したのは誰だ？"
    for width in (16, 24, 26):
        assert wrap_generated_lines(sentence, width, word_texts=list(sentence)) == wrap_generated_lines(sentence, width)
    # Punctuation glued to the next ASR word ('。3') still allows the sentence break.
    glued = ["山", "下", "盛", "彦", "に", "伝", "え", "ろ", "。3", "分", "以", "内", "に",
             "静", "川", "市", "南", "港", "の", "屋", "台", "に"]
    text = "".join(glued)
    assert wrap_generated_lines(text, 37, word_texts=glued) == wrap_generated_lines(text, 37) == [
        "山下盛彦に伝えろ。", "3分以内に静川市南港の屋台に"]
    # Words that are not the text's own pieces give no boundaries.
    assert wrap_generated_lines(sentence, 24, word_texts=["テーブル", "が"]) == wrap_generated_lines(sentence, 24)


def test_japanese_generation_pipeline_writes_utf8_srt_and_keeps_asr_evidence(tmp_path):
    words = _words(["今日", "は", "晴れ", "です。", "明日", "は", "雨", "です。"])
    audio_path = tmp_path / "日本語.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(json.dumps({"words": [word.model_dump() for word in words]}, ensure_ascii=False), encoding="utf-8")
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    output_path = tmp_path / "日本語.generated.srt"

    result = generate_srt_from_audio(
        audio_path,
        output_path,
        tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
        language="ja",
    )

    cues = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["今日は晴れです。", "明日は雨です。"]
    assert result.report["summary"]["cue_count"] == 2
    assert all(cue.start_ms >= 0 and cue.end_ms > cue.start_ms for cue in cues)
    saved_words = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))["words"]
    assert saved_words == [word.model_dump() for word in words]
