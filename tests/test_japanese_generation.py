from __future__ import annotations

import json

import pytest

from dubsync.models import Word
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
