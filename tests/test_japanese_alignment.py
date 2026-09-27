from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, Word


def test_japanese_phrase_words_match_source_characters_with_original_word_indices():
    cues = [Cue(index=1, start_ms=0, end_ms=2000, lines=["今日は学校へ行きます。"])]
    words = [
        Word(text="今日は", start=1.0, end=1.5),
        Word(text="学校へ", start=1.6, end=2.0),
        Word(text="行きます。", start=2.1, end=2.8),
    ]
    original = [word.model_dump() for word in words]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1, 2]}
    assert {match.asr_word_index for match in result.token_matches} == {0, 1, 2}
    assert result.anchor_regions[0].asr_word_indices == [0, 1, 2]
    assert result.anchor_regions[0].start == 1.0
    assert result.anchor_regions[0].end == 2.8
    assert [word.model_dump() for word in words] == original


def test_japanese_partial_word_change_retains_provider_timing_and_readable_span():
    cues = [Cue(index=1, start_ms=0, end_ms=2000, lines=["明日は東京に行きます"])]
    words = [Word(text="明日は大阪に行きます", start=1.1, end=2.7, confidence=0.97)]

    result = align_cues_to_words(cues, words)

    assert len(result.divergence_spans) == 1
    span = result.divergence_spans[0]
    assert span.srt_text == "東京"
    assert span.asr_text == "大阪"
    assert span.asr_word_indices == [0]
    assert (span.start, span.end) == (1.1, 2.7)
    assert result.cue_word_indices == {1: [0]}


def test_japanese_mixed_latin_and_separate_punctuation_keep_word_ownership():
    cues = [Cue(index=1, start_ms=0, end_ms=2000, lines=["OpenAIで日本語を学ぶ。"])]
    words = [
        Word(text="OpenAIで", start=0.1, end=0.5),
        Word(text="日本語を学ぶ", start=0.6, end=1.5),
        Word(text="。", start=1.5, end=1.5),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1]}


def test_shared_japanese_asr_word_does_not_invent_character_timestamps():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["今日は"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["晴れです"]),
    ]
    result = align_cues_to_words(cues, [Word(text="今日は晴れです", start=0.5, end=1.8)])

    assert result.anchor_coverage == 1.0
    assert result.cue_word_indices == {1: [0], 2: [0]}
    assert all((anchor.start, anchor.end) == (0.5, 1.8) for anchor in result.anchor_regions)


def test_japanese_omission_inside_one_provider_word_has_non_inverted_window():
    cues = [Cue(index=1, start_ms=0, end_ms=2000, lines=["今日はとても晴れです"])]
    result = align_cues_to_words(cues, [Word(text="今日は晴れです", start=0.5, end=1.8)])

    assert len(result.divergence_spans) == 1
    span = result.divergence_spans[0]
    assert span.srt_text == "とても"
    assert span.start <= span.end
    assert (span.start, span.end) == (0.5, 1.8)
