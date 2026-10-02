"""Final display limits preserve words and use actual acoustic boundaries."""

from collections import Counter

import pytest

from dubsync.models import Cue, Word
from dubsync.semantic_output import expand_output_flags, split_crowded_output_cues, wrap_semantic_lines
from dubsync.style_profile import StyleProfile
from dubsync.subtitle_annotations import speech_text_for_alignment
from dubsync.text_metrics import display_width
from dubsync.tokenize import alphanumeric_signature


def timed_words(text):
    return [Word(text=token, start=1 + position * .3, end=1.2 + position * .3)
            for position, token in enumerate(text.split())]


def test_timed_three_line_dialogue_splits_at_complete_clauses_without_word_loss():
    text = "We waited by the old gate, but nobody came to meet us."
    words = timed_words(text)
    source = Cue(index=7, start_ms=1000, end_ms=4900,
                 lines=["We waited by the old gate,", "but nobody came", "to meet us."],
                 speaker_id="actor", character="Mara", prompt_scene_id=3)
    ownership = {7: list(range(len(words)))}
    before = source.model_dump()
    result = split_crowded_output_cues([source], words, ownership,
                                     StyleProfile(max_chars_per_line=24, tail_ms=0))
    assert len(result.cues) == 2
    assert [speech_text_for_alignment(cue) for cue in result.cues] == [
        "We waited by the old gate,", "but nobody came to meet us."]
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    assert alphanumeric_signature(" ".join(cue.text for cue in result.cues)) == alphanumeric_signature(text)
    assert Counter(index for cue in result.cues for index in result.cue_word_indices[cue.index]) == Counter(range(len(words)))
    assert result.cues[0].start_ms == source.start_ms
    assert words[-1].end * 1000 <= result.cues[-1].end_ms <= words[-1].end * 1000 + 34
    for cue in result.cues:
        assert cue.speaker_id == "actor" and cue.character == "Mara" and cue.prompt_scene_id == 3
        assert all(cue.start_ms <= words[i].start * 1000 and cue.end_ms >= words[i].end * 1000
                   for i in result.cue_word_indices[cue.index])
    assert source.model_dump() == before and ownership == {7: list(range(len(words)))}


def test_unavailable_word_timing_reflows_two_lines_without_estimated_children():
    source = Cue(index=3, start_ms=5000, end_ms=6300,
                 lines=["This is the first authored sentence.", "This is the second sentence.",
                        "And this is the final sentence."])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_chars_per_line=20))
    assert len(result.cues) == 1 and len(result.cues[0].lines) == 2
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (5000, 6300)
    assert alphanumeric_signature(result.cues[0].text) == alphanumeric_signature(source.text)
    assert any(flag.kind == "output_line_limit_reflow" for flag in result.flags)


def test_one_provider_word_is_never_divided_into_timed_children():
    source = Cue(index=3, start_ms=1000, end_ms=2500,
                 lines=["The unusually long", "provider phrase contains", "several distinct words."])
    word = Word(text=source.plain_text, start=1, end=2.5)
    result = split_crowded_output_cues([source], [word], {3: [0]}, StyleProfile(max_chars_per_line=20))
    assert len(result.cues) == 1 and len(result.cues[0].lines) <= 2
    assert result.cue_word_indices == {3: [0]}
    assert alphanumeric_signature(result.cues[0].text) == alphanumeric_signature(source.text)


def test_visual_caption_pages_keep_order_and_known_envelope():
    source = Cue(index=53, start_ms=189300, end_ms=193180,
                 lines=["[Episódio 17]", "[Coragem é sentir medo e seguir em frente mesmo assim]"])
    result = split_crowded_output_cues([source], [], {53: []}, StyleProfile(max_chars_per_line=30))
    assert len(result.cues) >= 2
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    assert result.cues[0].start_ms == source.start_ms and result.cues[-1].end_ms == source.end_ms
    assert all(left.end_ms == right.start_ms for left, right in zip(result.cues, result.cues[1:]))
    assert alphanumeric_signature(" ".join(cue.text for cue in result.cues)) == alphanumeric_signature(source.text)
    assert all(not indices for indices in result.cue_word_indices.values())


def test_cap_also_applies_when_inferred_profile_allows_four_lines():
    source = Cue(index=1, start_ms=1000, end_ms=2200, lines=["One.", "Two.", "Three.", "Four."])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_lines_per_cue=4, max_chars_per_line=8))
    assert all(len(cue.lines) <= 2 for cue in result.cues)


def test_valid_two_line_cue_is_identity_and_reapplying_is_idempotent():
    source = Cue(index=1, start_ms=1000, end_ms=2200, lines=["Hello there.", "We waited."])
    profile = StyleProfile(max_chars_per_line=25)
    first = split_crowded_output_cues([source], [], {}, profile)
    second = split_crowded_output_cues(first.cues, [], first.cue_word_indices, profile)
    assert first.cues == second.cues == [source]
    assert first.cues[0] is source and not first.flags


def test_already_valid_authored_two_lines_override_alternative_semantic_wrap():
    source = Cue(index=1, start_ms=1000, end_ms=2200, lines=["A very old gate", "by the river."])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_chars_per_line=15))
    assert result.cues[0] is source and not result.flags


def test_protected_three_line_cue_never_splits_even_with_exact_words():
    source = Cue(index=7, start_ms=1000, end_ms=4900,
                 lines=["We waited by the old gate,", "but nobody came", "to meet us."])
    words = timed_words(source.plain_text)
    result = split_crowded_output_cues([source], words, {7: list(range(len(words)))},
                                     StyleProfile(max_chars_per_line=24), protected_cue_ids={7})
    assert len(result.cues) == 1 and len(result.cues[0].lines) == 2 and not result.expansions
    assert result.cue_word_indices == {7: list(range(len(words)))}
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 4900)


def test_flag_expansion_is_pure_deduplicated_and_only_for_actual_splits():
    from dubsync.models import QCFlag
    flag = QCFlag(kind="review", cue_ids=[7, 8, 7], message="A held source interval.")
    result = expand_output_flags([flag], {7: [7, 9], 8: [10]})
    assert result[0].cue_ids == [7, 9, 8]
    assert flag.cue_ids == [7, 8, 7]


def test_short_authored_lines_compact_before_creating_tiny_timed_children():
    source = Cue(index=2, start_ms=1600, end_ms=2250, lines=["Hello", "there."])
    words = [Word(text="Hello", start=1.6, end=1.85), Word(text="there.", start=1.9, end=2.2)]
    result = split_crowded_output_cues([source], words, {2: [0, 1]}, StyleProfile(max_chars_per_line=26), max_lines=1)
    assert result.cues == [source.with_lines(["Hello there."])]
    assert result.cue_word_indices == {2: [0, 1]} and not result.expansions


def test_overlapping_or_collapsed_owned_words_do_not_authorize_estimated_split_times():
    source = Cue(index=1, start_ms=1000, end_ms=2000,
                 lines=["We waited for a long time,", "but no one arrived", "before we left."])
    words = [Word(text=word, start=1, end=1.001) for word in source.plain_text.split()]
    result = split_crowded_output_cues([source], words, {1: list(range(len(words)))},
                                     StyleProfile(max_chars_per_line=20))
    assert len(result.cues) == 1 and len(result.cues[0].lines) == 2 and not result.expansions
    assert result.cue_word_indices == {1: list(range(len(words)))}
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 2000)


def test_inline_styling_remains_balanced_when_timed_splitting_is_unsafe():
    source = Cue(index=1, start_ms=1000, end_ms=5000,
                 lines=["<i>We waited for a long time,", "but no one arrived", "before we left.</i>"])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_chars_per_line=20))
    assert len(result.cues) == 1 and len(result.cues[0].lines) <= 2
    assert result.cues[0].text.count("<i>") == result.cues[0].text.count("</i>") == 1


def test_malformed_brackets_reflow_without_dropping_spoken_residue_or_gating_export():
    source = Cue(index=1, start_ms=1000, end_ms=3000,
                 lines=["[An unclosed caption", "still contains words", "and dialogue follows", "here."])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_chars_per_line=20))
    assert len(result.cues) == 1 and len(result.cues[0].lines) <= 2
    assert "".join(result.cues[0].text.split()) == "".join(source.text.split())
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 3000)


def test_timed_children_keep_complete_phrases_and_do_not_fill_long_speech_gaps():
    source = Cue(index=1, start_ms=1000, end_ms=3200,
                 lines=["Keep your", "bag close,", "bring the", "blue coat."])
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Keep", 1, 1.16), ("your", 1.2, 1.32), ("bag", 1.36, 1.52), ("close,", 1.6, 1.78),
        ("bring", 2.4, 2.58), ("the", 2.6, 2.68), ("blue", 2.72, 2.92), ("coat.", 3, 3.2),
    ]]
    result = split_crowded_output_cues([source], words, {1: list(range(8))},
                                     StyleProfile(max_lines_per_cue=4, max_chars_per_line=12, min_cue_dur=.1))
    assert [speech_text_for_alignment(cue) for cue in result.cues] == ["Keep your bag close,", "bring the blue coat."]
    assert 1780 <= result.cues[0].end_ms <= 1840
    assert result.cues[1].start_ms == 2400
    assert all(len(cue.lines) <= 2 for cue in result.cues)


def test_abbreviated_title_is_not_treated_as_a_sentence_boundary():
    source = Cue(index=1, start_ms=1000, end_ms=5000,
                 lines=["Please wait for Dr.", "Silva, and bring the", "entire medical report."])
    words = timed_words(source.plain_text)
    result = split_crowded_output_cues([source], words, {1: list(range(len(words)))},
                                     StyleProfile(max_chars_per_line=20, tail_ms=0))
    assert all(not cue.text.rstrip().endswith("Dr.") for cue in result.cues)


def test_embedded_physical_line_breaks_cannot_bypass_the_display_line_cap():
    source = Cue(index=1, start_ms=1000, end_ms=3000, lines=["One.\nTwo.\nThree."])
    result = split_crowded_output_cues([source], [], {}, StyleProfile(max_chars_per_line=26))
    assert all(len(cue.text.splitlines()) <= 2 for cue in result.cues)
    assert "".join(result.cues[0].text.split()) == "".join(source.text.split())


@pytest.mark.parametrize("count", [40, 160])
def test_long_exact_word_stream_only_wraps_chunks_that_can_fit_the_display_budget(monkeypatch, count):
    from dubsync import semantic_output
    calls = []
    original = semantic_output.wrap_semantic_lines

    def counted(text, width):
        size = len(text.split())
        calls.append(size)
        # The initial whole-cue reflow is necessary. Subsequent candidate
        # chunks need at most 53 visible columns (two 26-column lines).
        if len(calls) > 1:
            assert size <= 9, "An impossible candidate reached the expensive semantic wrapper."
        assert len(calls) <= count * 10 + 10
        return original(text, width)

    monkeypatch.setattr(semantic_output, "wrap_semantic_lines", counted)
    words = [Word(text="hello", start=i * .3, end=i * .3 + .2) for i in range(count)]
    source = Cue(index=1, start_ms=0, end_ms=count * 300, lines=[" ".join(["hello"] * count)])
    result = split_crowded_output_cues([source], words, {1: list(range(count))}, StyleProfile(max_chars_per_line=26))
    assert len(result.cues) == (count + 7) // 8
    assert Counter(i for cue in result.cues for i in result.cue_word_indices[cue.index]) == Counter(range(count))
    assert all(len(cue.lines) <= 2 for cue in result.cues)
    assert all(cue.start_ms <= words[i].start * 1000 + 1e-7 and cue.end_ms >= words[i].end * 1000 - 1e-7
               for cue in result.cues for i in result.cue_word_indices[cue.index])


@pytest.mark.parametrize("text,attachment", [
    ("kannst du dir deinen Geburtstagswunsch überlegen?", "deinen Geburtstagswunsch"),
    ("die letzte Wirtschaftskrise überstand,", "letzte Wirtschaftskrise"),
    ("wir bringen unsere alten roten Fahrräder mit", "alten roten Fahrräder"),
])
def test_german_determiner_and_adjectives_stay_with_their_capitalized_noun(text, attachment):
    words = timed_words(text)
    source = Cue(index=1, start_ms=1000, end_ms=1000 + len(words) * 300, lines=[text])
    result = split_crowded_output_cues([source], words, {1: list(range(len(words)))},
                                     StyleProfile(max_chars_per_line=26, min_cue_dur=.1), min_parts=2)
    spoken = [speech_text_for_alignment(cue) for cue in result.cues]
    assert any(attachment in phrase for phrase in spoken)
    assert " ".join(spoken) == text
    assert Counter(i for cue in result.cues for i in result.cue_word_indices[cue.index]) == Counter(range(len(words)))


@pytest.mark.parametrize("units,position", [("die letzte Wirtschaftskrise überstand,", 2),
                                          ("deinen schönen Geburtstagswunsch", 1),
                                          ("unsere alten roten Fahrräder", 2),
                                          ("unsere alten roten Fahrräder", 3)])
def test_attached_german_phrases_are_not_legal_timed_boundaries(units, position):
    from dubsync.semantic_output import _legal_break
    assert not _legal_break(units.split(), position)


@pytest.mark.parametrize("units,position", [("die letzte, dann gehen wir", 2),
                                          ("wir kamen später und gingen", 3),
                                          ("deinen. Danach gehen wir", 1)])
def test_german_attachment_guard_preserves_punctuation_and_unrelated_boundaries(units, position):
    from dubsync.semantic_output import _legal_break
    assert _legal_break(units.split(), position)


@pytest.mark.parametrize("cue_id,text,expected,start,end,first_owner,speaker", [
    (53, "大口を叩いてくれるじゃねぇか", ["大口を叩いて", "くれるじゃねぇか"],
     101000, 102934, 337, "chunk_1:3"),
    (53, "大口を叩いてくれるじゃねぇか", ["大口を叩いて", "くれるじゃねぇか"],
     101300, 103067, 333, "speaker_1"),
    (7, "テーブルをひっくり返したのは", ["テーブルを", "ひっくり返したのは"],
     5900, 7367, 33, "chunk_1:0"),
    (7, "テーブルをひっくり返したのは", ["テーブルを", "ひっくり返したのは"],
     5900, 7667, 32, "speaker_0"),
    (7, "テーブルをひっくり返したのは", ["テーブルを", "ひっくり返したのは"],
     6067, 7534, 34, "chunk_1:0"),
    (7, "テーブルをひっくり返したのは", ["テーブルを", "ひっくり返したのは"],
     6067, 7700, 34, "speaker_0"),
])
def test_frozen_japanese_orphans_reflow_at_phrases_with_exact_envelopes_and_owners(
        cue_id, text, expected, start, end, first_owner, speaker):
    # Geometry and provider indices come from the six policy-33 delivered cues.
    source = Cue(index=cue_id, start_ms=start, end_ms=end, lines=[text], speaker_id=speaker)
    words = [Word(text="unused", start=0, end=.1) for _ in range(first_owner)]
    step = (end - start) / len(text) / 1000
    words.extend(Word(text=char, start=start / 1000 + position * step,
                      end=start / 1000 + (position + .8) * step)
                 for position, char in enumerate(text))
    ownership = {cue_id: list(range(first_owner, first_owner + len(text)))}
    result = split_crowded_output_cues([source], words, ownership, StyleProfile(max_chars_per_line=26))
    assert result.cues == [source.with_lines(expected)]
    assert result.cue_word_indices == ownership and not result.expansions
    assert "".join(result.cues[0].lines) == text
    assert all(display_width(line) <= 26 for line in result.cues[0].lines)
    assert all(len(line) > 1 for line in result.cues[0].lines)
    assert source.lines == [text]


def test_legal_authored_japanese_break_keeps_identity():
    source = Cue(index=7, start_ms=5900, end_ms=7367,
                 lines=["テーブルをひっくり", "返したのは"])
    result = split_crowded_output_cues([source], [], {7: [33, 34]}, StyleProfile(max_chars_per_line=26))
    assert result.cues[0] is source and not result.flags
    assert result.cue_word_indices == {7: [33, 34]}


@pytest.mark.parametrize("text,width", [
    ("「か\u3099っ。」「ｶﾞ！」", 8),
    ("東京「晴天」です。", 8),
    ("キャットは、歩いていった。", 12),
    ("日本語👩\u200d💻で話す。", 10),
    ("日本語👍🏽で話す。", 10),
    ("日本語🇯🇵で話す。", 10),
])
def test_unspaced_semantic_lines_preserve_kana_punctuation_and_graphemes(text, width):
    import unicodedata
    lines = wrap_semantic_lines(text, width)
    assert "".join(lines) == text
    for left, right in zip(lines, lines[1:]):
        assert not left.endswith(("「", "\u200d"))
        assert not right.startswith(("\u3099", "ﾞ", "。", "！", "」", "っ", "ャ", "\u200d", "🏽"))
        assert not unicodedata.combining(right[0])
        assert not (left.endswith("🇯") and right.startswith("🇵"))


def test_impossibly_narrow_japanese_wrapper_keeps_visible_overflow_without_text_loss():
    from dubsync.verify import lint_cues
    source = Cue(index=1, start_ms=1000, end_ms=2000, lines=["「か\u3099っ。」"])
    profile = StyleProfile(max_chars_per_line=2)
    result = split_crowded_output_cues([source], [], {}, profile)
    assert len(result.cues) == 1 and len(result.cues[0].lines) <= 2
    assert "".join(result.cues[0].lines) == "".join(source.lines)
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 2000)
    assert any(issue.kind == "line_length" for issue in lint_cues(result.cues, profile))
