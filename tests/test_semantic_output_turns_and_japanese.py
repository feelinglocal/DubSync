"""Final display reflows and timed splits keep dialogue turns and Japanese phrases whole.

A line-initial dash opens one speaker's turn: a turn is never divided while
another turn shares the display, and its dash moves with it. Japanese timed
cuts obey the same line-break legality as Japanese wrapping, prefer phrase
boundaries and balance, and no reflow inserts or rewrites a space.
"""
from __future__ import annotations

import random
import re

import pytest

from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue, Word
from dubsync.qc_review import build_review
from dubsync.semantic_output import compact_lines, split_crowded_output_cues, wrap_semantic_lines
from dubsync.style_profile import StyleProfile
from dubsync.text_metrics import _NONSTARTING_KANA, _can_break_between, display_width

_TURN = re.compile(r"[-–—]\s")
_KANJI = re.compile(r"[㐀-䶿一-鿿]")
_KATAKANA = re.compile(r"[ァ-ヺー-ヿ]")


def _turn_words(text: str, start: float, end: float, pause_after: dict[int, float] | None = None) -> list[Word]:
    """Evenly spaced provider words for the spoken tokens (a dash is not spoken)."""
    tokens = [token for token in text.split() if token not in {"-", "–", "—"}]
    pause_after = pause_after or {}
    step = (end - start - sum(pause_after.values())) / len(tokens)
    words, time = [], start
    for position, token in enumerate(tokens):
        words.append(Word(text=token, start=round(time, 3), end=round(time + step * .85, 3)))
        time += step + pause_after.get(position, 0.0)
    return words


def _char_words(text: str, start: float = 1.0, step: float = .12,
                gaps: dict[int, float] | None = None) -> list[Word]:
    """One provider word per Japanese character, as MAI and Scribe return them."""
    words: list[Word] = []
    time = start
    for position, char in enumerate(text):
        if char in "、。？！" and words:
            words[-1] = words[-1].model_copy(update={"text": words[-1].text + char})
            continue
        time += (gaps or {}).get(position, 0.0)
        words.append(Word(text=char, start=round(time, 3), end=round(time + step * .9, 3), confidence=.99))
        time += step
    return words


def _turn_ids(lines: list[str]) -> list[tuple[str, int]]:
    """Every displayed token with the dialogue turn it belongs to."""
    tokens, turn = [], -1
    for line in lines:
        stripped = line.strip()
        if turn < 0 or _TURN.match(stripped):
            turn += 1
        tokens.extend((token, turn) for token in stripped.split())
    return tokens


def _assert_turns_intact(source: Cue, cues: list[Cue], width: int | None = None) -> None:
    expected = _turn_ids(source.lines)
    shown = [token for cue in cues for line in cue.lines for token in line.split()]
    assert shown == [token for token, _ in expected], "words, punctuation and dashes keep their order"
    sizes = {turn: sum(1 for _, other in expected if other == turn) for _, turn in expected}
    position = 0
    for cue in cues:
        assert len(cue.lines) <= 2
        count = sum(len(line.split()) for line in cue.lines)
        shared = [turn for _, turn in expected[position:position + count]]
        if len(set(shared)) > 1:
            assert all(shared.count(turn) == sizes[turn] for turn in set(shared)), (
                f"a divided turn shares a display with another turn: {cue.lines!r}")
        for line in cue.lines:
            count = len(line.split())
            turns = {turn for _, turn in expected[position:position + count]}
            assert len(turns) == 1, f"one line holds two speakers' turns: {line!r}"
            assert line.split()[-1] not in {"-", "–", "—"}, f"a dash dangles at the line end: {line!r}"
            if width is not None:
                assert display_width(line) <= width, f"line over the width: {line!r}"
            position += count


def _owned(words: list[Word]) -> list[int]:
    return list(range(len(words)))


_PROFILE = StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=.433)


@pytest.mark.parametrize("lines,words,enforce_width,expected", [
    # testing-008 cue 43: the valid two-turn layout stays, whatever the width.
    (["- Halt die Klappe, du Tr*ttel!", "- Ah!"], None, True,
     [["- Halt die Klappe, du Tr*ttel!", "- Ah!"]]),
    (["- Halt die Klappe, du Tr*ttel!", "- Ah!"], None, False,
     [["- Halt die Klappe, du Tr*ttel!", "- Ah!"]]),
    (["- Kommst du mit?", "- Nein, ich bleibe heute zu Hause."], None, True,
     [["- Kommst du mit?", "- Nein, ich bleibe heute zu Hause."]]),
    # Owned words: the split falls between the turns, and each dash moves with its turn.
    (["- Kommst du mit?", "- Nein, ich bleibe heute zu Hause."], (10.05, 12.9, {2: .3}), True, None),
    (["- Wohin gehst du?", "- Nach Hause.", "- Warte auf mich!"], (20.05, 22.9, {2: .3, 4: .3}), True, None),
    (["- A neve está ótima hoje de manhã.", "- Eu estou adorando minha prancha nova e cara."],
     (30.05, 33.9, {6: .3}), True, None),
    (["- Halt die Klappe, du Trottel!", "- Ah!"], (76.25, 77.40, {4: .25}), True, None),
    # Three lines under the customer's own width: still turn by turn.
    (["- Kommst du mit?", "- Nein, ich bleibe", "heute zu Hause."], (10.05, 12.9, {2: .3}), False, None),
    (["- Kommst du mit?", "- Nein, ich bleibe", "heute zu Hause."], None, False,
     [["- Kommst du mit?", "- Nein, ich bleibe heute zu Hause."]]),
])
def test_dash_dialogue_turns_are_never_merged_or_divided_by_reflow_or_timed_split(
        lines, words, enforce_width, expected):
    text = " ".join(lines)
    start_ms = int(words[0] * 1000) - 50 if words else 10000
    end_ms = int(words[1] * 1000) + 100 if words else 13000
    provider = _turn_words(text, *words) if words else []
    source = Cue(index=5, start_ms=start_ms, end_ms=end_ms, lines=lines)
    result = split_crowded_output_cues([source], provider, {5: _owned(provider)}, _PROFILE,
                                       enforce_width=enforce_width)
    _assert_turns_intact(source, result.cues)
    if expected is not None:
        assert [cue.lines for cue in result.cues] == expected
    if provider and len(result.cues) > 1:
        # A timed split keeps each turn's dash on the display of its words.
        assert all(_TURN.match(cue.lines[0]) for cue in result.cues)
        owned = [index for cue in result.cues for index in result.cue_word_indices[cue.index]]
        assert sorted(owned) == _owned(provider)


def test_more_dialogue_turns_than_lines_without_word_timing_join_only_whole_turns():
    source = Cue(index=7, start_ms=20000, end_ms=23000,
                 lines=["- Wohin gehst du?", "- Nach Hause.", "- Warte auf mich!"])
    result = split_crowded_output_cues([source], [], {7: []}, _PROFILE)
    assert len(result.cues) == 1 and len(result.cues[0].lines) == 2
    assert all(_TURN.match(line) for line in result.cues[0].lines)
    assert " ".join(result.cues[0].lines) == " ".join(source.lines)
    for turn in source.lines:
        assert any(turn in line for line in result.cues[0].lines), f"turn divided: {turn!r}"


def test_whole_turns_that_share_a_line_are_a_review_item():
    # Two speakers on one line need their attribution checked; the info-level
    # reflow alone says only that the layout changed (review W4C-2).
    source = Cue(index=7, start_ms=20000, end_ms=23000,
                 lines=["- Wohin gehst du?", "- Nach Hause.", "- Warte auf mich!"])
    result = split_crowded_output_cues([source], [], {7: []}, _PROFILE)
    joined = [flag for flag in result.flags if flag.kind == "output_dialogue_turns_joined"]
    assert [(flag.severity, flag.cue_ids, flag.new_text) for flag in joined] == [
        ("warning", [7], result.cues[0].text)]
    review = build_review(result.flags, [], result.cues, source_cues=[source])
    item, = [item for item in review.review if item.kind == "output_dialogue_turns_joined"]
    assert item.severity == "warning"

    # One line per turn, or a timed split between turns, is no such item.
    for lines in (["- Wohin gehst du?", "- Nach Hause."], ["- Kommst du mit?", "- Nein, ich bleibe", "heute zu Hause."]):
        cue = Cue(index=5, start_ms=10000, end_ms=13000, lines=lines)
        flags = split_crowded_output_cues([cue], [], {5: []}, _PROFILE).flags
        assert "output_dialogue_turns_joined" not in [flag.kind for flag in flags]


@pytest.mark.parametrize("text", [
    "- Halt die Klappe, du Tr*ttel! - Ah!",
    "- Kommst du mit? - Nein, ich bleibe heute zu Hause.",
])
def test_a_wrapped_turn_dash_never_dangles_at_a_line_end(text):
    for width in (12, 16, 20, 26):
        for lines in (wrap_semantic_lines(text, width), compact_lines(text, 2, width)):
            assert " ".join(lines) == text
            assert not any(line.split()[-1] in {"-", "–", "—"} for line in lines), lines


@pytest.mark.parametrize("text", [
    "Ich weiß nicht - vielleicht morgen oder übermorgen, mal sehen.",
    "Er hat es versucht – na ja, mehr oder weniger – und ist dann gegangen.",
])
def test_a_dash_inside_a_sentence_never_opens_a_line_like_a_new_speaker(text):
    for width in (12, 16, 20, 26):
        for lines in (wrap_semantic_lines(text, width), compact_lines(text, 2, width)):
            assert " ".join(lines) == text
            assert not any(_TURN.match(line) for line in lines), lines


def test_dash_dialogue_beside_a_caption_divides_only_between_turns():
    speech = Cue(index=11, start_ms=40000, end_ms=41500, lines=["- Você vem?", "- Não, fico aqui."])
    caption = Cue(index=12, start_ms=40200, end_ms=41200, lines=["[PLACA: SAÍDA]"])
    words = _turn_words("Você vem? Não, fico aqui.", 40.05, 41.4, {1: .2})
    composed = compose_bracketed_annotations([speech, caption], {11: _owned(words), 12: []},
                                             words=words, profile=_PROFILE)
    spoken = [cue.with_lines([line for line in cue.lines if not line.startswith("[")])
              for cue in composed.cues if not all(line.startswith("[") for line in cue.lines)]
    _assert_turns_intact(speech, spoken)
    assert [cue.lines for cue in spoken] == [["- Você vem?"], ["- Não, fico aqui."]]
    assert all(cue.lines[-1] == "[PLACA: SAÍDA]" and len(cue.lines) <= 2 for cue in composed.cues)
    assert all(left.end_ms <= right.start_ms for left, right in zip(composed.cues, composed.cues[1:]))


@pytest.mark.parametrize("enforce_width", [True, False])
@pytest.mark.parametrize("speech_lines,caption_lines,width", [
    (["Eu sei que você não queria vir hoje aqui,", "mas eu preciso muito falar com você agora."],
     ["[Hospital Central]"], 47),
    (["- Você vem?", "- Não, fico aqui."], ["[Rua das Flores]"], 47),
])
def test_full_two_line_speech_around_a_caption_never_makes_a_three_line_display(
        speech_lines, caption_lines, width, enforce_width):
    # W4R-1: the caption lies wholly inside speech that can be neither divided
    # nor fitted to one line. The two-line ceiling is never given up for it.
    speech = Cue(index=1, start_ms=1000, end_ms=4000, lines=speech_lines)
    caption = Cue(index=2, start_ms=1500, end_ms=3500, lines=caption_lines)
    profile = StyleProfile(fps=30, max_chars_per_line=width, min_cue_dur=.5)
    composed = compose_bracketed_annotations([speech, caption], {1: [], 2: []}, words=[], profile=profile,
                                             enforce_width=enforce_width)
    assert all(len(cue.lines) <= 2 for cue in composed.cues), [cue.lines for cue in composed.cues]
    assert all(left.end_ms <= right.start_ms for left, right in zip(composed.cues, composed.cues[1:]))
    shown = [line for cue in composed.cues for line in cue.lines]
    # Screen text and dialogue never share a line.
    assert all(line.startswith("[") == line.endswith("]") for line in shown), shown
    assert all(not ("[" in line and not line.startswith("[")) for line in shown), shown
    display, = [cue for cue in composed.cues if cue.start_ms < 4000 and 1000 < cue.end_ms]
    assert (display.index, display.start_ms, display.end_ms) == (1, 1000, 4000)
    if not enforce_width and not speech_lines[0].startswith("- "):
        # The width is the customer's advisory one: plain speech joins its own
        # lines, the caption keeps its time and line-length QC shows the width.
        assert display.lines == [" ".join(speech_lines), *caption_lines]
        reflow, = [flag for flag in composed.flags if flag.kind == "output_line_limit_reflow"]
        assert (reflow.cue_ids, reflow.severity, reflow.old_text) == ([1], "info", speech.text)
        assert not [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
        return
    # Dialogue turns or an enforced width keep the speech lines. The caption is
    # shown on its own just beside the speech (no display follows, so before it).
    assert display.lines == speech_lines
    assert all(display_width(line) <= width for line in shown), shown
    moved, = [cue for cue in composed.cues if cue.index != 1]
    assert (moved.start_ms, moved.end_ms, moved.lines) == (0, 1000, caption_lines)
    flag, = [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
    assert (flag.severity, flag.cue_ids, flag.old_text) == ("warning", [moved.index, 1], caption.text)
    assert "just before that speech" in flag.message
    assert composed.tracks[2]["display_intervals"] == [[0, 1000]]
    assert composed.tracks[2]["coverage_gaps_ms"] == [[1500, 3500]]
    assert composed.tracks[2]["early_extension_ms"] == 1500
    assert not [flag for flag in composed.flags if flag.kind == "output_line_limit_reflow"]


_FULL_SPEECH = {
    "plain": ["Eu não sei o que vou fazer da minha", "vida com essa empresa agora."],
    "short": ["Eu não sei.", "Vamos embora."],
    "dash": ["- Você vem com a gente amanhã?", "- Não, fico aqui em casa."],
    "mixed": ["[Esta empresa não é para você]", "A Lime não é para você."],
    "one": ["Eu não sei o que vou fazer."],
}
_NEIGHBORS = {
    "alone": [],
    # Displays touch the speech on both sides: no free interval beside it.
    "boxed": [Cue(index=5, start_ms=8000, end_ms=10000, lines=["Antes."]),
              Cue(index=6, start_ms=14000, end_ms=16000, lines=["Depois."])],
    # Too little room before; a free interval after, up to the next display.
    "room_after": [Cue(index=5, start_ms=9800, end_ms=9900, lines=["Antes."]),
                   Cue(index=6, start_ms=17000, end_ms=18000, lines=["Depois."])],
}


@pytest.mark.parametrize("neighbors", sorted(_NEIGHBORS))
@pytest.mark.parametrize("shape", sorted(_FULL_SPEECH))
@pytest.mark.parametrize("enforce_width", [True, False])
@pytest.mark.parametrize("max_lines", [1, 2])
def test_no_composed_display_exceeds_the_line_limit_around_a_contained_caption(
        max_lines, enforce_width, shape, neighbors):
    profile = StyleProfile(fps=25.0, max_chars_per_line=42, min_cue_dur=.5, max_lines_per_cue=max_lines)
    speech = Cue(index=1, start_ms=10000, end_ms=14000, lines=_FULL_SPEECH[shape])
    caption = Cue(index=2, start_ms=11000, end_ms=13000, lines=["[PLACA: SAÍDA DE EMERGÊNCIA]"])
    source = sorted([speech, caption, *_NEIGHBORS[neighbors]], key=lambda cue: (cue.start_ms, cue.index))
    # As in the pipeline: the line limit applies to every cue before captions are composed.
    segmented = split_crowded_output_cues(source, [], {cue.index: [] for cue in source}, profile,
                                          enforce_width=enforce_width)
    composed = compose_bracketed_annotations(segmented.cues, segmented.cue_word_indices, words=[], profile=profile,
                                             enforce_width=enforce_width)
    limit = min(2, max_lines)
    assert all(len(cue.lines) <= limit for cue in composed.cues), [cue.lines for cue in composed.cues]
    assert all(left.end_ms <= right.start_ms for left, right in zip(composed.cues, composed.cues[1:]))
    words = [token for cue in composed.cues for line in cue.lines for token in line.split()]
    spoken = [token for line in speech.lines for token in line.split()]
    assert words.count(spoken[-1]) == 1 and all(token in words for token in spoken)
    hidden = [flag for flag in composed.flags if flag.kind == "annotation_display_full" and flag.new_text is None]
    if "[PLACA:" in words:
        assert not hidden
        moved = [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
        if moved:
            display, = [cue for cue in composed.cues if "[PLACA:" in cue.lines[0]]
            assert display.end_ms <= speech.start_ms or display.start_ms >= speech.end_ms
            assert moved[0].severity == "warning"
    else:
        # Not shown at all only when no free interval beside the speech exists.
        assert neighbors == "boxed"
        flag, = hidden
        assert (flag.severity, flag.cue_ids, flag.old_text) == ("warning", [1], caption.text)
        assert composed.tracks[2]["display_intervals"] == [] and composed.tracks[2]["display_cue_ids"] == []
        assert composed.tracks[2]["coverage_gaps_ms"] == [[11000, 13000]]
        assert composed.tracks[2]["pages"][0]["delay_ms"] is None
        review = build_review(composed.flags, [], composed.cues, source_cues=source)
        item, = [item for item in review.review if item.kind == "annotation_display_full"]
        assert item.severity == "warning" and item.old_text == caption.text


def test_a_caption_moved_out_of_full_speech_prefers_the_free_interval_after_it():
    profile = StyleProfile(fps=25.0, max_chars_per_line=42, min_cue_dur=.5)
    source = [Cue(index=5, start_ms=9800, end_ms=9900, lines=["Antes."]),
              Cue(index=1, start_ms=10000, end_ms=14000, lines=_FULL_SPEECH["dash"]),
              Cue(index=2, start_ms=11000, end_ms=13000, lines=["[PLACA: SAÍDA DE EMERGÊNCIA]"]),
              Cue(index=6, start_ms=15000, end_ms=16000, lines=["Depois."])]
    composed = compose_bracketed_annotations(source, {5: [], 1: [], 2: [], 6: []}, words=[], profile=profile)
    assert [(cue.index, cue.start_ms, cue.end_ms, cue.lines) for cue in composed.cues] == [
        (5, 9800, 9900, ["Antes."]), (1, 10000, 14000, _FULL_SPEECH["dash"]),
        (2, 14000, 15000, ["[PLACA: SAÍDA DE EMERGÊNCIA]"]), (6, 15000, 16000, ["Depois."])]
    flag, = [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
    assert flag.cue_ids == [2, 1] and "just after that speech" in flag.message
    assert composed.tracks[2]["late_extension_ms"] == 2000
    review = build_review(composed.flags, [], composed.cues, source_cues=source)
    item, = [item for item in review.review if item.kind == "annotation_display_full"]
    assert item.srt_numbers == [2, 3]


def test_mixed_caption_dialogue_cue_keeps_its_lines_and_the_earlier_caption_keeps_its_own_time():
    # Matrix testlong-1 cue 658: never '[Esta empresa ...] A Lime ...' on one 54-column line.
    profile = StyleProfile(fps=30, max_chars_per_line=53, min_cue_dur=.5)
    mixed = Cue(index=658, start_ms=1931300, end_ms=1932467,
                lines=["[Esta empresa não é para você]", "A Lime não é para você."])
    caption = Cue(index=657, start_ms=1930550, end_ms=1931510, lines=["[Então procure outro emprego rápido]"])
    composed = compose_bracketed_annotations([caption, mixed], {657: [], 658: []}, words=[], profile=profile)
    by_id = {cue.index: cue for cue in composed.cues}
    assert by_id[658].lines == mixed.lines
    assert by_id[657].lines == caption.lines and by_id[657].end_ms <= mixed.start_ms
    assert all(display_width(line) <= 53 for cue in composed.cues for line in cue.lines)


def test_crowded_captions_beside_one_spoken_line_never_share_the_spoken_line():
    speech = Cue(index=1, start_ms=1000, end_ms=4000, lines=["Vamos embora daqui."])
    captions = [Cue(index=index, start_ms=1000, end_ms=4000, lines=[line])
                for index, line in ((2, "[Hospital Central]"), (3, "[Quarto 12]"), (4, "[Segundo andar]"))]
    profile = StyleProfile(max_chars_per_line=47, min_cue_dur=.5)
    composed = compose_bracketed_annotations([speech, *captions], {1: []}, words=[], profile=profile)
    assert [cue.lines for cue in composed.cues] == [
        ["Vamos embora daqui.", "[Hospital Central] [Quarto 12] [Segundo andar]"]]


def _assert_legal_japanese_cuts(text: str, cues: list[Cue]) -> None:
    children = ["".join(cue.lines) for cue in cues]
    assert "".join(children) == text, "every character is kept, without an inserted space"
    for left, right in zip(children, children[1:]):
        assert _can_break_between(left[-1], right[0]), (left, right)
        assert right[0] not in _NONSTARTING_KANA, (left, right)
        assert not (_KANJI.match(left[-1]) and _KANJI.match(right[0])), f"cut inside a kanji compound: {left}|{right}"
        assert not (_KATAKANA.match(left[-1]) and _KATAKANA.match(right[0])), f"cut inside a katakana word: {left}|{right}"
        assert left[-1] not in "っッ", (left, right)


@pytest.mark.parametrize("text,width,gaps,expected", [
    ("俺はただ正当防衛をしたまでだって言ってるだろうがよお前さん", 26, None,
     ["俺はただ正当防衛をしたまでだって", "言ってるだろうがよお前さん"]),
    ("俺はただ正当防衛をしたまでだって言ってるだろうがよお前さん", 26, {9: .35, 16: .35},
     ["俺はただ正当防衛をしたまでだって", "言ってるだろうがよお前さん"]),
    ("あのコンピューターシステムはもう完全にダメになってしまった", 26, None, None),
    ("彼女はずっと前からこの町に住んでいたけれど誰もその本当の名前を知らなかったらしい", 32, None, None),
    ("俺はただ正当防衛をしたまでだ、言ってるだろうがよお前さん", 26, None,
     ["俺はただ正当防衛をしたまでだ、", "言ってるだろうがよお前さん"]),
])
def test_japanese_timed_split_cuts_at_a_legal_balanced_phrase_boundary(text, width, gaps, expected):
    words = _char_words(text, gaps=gaps)
    source = Cue(index=7, start_ms=1000, end_ms=round(words[-1].end * 1000) + 200, lines=[text])
    result = split_crowded_output_cues([source], words, {7: _owned(words)},
                                       StyleProfile(max_chars_per_line=width))
    assert len(result.cues) >= 2
    _assert_legal_japanese_cuts(text, result.cues)
    # Balanced: no child is a fragment of a few characters beside a full display.
    sizes = [len("".join(cue.lines)) for cue in result.cues]
    assert min(sizes) >= len(text) / (2 * len(sizes)), sizes
    if expected is not None:
        assert ["".join(cue.lines) for cue in result.cues] == expected
    for cue in result.cues:
        owned = "".join(words[index].text for index in result.cue_word_indices[cue.index])
        assert owned == "".join(cue.lines)
        assert all(display_width(line) <= width for line in cue.lines)


def test_japanese_timed_split_prefers_the_customers_own_line_break():
    lines = ["あのコンピューター", "システムはもう完全に", "ダメになってしまった"]
    text = "".join(lines)
    words = _char_words(text)
    source = Cue(index=1, start_ms=1000, end_ms=round(words[-1].end * 1000) + 40, lines=lines)
    result = split_crowded_output_cues([source], words, {1: _owned(words)}, StyleProfile())
    _assert_legal_japanese_cuts(text, result.cues)
    children = ["".join(cue.lines) for cue in result.cues]
    authored = {len("".join(lines[:position])) for position in range(1, len(lines))}
    cuts = {len("".join(children[:position])) for position in range(1, len(children))}
    assert cuts <= authored, children
    # Each display made of the customer's own whole lines keeps them.
    assert [cue.lines for cue in result.cues] == [lines[:2], lines[2:]]


@pytest.mark.parametrize("lines", [
    ["今日は", "とてもいい天気ですね", "散歩でもしましょうか"],
    ["今日は", "とても", "いい天気ですね散歩でもしましょうか"],
    ["今日は", "とても", "いい天気ですね"],
    ["ここで山下様を", "もてなすんだ", "わかったか"],
])
@pytest.mark.parametrize("owned", [False, True])
def test_japanese_reflow_never_inserts_a_space_or_breaks_inside_desu(lines, owned):
    text = "".join(lines)
    words = _char_words(text) if owned else []
    source = Cue(index=1, start_ms=1000, end_ms=round(words[-1].end * 1000) + 40 if owned else 4000, lines=lines)
    for enforce_width in (True, False):
        result = split_crowded_output_cues([source], words, {1: _owned(words)}, StyleProfile(),
                                           enforce_width=enforce_width)
        shown = [line for cue in result.cues for line in cue.lines]
        assert "".join(shown) == text and not any(" " in line for line in shown), shown
        assert all(len(cue.lines) <= 2 for cue in result.cues)
        for cue in result.cues:
            for left, right in zip(cue.lines, cue.lines[1:]):
                assert left[-1] + right[0] not in {"です", "でし", "ます", "すね"}, cue.lines
                assert _can_break_between(left[-1], right[0]), cue.lines
    assert "".join(compact_lines("\n".join(lines), 2, 26)) == text


@pytest.mark.parametrize("line", [
    "今夜 俺の兄貴がここで山下様をもてなすんだ",
    "お前に 山下様の名前を口にする資格はない！",
])
def test_spaced_japanese_phrase_wider_than_the_line_breaks_at_a_legal_character(line):
    source = Cue(index=1, start_ms=1000, end_ms=4000, lines=[line])
    result = split_crowded_output_cues([source], [], {1: []}, StyleProfile())
    shown = result.cues[0].lines
    assert all(display_width(part) <= 26 for part in shown), shown
    assert all(len(part) > 1 for part in shown), shown
    assert "".join(shown).replace(" ", "") == line.replace(" ", "")
    assert sum(part.count(" ") for part in shown) <= line.count(" ")
    for left, right in zip(shown, shown[1:]):
        assert _can_break_between(left[-1], right[0])


@pytest.mark.parametrize("line,kept,breakable", [
    ("お前に\u3000山下様の名前を\u3000口にする資格はない！", "\u3000", True),
    ("Qu'est-ce que tu fais ici\u00a0? Je ne sais vraiment pas\u00a0!", "\u00a0", False),
    ("Mais pourquoi tu ne m'as rien dit avant\u202f?", "\u202f", False),
])
def test_reflow_keeps_ideographic_and_no_break_spaces(line, kept, breakable):
    source = Cue(index=1, start_ms=1000, end_ms=4000, lines=[line])
    result = split_crowded_output_cues([source], [], {1: []}, StyleProfile())
    shown = result.cues[0].lines
    assert len(shown) == 2
    rejoined = "".join(shown)
    assert "".join(line.split()) == "".join(rejoined.split())
    # Only a line break may consume a breakable space; the rest keep their character.
    assert line.count(kept) - rejoined.count(kept) == (1 if breakable else 0), shown
    assert sum(part.count(" ") for part in shown) <= line.count(" "), shown
    assert not any(part[0] in "?!:;" for part in shown), shown


def test_punctuation_set_off_by_a_space_never_starts_a_line():
    rng = random.Random(7)
    vocabulary = "je tu il elle nous vous ils elles ce pas vraiment maintenant faire dire".split()
    for trial in range(400):
        words = [rng.choice(vocabulary) for _ in range(rng.randint(4, 12))]
        words.insert(rng.randint(1, len(words)), rng.choice("?!:;"))
        for separator in (" ", "\u00a0"):
            text = " ".join(words).replace(" ?", separator + "?").replace(" !", separator + "!")
            for width in (16, 18, 26):
                for lines in (wrap_semantic_lines(text, width), compact_lines(text, 2, width)):
                    assert "".join("".join(lines).split()) == "".join(text.split())
                    assert not any(line[:1] in "?!:;" for line in lines), (trial, text, lines)


_LATIN = ("Kommst du heute mit uns ins Kino oder bleibst du lieber zu Hause bei deiner Familie "
          "A neve está ótima hoje de manhã e eu estou adorando minha prancha nova").split()
_JAPANESE = ["今日は", "とても", "いい天気ですね", "散歩でも", "しましょうか", "俺はただ", "正当防衛を",
             "したまでだって", "言ってるだろうが", "あのコンピューター", "システムは", "もう完全に", "ダメに",
             "なってしまった", "ちょっと待ってくれよ", "そんなことは", "聞いてないぞ"]


def _random_case(rng: random.Random) -> tuple[Cue, list[Word]]:
    if rng.random() < .5:
        lines = ["".join(rng.choice(_JAPANESE) for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 4))]
        lines = [line for line in lines if line] or ["今日は"]
        words = _char_words("".join(lines))
    else:
        turns = rng.randint(1, 3)
        lines = []
        for turn in range(turns):
            body = " ".join(rng.choice(_LATIN) for _ in range(rng.randint(1, 7))) + rng.choice([".", "?", "!", ","])
            body = body[0].upper() + body[1:]
            wrap = wrap_semantic_lines(body, rng.choice([14, 20, 30]))
            if turns > 1:
                wrap[0] = "- " + wrap[0]
            lines.extend(wrap)
        spoken = [token for line in lines for token in line.split() if token != "-"]
        words, time = [], 1.0
        for token in spoken:
            words.append(Word(text=token, start=round(time, 3), end=round(time + .25, 3)))
            time += .3 + (.25 if token[-1] in ".?!" else 0)
    end_ms = round(words[-1].end * 1000) + 200
    return Cue(index=1, start_ms=1000, end_ms=end_ms, lines=lines), words


@pytest.mark.parametrize("seed", range(6))
def test_reflow_and_split_preserve_every_character_turn_and_legal_break(seed):
    rng = random.Random(seed)
    for _ in range(40):
        source, words = _random_case(rng)
        width = rng.choice([16, 20, 26])
        owned = _owned(words) if rng.random() < .6 else []
        result = split_crowded_output_cues([source], words, {1: owned}, StyleProfile(max_chars_per_line=width),
                                           enforce_width=rng.random() < .5)
        shown = [line for cue in result.cues for line in cue.lines]
        assert "".join("".join(shown).split()) == "".join("".join(source.lines).split()), (source.lines, shown)
        assert all(len(cue.lines) <= 2 for cue in result.cues)
        japanese = any(_KANJI.search(line) or "぀" <= line[0] <= "ヿ" for line in source.lines)
        if japanese:
            assert not any(" " in line for line in shown), shown
            assert not any(line[0] in _NONSTARTING_KANA for line in shown), shown
        else:
            turns = sum(1 for line in source.lines if _TURN.match(line))
            if turns <= 2 or len(result.cues) > 1:
                _assert_turns_intact(source, result.cues)
            assert not any(line.split()[-1] == "-" for line in shown)
        changed = {flag.kind for flag in result.flags}
        if "output_line_limit_split" in changed or any(flag.message.startswith("The complete spoken phrase fits")
                                                       for flag in result.flags):
            assert all(display_width(line) <= width for line in shown), (source.lines, shown)
