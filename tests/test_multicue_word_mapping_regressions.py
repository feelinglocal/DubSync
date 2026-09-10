from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.style_profile import StyleProfile


def _decision(span: DivergenceSpan, text: str) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id=span.case_id, verdict="hybrid", final_text=text,
        confidence=0.95, reason="confirmed wording within the saved audio span",
    )


def test_episode11_case115_preserves_the_last_word_before_the_acoustic_gap():
    # Authoritative source cues 330/331 and saved final/11 case-115. Source
    # indices 1479..1487 and ASR indices 1357..1365 are rebased to this slice.
    cues = [
        Cue(index=330, start_ms=1002750, end_ms=1003920,
            lines=["Lá em cima não tem banheiro."]),
        Cue(index=331, start_ms=1009230, end_ms=1010080,
            lines=["Vou segurar mais um pouco."]),
    ]
    span = DivergenceSpan(
        case_id="case-115", cue_ids=[330, 331],
        srt_text="Lá em cima não tem banheiro Vou segurar mais",
        asr_text="Tá bom, então vamo lá. Devo esperar",
        srt_token_indices=list(range(9)), asr_word_indices=list(range(7)),
        start=1002.912, end=1009.822, confidence=1.0,
    )
    decision = _decision(span, "Tá bom, então vamo lá. Eu vou esperar")
    alignment = AlignmentResult(cue_word_indices={330: [], 331: [7, 8]})

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    aligned = _alignment_with_decision_words(alignment, [decision], [span], source_cues=cues)

    assert [cue.plain_text for cue in changed] == [
        "Tá bom, então vamo lá.", "Eu vou esperar um pouco.",
    ]
    assert aligned.cue_word_indices == {330: [0, 1, 2, 3, 4], 331: [5, 6, 7, 8]}
    assert not any(flag.kind == "adjudication_word_mapping_held" for flag in aligned.flags)
    words = [
        Word(text=text, start=start, end=end, confidence=1.0, speaker_id="speaker_1")
        for text, start, end in [
            ("Tá", 1002.912, 1003.062), ("bom,", 1003.082, 1003.182),
            ("então", 1003.262, 1003.362), ("vamo", 1003.402, 1003.562),
            ("lá.", 1003.572, 1003.702), ("Devo", 1009.352, 1009.522),
            ("esperar", 1009.542, 1009.822), ("um", 1009.842, 1009.942),
            ("pouco,", 1009.962, 1010.142),
        ]
    ]
    with_words = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words,
    )
    assert with_words.cue_word_indices == aligned.cue_word_indices
    assert max(words[index].end for index in with_words.cue_word_indices[330]) == 1003.702
    assert min(words[index].start for index in with_words.cue_word_indices[331]) == 1009.352


def test_episode11_case199_keeps_the_bounded_name_alias_before_the_invitation():
    cues = [
        Cue(index=506, start_ms=1424640, end_ms=1425120, lines=["Luan Nian."]),
        Cue(index=507, start_ms=1425120, end_ms=1426280,
            lines=["- Luke, foto. - Olha para a câmera."]),
    ]
    span = DivergenceSpan(
        case_id="case-199", cue_ids=[506, 507], srt_text="Luan Nian Luke",
        asr_text="Yeon Su. Vem, vamo tirar uma",
        srt_token_indices=[0, 1, 2], asr_word_indices=list(range(6)),
        start=1425.218, end=1425.808, confidence=1.0,
        speaker_ids=["speaker_4", "speaker_5"],
    )
    decision = _decision(span, "Luan Nian. Vem, vamo tirar uma")
    alignment = AlignmentResult(cue_word_indices={506: [], 507: [6, 7, 8, 9, 10]})

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    aligned = _alignment_with_decision_words(alignment, [decision], [span], source_cues=cues)

    assert changed[0].plain_text == "Luan Nian."
    assert changed[1].plain_text == "- Vem, vamo tirar uma, foto. - Olha para a câmera."
    assert aligned.cue_word_indices == {506: [0, 1], 507: [2, 3, 4, 5, 6, 7, 8, 9, 10]}


def test_multicue_acoustic_boundary_is_held_without_retained_lexical_evidence():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old phrase"]),
        Cue(index=2, start_ms=4000, end_ms=5000, lines=["ending"]),
    ]
    span = DivergenceSpan(
        case_id="unanchored", cue_ids=[1, 2], srt_text="old phrase ending",
        asr_text="foo bar", srt_token_indices=[0, 1, 2],
        asr_word_indices=[10, 11], confidence=1.0,
    )
    alignment = AlignmentResult(cue_word_indices={1: [9], 2: [12]})

    aligned = _alignment_with_decision_words(
        alignment, [_decision(span, "alpha gamma delta")], [span], source_cues=cues,
    )

    assert aligned.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in aligned.flags] == ["adjudication_word_mapping_held"]
    assert aligned.flags[0].cue_ids == [1, 2]
    assert not alignment.flags


def test_repeated_lexical_anchors_do_not_resolve_an_ambiguous_word_boundary():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old"]),
        Cue(index=2, start_ms=2000, end_ms=3000, lines=["other"]),
    ]
    span = DivergenceSpan(
        case_id="repeated", cue_ids=[1, 2], srt_text="old other",
        asr_text="yes yes yes", srt_token_indices=[0, 1],
        asr_word_indices=[10, 11, 12], confidence=1.0,
    )
    alignment = AlignmentResult(cue_word_indices={1: [9], 2: [13]})

    aligned = _alignment_with_decision_words(
        alignment, [_decision(span, "yes yes")], [span], source_cues=cues,
    )

    assert aligned.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in aligned.flags] == ["adjudication_word_mapping_held"]


def test_exact_lexical_boundary_respects_multitoken_asr_word_entries():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old phrase"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["other ending"]),
    ]
    span = DivergenceSpan(
        case_id="multitoken-word", cue_ids=[1, 2], srt_text="old phrase other ending",
        asr_text="I really mean it", srt_token_indices=[0, 1, 2, 3],
        asr_word_indices=[0, 1, 2], confidence=1.0,
    )
    words = [
        Word(text="I really", start=0.1, end=0.8),
        Word(text="mean", start=1.1, end=1.3),
        Word(text="it", start=1.3, end=1.5),
    ]

    aligned = _alignment_with_decision_words(
        AlignmentResult(), [_decision(span, "I really mean it")], [span],
        source_cues=cues, words=words,
    )

    assert aligned.cue_word_indices == {1: [0], 2: [1, 2]}
    assert not aligned.flags


def test_a_replacement_cannot_divide_one_acoustic_word_between_two_cues():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["other phrase"]),
    ]
    span = DivergenceSpan(
        case_id="indivisible-word", cue_ids=[1, 2], srt_text="old other phrase",
        asr_text="can't go now", srt_token_indices=[0, 1, 2],
        asr_word_indices=[0, 1], confidence=1.0,
    )
    words = [Word(text="can't go", start=0.2, end=1.3), Word(text="now", start=1.5, end=1.8)]
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [1]})

    aligned = _alignment_with_decision_words(
        alignment, [_decision(span, "can't go now")], [span], source_cues=cues, words=words,
    )

    assert aligned.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in aligned.flags] == ["adjudication_word_mapping_held"]


@pytest.mark.parametrize("anchor_confidence", [0.2, 0.95])
def test_changed_phrase_needs_a_confident_retained_anchor(anchor_confidence):
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old"]),
        Cue(index=2, start_ms=4000, end_ms=5000, lines=["other phrase"]),
    ]
    span = DivergenceSpan(
        case_id="anchor-confidence", cue_ids=[1, 2], srt_text="old other phrase",
        asr_text="anchor changed", srt_token_indices=[0, 1, 2],
        asr_word_indices=[0, 1], confidence=1.0,
    )
    words = [
        Word(text="anchor", start=0.2, end=0.4, confidence=anchor_confidence),
        Word(text="changed", start=4.2, end=4.4, confidence=1.0),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [], 2: []})

    aligned = _alignment_with_decision_words(
        alignment, [_decision(span, "anchor new words")], [span], source_cues=cues, words=words,
    )

    if anchor_confidence < 0.8:
        assert aligned.cue_word_indices == alignment.cue_word_indices
        assert [flag.kind for flag in aligned.flags] == ["adjudication_word_mapping_held"]
    else:
        assert aligned.cue_word_indices == {1: [0], 2: [1]}
        assert not aligned.flags


def test_sentence_partition_excludes_words_outside_the_unique_retained_phrase():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old."]),
        Cue(index=2, start_ms=4000, end_ms=6000, lines=["other."]),
    ]
    span = DivergenceSpan(
        case_id="retained-phrase", cue_ids=[1, 2], srt_text="old other",
        asr_text="yes. YES next", srt_token_indices=[0, 1],
        asr_word_indices=[0, 1, 2], confidence=1.0,
    )
    words = [
        Word(text="yes.", start=0.1, end=0.2),
        Word(text="YES", start=4.1, end=4.2),
        Word(text="next", start=5.1, end=5.2),
    ]

    aligned = _alignment_with_decision_words(
        AlignmentResult(), [_decision(span, "yes. next")], [span], source_cues=cues, words=words,
    )

    assert aligned.cue_word_indices == {1: [0], 2: [2]}
    assert not aligned.flags


def test_retained_phrase_with_two_possible_word_windows_holds_timing():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old."]),
        Cue(index=2, start_ms=4000, end_ms=6000, lines=["other."]),
    ]
    span = DivergenceSpan(
        case_id="repeated-phrase", cue_ids=[1, 2], srt_text="old other",
        asr_text="call. yes yes", srt_token_indices=[0, 1],
        asr_word_indices=[0, 1, 2], confidence=1.0,
    )
    words = [
        Word(text="call.", start=0.1, end=0.2),
        Word(text="yes", start=4.1, end=4.2),
        Word(text="yes", start=5.1, end=5.2),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [], 2: []})

    aligned = _alignment_with_decision_words(
        alignment, [_decision(span, "call. yes")], [span], source_cues=cues, words=words,
    )

    assert aligned.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in aligned.flags] == ["adjudication_word_mapping_held"]
