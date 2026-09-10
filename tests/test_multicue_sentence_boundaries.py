from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions, indexed_multi_cue_replacements
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.style_profile import StyleProfile


def _case73() -> tuple[list[Cue], DivergenceSpan]:
    # Frozen source cues and alignment evidence from episode 11; token indices
    # 1022..1031 are rebased to this two-cue slice. No human-edited text is input.
    cues = [
        Cue(index=213, start_ms=659110, end_ms=660680,
            lines=["você nunca vai ficar sem flores no dia a dia."]),
        Cue(index=214, start_ms=660680, end_ms=662200,
            lines=["Produção própria, consumo próprio, pura alegria."]),
    ]
    return cues, DivergenceSpan(
        case_id="case-73", cue_ids=[213, 214],
        srt_text="no dia a dia Produção própria consumo próprio pura alegria",
        asr_text="Você será muito feliz.", srt_token_indices=list(range(6, 16)),
        asr_word_indices=[908, 909, 910, 911], start=660.852, end=662.072,
    )


def test_case73_whole_sentence_stays_in_its_observed_source_time_window():
    cues, span = _case73()

    assert indexed_multi_cue_replacements(cues, span, "Você será muito feliz.") == {
        213: (6, 10, ""), 214: (0, 6, "Você será muito feliz."),
    }

    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=1.0, reason="confirmed speech in the indexed span",
    )
    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    assert [cue.plain_text for cue in changed] == [
        "você nunca vai ficar sem flores.", "Você será muito feliz.",
    ]
    assert [(cue.start_ms, cue.end_ms) for cue in changed] == [
        (cue.start_ms, cue.end_ms) for cue in cues
    ]


@pytest.mark.parametrize("updates", [
    {"start": None, "end": None},
    {"start": 660.5},
    {"end": 662.3},
    {"start": 662.1, "end": 660.9},
])
def test_whole_sentence_assignment_requires_unambiguous_timing(updates):
    cues, span = _case73()
    span = span.model_copy(update=updates)

    assert indexed_multi_cue_replacements(cues, span, "Você será muito feliz.") == {
        213: (6, 10, "Você será"), 214: (0, 6, "muito feliz."),
    }


def test_whole_sentence_assignment_requires_a_complete_target_source_cue():
    cues, span = _case73()
    cues[1] = cues[1].with_lines(["Produção própria, consumo próprio, pura alegria. Depois."])

    assert indexed_multi_cue_replacements(cues, span, "Você será muito feliz.") == {
        213: (6, 10, "Você será"), 214: (0, 6, "muito feliz."),
    }


def test_case115_keeps_the_sentence_break_and_unedited_suffix():
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
        srt_token_indices=list(range(9)), start=1002.912, end=1009.822,
    )

    assert indexed_multi_cue_replacements(
        cues, span, "Tá bom, então vamo lá. Eu vou esperar",
    ) == {330: (0, 6, "Tá bom, então vamo lá."), 331: (0, 3, "Eu vou esperar")}


def test_case199_corrected_name_keeps_its_sentence_and_speaker_turn():
    cues = [
        Cue(index=506, start_ms=1424640, end_ms=1425120, lines=["Luan Nian."]),
        Cue(index=507, start_ms=1425120, end_ms=1426280,
            lines=["- Luke, foto. - Olha para a câmera."]),
    ]
    span = DivergenceSpan(
        case_id="case-199", cue_ids=[506, 507], srt_text="Luan Nian Luke",
        asr_text="Yeon Su. Vem, vamo tirar uma", srt_token_indices=[0, 1, 2],
        start=1425.218, end=1425.808, speaker_ids=["speaker_4", "speaker_5"],
    )

    assert indexed_multi_cue_replacements(
        cues, span, "Luan Nian. Vem, vamo tirar uma",
    ) == {506: (0, 2, "Luan Nian."), 507: (0, 1, "Vem, vamo tirar uma")}


@pytest.mark.parametrize("final_text, expected", [
    ("One two. Three four five six", ["One two.", "Three four five six"]),
    ("One. Two three four five six", ["One. Two three four", "five six"]),
    ("One, two three four five six", ["One, two three four", "five six"]),
    ("One two aren't. Three four five", ["One two aren't.", "Three four five"]),
])
def test_sentence_preference_is_bounded_and_keeps_lexical_units(final_text, expected):
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Old sentence."]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["Other suffix."]),
    ]
    span = DivergenceSpan(
        case_id="bounded", cue_ids=[1, 2], srt_text="Old sentence Other",
        asr_text=final_text, srt_token_indices=[0, 1, 2],
    )

    edits = indexed_multi_cue_replacements(cues, span, final_text)
    assert edits is not None
    assert [edit[2] for edit in edits.values()] == expected


def test_sentence_preference_needs_a_matching_source_sentence_boundary():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Old unfinished"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["sentence."]),
    ]
    span = DivergenceSpan(
        case_id="unfinished", cue_ids=[1, 2], srt_text="Old unfinished sentence",
        srt_token_indices=[0, 1, 2], asr_text="One two. Three four five six.",
    )

    assert indexed_multi_cue_replacements(cues, span, span.asr_text) == {
        1: (0, 2, "One two. Three four"), 2: (0, 1, "five six."),
    }


def test_equally_near_sentence_boundaries_keep_the_source_proportion():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Old."]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["Other."]),
    ]
    span = DivergenceSpan(
        case_id="tied", cue_ids=[1, 2], srt_text="Old Other",
        srt_token_indices=[0, 1], asr_text="One two. Three four five six. Seven eight.",
    )

    assert indexed_multi_cue_replacements(cues, span, span.asr_text) == {
        1: (0, 1, "One two. Three four"), 2: (0, 1, "five six. Seven eight."),
    }


def test_three_cue_partition_does_not_reuse_a_sentence_boundary():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["First."]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["Second."]),
        Cue(index=3, start_ms=2000, end_ms=3000, lines=["Third."]),
    ]
    span = DivergenceSpan(
        case_id="three", cue_ids=[1, 2, 3], srt_text="First Second Third",
        srt_token_indices=[0, 1, 2], asr_text="One two three. Four five six.",
    )

    assert indexed_multi_cue_replacements(cues, span, span.asr_text) == {
        1: (0, 1, "One two three."), 2: (0, 1, "Four"), 3: (0, 1, "five six."),
    }


def test_internal_source_title_punctuation_is_not_a_new_cue_boundary():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Dr. Smith."]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["Other suffix."]),
    ]
    span = DivergenceSpan(
        case_id="title", cue_ids=[1, 2], srt_text="Dr Smith Other",
        srt_token_indices=[0, 1, 2], asr_text="Dr. Smith arrived safely today",
    )

    assert indexed_multi_cue_replacements(cues, span, span.asr_text) == {
        1: (0, 2, "Dr. Smith arrived"), 2: (0, 1, "safely today"),
    }


def _case164() -> tuple[list[Cue], DivergenceSpan]:
    # Frozen episode 11 source/evidence; global source 2048..2051 is rebased.
    cues = [
        Cue(index=452, start_ms=1321160, end_ms=1321680, lines=["Vamos."]),
        Cue(index=453, start_ms=1321750, end_ms=1322310, lines=["Vamos rápido."]),
        Cue(index=454, start_ms=1322310, end_ms=1323030, lines=["Yuanzhu vai pagar."]),
    ]
    return cues, DivergenceSpan(
        case_id="case-164", cue_ids=[452, 453, 454],
        srt_text="Vamos Vamos rápido Yuanzhu", asr_text="E o Yeon Su",
        srt_token_indices=[0, 1, 2, 3], asr_word_indices=[0, 1, 2, 3],
        start=1322.278, end=1322.738, speaker_ids=["speaker_4"],
    )


def test_case164_unfinished_phrase_follows_its_retained_source_anchor():
    cues, span = _case164()
    final_text = "E o Yuanzhu"

    assert indexed_multi_cue_replacements(cues, span, final_text) == {
        452: (0, 1, ""), 453: (0, 2, ""), 454: (0, 1, final_text),
    }
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="hybrid", final_text=final_text,
        confidence=0.95, reason="confirmed unfinished phrase with retained source name",
    )
    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    assert [(cue.index, cue.plain_text) for cue in changed] == [
        (454, "E o Yuanzhu vai pagar."),
    ]
    assert (changed[0].start_ms, changed[0].end_ms) == (1322310, 1323030)
    aligned = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={452: [], 453: [], 454: [4, 5]}),
        [decision], [span], source_cues=cues,
    )
    assert aligned.cue_word_indices == {452: [], 453: [], 454: [0, 1, 2, 3, 4, 5]}


@pytest.mark.parametrize("final_text, speakers", [
    ("E o Yuanzhu", []),
    ("E o Yuanzhu", ["speaker_4", "speaker_5"]),
    ("E o outro", ["speaker_4"]),
    ("E o Yuanzhu.", ["speaker_4"]),
    ("E. O Yuanzhu", ["speaker_4"]),
])
def test_continuation_assignment_requires_one_unfinished_anchored_phrase(final_text, speakers):
    cues, span = _case164()
    span = span.model_copy(update={"speaker_ids": speakers})

    edits = indexed_multi_cue_replacements(cues, span, final_text)
    assert edits is not None
    assert any(edits[cue_id][2] for cue_id in (452, 453))


def test_continuation_assignment_cannot_consume_a_retained_first_cue_prefix():
    cues, span = _case164()
    cues[0] = cues[0].with_lines(["Então Vamos."])
    span = span.model_copy(update={"srt_token_indices": [1, 2, 3, 4]})

    assert indexed_multi_cue_replacements(cues, span, "E o Yuanzhu") == {
        452: (1, 2, "E"), 453: (0, 2, "o"), 454: (0, 1, "Yuanzhu"),
    }


@pytest.mark.parametrize("target_text", ["Yuanzhu.", "Yuanzhu. Ele vai pagar."])
def test_continuation_assignment_requires_an_unfinished_source_prefix(target_text):
    cues, span = _case164()
    cues[2] = cues[2].with_lines([target_text])

    assert indexed_multi_cue_replacements(cues, span, "E o Yuanzhu") == {
        452: (0, 1, "E"), 453: (0, 2, "o"), 454: (0, 1, "Yuanzhu"),
    }
