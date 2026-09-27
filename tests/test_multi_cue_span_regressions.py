from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.style_profile import StyleProfile


def _audio_decision(text: str) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id="case-68", verdict="use_audio", final_text=text,
        confidence=1.0, reason="confirmed speech in the indexed span",
    )


def test_episode11_case68_preserves_source_outside_the_multicue_span():
    # Frozen authoritative test fix/11.srt cues 206/207, with global token
    # indices 985..989 rebased to this slice. The human-edited SRT is not input.
    cues = [
        Cue(index=206, start_ms=648830, end_ms=649710,
            lines=["O que vocês duas estão fazendo?"]),
        Cue(index=207, start_ms=650160, end_ms=651400,
            lines=["Estão achando que eu sou invisível?"]),
    ]
    span = DivergenceSpan(
        case_id="case-68", cue_ids=[206, 207],
        srt_text="vocês duas estão fazendo Estão", asr_text="tão falando? Tão",
        srt_token_indices=[2, 3, 4, 5, 6], asr_word_indices=[877, 878, 879],
        start=649.152, end=650.452,
    )

    changed, flags = apply_adjudication_decisions(
        cues, [span], [_audio_decision("tão falando? Tão")], StyleProfile(),
    )

    assert [cue.plain_text for cue in changed] == [
        "O que tão falando?", "Tão achando que eu sou invisível?",
    ]
    assert [cue.index for cue in changed] == [206, 207]
    assert len(flags) == 1
    assert "O que" in flags[0].new_text
    assert "achando que eu sou invisível?" in flags[0].new_text


def test_multicue_replacement_reassigns_evidence_to_the_same_text_partition():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Before old"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["a b c d e after."]),
    ]
    span = DivergenceSpan(
        case_id="case-68", cue_ids=[1, 2], srt_text="old a b c d e",
        asr_text="one two three four", srt_token_indices=list(range(1, 7)),
        asr_word_indices=[100, 101, 102, 103],
    )
    decision = _audio_decision("one two three four")
    alignment = AlignmentResult(cue_word_indices={1: [99, 101, 102], 2: [100, 103, 104]})

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    aligned = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues,
    )

    assert [cue.plain_text for cue in changed] == ["Before one", "two three four after."]
    assert aligned.cue_word_indices == {1: [99, 100], 2: [101, 102, 103, 104]}
    assigned = [word for indices in aligned.cue_word_indices.values() for word in indices]
    assert all(assigned.count(word) == 1 for word in span.asr_word_indices)
    assert alignment.cue_word_indices == {1: [99, 101, 102], 2: [100, 103, 104]}


def test_multicue_indexed_edit_holds_when_source_indices_are_not_contiguous():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old retained"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["other after."]),
    ]
    span = DivergenceSpan(
        case_id="case-68", cue_ids=[1, 2], srt_text="old other",
        asr_text="new speech", srt_token_indices=[0, 2], asr_word_indices=[10, 11],
    )
    alignment = AlignmentResult(cue_word_indices={1: [9], 2: [12]})
    decision = _audio_decision("new speech")

    changed, flags = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    aligned = _alignment_with_decision_words(alignment, [decision], [span], source_cues=cues)

    assert changed == cues
    assert [flag.kind for flag in flags] == ["adjudication_span_edit_held"]
    assert aligned.cue_word_indices == alignment.cue_word_indices


@pytest.mark.parametrize("replacement", ["aren't", "aren’t"])
def test_multicue_replacement_keeps_a_contraction_with_its_acoustic_word(replacement):
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["We are"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["not going."]),
    ]
    span = DivergenceSpan(
        case_id="case-68", cue_ids=[1, 2], srt_text="are not",
        asr_text=replacement, srt_token_indices=[1, 2], asr_word_indices=[1],
    )
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [1, 2]})
    decision = _audio_decision(replacement)

    changed, flags = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    aligned = _alignment_with_decision_words(alignment, [decision], [span], source_cues=cues)

    assert [cue.plain_text for cue in changed] == [f"We {replacement}", "going."]
    assert [flag.kind for flag in flags] == ["text_changed"]
    assert aligned.cue_word_indices == {1: [0, 1], 2: [2]}
