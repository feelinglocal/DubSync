from dubsync.adjudication_parts import (
    build_adjudication_plan,
    combine_adjudication_parts,
)
from dubsync.models import AdjudicationDecision, DivergenceSpan, Word
from dubsync.tokenize import alphanumeric_signature


def _long_insertion() -> tuple[DivergenceSpan, list[Word]]:
    words: list[Word] = []
    for index in range(99):
        text = f"word{index}"
        if index in {24, 49, 74, 98}:
            text += "."
        start = 54.68 + index * 0.54
        words.append(
            Word(
                text=text,
                start=round(start, 3),
                end=round(start + 0.32, 3),
                confidence=0.98,
                speaker_id="A" if index < 75 else "B",
            )
        )
    span = DivergenceSpan(
        case_id="case-16",
        cue_ids=[],
        srt_text="",
        asr_text=" ".join(word.text for word in words),
        start=words[0].start,
        end=words[-1].end,
        confidence=0.98,
        speaker_ids=["A", "B"],
        asr_word_indices=list(range(len(words))),
        left_anchor_cue_id=26,
        left_anchor_end=53.966,
        left_anchor_speaker_id="A",
    )
    return span, words


def test_long_pure_insertion_is_partitioned_without_clipping_or_word_loss():
    parent, words = _long_insertion()

    plan = build_adjudication_plan(
        [parent],
        words,
        pad_seconds=2.0,
        max_duration_seconds=20.0,
    )

    assert plan.parent_spans == (parent,)
    assert plan.partitioned_parent_ids == ("case-16",)
    assert 2 <= len(plan.provider_spans) <= 8
    assert plan.part_ids("case-16") == tuple(span.case_id for span in plan.provider_spans)
    assert [index for span in plan.provider_spans for index in span.asr_word_indices] == list(
        range(99)
    )
    assert all((span.end or 0) - (span.start or 0) + 4.0 <= 20.001 for span in plan.provider_spans)
    assert all(span.case_id.startswith("case-16.part-") for span in plan.provider_spans)
    assert all(span.cue_ids == [] and span.srt_token_indices == [] for span in plan.provider_spans)


def test_source_backed_long_span_is_not_partitioned():
    parent, words = _long_insertion()
    source_backed = parent.model_copy(
        update={
            "cue_ids": [26],
            "srt_text": "source text",
            "srt_token_indices": [0, 1],
        }
    )

    plan = build_adjudication_plan(
        [source_backed],
        words,
        pad_seconds=2.0,
        max_duration_seconds=20.0,
    )

    assert plan.provider_spans == (source_backed,)
    assert plan.partitioned_parent_ids == ()
    assert plan.part_ids("case-16") == ("case-16",)


def test_partition_decisions_recombine_in_acoustic_order_with_full_text_coverage():
    parent, words = _long_insertion()
    plan = build_adjudication_plan(
        [parent],
        words,
        pad_seconds=2.0,
        max_duration_seconds=20.0,
    )
    part_decisions = [
        AdjudicationDecision(
            case_id=span.case_id,
            verdict="use_audio",
            final_text=span.asr_text,
            confidence=0.99 - position * 0.01,
            speaker=span.speaker_ids[0] if len(span.speaker_ids) == 1 else None,
            character="unknown",
            reason="bounded audio confirms the complete part",
        )
        for position, span in enumerate(plan.provider_spans)
    ]

    decisions, flags = combine_adjudication_parts(
        plan,
        list(reversed(part_decisions)),
        [],
        confidence_gate=0.7,
    )

    assert len(decisions) == 1
    assert decisions[0].case_id == "case-16"
    assert decisions[0].verdict == "use_audio"
    assert alphanumeric_signature(decisions[0].final_text) == alphanumeric_signature(parent.asr_text)
    assert decisions[0].confidence == min(decision.confidence for decision in part_decisions)
    assert [flag.kind for flag in flags] == ["adjudication_span_partitioned"]


def test_partial_middle_only_part_response_atomically_holds_the_parent():
    parent, words = _long_insertion()
    plan = build_adjudication_plan(
        [parent],
        words,
        pad_seconds=2.0,
        max_duration_seconds=20.0,
    )
    part_decisions = [
        AdjudicationDecision(
            case_id=span.case_id,
            verdict="use_audio",
            final_text=(span.asr_text if position else words[span.asr_word_indices[0]].text),
            confidence=0.99,
            speaker=None,
            character="unknown",
            reason="provider response",
        )
        for position, span in enumerate(plan.provider_spans)
    ]

    decisions, flags = combine_adjudication_parts(
        plan,
        part_decisions,
        [],
        confidence_gate=0.7,
    )

    assert decisions[0].case_id == "case-16"
    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == ""
    assert any(flag.kind == "adjudication_parts_incomplete" for flag in flags)
    assert not any(flag.kind == "adjudication_span_partitioned" for flag in flags)
