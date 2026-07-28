from __future__ import annotations

from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile


def test_two_replacement_tokens_are_distributed_across_two_affected_cues():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old first"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["old second"]),
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2],
        srt_text="old first old second",
        asr_text="spoken replacement",
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="spoken replacement",
        confidence=1.0,
        reason="audio contains the replacement",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert [cue.plain_text for cue in transformed] == ["spoken", "replacement"]


def test_adjudicated_cues_with_fewer_lexical_units_round_trip_through_srt():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old first"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["old second"]),
        Cue(index=3, start_ms=2000, end_ms=3000, lines=["old third"]),
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2, 3],
        srt_text="old first old second old third",
        asr_text="brief replacement",
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="brief replacement",
        confidence=1.0,
        reason="audio contains fewer lexical units than the source cues",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )
    assert transformed
    rendered = write_srt(transformed)

    reparsed = parse_srt_text(rendered)

    assert reparsed
    assert [cue.plain_text for cue in reparsed] == [
        cue.plain_text for cue in transformed
    ]
    assert all(cue.plain_text for cue in transformed)
    assert all(cue.plain_text for cue in reparsed)


def test_multiple_reordered_word_corrections_compose_in_one_cue():
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=3000,
            lines=["Ja, hier wirst du nicht gebraucht."],
        )
    ]
    spans = [
        DivergenceSpan(
            case_id="case-1",
            cue_ids=[1],
            srt_text="hier",
            asr_text="du",
            srt_token_indices=[1],
            asr_word_indices=[1],
        ),
        DivergenceSpan(
            case_id="case-2",
            cue_ids=[1],
            srt_text="du",
            asr_text="hier",
            srt_token_indices=[3],
            asr_word_indices=[3],
        ),
    ]
    decisions = [
        AdjudicationDecision(
            case_id="case-1",
            verdict="use_audio",
            final_text="du",
            confidence=1.0,
            reason="audio confirms the reordered subject",
        ),
        AdjudicationDecision(
            case_id="case-2",
            verdict="use_audio",
            final_text="hier",
            confidence=1.0,
            reason="audio confirms the reordered adverb",
        ),
    ]

    transformed, _ = apply_adjudication_decisions(
        cues,
        spans,
        decisions,
        StyleProfile(),
    )

    assert [cue.plain_text for cue in transformed] == [
        "Ja, du wirst hier nicht gebraucht."
    ]


def test_empty_partial_audio_decision_removes_only_unspoken_words():
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=3000,
            lines=["Das bildest du dir nur etwas ein, Nova."],
        )
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="etwas",
        asr_text="",
        srt_token_indices=[5],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="",
        confidence=1.0,
        reason="the actor omitted this word",
    )

    transformed, flags = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(drop_policy="keep_flagged"),
    )

    assert [cue.plain_text for cue in transformed] == [
        "Das bildest du dir nur ein, Nova."
    ]
    assert [flag.kind for flag in flags] == ["text_changed"]


def test_deleting_final_unspoken_word_does_not_leave_space_before_period():
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=3000,
            lines=["Matthew, schmarotz nicht bei uns rum."],
        )
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="rum",
        asr_text="",
        srt_token_indices=[5],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="",
        confidence=1.0,
        reason="the final word is not spoken",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert transformed[0].plain_text == "Matthew, schmarotz nicht bei uns."


def test_replacing_contraction_suffix_removes_orphan_apostrophe():
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=3000,
            lines=["Hier gibt's keine Monster."],
        )
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="s",
        asr_text="es",
        srt_token_indices=[2],
        asr_word_indices=[2],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="es",
        confidence=1.0,
        reason="the audio contains the expanded word",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert transformed[0].plain_text == "Hier gibt es keine Monster."


def test_context_only_final_text_deletes_indexed_middle_span_without_duplication():
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=3000,
            lines=["hello unwanted world"],
        )
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="unwanted",
        asr_text="",
        srt_token_indices=[1],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="hello world",
        confidence=1.0,
        reason="the final response includes unchanged cue context",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert transformed[0].plain_text == "hello world"
