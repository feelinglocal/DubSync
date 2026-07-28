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
