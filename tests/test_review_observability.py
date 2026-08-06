from __future__ import annotations

from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.observability import name_spelling_inconsistency_flags, span_coverage_flags
from dubsync.pipeline import (
    _generated_adlib_rejection_flag,
    _is_repetitive_generated_text,
)
from dubsync.srt_io import parse_srt_text


def test_repetition_guard_keeps_short_emphatic_dialogue():
    assert _is_repetitive_generated_text("Nein, nein, nein, nein, nein, nein, nein, nein") is False


def test_repetition_guard_still_rejects_long_repeated_credit_song():
    text = "la la la " * 8 + "oh oh oh " * 8

    assert _is_repetitive_generated_text(text) is True


def test_repetitive_adlib_requires_dialogue_envelope_context_for_short_spans():
    source = parse_srt_text("1\n00:00:00,000 --> 00:00:10,000\nNein!\n\n")
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="Nein " * 8, start=1.0, end=3.0)

    assert _generated_adlib_rejection_flag(source, span, "Nein " * 8) is None


def test_name_spelling_inconsistency_flags_source_observed_near_match_without_changing_text():
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nDeanna kommt.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nDeanna bleibt.\n\n"
    )
    output = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nDiana kommt.\n\n")

    flags = name_spelling_inconsistency_flags(source, output)

    assert len(flags) == 1
    assert flags[0].kind == "name_spelling_inconsistency"
    assert flags[0].severity == "warning"
    assert "Diana" in flags[0].message
    assert "Deanna" in flags[0].message


def test_span_coverage_flag_surfaces_compressed_replacement_without_reconciliation_change():
    source = [
        Cue(index=1, start_ms=10_000, end_ms=12_000, lines=["alpha beta"]),
        Cue(index=2, start_ms=12_000, end_ms=14_000, lines=["gamma delta"]),
    ]
    rebuilt = [
        Cue(index=1, start_ms=10_000, end_ms=10_700, lines=["alpha"]),
        Cue(index=2, start_ms=10_700, end_ms=11_400, lines=["beta"]),
    ]
    spans = [
        DivergenceSpan(case_id="case-1", cue_ids=[1, 2], srt_text="alpha beta gamma delta", asr_text="alpha", start=10.0, end=11.0)
    ]
    decisions = [
        AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="alpha", confidence=0.95, reason="fixture")
    ]

    flags = span_coverage_flags(source, rebuilt, spans, decisions)

    assert [flag.kind for flag in flags] == ["span_coverage_low"]
    assert flags[0].severity == "error"
    assert "35%" in flags[0].message
