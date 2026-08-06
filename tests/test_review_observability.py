from __future__ import annotations

import json

from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.observability import name_spelling_inconsistency_flags, span_coverage_flags
from dubsync.pipeline import (
    _generated_adlib_rejection_flag,
    _is_repetitive_generated_text,
    _load_asr_artifact_with_repair,
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


def test_repetitive_adlib_is_rejected_even_without_span_timing():
    source = parse_srt_text("1\n00:00:00,000 --> 00:00:10,000\nDialogue.\n\n")
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="la " * 18, start=None, end=None)

    flag = _generated_adlib_rejection_flag(source, span, "la " * 18)

    assert flag is not None
    assert flag.kind == "adlib_rejected_repetitive_content"


def test_name_spelling_inconsistency_flags_source_observed_near_match_without_changing_text():
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nIch sehe Deanna.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nIch kenne Deanna.\n\n"
    )
    output = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nIch sehe Diana.\n\n")

    flags = name_spelling_inconsistency_flags(source, output)

    assert len(flags) == 1
    assert flags[0].kind == "name_spelling_inconsistency"
    assert flags[0].severity == "warning"
    assert "Diana" in flags[0].message
    assert "Deanna" in flags[0].message


def test_near_match_to_lowercase_source_word_is_labeled_word_substitution_not_name():
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nWir gehen nach Hause.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSie gehen weiter.\n\n"
    )
    output = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nWir sehen nach Hause.\n\n")

    flags = name_spelling_inconsistency_flags(source, output)

    assert [flag.kind for flag in flags] == ["unsourced_word_substitution"]
    assert "word substitution" in flags[0].message


def test_sentence_initial_capitalized_common_word_is_not_labeled_name_drift():
    source = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nFrau Holle kommt.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nFrau Holle bleibt.\n\n"
    )
    output = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nFrag Holle kommt.\n\n")

    flags = name_spelling_inconsistency_flags(source, output)

    assert [flag.kind for flag in flags] == ["unsourced_word_substitution"]


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


def test_resume_asr_artifact_repairs_words_and_merges_persisted_flags(tmp_path):
    artifact_path = tmp_path / "asr.json"
    artifact_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "", "start": 0.0, "end": 0.1},
                    {"text": "beta", "start": 2.0, "end": 2.0},
                    {"text": "alpha", "start": 1.0, "end": 1.5},
                ],
                "repair_flags": [
                    {
                        "kind": "word_stream_repaired",
                        "cue_ids": [],
                        "message": "Persisted provider repair.",
                        "severity": "warning",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    words, flags = _load_asr_artifact_with_repair(artifact_path)

    assert [word.text for word in words] == ["alpha", "beta"]
    assert words[1].end > words[1].start
    assert [flag.message for flag in flags] == [
        "Persisted provider repair.",
        "ASR resume artifact word stream was repaired before alignment: "
        "1 blank dropped, 0 invalid dropped, 1 timing clamped, 2 reordered; 2 usable words remain.",
    ]
