from __future__ import annotations

from dubsync.aligner import align_cues_to_words
from dubsync.evaluation import evaluate_against_golden
from dubsync.models import Word
from dubsync.srt_io import parse_srt_text


def test_alignment_uses_cue_timing_prior_for_repeated_word():
    cues = parse_srt_text("1\n00:00:50,000 --> 00:00:51,000\nja\n\n")
    words = [
        Word(text="ja", start=1.0, end=1.2, confidence=0.99),
        Word(text="ja", start=50.1, end=50.3, confidence=0.99),
    ]

    result = align_cues_to_words(cues, words)

    assert result.cue_word_indices == {1: [1]}


def test_evaluation_matches_golden_content_after_inserted_predicted_cue():
    predicted = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,400\nalpha\n\n"
        "2\n00:00:00,500 --> 00:00:00,900\ninserted\n\n"
        "3\n00:00:01,000 --> 00:00:01,400\nbeta\n\n"
        "4\n00:00:02,000 --> 00:00:02,400\ngamma\n\n"
    )
    golden = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,400\nalpha\n\n"
        "2\n00:00:01,000 --> 00:00:01,400\nbeta\n\n"
        "3\n00:00:02,000 --> 00:00:02,400\ngamma\n\n"
    )

    metrics = evaluate_against_golden(predicted, golden, fps=30.0)

    assert metrics["matched_cues"] == 3
    assert metrics["start_mae_ms"] == 0.0
    assert metrics["starts_within_1_frame_ratio"] == 1.0
