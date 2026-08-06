from __future__ import annotations

import dubsync.evaluation as evaluation_module
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


def test_alignment_flags_large_drift_relative_to_episode_offset():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nalpha\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nbeta\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\ngamma\n\n"
        "4\n00:00:03,000 --> 00:00:03,500\nneedle\n\n"
    )
    words = [
        Word(text="alpha", start=0.1, end=0.2),
        Word(text="beta", start=1.1, end=1.2),
        Word(text="gamma", start=2.1, end=2.2),
        *[
            Word(text=f"filler{index}", start=3.0 + index * 0.15, end=3.1 + index * 0.15)
            for index in range(300)
        ],
        Word(text="needle", start=50.1, end=50.3),
    ]

    result = align_cues_to_words(cues, words)

    assert result.cue_word_indices[4] == [303]
    outliers = [flag for flag in result.flags if flag.kind == "alignment_outlier"]
    assert [flag.cue_ids for flag in outliers] == [[4]]


def test_alignment_does_not_flag_uniform_episode_shift_as_an_outlier():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:00,500\nalpha\n\n"
        "2\n00:00:01,000 --> 00:00:01,500\nbeta\n\n"
        "3\n00:00:02,000 --> 00:00:02,500\ngamma\n\n"
    )
    words = [
        Word(text="alpha", start=50.1, end=50.2),
        Word(text="beta", start=51.1, end=51.2),
        Word(text="gamma", start=52.1, end=52.2),
    ]

    result = align_cues_to_words(cues, words)

    assert not any(flag.kind == "alignment_outlier" for flag in result.flags)


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


def test_evaluation_precomputes_each_cue_signature_once(monkeypatch):
    predicted = [
        parse_srt_text(f"1\n00:00:{index:02d},000 --> 00:00:{index:02d},500\nword {index}\n\n")[0]
        for index in range(40)
    ]
    golden = [cue.model_copy() for cue in predicted]
    original = evaluation_module._text_signature
    calls = 0

    def counted_signature(cue):
        nonlocal calls
        calls += 1
        return original(cue)

    monkeypatch.setattr(evaluation_module, "_text_signature", counted_signature)

    matches = evaluation_module._match_cues_by_content(predicted, golden)

    assert len(matches) == 40
    assert calls == len(predicted) + len(golden)
