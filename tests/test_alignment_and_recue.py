from __future__ import annotations

from dubsync import aligner
from dubsync.aligner import align_cues_to_words
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.recue import rebuild_cues
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.verify import lint_cues


def test_shifted_timing_gets_full_anchor_coverage(shifted_srt_text, shifted_wordstream):
    cues = parse_srt_text(shifted_srt_text)
    words = [Word.model_validate(item) for item in shifted_wordstream]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.unmatched_cue_ids == []


def test_shifted_timing_outputs_anchor_regions(shifted_srt_text, shifted_wordstream):
    cues = parse_srt_text(shifted_srt_text)
    words = [Word.model_validate(item) for item in shifted_wordstream]

    result = align_cues_to_words(cues, words)

    assert len(result.anchor_regions) == 1
    anchor = result.anchor_regions[0]
    assert anchor.anchor_id == "anchor-1"
    assert anchor.cue_ids == [1, 2]
    assert anchor.srt_token_indices == [0, 1, 2, 3]
    assert anchor.asr_word_indices == [0, 1, 2, 3]
    assert anchor.srt_text == "hello there general kenobi"
    assert anchor.asr_text == "hello there general kenobi"
    assert anchor.start == 1.0
    assert anchor.end == 2.8
    assert anchor.score == 1.0


def test_alignment_normalizes_digits_and_spoken_number_words():
    cues = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nyou have 2 choices\n\n")
    words = [
        Word(text="you", start=0.00, end=0.10),
        Word(text="have", start=0.12, end=0.22),
        Word(text="two", start=0.24, end=0.36),
        Word(text="choices", start=0.38, end=0.70),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1, 2, 3]}


def test_alignment_normalizes_german_hyphen_compounds_and_ordinals():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nLevel-1-Versager dritte Prufung\n\n"
    )
    words = [
        Word(text="Level-eins-Versager", start=0.0, end=0.5),
        Word(text="dritte", start=0.6, end=0.8),
        Word(text="Prufung", start=0.82, end=1.0),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []


def test_alignment_uses_banded_dp_for_long_same_text_episode(monkeypatch):
    calls = 0
    original_similarity = aligner._similarity

    def counting_similarity(left: str, right: str) -> float:
        nonlocal calls
        calls += 1
        return original_similarity(left, right)

    monkeypatch.setattr(aligner, "_similarity", counting_similarity)
    token_count = 300
    cues = [
        Cue(index=index + 1, start_ms=index * 500, end_ms=index * 500 + 400, lines=[f"token{index}"])
        for index in range(token_count)
    ]
    words = [
        Word(text=f"token{index}", start=index * 0.5, end=index * 0.5 + 0.3, confidence=0.99)
        for index in range(token_count)
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert calls < (token_count * token_count) // 2


def test_injected_improv_span_is_isolated_to_changed_cue(shifted_srt_text):
    cues = parse_srt_text(shifted_srt_text)
    words = [
        Word(text="hello", start=1.00, end=1.20, confidence=0.98, speaker_id="A"),
        Word(text="there", start=1.23, end=1.45, confidence=0.97, speaker_id="A"),
        Word(text="you", start=2.00, end=2.10, confidence=0.98, speaker_id="A"),
        Word(text="are", start=2.12, end=2.22, confidence=0.96, speaker_id="A"),
        Word(text="early", start=2.24, end=2.58, confidence=0.99, speaker_id="A"),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 0.5
    assert len(result.divergence_spans) == 1
    span = result.divergence_spans[0]
    assert span.cue_ids == [2]
    assert span.srt_text == "general kenobi"
    assert span.asr_text == "you are early"


def test_multi_cue_changed_span_distributes_spoken_word_indices_per_cue():
    alignment = AlignmentResult()
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2],
        srt_text="old first old second",
        asr_text="new spoken first second",
        asr_word_indices=[10, 11, 12, 13],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="new spoken first second",
        confidence=0.91,
        speaker="A",
        character="unknown",
        reason="actor improvised",
    )

    updated = _alignment_with_decision_words(alignment, [decision], [span])

    assert updated.cue_word_indices == {1: [10, 11], 2: [12, 13]}


def test_delete_only_divergence_inherits_surrounding_anchor_window():
    cues = parse_srt_text("1\n00:00:00,000 --> 00:00:03,000\nalpha missing omega\n\n")
    words = [
        Word(text="alpha", start=1.00, end=1.20, confidence=0.98, speaker_id="A"),
        Word(text="omega", start=2.00, end=2.30, confidence=0.97, speaker_id="A"),
    ]

    result = align_cues_to_words(cues, words)

    assert len(result.divergence_spans) == 1
    span = result.divergence_spans[0]
    assert span.cue_ids == [1]
    assert span.srt_text == "missing"
    assert span.asr_text == ""
    assert span.asr_word_indices == []
    assert span.start == 1.20
    assert span.end == 2.00


def test_recue_preserves_unchanged_segmentation_and_snaps_to_grid(shifted_srt_text, shifted_wordstream):
    cues = parse_srt_text(shifted_srt_text)
    words = [Word.model_validate(item) for item in shifted_wordstream]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=0.5)

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)
    issues = lint_cues(rebuilt, profile)

    assert [cue.text for cue in rebuilt] == [cue.text for cue in cues]
    assert [cue.index for cue in rebuilt] == [1, 2]
    assert rebuilt[0].start_ms == 1000
    assert rebuilt[0].end_ms == 1500
    assert rebuilt[1].start_ms == 2000
    assert rebuilt[1].end_ms == 2866
    assert flags == []
    assert issues == []


def test_recue_clamps_lead_in_to_zero_timestamp():
    cues = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
    words = [Word(text="hello", start=0.02, end=0.20, confidence=0.98, speaker_id="A")]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, lead_in_ms=100)

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    assert flags == []
    assert rebuilt[0].start_ms == 0
    assert rebuilt[0].end_ms >= 500


def test_recue_extends_min_duration_only_into_available_gap():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\ntwo\n\n"
    )
    words = [
        Word(text="one", start=1.00, end=1.10, confidence=0.98, speaker_id="A"),
        Word(text="two", start=1.20, end=1.30, confidence=0.97, speaker_id="A"),
    ]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5)

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)
    issues = lint_cues(rebuilt, profile)

    assert flags == []
    assert rebuilt[0].start_ms == 1000
    assert rebuilt[0].end_ms == 1200
    assert rebuilt[1].start_ms == 1200
    assert any(issue.kind == "min_duration" and issue.cue_id == 1 for issue in issues)
    assert not any(issue.kind == "overlap" for issue in issues)


def test_recue_propagates_dominant_speaker_id(shifted_srt_text, shifted_wordstream):
    cues = parse_srt_text(shifted_srt_text)
    words = [Word.model_validate(item) for item in shifted_wordstream]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=0.5)

    rebuilt, _ = rebuild_cues(cues, words, alignment, profile)

    assert rebuilt[0].speaker_id == "A"
    assert rebuilt[1].speaker_id == "A"


def test_recue_default_keep_flagged_preserves_unmatched_cue():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nmatched line\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nabsent phrase\n\n"
    )
    words = [Word(text="matched", start=0.0, end=0.2), Word(text="line", start=0.25, end=0.5)]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, drop_policy="keep_flagged")

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    assert [cue.index for cue in rebuilt] == [1, 2]
    assert any(flag.kind == "unmatched_cue" and flag.cue_ids == [2] for flag in flags)
    assert any(flag.kind == "interpolated_timing" and flag.cue_ids == [2] for flag in flags)
    assert rebuilt[1].start_ms != 1000
    assert rebuilt[1].end_ms != 2000


def test_recue_drop_policy_remove_drops_unmatched_cue_with_qc_flag():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nmatched line\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nabsent phrase\n\n"
    )
    words = [Word(text="matched", start=0.0, end=0.2), Word(text="line", start=0.25, end=0.5)]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, drop_policy="remove")

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    assert [cue.index for cue in rebuilt] == [1]
    assert flags[0].kind == "dropped_unmatched_cue"
    assert flags[0].cue_ids == [2]
    assert flags[0].old_text == "absent phrase"
    assert flags[0].start == 1.0
    assert flags[0].end == 2.0


def test_recue_preserves_different_speaker_overlap_with_stack_policy():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nalpha one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nbeta two\n\n"
    )
    words = [
        Word(text="alpha", start=1.00, end=1.20, speaker_id="A"),
        Word(text="one", start=1.22, end=1.70, speaker_id="A"),
        Word(text="beta", start=1.30, end=1.50, speaker_id="B"),
        Word(text="two", start=1.52, end=1.90, speaker_id="B"),
    ]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, overlap_policy="stack")

    rebuilt, _ = rebuild_cues(cues, words, alignment, profile)

    assert rebuilt[0].speaker_id == "A"
    assert rebuilt[1].speaker_id == "B"
    assert rebuilt[1].start_ms < rebuilt[0].end_ms


def test_style_lint_allows_stacked_overlap_for_different_known_speakers():
    cues = [
        parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nalpha one\n\n")[0].model_copy(update={"speaker_id": "A"}),
        parse_srt_text("2\n00:00:01,500 --> 00:00:02,500\nbeta two\n\n")[0].model_copy(update={"speaker_id": "B"}),
    ]

    issues = lint_cues(cues, StyleProfile(fps=30.0, overlap_policy="stack"))

    assert not any(issue.kind == "overlap" for issue in issues)


def test_recue_chains_same_speaker_overlap():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nalpha one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nbeta two\n\n"
    )
    words = [
        Word(text="alpha", start=1.00, end=1.20, speaker_id="A"),
        Word(text="one", start=1.22, end=1.70, speaker_id="A"),
        Word(text="beta", start=1.30, end=1.50, speaker_id="A"),
        Word(text="two", start=1.52, end=1.90, speaker_id="A"),
    ]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, overlap_policy="stack")

    rebuilt, _ = rebuild_cues(cues, words, alignment, profile)

    assert rebuilt[1].start_ms == rebuilt[0].end_ms
