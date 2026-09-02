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


def test_alignment_ignores_bracket_only_visual_text_for_spoken_timing():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[Station of Beijing]\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nhello there\n\n"
    )
    words = [
        Word(text="hello", start=10.00, end=10.20),
        Word(text="there", start=10.25, end=10.50),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.unmatched_cue_ids == []
    assert result.cue_word_indices == {2: [0, 1]}


def test_alignment_uses_only_spoken_text_from_mixed_bracket_cue():
    cues = parse_srt_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "[This company is not for you]\n"
        "Lime is not for you.\n"
        "\n"
    )
    words = [
        Word(text="Lime", start=3.00, end=3.20),
        Word(text="is", start=3.25, end=3.32),
        Word(text="not", start=3.35, end=3.45),
        Word(text="for", start=3.50, end=3.62),
        Word(text="you", start=3.70, end=3.90),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1, 2, 3, 4]}


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


def test_alignment_ignores_bracketed_screen_text_lines_without_losing_spoken_lines():
    cues = parse_srt_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "[Document: Final admission notice]\n"
        "I accept.\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "[Episode 1]\n"
        "\n"
        "3\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "We go now.\n"
        "\n"
    )
    words = [
        Word(text="I", start=10.00, end=10.10, confidence=0.98),
        Word(text="accept", start=10.12, end=10.40, confidence=0.98),
        Word(text="We", start=20.00, end=20.10, confidence=0.98),
        Word(text="go", start=20.12, end=20.22, confidence=0.98),
        Word(text="now", start=20.24, end=20.50, confidence=0.98),
    ]

    alignment = align_cues_to_words(cues, words)
    rebuilt, flags = rebuild_cues(cues, words, alignment, StyleProfile(fps=30.0, min_cue_dur=0.5))

    assert alignment.anchor_coverage == 1.0
    assert alignment.divergence_spans == []
    assert alignment.cue_word_indices == {1: [0, 1], 3: [2, 3, 4]}
    assert alignment.unmatched_cue_ids == []
    assert rebuilt[0].text == "[Document: Final admission notice]\nI accept."
    assert rebuilt[0].start_ms == 10000
    assert rebuilt[0].end_ms == 10500
    assert rebuilt[1].plain_text == "[Episode 1]"
    assert rebuilt[1].start_ms == 1000
    assert rebuilt[1].end_ms == 2000
    assert not any(flag.kind == "interpolated_timing" and flag.cue_ids == [2] for flag in flags)


def test_fuzzy_name_variant_stays_visible_for_audio_adjudication():
    cues = parse_srt_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "Komm jetzt, Tristen, bitte.\n"
        "\n"
    )
    words = [
        Word(text=text, start=index * 0.2, end=index * 0.2 + 0.15)
        for index, text in enumerate(["Komm", "jetzt", "Tristan", "bitte"])
    ]

    result = align_cues_to_words(cues, words)

    assert len(result.divergence_spans) == 1
    assert result.divergence_spans[0].srt_text == "Tristen"
    assert result.divergence_spans[0].asr_text == "Tristan"


def test_divergence_with_unknown_asr_confidence_remains_unscored():
    cues = parse_srt_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Tristen\n"
        "\n"
    )
    words = [Word(text="Tristan", start=0.1, end=0.6, confidence=None)]

    result = align_cues_to_words(cues, words)

    assert len(result.divergence_spans) == 1
    assert result.divergence_spans[0].confidence == 0.0


def test_alignment_rejects_an_implausibly_overlong_single_word_match():
    cues = [Cue(index=771, start_ms=2_087_800, end_ms=2_088_670, lines=["Flora,"])]
    words = [
        Word(
            text="Flora",
            start=2_083.496,
            end=2_088.276,
            confidence=0.99,
            speaker_id="A",
        )
    ]

    result = align_cues_to_words(cues, words)

    assert result.cue_word_indices == {}
    assert result.unmatched_cue_ids == [771]
    assert result.diagnostics.missing_audio_cue_ids == [771]
    assert result.diagnostics.missing_audio_guard_version == 3
    duration_flag = next(
        flag for flag in result.flags if flag.kind == "implausible_matched_word_duration"
    )
    assert duration_flag.cue_ids == [771]
    assert duration_flag.severity == "error"
    assert any(
        flag.kind == "missing_audio_timing_held" and flag.cue_ids == [771]
        for flag in result.flags
    )


def test_alignment_accepts_a_long_word_when_the_source_cue_is_also_long():
    cues = [Cue(index=1, start_ms=0, end_ms=5_000, lines=["Flora,"])]
    words = [Word(text="Flora", start=0.1, end=4.88, confidence=0.99, speaker_id="A")]

    result = align_cues_to_words(cues, words)

    assert result.cue_word_indices == {1: [0]}
    assert result.unmatched_cue_ids == []
    assert result.diagnostics.missing_audio_cue_ids == []
    assert not any(flag.kind == "implausible_matched_word_duration" for flag in result.flags)


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


def test_alignment_budget_exhaustion_rejects_a_bounded_run_that_misses_a_unique_exact_pair(monkeypatch):
    monkeypatch.setattr(aligner, "ALIGNMENT_CELL_BUDGET", 50_000)
    token_count = 300
    cues = [
        Cue(index=index + 1, start_ms=index * 200, end_ms=index * 200 + 120, lines=["filler"])
        for index in range(token_count)
    ]
    words = [
        Word(text="filler", start=index * 0.2, end=index * 0.2 + 0.1, confidence=0.99)
        for index in range(token_count)
    ]
    cues[20] = cues[20].with_lines(["needle"])
    words[280] = words[280].model_copy(update={"text": "needle"})

    result = align_cues_to_words(cues, words)

    assert result.token_matches == []
    assert len(result.divergence_spans) == 1
    assert result.divergence_spans[0].srt_text.startswith("filler")
    assert result.divergence_spans[0].asr_text.startswith("filler")
    assert result.diagnostics.unbanded_fallback is False
    assert result.diagnostics.band_limited is True
    assert result.diagnostics.unresolved is True
    assert any(flag.kind == "alignment_band_limited" for flag in result.flags)
    assert any(flag.kind == "alignment_unresolved" and flag.severity == "error" for flag in result.flags)


def test_alignment_budget_accepts_a_unique_pair_explained_by_an_adjacent_transposition(monkeypatch):
    monkeypatch.setattr(aligner, "ALIGNMENT_CELL_BUDGET", 50_000)
    cues = [
        Cue(index=index + 1, start_ms=index * 200, end_ms=index * 200 + 120, lines=["filler"])
        for index in range(300)
    ]
    cues[20] = cues[20].with_lines(["um novo inquilino"])
    cues[100] = cues[100].with_lines(["novo"])
    tokens = aligner.tokenize_cues(cues)
    words_norm = [token.normalized for token in tokens]
    words_norm[21], words_norm[22] = words_norm[22], words_norm[21]

    run = aligner._align_tokens_detailed(tokens, words_norm)

    assert run.unresolved is False
    assert run.band_limited is False
    assert sum(op.kind == "match" for op in run.ops) == len(tokens) - 1
    assert any(op.kind == "match" and op.srt_index == 21 and op.asr_index == 22 for op in run.ops)


def test_local_transposition_check_uses_constant_radius_membership():
    class MembershipOnlyPairs:
        def __init__(self, pairs):
            self.pairs = set(pairs)

        def __contains__(self, pair):
            return pair in self.pairs

        def __iter__(self):
            raise AssertionError("transposition proof must not scan every matched pair")

    matched_pairs = MembershipOnlyPairs({(9, 9), (11, 10), (12, 12)})

    assert aligner._is_locally_explained_transposition((10, 11), matched_pairs)


def test_successful_timing_prior_replaces_failed_preliminary_unresolved_status(monkeypatch):
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\necho one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\necho two\n\n"
    )
    words = [
        Word(text="echo", start=0.1, end=0.2),
        Word(text="one", start=0.3, end=0.4),
        Word(text="echo", start=1.1, end=1.2),
        Word(text="two", start=1.3, end=1.4),
    ]
    successful_ops = [aligner._Op("match", index, index, 1.0) for index in range(4)]
    runs = iter(
        [
            aligner._AlignmentRun(
                ops=aligner._fully_divergent_ops(4, 4),
                band_limited=True,
                unresolved=True,
            ),
            aligner._AlignmentRun(ops=successful_ops),
        ]
    )
    monkeypatch.setattr(aligner, "_align_tokens_detailed", lambda *args, **kwargs: next(runs))

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.diagnostics.prior_used is True
    assert result.diagnostics.band_limited is True
    assert result.diagnostics.unresolved is False
    assert any(flag.kind == "alignment_band_limited" for flag in result.flags)
    assert not any(flag.kind == "alignment_unresolved" for flag in result.flags)


def test_alignment_retry_margins_progress_without_automatic_full_width():
    assert aligner._retry_margins(64, 5_000) == [64, 256, 1_024]
    assert aligner._retry_margins(64, 500) == [64, 256, 500]


def test_small_initial_alignment_is_not_reported_as_an_unbanded_fallback():
    cues = [Cue(index=1, start_ms=0, end_ms=1_000, lines=["alpha"])]
    words = [Word(text="alpha", start=0.1, end=0.3)]

    result = align_cues_to_words(cues, words)

    assert result.diagnostics.unbanded_fallback is False


def test_alignment_budget_exhaustion_preserves_a_reviewable_whole_span(monkeypatch):
    monkeypatch.setattr(aligner, "ALIGNMENT_CELL_BUDGET", 0)
    cues = [Cue(index=1, start_ms=0, end_ms=1_000, lines=["alpha beta"])]
    words = [
        Word(text="alpha", start=0.1, end=0.3),
        Word(text="beta", start=0.4, end=0.6),
    ]

    result = align_cues_to_words(cues, words)

    assert result.token_matches == []
    assert len(result.divergence_spans) == 1
    assert result.divergence_spans[0].srt_text == "alpha beta"
    assert result.divergence_spans[0].asr_text == "alpha beta"
    assert result.diagnostics.band_limited is True
    assert result.diagnostics.unresolved is True
    assert any(flag.kind == "alignment_band_limited" for flag in result.flags)
    unresolved = [flag for flag in result.flags if flag.kind == "alignment_unresolved"]
    assert len(unresolved) == 1
    assert unresolved[0].severity == "error"


def test_similarity_uses_score_cutoff_without_rejecting_near_length_german_pairs(monkeypatch):
    calls: list[tuple[str, str, float | None]] = []

    def fake_ratio(left: str, right: str, score_cutoff: float | None = None) -> float:
        calls.append((left, right, score_cutoff))
        return 88.9

    monkeypatch.setattr(aligner.fuzz, "ratio", fake_ratio)

    assert aligner._similarity("gehen", "gehe") == 0.889
    assert aligner._similarity("Haus", "Hause") == 0.889
    assert calls == [
        ("gehen", "gehe", 85.0),
        ("Haus", "Hause", 85.0),
    ]


def test_band_windows_keep_distant_priors_disjoint_and_bounded():
    windows = aligner._band_windows(
        row=50,
        token_count=100,
        word_count=1_000,
        margin=5,
        prior_centers=(100, 900),
    )

    assert len(windows) == 3
    assert windows == [(95, 105), (495, 505), (895, 905)]


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
    assert not any(flag.kind == "interpolated_timing" and flag.cue_ids == [2] for flag in flags)
    assert rebuilt[1].start_ms == 1000
    assert rebuilt[1].end_ms == 2000


def test_recue_preserves_missing_middle_source_timing_between_matched_neighbors():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nalpha one\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nmissing middle\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nomega three\n\n"
    )
    words = [
        Word(text="alpha", start=10.00, end=10.20),
        Word(text="one", start=10.25, end=10.50),
        Word(text="omega", start=20.00, end=20.20),
        Word(text="three", start=20.25, end=20.50),
    ]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, drop_policy="keep_flagged")

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    missing = next(cue for cue in rebuilt if cue.index == 2)
    assert alignment.unmatched_cue_ids == [2]
    assert alignment.diagnostics.missing_audio_cue_ids == [2]
    assert missing.start_ms == 2000
    assert missing.end_ms == 3000
    assert not any(flag.kind == "interpolated_timing" and flag.cue_ids == [2] for flag in flags)


def test_alignment_does_not_assign_partial_repeated_missing_cue_to_later_sentence():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\ntake the card\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\ntake the ring\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\ntake the sword\n\n"
    )
    words = [
        Word(text="take", start=10.00, end=10.10),
        Word(text="the", start=10.15, end=10.20),
        Word(text="card", start=10.25, end=10.50),
        Word(text="take", start=20.00, end=20.10),
        Word(text="the", start=20.15, end=20.20),
        Word(text="sword", start=20.25, end=20.50),
    ]

    alignment = align_cues_to_words(cues, words)

    assert alignment.cue_word_indices == {1: [0, 1, 2], 3: [3, 4, 5]}
    assert alignment.unmatched_cue_ids == [2]


def test_recue_keeps_bracket_only_visual_text_at_source_timing_without_interpolation():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\nmatched line\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n[Station of Beijing]\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nnext line\n\n"
    )
    words = [
        Word(text="matched", start=10.0, end=10.2),
        Word(text="line", start=10.25, end=10.5),
        Word(text="next", start=20.0, end=20.2),
        Word(text="line", start=20.25, end=20.5),
    ]
    alignment = align_cues_to_words(cues, words)
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5, drop_policy="remove")

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    visual = next(cue for cue in rebuilt if cue.index == 2)
    assert visual.start_ms == 2000
    assert visual.end_ms == 4000
    assert visual.text == "[Station of Beijing]"
    assert not any(flag.cue_ids == [2] for flag in flags)


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
