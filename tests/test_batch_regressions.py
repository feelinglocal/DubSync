from __future__ import annotations

from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SpeechRegion, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.changes import apply_adjudication_decisions
from dubsync.pipeline import _adlib_cue_ids_by_case, _alignment_with_decision_words, _without_stale_verify_flags
from dubsync.recue import rebuild_cues
from dubsync.source_order import sort_cues_chronologically
from dubsync.style_profile import StyleProfile
from dubsync.text_metrics import display_width
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity


def test_keep_srt_divergence_still_extends_to_spoken_span_words():
    alignment = AlignmentResult(cue_word_indices={18: [58, 59, 60, 61, 62]})
    span = DivergenceSpan(
        case_id="case-4",
        cue_ids=[18],
        srt_text="15",
        asr_text="funfzehn.",
        asr_word_indices=[63],
    )
    decision = AdjudicationDecision(
        case_id="case-4",
        verdict="keep_srt",
        final_text="15",
        confidence=1.0,
        reason="source spelling is preferred",
    )

    updated = _alignment_with_decision_words(alignment, [decision], [span])

    assert updated.cue_word_indices[18] == [58, 59, 60, 61, 62, 63]


def test_recue_ceil_snaps_end_so_last_syllable_is_not_cut():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Das Ding ist mindestens Level 15."])]
    words = [Word(text="funfzehn.", start=0.25, end=0.501, confidence=0.95)]
    alignment = AlignmentResult(cue_word_indices={1: [0]})
    profile = StyleProfile(fps=30.0, min_cue_dur=0.1, tail_ms=0)

    rebuilt, flags = rebuild_cues(cues, words, alignment, profile)

    assert flags == []
    assert rebuilt[0].end_ms == 533


def test_recue_trims_impossible_word_cluster_before_timing():
    cues = [Cue(index=6, start_ms=32000, end_ms=45000, lines=["Mein Drache!"])]
    words = [
        Word(text="Mein", start=32.96, end=43.76, confidence=1.0),
        Word(text="Drache!", start=43.82, end=44.98, confidence=1.0),
    ]
    alignment = AlignmentResult(cue_word_indices={6: [0, 1]})
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5)

    rebuilt, flags = rebuild_cues(
        cues,
        words,
        alignment,
        profile,
        max_word_duration=2.0,
        max_intra_cue_gap=1.5,
    )

    assert rebuilt[0].start_ms == 43800
    assert rebuilt[0].end_ms == 45033
    assert flags[0].kind == "timing_outlier_trimmed"


def test_unmatched_kept_cue_is_interpolated_instead_of_source_timed():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["before"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["missing"]),
        Cue(index=3, start_ms=2000, end_ms=3000, lines=["after"]),
    ]
    words = [
        Word(text="before", start=10.0, end=10.5, confidence=0.9),
        Word(text="after", start=20.0, end=20.5, confidence=0.9),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0], 3: [1]}, unmatched_cue_ids=[2])

    rebuilt, flags = rebuild_cues(cues, words, alignment, StyleProfile(fps=30.0, min_cue_dur=0.5))

    interpolated = next(cue for cue in rebuilt if cue.index == 2)
    assert interpolated.start_ms != 1000
    assert interpolated.end_ms != 2000
    assert any(flag.kind == "interpolated_timing" and flag.cue_ids == [2] for flag in flags)


def test_adlib_reconciliation_reuses_unmatched_source_cue():
    cues = [Cue(index=42, start_ms=72166, end_ms=73500, lines=["seine Gefuhle", "beeinflussen,"])]
    span = DivergenceSpan(
        case_id="case-6",
        cue_ids=[],
        srt_text="",
        asr_text="seine Gefuhle beeinflussen,",
        start=72.199,
        end=73.519,
        asr_word_indices=[157, 158, 159],
    )
    decision = AdjudicationDecision(
        case_id="case-6",
        verdict="use_audio",
        final_text="seine Gefuhle beeinflussen,",
        confidence=1.0,
        reason="spoken insertion matched source cue",
    )

    cue_ids, flags = _adlib_cue_ids_by_case(cues, [span], [decision], [42])
    changed, change_flags = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(max_chars_per_line=26),
        adlib_cue_ids_by_case=cue_ids,
    )

    assert cue_ids == {"case-6": 42}
    assert flags[0].kind == "adlib_reconciled"
    assert [cue.index for cue in changed] == [42]
    assert changed[0].plain_text == "seine Gefuhle beeinflussen,"
    assert change_flags == []


def test_adlib_reconciliation_does_not_reuse_unrelated_nearby_unmatched_cue():
    cues = [Cue(index=42, start_ms=1000, end_ms=2000, lines=["open the door"])]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[],
        srt_text="",
        asr_text="surprise attack",
        start=1.2,
        end=1.8,
        asr_word_indices=[0, 1],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="surprise attack",
        confidence=1.0,
        reason="spoken ad-lib",
    )

    cue_ids, flags = _adlib_cue_ids_by_case(cues, [span], [decision], [42])

    assert cue_ids["case-1"] != 42
    assert not flags


def test_word_anchored_vad_refinement_ignores_later_regions_from_bad_word_end():
    cue = Cue(index=16, start_ms=48266, end_ms=67400, lines=["Los geht's."])
    words = [
        Word(text="Los", start=48.299, end=48.5, confidence=0.96),
        Word(text="geht's.", start=48.599, end=67.379, confidence=0.91),
    ]
    alignment = AlignmentResult(cue_word_indices={16: [0, 1]})
    regions = [
        SpeechRegion(start=48.2, end=48.9),
        SpeechRegion(start=67.0, end=69.0),
    ]

    refined, flags = refine_cues_to_speech_activity(
        [cue],
        regions,
        StyleProfile(fps=30.0, min_cue_dur=0.5),
        BoundaryRefinementConfig(start_pad_ms=40, end_pad_ms=40),
        words=words,
        alignment=alignment,
    )

    assert refined[0].end_ms <= 49000
    assert refined[0].duration_ms < 1000
    assert flags[0].kind == "timing_refined"


def test_source_cues_are_sorted_chronologically_and_flag_moved_cues():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["first"]),
        Cue(index=2, start_ms=3000, end_ms=3500, lines=["third"]),
        Cue(index=42, start_ms=1500, end_ms=2500, lines=["second"]),
    ]

    sorted_cues, flags = sort_cues_chronologically(cues)

    assert [cue.index for cue in sorted_cues] == [1, 42, 2]
    assert flags[0].kind == "source_out_of_order"
    assert flags[0].cue_ids == [42, 2]


def test_final_output_guard_sorts_and_merges_duplicate_overlapping_cues():
    cues = [
        Cue(index=37, start_ms=72166, end_ms=73600, lines=["seine Gefuhle beeinflussen,"]),
        Cue(index=36, start_ms=72166, end_ms=72666, lines=["seine Gefuhle beeinflussen,"]),
        Cue(index=38, start_ms=73600, end_ms=75100, lines=["ja sogar seinen Willen steuern."]),
    ]

    finalized, flags = finalize_cues_for_output(cues, StyleProfile(fps=30.0), no_overlaps=True)

    assert [cue.plain_text for cue in finalized] == [
        "seine Gefuhle beeinflussen,",
        "ja sogar seinen Willen steuern.",
    ]
    assert finalized[0].start_ms == 72166
    assert finalized[0].end_ms == 73600
    assert flags[0].kind == "duplicate_cue_merged"


def test_final_output_guard_extends_fast_cue_without_overlap():
    cues = [
        Cue(index=14, start_ms=36433, end_ms=37233, lines=["Dann schaffen wir es an die"]),
        Cue(index=15, start_ms=38000, end_ms=39100, lines=["Diamant-Akademie!"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0, min_cue_dur=0.5),
        no_overlaps=True,
        max_cps=30,
    )

    assert finalized[0].end_ms > 37233
    assert finalized[0].end_ms <= finalized[1].start_ms
    assert not any(left.end_ms > right.start_ms for left, right in zip(finalized, finalized[1:]))
    assert flags[0].kind == "cps_duration_extended"


def test_final_output_guard_resolves_different_speaker_overlap_when_no_overlaps_enabled():
    cues = [
        Cue(index=74, start_ms=108266, end_ms=108766, lines=["Moi..."], speaker_id="speaker_2"),
        Cue(index=75, start_ms=108533, end_ms=109666, lines=["Mme la Gouverneure,"], speaker_id="speaker_4"),
    ]

    finalized, flags = finalize_cues_for_output(cues, StyleProfile(fps=30.0), no_overlaps=True)

    assert finalized[0].end_ms < finalized[1].start_ms
    assert finalized[1].plain_text == "Mme la Gouverneure,"
    assert flags[0].kind == "output_overlap_resolved"


def test_final_output_guard_extends_one_more_frame_when_cps_snap_is_still_fast():
    cues = [
        Cue(index=24, start_ms=31933, end_ms=32000, lines=["x" * 40]),
        Cue(index=25, start_ms=34000, end_ms=35000, lines=["next"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0),
        no_overlaps=True,
        max_cps=30,
    )

    cps = display_width(finalized[0].plain_text) / (finalized[0].duration_ms / 1000.0)
    assert cps <= 30
    assert finalized[0].end_ms > 33266
    assert flags[0].kind == "cps_duration_extended"


def test_final_output_guard_leaves_fast_cue_when_no_gap_exists_without_shifting_following():
    cues = [
        Cue(index=24, start_ms=31933, end_ms=33266, lines=["x" * 40]),
        Cue(index=25, start_ms=33266, end_ms=34466, lines=["y" * 35]),
        Cue(index=26, start_ms=35000, end_ms=36000, lines=["next"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0),
        no_overlaps=True,
        max_cps=30,
    )

    assert finalized[0].end_ms == finalized[1].start_ms
    assert [cue.start_ms for cue in finalized] == [31933, 33266, 35000]
    assert not any(flag.kind == "cps_duration_extended" for flag in flags)


def test_final_output_guard_does_not_delay_following_cue_for_cps_extension():
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["x" * 40]),
        Cue(index=2, start_ms=500, end_ms=3000, lines=["following speech"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=0.5),
        no_overlaps=True,
        max_cps=30,
    )

    assert [cue.start_ms for cue in finalized] == [0, 500]
    assert [cue.plain_text for cue in finalized] == ["x" * 40, "following speech"]
    assert not any(flag.kind == "cps_duration_extended" for flag in flags)


def test_resume_verify_drops_stale_recomputed_flags_only():
    flags = [
        QCFlag(kind="text_changed", cue_ids=[1], message="Earlier adjudication evidence."),
        QCFlag(kind="interpolated_timing", cue_ids=[2], message="Earlier rebuild evidence."),
        QCFlag(kind="impossible_cps_fast", cue_ids=[3], message="Stale verify flag."),
        QCFlag(kind="output_overlap_resolved", cue_ids=[4, 5], message="Stale output flag."),
        QCFlag(kind="speaker_transition_gap_inserted", cue_ids=[6, 7], message="Stale speaker gap flag."),
    ]

    filtered = _without_stale_verify_flags(flags)

    assert [flag.kind for flag in filtered] == ["text_changed", "interpolated_timing"]


def test_final_output_guard_does_not_pull_later_cue_earlier_when_gap_exists():
    cues = [
        Cue(index=46, start_ms=87066, end_ms=87766, lines=["trainieren hier nicht."]),
        Cue(index=47, start_ms=101633, end_ms=102733, lines=["Das Drachenei ist", "tatsachlich"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0, min_cue_dur=0.5),
        no_overlaps=True,
        max_cps=30,
    )

    assert finalized[0].end_ms > 87766
    assert finalized[1].start_ms == 101633
    assert flags[0].kind == "cps_duration_extended"


def test_final_output_guard_merges_short_adjacent_phrase_when_extension_cannot_fit():
    cues = [
        Cue(index=35, start_ms=73300, end_ms=73800, lines=["Ihr wisst schon,"]),
        Cue(index=29, start_ms=73800, end_ms=74300, lines=["Wahrnehmung!"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=0.5),
        no_overlaps=True,
        max_cps=30,
    )

    assert len(finalized) == 1
    assert finalized[0].lines == ["Ihr wisst schon,", "Wahrnehmung!"]
    assert flags[0].kind == "cps_cue_merged"


def test_final_output_guard_does_not_merge_fast_cues_from_different_speakers():
    cues = [
        Cue(index=74, start_ms=108266, end_ms=108766, lines=["xxxxxxxxxxxxxxxxxxxx"], speaker_id="speaker_2"),
        Cue(index=75, start_ms=108766, end_ms=109266, lines=["next"], speaker_id="speaker_4"),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0, max_chars_per_line=26, min_cue_dur=0.5),
        no_overlaps=True,
        max_cps=30,
    )

    assert [cue.plain_text for cue in finalized] == ["xxxxxxxxxxxxxxxxxxxx", "next"]
    assert not any(flag.kind == "cps_cue_merged" for flag in flags)
