from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.pipeline import _adlib_cue_ids_by_case, _alignment_with_decision_words
from dubsync.recue import rebuild_cues
from dubsync.source_order import sort_cues_chronologically
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity
from dubsync.verify import cps_sanity_flags

pytestmark = pytest.mark.regression


def _fixture() -> dict[str, object]:
    path = Path(__file__).parents[1] / "fixtures" / "regression" / "batch_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ep7_out_of_order_source_reconciles_adlib_without_duplicate_overlap():
    data = _fixture()["ep7"]
    cues = [Cue.model_validate(row) for row in data["source_cues"]]
    span = DivergenceSpan.model_validate(data["adlib_span"])
    decision = AdjudicationDecision.model_validate(data["adlib_decision"])

    sorted_cues, source_flags = sort_cues_chronologically(cues)
    cue_ids, reconcile_flags = _adlib_cue_ids_by_case(sorted_cues, [span], [decision], [42])
    changed, change_flags = apply_adjudication_decisions(
        sorted_cues,
        [span],
        [decision],
        StyleProfile(max_chars_per_line=26),
        adlib_cue_ids_by_case=cue_ids,
    )
    alignment = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={35: [0], 36: [3]}, unmatched_cue_ids=[42]),
        [decision],
        [span],
        cue_ids,
    )
    words = [
        Word(text="seine", start=72.2, end=72.45),
        Word(text="Gefuhle", start=72.5, end=72.95),
        Word(text="beeinflussen", start=73.0, end=73.52),
        Word(text="steuern", start=73.6, end=74.1),
    ]
    rebuilt, _ = rebuild_cues(changed, words, alignment, StyleProfile(fps=30.0, min_cue_dur=0.5))
    finalized, output_flags = finalize_cues_for_output(rebuilt, StyleProfile(fps=30.0), no_overlaps=True)

    assert source_flags[0].kind == "source_out_of_order"
    assert reconcile_flags[0].kind == "adlib_reconciled"
    assert cue_ids == {"case-6": 42}
    assert change_flags == []
    assert [cue.plain_text for cue in finalized].count("seine Gefuhle beeinflussen,") == 1
    assert not any(flag.kind == "duplicate_cue_merged" for flag in output_flags)


def test_ep9_keep_srt_number_spans_cover_spoken_word_ends():
    data = _fixture()["ep9"]
    cues = [Cue.model_validate(row) for row in data["cues"]]
    words = [Word.model_validate(row) for row in data["words"]]
    spans = [DivergenceSpan.model_validate(row) for row in data["spans"]]
    decisions = [AdjudicationDecision.model_validate(row) for row in data["decisions"]]
    alignment = AlignmentResult(cue_word_indices={18: [0, 1, 2, 3, 4], 20: [6, 7, 8]})

    updated = _alignment_with_decision_words(alignment, decisions, spans)
    rebuilt, flags = rebuild_cues(cues, words, updated, StyleProfile(fps=30.0, min_cue_dur=0.5))

    by_id = {cue.index: cue for cue in rebuilt}
    assert flags == []
    assert by_id[18].end_ms >= 53859
    assert by_id[20].end_ms >= 59619
    assert cps_sanity_flags(rebuilt, max_cps=30, min_cps=2) == []


def test_ep10_bad_asr_word_end_trims_to_own_vad_region_and_passes_cps_net():
    data = _fixture()["ep10"]
    cue = Cue.model_validate(data["cue"])
    words = [Word.model_validate(row) for row in data["words"]]
    regions = [SpeechRegion.model_validate(row) for row in data["vad"]]
    alignment = AlignmentResult(cue_word_indices={16: [0, 1]})

    refined, flags = refine_cues_to_speech_activity(
        [cue],
        regions,
        StyleProfile(fps=30.0, min_cue_dur=0.5),
        BoundaryRefinementConfig(start_pad_ms=40, end_pad_ms=40),
        words=words,
        alignment=alignment,
    )

    assert refined[0].end_ms <= 49000
    assert all(left.start_ms <= right.start_ms for left, right in zip(refined, refined[1:]))
    assert flags[0].kind == "timing_refined"
    assert cps_sanity_flags(refined, max_cps=30, min_cps=2) == []


def test_ep10_prefix_giant_cue_trips_impossible_cps_net():
    data = _fixture()["ep10"]
    cue = Cue.model_validate(data["cue"])

    flags = cps_sanity_flags([cue], max_cps=30, min_cps=2)

    assert [flag.kind for flag in flags] == ["impossible_cps_slow"]
