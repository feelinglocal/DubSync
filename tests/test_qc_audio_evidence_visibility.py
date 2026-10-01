"""Exercise model-evidence holds through the actual customer QC classifier."""
from __future__ import annotations

import pytest

from dubsync.adjudication import confidence_gated_decision
from dubsync.adjudication_policy import DeterministicAdjudicationPolicy
from dubsync.asr_crosscheck import classify_spans, compare_word_streams
from dubsync.asr_crosscheck_runtime import cross_check_decision_gate
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag, Word
from dubsync.qc_review import build_review


def _cue(index=15):
    return Cue(index=index, start_ms=36400, end_ms=37000, lines=["Damien"])


def _evidence_flag(cue, evidence, heard, verdict):
    span = DivergenceSpan(case_id=f"case-{cue.index}", cue_ids=[cue.index], srt_text=cue.plain_text,
                          asr_text="Damian", start=cue.start_ms / 1000, end=cue.end_ms / 1000)
    decision = AdjudicationDecision(case_id=span.case_id, verdict=verdict,
                                    final_text=cue.plain_text if verdict == "keep_srt" else "Damian",
                                    confidence=1, reason="Audio was reviewed.", evidence=evidence, heard_text=heard)
    held, flag = confidence_gated_decision(span, decision, 0, policy=DeterministicAdjudicationPolicy(language="de"))
    assert held.verdict == "keep_srt" and held.final_text == cue.plain_text
    assert flag is not None and flag.confidence == 0
    return flag


@pytest.mark.parametrize("evidence,heard,verdict", [
    ("heard_unclear", "Dam...", "use_audio"),
    ("heard_unclear", "Dam...", "keep_srt"),
    ("not_audible", "", "keep_srt"),
    ("heard_clearly", "Damian", "keep_srt"),
])
def test_zero_confidence_model_evidence_is_customer_review_not_clean(evidence, heard, verdict):
    cue = _cue()
    flag = _evidence_flag(cue, evidence, heard, verdict)
    before = flag.model_dump()
    review = build_review([flag], [], [cue], source_cues=[cue])
    assert review.verdict == "check"
    assert review.counts["review_item_count"] == 1
    assert review.counts["review_cue_count"] == 1
    item, = review.review
    assert item.kind == "low_confidence_adjudication" and item.severity == "warning"
    assert item.srt_numbers == [1] and item.cue_ids == [15] and item.raw_flags == [0]
    assert item.action and "Listen" in item.action
    assert review.diagnostics == []
    assert flag.model_dump() == before


def test_duplicate_native_uncertainty_is_grouped_without_disappearing():
    cue = _cue()
    flags = [_evidence_flag(cue, "heard_unclear", "Dam...", "keep_srt"),
             _evidence_flag(cue, "heard_clearly", "Damian", "keep_srt")]
    review = build_review(flags, [], [cue], source_cues=[cue])
    assert review.verdict == "check" and review.counts["review_item_count"] == 1
    assert review.review[0].raw_flags == [0, 1]
    assert review.review[0].srt_numbers == [1]
    assert not review.diagnostics


def test_four_actual_german_name_mismatches_do_not_report_clean():
    name_cue_ids = {15, 19, 28, 35}
    cues = [_cue(index).model_copy(update={"start_ms": index*2000, "end_ms": index*2000+800})
            for index in range(1, 36)]
    flags = [_evidence_flag(cue, "heard_clearly", "Damian", "keep_srt")
             for cue in cues if cue.index in name_cue_ids]
    review = build_review(flags, [], cues, source_cues=cues)
    assert review.verdict == "check"
    assert review.counts["review_item_count"] == 4 and review.counts["review_cue_count"] == 4
    assert [item.srt_numbers for item in review.review] == [[15], [19], [28], [35]]
    assert not review.diagnostics


@pytest.mark.parametrize("reason", [
    "The source cue had no trustworthy local speech evidence.",
    "Punctuation/casing-only difference; preserve the source.",
    "No model proposal was made.",
])
def test_synthetic_zero_confidence_pipeline_keeps_remain_diagnostics(reason):
    cue = _cue()
    span = DivergenceSpan(case_id="synthetic", cue_ids=[cue.index], srt_text=cue.plain_text, asr_text="Damian")
    decision = AdjudicationDecision(case_id=span.case_id, verdict="keep_srt", final_text=cue.plain_text,
                                    confidence=0, reason=reason)
    _, flag = confidence_gated_decision(span, decision, .7)
    review = build_review([flag], [], [cue], source_cues=[cue])
    assert review.verdict == "clean" and review.review == []
    assert len(review.diagnostics) == 1 and review.diagnostics[0].raw_flags == [0]


def test_model_evidence_is_not_absorbed_into_an_absent_lyric_note():
    cue = _cue().model_copy(update={"lines": ["♪Damien♪"]})
    flag = _evidence_flag(cue, "heard_unclear", "Dam...", "keep_srt")
    missing = QCFlag(kind="missing_audio_timing_held", cue_ids=[cue.index], severity="error",
                     message="No trustworthy local speech evidence was available for this source cue.")
    review = build_review([missing, flag], [], [cue], source_cues=[cue])
    assert review.verdict == "check"
    assert review.review[0].kind == "low_confidence_adjudication"
    assert review.review[0].raw_flags == [1]
    assert review.notes[0].raw_flags == [0]


def test_real_dual_asr_conflict_hold_also_remains_actionable_at_zero_confidence():
    cue = _cue()
    span = DivergenceSpan(case_id="dual", cue_ids=[cue.index], srt_text="Damien", asr_text="Damian",
                          start=36.465, end=36.919, asr_word_indices=[0])
    primary = [Word(text="Damian", start=span.start, end=span.end)]
    secondary = [Word(text="Damien", start=span.start, end=span.end)]
    checks = classify_spans([span], compare_word_streams(primary, secondary))
    proposed = AdjudicationDecision(case_id=span.case_id, verdict="use_audio", final_text="Damian",
                                    confidence=.99, reason="Legacy numeric confidence only.")
    _, flags = cross_check_decision_gate([span], [proposed], checks, policy=DeterministicAdjudicationPolicy(language="de"))
    assert len(flags) == 1 and flags[0].confidence == 0
    review = build_review(flags, [], [cue], source_cues=[cue])
    assert review.verdict == "check" and review.counts["review_item_count"] == 1
    assert not review.diagnostics
