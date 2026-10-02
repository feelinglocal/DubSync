from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest

from dubsync.asr_crosscheck_config import cross_check_context
from dubsync.collapsed_singleton_timing import (
    build_collapsed_singleton_timing_questions,
    reconcile_collapsed_singleton_timing,
)
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SpeechRegion, TokenMatch, Word
from dubsync.llm_providers import _adjudication_prompt
from dubsync.style_profile import StyleProfile


def _case():
    """EP11-shaped evidence, shifted by 1640 s; hearing below is a test fixture."""
    sources = [Cue(index=600, start_ms=90000, end_ms=90500, lines=["né?"]),
               Cue(index=601, start_ms=100000, end_ms=100520, lines=["É."]),
               Cue(index=602, start_ms=110000, end_ms=110610, lines=["Tomam."])]
    current = [sources[0], Cue(index=975, start_ms=3034, end_ms=3567, lines=["Hum."]),
               Cue(index=991, start_ms=3900, end_ms=4400, lines=["Ó."]), sources[1],
               sources[2].with_lines(["Tomem."])]
    primary_rows = [("né?", 1.93, 2.105), ("Hum.", 3.055, 3.505), ("Ó.", 3.905, 4.165),
                    ("É.", 8.630, 8.631), ("Tomem.", 8.655, 9.045)]
    secondary_rows = [("né?", 1.92, 2.159), ("Hum.", 3.12, 3.499), ("Oh.", 3.96, 4.30),
                      ("É.", 7.04, 7.32), ("Tomem", 8.639, 9.079)]
    words = [Word(text=t, start=a, end=b, speaker_id=f"primary_{i == 4}")
             for i, (t, a, b) in enumerate(primary_rows)]
    secondary = [Word(text=t, start=a, end=b, confidence=None, speaker_id=f"secondary_{i == 4}")
                 for i, (t, a, b) in enumerate(secondary_rows)]
    regions = [SpeechRegion(start=a, end=b) for a, b in [
        (1.90, 2.105), (3.055, 3.505), (3.905, 4.165), (4.605, 5.305),
        (5.405, 5.755), (6.215, 6.315), (6.515, 6.585), (6.975, 7.245), (8.655, 9.045),
    ]]
    alignment = AlignmentResult(
        cue_word_indices={600: [0], 601: [3], 602: [4], 975: [1], 991: [2]},
        token_matches=[TokenMatch(cue_id=600, srt_token_index=0, asr_word_index=0, score=1),
                       TokenMatch(cue_id=601, srt_token_index=1, asr_word_index=3, score=1)],
        divergence_spans=[DivergenceSpan(case_id="ordinary-tomem", cue_ids=[602],
            srt_text="Tomam", srt_token_indices=[2], asr_text="Tomem.", asr_word_indices=[4],
            start=8.655, end=9.045)],
    )
    ordinary = [AdjudicationDecision(case_id="ordinary-tomem", verdict="use_audio", final_text="Tomem",
        evidence="heard_clearly", heard_text="Tomem", confidence=1, reason="Unit fixture of accepted anchor wording.")]
    return {"current_cues": current, "source_cues": sources, "alignment": alignment,
            "words": words, "regions": regions, "secondary_words": secondary, "decisions": ordinary,
            "audio_duration_seconds": 12.0, "uncertain_word_indices": set(),
            "protected_cue_ids": set(), "resolved_cue_ids": set()}


def _context(case):
    return cross_check_context(case["secondary_words"], {"asr": {
        "provider": "openrouter", "model": "microsoft/mai-transcribe-2", "language_code": "pt"}})


def _ask(case, context=None):
    return build_collapsed_singleton_timing_questions(**case, secondary_context=_context(case) if context is None else context)


def _hear(question, *, text="É.", evidence="heard_clearly", final=None):
    return AdjudicationDecision(case_id=question.span.case_id, verdict="keep_srt",
        final_text=question.span.srt_text if final is None else final, evidence=evidence,
        heard_text=text, confidence=1, reason="SYNTHETIC UNIT FIXTURE ONLY; no native audio call.")


def _resolve(case, questions, hearings=(), **overrides):
    arguments = {**case, "secondary_context": _context(case), "questions": questions,
                 "hearing_decisions": list(hearings), "profile": StyleProfile(fps=30, tail_ms=0), **overrides}
    return reconcile_collapsed_singleton_timing(**arguments)


def test_collapsed_complete_cue_uses_unique_raw_secondary_vad_after_fresh_hearing_only():
    case = _case()
    before = deepcopy(case)
    questions = _ask(case)
    assert len(questions) == 1
    question = questions[0]
    assert question.span.case_id.startswith("collapsed-singleton-timing-v1-")
    assert question.purpose == "collapsed_singleton_timing"
    assert question.span.cue_ids == [601] and question.span.srt_token_indices == [3]
    assert question.span.srt_text == "É." and question.target_word_indices == (3,)
    assert question.span.left_anchor_cue_id == 600 and question.span.right_anchor_cue_id == 602
    assert "collapsed_singleton_proof" in question.record()
    with pytest.raises(FrozenInstanceError):
        question.purpose = "missing_dialogue"

    pending = _resolve(case, questions)
    assert pending.resolved_cue_ids == set() and pending.cues == case["current_cues"]
    assert pending.outcomes[0]["outcome"] == "pending_audio_question"
    result = _resolve(case, questions, [_hear(question)])
    assert result.resolved_cue_ids == {601}
    assert result.spoken_spans[601] == (6975, 7245)
    target = next(c for c in result.cues if c.index == 601)
    assert target.lines == ["É."] and target.start_ms <= 6975 < 7245 <= target.end_ms
    assert [c for c in result.cues if c.index != 601] == [c for c in case["current_cues"] if c.index != 601]
    assert result.alignment is case["alignment"]
    assert [f.kind for f in result.flags] == ["collapsed_singleton_audio_reconciled"]
    assert result.flags[0].cue_ids == [601]
    assert case == before


def test_late_hearing_prompt_uses_current_token_ownership_after_generated_cues_and_anchor_correction():
    case = _case()
    question = _ask(case)[0]
    prompt = json.loads(_adjudication_prompt(
        [question.span], episode_context=case["current_cues"], episode_words=case["words"], language="pt",
    ))
    owned = prompt["spans"][0]["source_cue_ownership"]
    assert [row["cue_id"] for row in owned] == [601]
    assert owned[0]["marked_text"] == "<editable>É</editable>."
    assert owned[0]["tokens"] == [{"token_index": 3, "text": "É", "editable_here": True}]
    proof = question.collapsed_singleton_proof
    assert proof["source_token_indices"] == [1]
    assert proof["current_token_indices"] == [3]
    assert proof["right_anchor"]["authorization"]["case"]["srt_token_indices"] == [2]


@pytest.mark.parametrize("fault", [
    "no_secondary", "bad_hash", "wrong_provider", "old_context", "duplicate_secondary_target",
    "secondary_accent", "partial_secondary_target", "unknown_secondary_speaker", "low_secondary_confidence",
    "secondary_placeholder", "secondary_invalid", "secondary_outside_bracket", "reversed_secondary_order",
    "ambiguous_anchor", "distant_anchor", "uncertain_anchor", "shared_target", "shared_anchor",
    "normal_primary", "ten_ms_primary", "partial_primary", "two_primary_words", "source_target_changed",
    "current_target_changed", "protected", "resolved", "unresolved_alignment", "annotation", "song",
    "duplicate_source_slot", "missing_current_anchor", "source_anchor_unmatched", "missing_anchor_decision",
    "wrong_anchor_case", "unclear_anchor_decision", "wrong_anchor_heard", "partial_anchor_case",
    "stale_anchor_case", "duplicate_anchor_decision", "deterministic_anchor_decision", "unowned_primary_in_gap", "unowned_secondary_in_gap",
    "foreign_primary_on_target", "foreign_secondary_on_target", "two_vad_matches", "no_vad_match",
    "shared_target_anchor_region", "attached_unknown_region", "invalid_region", "outside_audio", "too_long",
])
def test_question_rejects_ambiguous_ownership_spelling_anchors_or_acoustic_evidence(fault):
    case = _case()
    words, secondary, alignment, regions = (case[k] for k in ("words", "secondary_words", "alignment", "regions"))
    context = None
    if fault == "no_secondary":
        case["secondary_words"] = []
    elif fault in {"bad_hash", "wrong_provider", "old_context"}:
        key, value = {"bad_hash": ("words_sha256", "0" * 64), "wrong_provider": ("provider", "unknown"),
                      "old_context": ("policy_version", -1)}[fault]
        context = {**_context(case), key: value}
    elif fault == "duplicate_secondary_target":
        secondary.insert(3, Word(text="É", start=6.23, end=6.31, speaker_id="secondary_False"))
    elif fault in {"secondary_accent", "partial_secondary_target"}:
        secondary[3] = secondary[3].model_copy(update={"text": "e" if fault == "secondary_accent" else "É sim"})
    elif fault == "unknown_secondary_speaker":
        secondary[3] = secondary[3].model_copy(update={"speaker_id": None})
    elif fault == "low_secondary_confidence":
        secondary[3] = secondary[3].model_copy(update={"confidence": .1})
    elif fault == "secondary_placeholder":
        secondary[3] = secondary[3].model_copy(update={"end": secondary[3].start + .001})
    elif fault == "secondary_invalid":
        secondary[3] = secondary[3].model_copy(update={"end": float("nan")})
        context = _context(_case())
    elif fault == "secondary_outside_bracket":
        secondary[3] = secondary[3].model_copy(update={"start": 9.4, "end": 9.6})
    elif fault == "reversed_secondary_order":
        secondary[0], secondary[4] = secondary[4], secondary[0]
    elif fault == "ambiguous_anchor":
        secondary.insert(0, Word(text="né?", start=1.8, end=2.0, speaker_id="secondary_False"))
    elif fault == "distant_anchor":
        secondary[4] = secondary[4].model_copy(update={"start": 9.2, "end": 9.5})
    elif fault == "uncertain_anchor":
        case["uncertain_word_indices"] = {4}
    elif fault in {"shared_target", "shared_anchor"}:
        alignment.cue_word_indices[999] = [3 if fault == "shared_target" else 4]
    elif fault in {"normal_primary", "ten_ms_primary"}:
        words[3] = words[3].model_copy(update={"end": 8.64 if fault == "ten_ms_primary" else 8.9})
    elif fault == "partial_primary":
        words[3] = words[3].model_copy(update={"text": "e"})
    elif fault == "two_primary_words":
        alignment.cue_word_indices[601] = [2, 3]
    elif fault == "source_target_changed":
        case["source_cues"][1] = case["source_cues"][1].with_lines(["É sim."])
    elif fault == "current_target_changed":
        case["current_cues"][3] = case["current_cues"][3].with_lines(["Não."])
    elif fault in {"protected", "resolved"}:
        case[f"{fault}_cue_ids"] = {601}
    elif fault == "unresolved_alignment":
        alignment.diagnostics.unresolved = True
    elif fault in {"annotation", "song"}:
        text = "[É.]" if fault == "annotation" else "♪É.♪"
        case["source_cues"][1] = case["source_cues"][1].with_lines([text])
        case["current_cues"][3] = case["current_cues"][3].with_lines([text])
    elif fault == "duplicate_source_slot":
        case["source_cues"].insert(2, Cue(index=699, start_ms=1, end_ms=2, lines=["É."]))
    elif fault == "missing_current_anchor":
        case["current_cues"].pop()
    elif fault == "source_anchor_unmatched":
        alignment.token_matches = [m for m in alignment.token_matches if m.cue_id != 600]
    elif fault == "missing_anchor_decision":
        case["decisions"] = []
    elif fault == "wrong_anchor_case":
        case["decisions"][0] = case["decisions"][0].model_copy(update={"case_id": "unrelated"})
    elif fault == "unclear_anchor_decision":
        case["decisions"][0] = case["decisions"][0].model_copy(update={"evidence": "heard_unclear", "confidence": 0})
    elif fault == "wrong_anchor_heard":
        case["decisions"][0] = case["decisions"][0].model_copy(update={"heard_text": "Tomam"})
    elif fault == "partial_anchor_case":
        alignment.divergence_spans[0].srt_token_indices = []
    elif fault == "stale_anchor_case":
        alignment.divergence_spans[0].asr_word_indices = [3]
    elif fault == "duplicate_anchor_decision":
        case["decisions"].append(deepcopy(case["decisions"][0]))
    elif fault == "deterministic_anchor_decision":
        case["decisions"][0] = case["decisions"][0].model_copy(update={
            "reason": "Dual ASR cross-check: exact primary wording corroborated.",
        })
    elif fault == "unowned_primary_in_gap":
        del alignment.cue_word_indices[991]
    elif fault == "unowned_secondary_in_gap":
        secondary.insert(3, Word(text="Outro", start=5.45, end=5.65, speaker_id="secondary_False"))
    elif fault == "foreign_primary_on_target":
        words.append(Word(text="Outro", start=7.0, end=7.15, speaker_id="foreign"))
    elif fault == "foreign_secondary_on_target":
        secondary.insert(3, Word(text="Outro", start=7.0, end=7.15, speaker_id="foreign"))
    elif fault == "two_vad_matches":
        regions[7:8] = [SpeechRegion(start=6.975, end=7.12), SpeechRegion(start=7.18, end=7.35)]
    elif fault == "no_vad_match":
        regions.pop(7)
    elif fault == "shared_target_anchor_region":
        regions[7:] = [SpeechRegion(start=6.975, end=9.045)]
    elif fault == "attached_unknown_region":
        regions.insert(7, SpeechRegion(start=6.8, end=6.9))
    elif fault == "invalid_region":
        regions[7] = regions[7].model_copy(update={"end": float("nan")})
    elif fault == "outside_audio":
        case["audio_duration_seconds"] = 8
    elif fault == "too_long":
        words[0] = words[0].model_copy(update={"start": .1, "end": .3})
        secondary[0] = secondary[0].model_copy(update={"start": .1, "end": .3})
        for collection in [words, secondary]:
            for i in range(1, len(collection)):
                collection[i] = collection[i].model_copy(update={"start": collection[i].start + 20, "end": collection[i].end + 20})
        case["audio_duration_seconds"] = 40
    assert _ask(case, context=context) == []


@pytest.mark.parametrize("heard,evidence,final", [
    ("", "not_audible", ""), ("É.", "heard_unclear", "É."), ("e", "heard_clearly", "É."),
    ("É É", "heard_clearly", "É."), ("É Tomem", "heard_clearly", "É."),
    ("Não", "heard_clearly", "É."), ("É.", "heard_clearly", "Sim."),
])
def test_unconfirmed_incomplete_repeated_or_different_hearing_never_retimes(heard, evidence, final):
    case = _case()
    questions = _ask(case)
    result = _resolve(case, questions, [_hear(questions[0], text=heard, evidence=evidence, final=final)])
    assert not result.resolved_cue_ids and result.cues == case["current_cues"] and not result.spoken_spans


@pytest.mark.parametrize("fault", ["proof", "primary", "secondary", "regions", "owner", "anchor_decision", "question_span", "duplicate_question", "duplicate_answer"])
def test_issued_proof_and_answer_are_rejected_after_any_bound_evidence_changes(fault):
    case = _case()
    questions = _ask(case)
    answers = [_hear(questions[0])]
    if fault == "proof":
        questions[0].collapsed_singleton_proof["injected"] = True
    elif fault == "primary":
        case["words"][3] = case["words"][3].model_copy(update={"start": 8.62, "end": 8.621})
    elif fault == "secondary":
        case["secondary_words"][3] = case["secondary_words"][3].model_copy(update={"end": 7.31})
    elif fault == "regions":
        case["regions"][7] = SpeechRegion(start=6.985, end=7.245)
    elif fault == "owner":
        case["alignment"].cue_word_indices[601] = []
    elif fault == "anchor_decision":
        case["decisions"] = []
    elif fault == "question_span":
        questions[0].span.right_anchor_start += .1
    elif fault == "duplicate_question":
        questions.append(questions[0])
    elif fault == "duplicate_answer":
        answers.append(answers[0])
    result = _resolve(case, questions, answers)
    assert not result.resolved_cue_ids and result.cues == case["current_cues"]


def test_source_time_shift_does_not_determine_recovered_acoustic_envelope():
    original = _case()
    shifted = deepcopy(original)
    shifted["source_cues"] = [c.with_timing(c.start_ms + 800000, c.end_ms + 800000) for c in shifted["source_cues"]]
    shifted["current_cues"] = [c.with_timing(c.start_ms + 800000, c.end_ms + 800000) for c in shifted["current_cues"]]
    a, b = _ask(original), _ask(shifted)
    assert a and b
    assert _resolve(original, a, [_hear(a[0])]).spoken_spans == _resolve(shifted, b, [_hear(b[0])]).spoken_spans


@pytest.mark.parametrize("kind", ["audio_snippet_unavailable", "adjudication_audio_unavailable", "invalid_llm_response", "llm_provider_unavailable"])
def test_provider_failure_flag_cannot_be_overridden_by_a_positive_hearing(kind):
    case = _case()
    questions = _ask(case)
    flag = QCFlag(kind=kind, cue_ids=[601], message="Fixture provider failure.")
    result = _resolve(case, questions, [_hear(questions[0])], flags=[flag])
    assert not result.resolved_cue_ids and result.flags == [flag]
