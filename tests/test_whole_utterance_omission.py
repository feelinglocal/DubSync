from __future__ import annotations

from copy import deepcopy

import pytest

from dubsync.missing_dialogue_reconciliation import build_missing_dialogue_questions, reconcile_missing_dialogue
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
from dubsync.style_profile import StyleProfile


def _case(*, punctuation=False):
    cues = [Cue(index=1, start_ms=900, end_ms=1300, lines=["Wait here."]),
            Cue(index=2, start_ms=2200, end_ms=2600, lines=["Tao,"]),
            Cue(index=3, start_ms=2700, end_ms=3000, lines=["Come closer."])]
    words = [Word(text=text, start=start, end=end, confidence=1) for text, start, end in [
        ("Wait", 1, 1.15), ("here", 1.16, 1.3), ("Come", 2.1, 2.2), ("closer", 2.2, 2.4),
    ]]
    if punctuation:
        words.append(Word(text="。", start=1.44, end=1.479, confidence=0))
    parent = DivergenceSpan(case_id="parent", cue_ids=[2], srt_text="Tao", srt_token_indices=[2],
                            asr_text="。" if punctuation else "", asr_word_indices=[4] if punctuation else [],
                            start=1.44 if punctuation else 1.3, end=1.479 if punctuation else 2.1,
                            left_anchor_cue_id=1, right_anchor_cue_id=3, left_anchor_end=1.3, right_anchor_start=2.1)
    alignment = AlignmentResult(cue_word_indices={1: [0, 1], 2: [], 3: [2, 3]},
        token_matches=[TokenMatch(cue_id=cue, srt_token_index=token, asr_word_index=word, score=1)
                       for cue, token, word in [(1, 0, 0), (1, 1, 1), (3, 3, 2), (3, 4, 3)]],
        divergence_spans=[parent], unmatched_cue_ids=[2], diagnostics={"missing_audio_cue_ids": [] if punctuation else [2]})
    regions = [SpeechRegion(start=1, end=1.3), SpeechRegion(start=2.1 if punctuation else 1.875, end=2.4)]
    return cues, words, alignment, regions


def _questions(case, **kwargs):
    return build_missing_dialogue_questions(case[0], case[2], case[1], case[3], audio_duration_seconds=4, **kwargs)


def _resolve(case, questions, *, evidence="heard_clearly"):
    decisions = [AdjudicationDecision(case_id=q.span.case_id, verdict="use_audio", final_text="", heard_text="",
                                     evidence=evidence, confidence=1, reason="Unit fixture absence only.") for q in questions]
    return reconcile_missing_dialogue(case[0], case[0], case[2], case[1], case[3], questions, decisions,
                                      StyleProfile(fps=30), flags=[])


@pytest.mark.parametrize("punctuation", [False, True], ids=["225ms-owned-neighbor-preroll", "nonlexical-asr-without-missing-diagnostic"])
def test_whole_native_absence_can_remove_only_the_target_with_complete_neighbor_ownership(punctuation):
    case = _case(punctuation=punctuation)
    before = deepcopy(case)
    questions = _questions(case)
    assert len(questions) == 1
    assert questions[0].span.case_id == "missing-dialogue-v1-parent-cue-2"
    assert questions[0].span.srt_text == "Tao," and questions[0].span.asr_word_indices == []
    result = _resolve(case, questions)
    assert result.cues == [case[0][0], case[0][2]]
    assert result.resolved_cue_ids == {2} and result.spoken_spans == {}
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert case == before


@pytest.mark.parametrize("fault", ["unclear", "independent_activity", "connected_unknown_member", "foreign_chain_word",
    "shared_neighbor", "incomplete_neighbor", "long_preroll", "long_tail", "outside_question", "placeholder_neighbor",
    "insufficient_overlap", "unknown_whole_chain", "lexical_parent", "symbol_parent", "owned_lexical_target"])
def test_absence_cannot_discard_unknown_or_incompletely_owned_activity(fault):
    case = _case(punctuation=fault in {"lexical_parent", "symbol_parent", "owned_lexical_target"})
    cues, words, alignment, regions = case
    evidence = "heard_clearly"
    if fault == "unclear":
        evidence = "heard_unclear"
    elif fault == "independent_activity":
        regions.insert(1, SpeechRegion(start=1.5, end=1.6))
    elif fault == "connected_unknown_member":
        regions.insert(1, SpeechRegion(start=1.82, end=1.84))
    elif fault == "foreign_chain_word":
        words.append(Word(text="Other", start=2.25, end=2.35))
    elif fault == "shared_neighbor":
        alignment.cue_word_indices[8] = [2]
    elif fault == "incomplete_neighbor":
        alignment.token_matches = alignment.token_matches[:-1]
    elif fault == "long_preroll":
        regions[-1] = SpeechRegion(start=1.799, end=2.4)
    elif fault == "long_tail":
        regions[-1] = SpeechRegion(start=1.875, end=2.75)
    elif fault == "outside_question":
        regions[-1] = SpeechRegion(start=1.875, end=2.45)
    elif fault == "placeholder_neighbor":
        words[2] = words[2].model_copy(update={"end": 2.101})
    elif fault == "insufficient_overlap":
        regions[-1] = SpeechRegion(start=1.875, end=2.1001)
    elif fault == "unknown_whole_chain":
        regions[:] = [SpeechRegion(start=1, end=2.4)]
    elif fault == "lexical_parent":
        words[-1] = words[-1].model_copy(update={"text": "Tao"})
        alignment.divergence_spans[0] = alignment.divergence_spans[0].model_copy(update={"asr_text": "Tao"})
    elif fault == "symbol_parent":
        words[-1] = words[-1].model_copy(update={"text": "%"})
        alignment.divergence_spans[0] = alignment.divergence_spans[0].model_copy(update={"asr_text": "%"})
    elif fault == "owned_lexical_target":
        alignment.cue_word_indices[2] = [0]
    questions = _questions(case)
    result = _resolve(case, questions, evidence=evidence)
    assert result.cues == cues and result.resolved_cue_ids == set()


def test_neighbor_edge_allowance_is_bound_to_the_question_and_not_relaxed_per_case():
    case = _case()
    questions = _questions(case, max_neighbor_boundary_overrun=.2)
    result = _resolve(case, questions)
    assert result.cues == case[0] and result.resolved_cue_ids == set()
