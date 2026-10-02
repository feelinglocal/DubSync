from __future__ import annotations

from copy import deepcopy

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.models import Cue, SpeechRegion, Word
from dubsync.whole_utterance_timing import build_whole_utterance_timing_questions
from test_whole_utterance_timing import _case, _resolve, _secondary_context, _secondary_split


def _ambiguous_case():
    cues = [Cue(index=1, start_ms=18333, end_ms=19133, lines=["ことなら"]),
            Cue(index=2, start_ms=22300, end_ms=22933, lines=["ここで跪いて"]),
            Cue(index=3, start_ms=23533, end_ms=25333, lines=["落ちてる食べ物を"])]
    left = [("ことなら", 18.5, 19.0)]
    primary = left + [("こ", 21.8, 21.98), ("こ", 21.98, 22.12), ("で", 22.12, 22.26),
                      ("ひ", 22.26, 22.42), ("ざ", 22.42, 22.54), ("ま", 22.54, 22.66), ("ず", 22.66, 22.76),
                      ("い", 22.76, 22.96), ("て", 22.96, 23.54),
                      ("落", 23.6, 23.7), ("ち", 23.7, 23.8), ("て", 23.8, 23.88), ("る", 23.88, 24.02),
                      ("食", 24.02, 24.12), ("べ", 24.12, 24.24), ("物", 24.24, 24.4), ("を", 24.4, 24.52)]
    secondary = left + [("こ", 21.76, 21.839), ("こ", 21.88, 21.96), ("で", 22., 22.08),
                        ("跪", 22.24, 22.319), ("い", 22.72, 22.8), ("て", 22.84, 22.92),
                        ("落", 23.56, 23.64), ("ち", 23.64, 23.719), ("て", 23.72, 23.8), ("る", 23.8, 23.879),
                        ("食", 23.96, 24.039), ("べ", 24.04, 24.12), ("物", 24.16, 24.24), ("を", 24.32, 24.4)]
    def words(rows, speaker):
        return [Word(text=text, start=start, end=end, confidence=1, speaker_id=speaker) for text, start, end in rows]
    primary, secondary = words(primary, "primary_1"), words(secondary, "secondary_1")
    alignment = align_cues_to_words(cues, primary, language="ja")
    regions = [SpeechRegion(start=18.4, end=19.1), SpeechRegion(start=21.715, end=23.075),
               SpeechRegion(start=23.415, end=25.475)]
    return (cues, primary, alignment, regions), secondary, {9}


def _ask(case, secondary, uncertain=(), context=None):
    return build_whole_utterance_timing_questions(case[0], case[2], case[1], case[3], audio_duration_seconds=100,
        uncertain_word_indices=set(uncertain), secondary_words=secondary,
        secondary_context=_secondary_context(secondary) if context is None else context)


@pytest.mark.parametrize("kind", ["split", "ambiguous"])
def test_verified_secondary_evidence_recovers_only_the_existing_whole_cue_region(kind):
    if kind == "split":
        case, secondary, uncertain = _case("split"), _secondary_split(), set()
        expected = (64375, 65135)
    else:
        case, secondary, uncertain = _ambiguous_case()
        expected = (21715, 23075)
    before = deepcopy((case, secondary))
    questions = _ask(case, secondary, uncertain)
    assert len(questions) == 1
    assert questions[0].record()["secondary_acoustic_proof"]["context"] == _secondary_context(secondary)
    result = _resolve(case, questions)
    assert result.resolved_cue_ids == {2}
    assert result.spoken_spans[2][0] == expected[0]
    assert expected[1] <= result.spoken_spans[2][1] <= expected[1] + 1
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert (case, secondary) == before


@pytest.mark.parametrize("fault", ["missing_context", "changed_hash", "wrong_provider", "unknown_speaker",
    "mixed_speakers", "short_secondary", "low_confidence", "invalid_secondary", "foreign_edge_word",
    "unexplained_neighbor_tail", "source_boundary_mismatch"])
def test_neighbor_edge_proof_requires_verified_unambiguous_secondary_source_ownership(fault):
    case, secondary = _case("split"), _secondary_split()
    context = None
    if fault == "missing_context":
        context = {}
    elif fault == "changed_hash":
        context = {**_secondary_context(secondary), "words_sha256": "0" * 64}
    elif fault == "wrong_provider":
        context = {**_secondary_context(secondary), "provider": "unsupported"}
    elif fault == "unknown_speaker":
        secondary[3] = secondary[3].model_copy(update={"speaker_id": None})
    elif fault == "mixed_speakers":
        secondary[5] = secondary[5].model_copy(update={"speaker_id": "secondary_2"})
    elif fault == "short_secondary":
        secondary[3] = secondary[3].model_copy(update={"end": secondary[3].start + .001})
    elif fault == "low_confidence":
        secondary[3] = secondary[3].model_copy(update={"confidence": .1})
    elif fault == "invalid_secondary":
        secondary[3] = secondary[3].model_copy(update={"end": secondary[3].start})
    elif fault == "foreign_edge_word":
        secondary.insert(3, Word(text="別", start=70.64, end=70.675, speaker_id="secondary_2"))
    elif fault == "unexplained_neighbor_tail":
        case[3][-1] = SpeechRegion(start=70.2, end=72.095)
    elif fault == "source_boundary_mismatch":
        secondary[3] = secondary[3].model_copy(update={"text": "川"})
    assert _ask(case, secondary, context=context) == []


@pytest.mark.parametrize("fault", ["no_uncertainty", "no_secondary", "partial_secondary", "repeated_secondary_word",
    "too_far", "foreign_primary", "shared_primary", "extra_vad", "split_secondary_vad"])
def test_ambiguous_primary_tail_requires_unique_bounded_whole_source_corroboration(fault):
    case, secondary, uncertain = _ambiguous_case()
    if fault == "no_uncertainty":
        uncertain.clear()
    elif fault == "no_secondary":
        secondary = []
    elif fault == "partial_secondary":
        secondary[4] = secondary[4].model_copy(update={"text": "座"})
    elif fault == "repeated_secondary_word":
        secondary.insert(7, Word(text="て", start=23.05, end=23.1, confidence=1, speaker_id="secondary_1"))
    elif fault == "too_far":
        secondary[6] = secondary[6].model_copy(update={"start": 22.5, "end": 22.6})
    elif fault == "foreign_primary":
        case[1].append(Word(text="別", start=23., end=23.04, confidence=1, speaker_id="primary_2"))
    elif fault == "shared_primary":
        case[2].cue_word_indices[99] = [9]
    elif fault == "extra_vad":
        case[3].insert(1, SpeechRegion(start=20, end=20.5))
    elif fault == "split_secondary_vad":
        case[3][1:2] = [SpeechRegion(start=21.715, end=22.4), SpeechRegion(start=22.7, end=23.075)]
    assert _ask(case, secondary, uncertain) == []


def test_secondary_proof_cannot_be_changed_after_the_question_is_issued():
    case, secondary, uncertain = _ambiguous_case()
    questions = _ask(case, secondary, uncertain)
    assert questions
    questions[0].secondary_acoustic_proof["context"]["words_sha256"] = "0" * 64
    result = _resolve(case, questions)
    assert result.resolved_cue_ids == set() and result.cues == case[0]
