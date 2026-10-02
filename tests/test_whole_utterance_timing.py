from __future__ import annotations

from copy import deepcopy

import pytest

from dubsync.missing_dialogue_reconciliation import reconcile_missing_dialogue
from dubsync.asr_crosscheck_config import cross_check_context
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SpeechRegion, TokenMatch, Word
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import tokenize_cues
from dubsync.whole_utterance_timing import build_whole_utterance_timing_questions


def _case(kind="empty"):
    """Actual problematic gap geometry; words stay immutable provider evidence."""
    if kind == "split":
        labels = ["三分だ", "三分？", "山下森彦に伝えろ"]
        entries = [("三分だ", 63.74, 64.099), ("三", 64.739, 64.760), ("分", 71.14, 71.16),
                   ("山", 71.28, 71.32), ("下", 71.36, 71.361), ("森", 71.46, 71.48), ("彦", 71.48, 71.6),
                   ("に", 71.6, 71.82), ("伝", 71.82, 71.9), ("え", 71.9, 72.02), ("ろ", 72.02, 72.08)]
        ownership = {1: [0], 2: [1, 2], 3: list(range(3, 11))}
        regions = [(61.095, 64.125), (64.375, 65.135), (70.615, 72.095)]
        target_times = (64666, 65466)
    elif kind == "mismatch":
        labels = ["弁償しろだと？", "ああ", "いいだろ"]
        entries = [(labels[0], 72.120, 72.875), ("う", 73.220, 73.420), ("ん", 73.420, 73.540), (labels[2], 73.635, 74.160)]
        ownership = {1: [0], 2: [], 3: [3]}
        regions = [(72.1, 72.9), (73.165, 73.555), (73.635, 74.225)]
        target_times = (73560, 73960)
    elif kind == "laugh":
        labels = ["いいでしょう？", "ははは", "お嬢さんよ"]
        entries = [(labels[0], 41.46, 41.799), ("。", 41.840, 41.879), (labels[2], 42.775, 43.559)]
        ownership = {1: [0], 2: [], 3: [2]}
        regions = [(41.395, 41.855), (42.005, 42.105), (42.185, 42.325), (42.425, 42.545), (42.775, 44.735)]
        target_times = (42800, 43130)
    else:
        labels = ["Wait here.", "Tao", "Come."]
        entries = [("Wait", 1.0, 1.15), ("here", 1.16, 1.3), ("Come", 2.1, 2.4)]
        ownership = {1: [0, 1], 2: [], 3: [2]}
        regions = [(1, 1.35), (1.6, 1.75), (2.05, 2.4)]
        target_times = (2000, 2300)
    cues = [Cue(index=1, start_ms=100, end_ms=400, lines=[labels[0]]),
            Cue(index=2, start_ms=target_times[0], end_ms=target_times[1], lines=[labels[1]]),
            Cue(index=3, start_ms=3000, end_ms=4000, lines=[labels[2]])]
    words = [Word(text=text, start=start, end=end, confidence=1) for text, start, end in entries]
    tokens = tokenize_cues(cues)
    matches = []
    for cue in cues:
        own_tokens = [t for t in tokens if t.cue_id == cue.index]
        indices = ownership[cue.index]
        for offset, token in enumerate(own_tokens):
            if indices:
                matches.append(TokenMatch(cue_id=cue.index, srt_token_index=token.token_index,
                                          asr_word_index=indices[min(offset, len(indices) - 1)], score=1))
    target_tokens = [t.token_index for t in tokens if t.cue_id == 2]
    evidence = [1, 2] if kind == "mismatch" else [1] if kind == "laugh" else []
    spans = [] if kind == "split" else [DivergenceSpan(
        case_id="parent", cue_ids=[2], srt_text=labels[1], srt_token_indices=target_tokens,
        asr_text="うん" if kind == "mismatch" else "。" if kind == "laugh" else "",
        asr_word_indices=evidence, start=words[ownership[1][-1]].end,
        end=words[ownership[3][0]].start, left_anchor_cue_id=1, right_anchor_cue_id=3,
    )]
    alignment = AlignmentResult(cue_word_indices=ownership, token_matches=matches, divergence_spans=spans,
                                unmatched_cue_ids=[] if kind == "split" else [2], diagnostics={"missing_audio_cue_ids": []})
    return cues, words, alignment, [SpeechRegion(start=a, end=b) for a, b in regions]


def _secondary_split():
    entries = [("三分だ", 63.6, 64.1), ("3", 64.52, 64.599), ("分", 64.8, 64.879),
               ("山", 70.68, 70.759), ("下", 70.88, 70.96), ("盛", 71.08, 71.159), ("彦", 71.319, 71.399),
               ("に", 71.52, 71.599), ("伝", 71.68, 71.759), ("え", 71.84, 71.919), ("ろ", 71.92, 71.999)]
    return [Word(text=text, start=start, end=end, confidence=1, speaker_id="secondary_1") for text, start, end in entries]


def _secondary_context(words):
    return cross_check_context(words, {"asr": {"provider": "openrouter", "model": "microsoft/mai-transcribe-2", "language_code": "ja"}})


def _questions(case, *, secondary=False):
    words = _secondary_split() if secondary else None
    return build_whole_utterance_timing_questions(case[0], case[2], case[1], case[3], audio_duration_seconds=100,
                                                secondary_words=words, secondary_context=_secondary_context(words) if words else None)


def _resolve(case, questions, *, heard=None, evidence="heard_clearly", flags=()):
    decisions = [AdjudicationDecision(case_id=q.span.case_id, verdict="use_audio" if heard == "" else "keep_srt",
                                    final_text="" if heard == "" else q.span.srt_text,
                                    heard_text=q.span.srt_text if heard is None else heard, evidence=evidence,
                                    confidence=1, reason="Unit fixture hearing only.") for q in questions]
    return reconcile_missing_dialogue(case[0], case[0], case[2], case[1], case[3], questions, decisions,
                                      StyleProfile(fps=30), flags=list(flags))


@pytest.mark.parametrize("kind,spoken", [("split", (64375, 65135)), ("mismatch", (73165, 73555)), ("laugh", (42005, 42545))])
def test_whole_hearing_recovers_the_complete_independent_utterance_without_rewriting_words(kind, spoken):
    case = _case(kind)
    original = deepcopy(case)
    questions = _questions(case, secondary=kind == "split")
    assert len(questions) == 1
    q = questions[0]
    assert q.span.cue_ids == [2] and q.span.srt_text == case[0][1].plain_text
    assert q.span.srt_token_indices == [t.token_index for t in tokenize_cues(case[0]) if t.cue_id == 2]
    result = _resolve(case, questions, flags=[QCFlag(kind="timing_evidence_held", cue_ids=[2], message="Bad ASR timing.")])
    assert result.spoken_spans[2][0] == spoken[0]
    assert spoken[1] <= result.spoken_spans[2][1] <= spoken[1] + 1  # Conservative millisecond ceiling.
    assert result.resolved_cue_ids == {2}
    target = result.cues[1]
    assert target.lines == case[0][1].lines
    assert target.start_ms <= spoken[0] < target.end_ms
    assert target.end_ms >= spoken[1]
    assert result.cues[0] == case[0][0] and result.cues[2] == case[0][2]
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert not any(flag.kind == "timing_evidence_held" for flag in result.flags)
    assert case == original


def test_shifted_customer_envelopes_do_not_move_the_recovered_speech():
    case = _case("split")
    shifted = ([c.with_timing(c.start_ms + 20000, c.end_ms + 20000) for c in case[0]], *case[1:])
    assert _resolve(case, _questions(case, secondary=True)).spoken_spans == _resolve(shifted, _questions(shifted, secondary=True)).spoken_spans


def test_split_placeholder_stays_held_without_independent_neighbor_boundary_evidence():
    assert _questions(_case("split")) == []


def test_clear_source_wording_cannot_echo_the_neighbor_into_an_untranscribed_burst():
    case = _case()
    case[0][1] = case[0][1].with_lines(["Come."])
    case[2].divergence_spans[0] = case[2].divergence_spans[0].model_copy(update={"srt_text": "Come."})
    questions = _questions(case)
    result = _resolve(case, questions)
    assert result.resolved_cue_ids == set()
    assert result.cues == case[0]


@pytest.mark.parametrize("left_end", [41.760, 41.840], ids=["mai", "scribe"])
def test_actual_long_neighbor_tail_cannot_discard_the_first_chuckle(left_end):
    case = _case("laugh")
    case[1][0] = case[1][0].model_copy(update={"end": left_end})
    # Captured 1B VAD: the first chuckle remains in the left raw region.
    # Only retaining the final 200ms pulse would clip a confirmed laugh.
    case[3][:] = [SpeechRegion(start=39.565, end=42.185), SpeechRegion(start=42.265, end=42.465),
                  SpeechRegion(start=42.775, end=44.735)]
    assert _questions(case) == []


@pytest.mark.parametrize("fault", ["shared_burst", "shared_owner", "foreign_word", "invalid_anchor", "reordered_anchor",
    "reordered_indices", "collapsed_anchor", "low_overlap_anchor", "unknown_speaker_shared_burst", "embedded_target",
    "missing_regions", "unresolved", "song", "annotation", "partial_target", "changed_owned_word", "invalid_word"])
def test_no_question_when_acoustic_or_source_ownership_is_ambiguous(fault):
    case = _case("split") if fault in {"embedded_target", "changed_owned_word", "invalid_word"} else _case()
    cues, words, alignment, regions = case
    if fault in {"shared_burst", "unknown_speaker_shared_burst"}:
        regions[:] = [SpeechRegion(start=1, end=2.4)]
    elif fault == "shared_owner":
        alignment.cue_word_indices[9] = [0]
    elif fault == "foreign_word":
        words.append(Word(text="Other", start=1.65, end=1.7))
    elif fault == "invalid_anchor":
        words[1] = words[1].model_copy(update={"end": words[1].start})
    elif fault == "reordered_anchor":
        words[1] = words[1].model_copy(update={"start": .9, "end": .99})
    elif fault == "reordered_indices":
        alignment.cue_word_indices[1] = [1, 0]
    elif fault == "collapsed_anchor":
        words[1] = words[1].model_copy(update={"end": words[1].start + .001})
    elif fault == "low_overlap_anchor":
        regions[0] = SpeechRegion(start=1, end=1.1601)
    elif fault == "embedded_target":
        regions[:] = [SpeechRegion(start=61.095, end=65.135), SpeechRegion(start=70.615, end=72.095)]
    elif fault == "missing_regions":
        regions.clear()
    elif fault == "unresolved":
        alignment.diagnostics.unresolved = True
    elif fault == "song":
        cues[1] = cues[1].with_lines(["♪Tao"])
    elif fault == "annotation":
        cues[1] = cues[1].with_lines(["[Tao]"])
    elif fault == "partial_target":
        cues[1] = cues[1].with_lines(["Tao wait"])
    elif fault == "changed_owned_word":
        words[2] = words[2].model_copy(update={"end": words[2].start + .15})
    elif fault == "invalid_word":
        words[1] = words[1].model_copy(update={"end": float("inf")})
    assert _questions(case, secondary=fault in {"embedded_target", "changed_owned_word", "invalid_word"}) == []


@pytest.mark.parametrize("gap_ms,accepted", [(199, True), (200, False), (201, False)])
def test_every_remaining_burst_must_form_one_group_with_strict_existing_gap(gap_ms, accepted):
    case = _case()
    case[3][1:2] = [SpeechRegion(start=1.45, end=1.5), SpeechRegion(start=(1500 + gap_ms) / 1000, end=1.85)]
    assert bool(_questions(case)) is accepted


@pytest.mark.parametrize("fault", ["unclear", "empty", "different_text", "changed_ownership", "changed_words", "changed_regions", "unavailable"])
def test_fresh_positive_whole_cue_hearing_and_unchanged_bound_evidence_are_required(fault):
    case = _case("split")
    questions = _questions(case, secondary=True)
    assert questions
    heard, evidence, flags = None, "heard_clearly", []
    if fault == "unclear":
        evidence = "heard_unclear"
    elif fault == "empty":
        heard = ""
    elif fault == "different_text":
        heard = "十分"
    elif fault == "changed_ownership":
        case[2].cue_word_indices[2] = [1]
    elif fault == "changed_words":
        case[1][1] = case[1][1].model_copy(update={"end": 64.9})
    elif fault == "changed_regions":
        case[3].append(SpeechRegion(start=66, end=66.2))
    elif fault == "unavailable":
        flags = [QCFlag(kind="adjudication_audio_unavailable", cue_ids=[2], message="No audio.")]
    result = _resolve(case, questions, heard=heard, evidence=evidence, flags=flags)
    assert result.cues == case[0] and result.resolved_cue_ids == set()
