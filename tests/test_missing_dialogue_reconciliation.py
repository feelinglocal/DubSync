"""Missing ASR text is a question, never evidence that a source line is absent."""
from __future__ import annotations

import json
from copy import deepcopy
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine
from dubsync.missing_dialogue_reconciliation import (
    MissingDialogueEvidence,
    with_missing_dialogue_residual_questions,
    build_missing_dialogue_questions,
    reconcile_missing_dialogue,
    reconciliation_context,
    validate_reconciliation_artifact,
)
from dubsync.models import (
    AdjudicationDecision, AlignmentResult, AudioSnippet, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word,
)
from dubsync.srt_io import write_srt
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import SpeechEvidence
from dubsync.tokenize import tokenize_cues


def _case(*, bursts=((1.6, 1.9),), target="Tao,"):
    cues = [
        Cue(index=1, start_ms=900, end_ms=1300, lines=["Wait here."]),
        Cue(index=2, start_ms=2000, end_ms=2300, lines=[target]),
        Cue(index=3, start_ms=2300, end_ms=2800, lines=["come closer."]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Wait", 1.0, 1.15), ("here", 1.16, 1.3),
        ("closer", 2.1, 2.4),
    ]]
    tokens = tokenize_cues(cues)
    target_indices = [t.token_index for t in tokens if t.cue_id == 2]
    neighbor = next(t.token_index for t in tokens if t.cue_id == 3)
    span = DivergenceSpan(case_id="case-mixed", cue_ids=[2, 3], srt_text=f"{target} come", asr_text="",
                          srt_token_indices=[*target_indices, neighbor], start=1.3, end=2.1,
                          left_anchor_cue_id=1, right_anchor_cue_id=3,
                          left_anchor_end=1.3, right_anchor_start=2.1)
    alignment = AlignmentResult(
        cue_word_indices={1: [0, 1], 2: [], 3: [2]},
        token_matches=[TokenMatch(cue_id=1, srt_token_index=0, asr_word_index=0, score=1),
                       TokenMatch(cue_id=1, srt_token_index=1, asr_word_index=1, score=1),
                       TokenMatch(cue_id=3, srt_token_index=neighbor + 1, asr_word_index=2, score=1)],
        divergence_spans=[span], unmatched_cue_ids=[2], diagnostics={
            "missing_audio_cue_ids": [2], "missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION,
        },
    )
    regions = [SpeechRegion(start=1, end=1.3),
               *(SpeechRegion(start=a, end=b) for a, b in bursts), SpeechRegion(start=2.1, end=2.4)]
    return cues, words, alignment, regions


def _questions(case, *, audio_duration_seconds=4.0):
    cues, words, alignment, regions = case
    return build_missing_dialogue_questions(cues, alignment, words, regions,
                                           audio_duration_seconds=audio_duration_seconds)


def _fragmented_laugh_case():
    """Captured JA11 activity geometry, with simple independently matched anchors."""
    cues, words, alignment, _ = _case(bursts=(), target="ははは")
    cues = [cues[0].with_timing(42030, 42790), cues[1].with_timing(42800, 43130),
            cues[2].with_timing(43500, 44200)]
    words = [word.model_copy(update={"start": start, "end": end}) for word, (start, end) in zip(
        words, ((41.46, 41.6), (41.61, 41.855), (42.775, 43.7)),
    )]
    parent = alignment.divergence_spans[0].model_copy(update={
        "start": 41.855, "end": 42.775, "left_anchor_end": 41.855, "right_anchor_start": 42.775,
    })
    alignment = alignment.model_copy(update={"divergence_spans": [parent]})
    regions = [SpeechRegion(start=start, end=end) for start, end in (
        (41.46, 41.855), (42.005, 42.105), (42.185, 42.325), (42.425, 42.545), (42.775, 43.7),
    )]
    return cues, words, alignment, regions


def _decision(question, text="Tao", *, evidence="heard_clearly", verdict="use_audio"):
    return AdjudicationDecision(case_id=question.span.case_id, verdict=verdict, final_text=text,
                                evidence=evidence, heard_text=text, confidence=1, reason="Fixture native hearing.")


def _reconcile(case, questions, decisions, *, flags=()):
    cues, words, alignment, regions = case
    return reconcile_missing_dialogue(cues, cues, alignment, words, regions, questions, decisions,
                                     StyleProfile(fps=30), flags=list(flags))


def test_question_owns_only_the_complete_missing_cue_and_keeps_neighbors_read_only():
    case = _case()
    original = deepcopy(case)
    questions = _questions(case)
    assert len(questions) == 1
    q = questions[0]
    assert q.span.cue_ids == [2] and q.span.srt_text == "Tao,"
    assert q.span.srt_token_indices == [2] and q.span.asr_word_indices == []
    assert q.span.start == 1.0 and q.span.end == 2.4  # Complete anchors, not only the gap.
    assert q.span.left_anchor_end == 1.3 and q.span.right_anchor_start == 2.1
    assert q.span.context_before[-1].text == "Wait here."
    assert q.span.context_after[0].text == "come closer."
    assert case == original


def test_confirmed_omission_deletes_only_target_without_assigning_neighbor_words():
    case = _case(bursts=())
    questions = _questions(case)
    original = deepcopy(case)
    result = _reconcile(case, questions, [_decision(questions[0], "")])
    assert result.cues == [case[0][0], case[0][2]]
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert result.alignment.diagnostics.missing_audio_cue_ids == []
    assert result.alignment.unmatched_cue_ids == []
    assert result.resolved_cue_ids == {2}
    assert result.outcomes[0]["outcome"] == "audio_confirmed_omission"
    residual = [flag for flag in result.flags if flag.kind == "missing_audio_source_cue_held"]
    assert len(residual) == 1 and residual[0].cue_ids == [3]
    assert residual[0].old_text == "come"
    assert case == original


@pytest.mark.parametrize("text,verdict", [("Hey!", "use_audio"), ("Tao,", "keep_srt")])
def test_clear_whole_cue_hearing_uses_only_one_independent_burst(text, verdict):
    case = _case()
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], text, verdict=verdict)])
    target = next(cue for cue in result.cues if cue.index == 2)
    assert target.plain_text == text
    assert target.start_ms == 1600 and target.end_ms == 1967  # 40ms tail, then the existing frame grid.
    assert result.spoken_spans == {2: (1600, 1900)}
    assert result.cues[0] == case[0][0] and result.cues[2] == case[0][2]
    assert result.alignment.cue_word_indices == case[2].cue_word_indices


@pytest.mark.parametrize("fault", [
    "no_decision", "legacy_confidence", "unclear", "inaudible", "no_burst", "multiple_bursts",
    "crossing_burst", "foreign_word", "neighbor_echo", "neighbor_fragment_echo", "different_hearing",
    "neighbor_prefix", "neighbor_suffix",
])
def test_clear_native_wording_cannot_create_or_borrow_acoustic_ownership(fault):
    case = _case()
    if fault == "no_burst":
        case = _case(bursts=())
    elif fault == "multiple_bursts":
        case = _case(bursts=((1.5, 1.6), (1.85, 1.9)))
    elif fault == "crossing_burst":
        case = _case(bursts=((1.2, 1.9),))
    questions = _questions(case)
    assert questions
    decisions = [_decision(questions[0])]
    if fault == "no_decision":
        decisions = []
    elif fault == "legacy_confidence":
        decisions = [AdjudicationDecision(case_id=questions[0].span.case_id, verdict="use_audio",
                                          final_text="Tao", confidence=1, reason="A model score is not hearing.")]
    elif fault == "unclear":
        decisions = [_decision(questions[0], evidence="heard_unclear")]
    elif fault == "inaudible":
        decisions = [_decision(questions[0], "", evidence="not_audible")]
    elif fault == "foreign_word":
        case[1].append(Word(text="Other", start=1.7, end=1.8))
    elif fault == "neighbor_echo":
        decisions = [_decision(questions[0], "closer")]
    elif fault == "neighbor_fragment_echo":
        decisions = [_decision(questions[0], "Tao come")]
    elif fault == "neighbor_prefix":
        decisions = [_decision(questions[0], "Wait here Tao")]
    elif fault == "neighbor_suffix":
        decisions = [_decision(questions[0], "Tao closer")]
    elif fault == "different_hearing":
        decisions[0] = decisions[0].model_copy(update={"heard_text": "Bob"})
    result = _reconcile(case, questions, decisions)
    assert result.cues == case[0]
    assert result.resolved_cue_ids == set()
    assert result.alignment == case[2]


@pytest.mark.parametrize("source,heard", [("Wait here Tao", "Wait here Tao"), ("Tao closer", "Tao closer"),
                                           ("Tao", "Please Tao")])
def test_neighbor_echo_check_preserves_source_backed_repetition_and_unrelated_improvisation(source, heard):
    # Long enough to voice three words (a 0.3 s burst is too short for 11 letters).
    case = _case(target=source, bursts=((1.45, 1.95),))
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], heard)])
    assert result.resolved_cue_ids == {2}
    assert next(cue for cue in result.cues if cue.index == 2).plain_text == heard


@pytest.mark.parametrize("heard,verdict", [("ははは", "keep_srt"), ("あはは", "use_audio")])
def test_fragmented_whole_cue_hearing_uses_all_three_contained_laugh_bursts(heard, verdict):
    case = _fragmented_laugh_case()
    original = deepcopy(case)
    questions = _questions(case, audio_duration_seconds=45)
    result = _reconcile(case, questions, [_decision(questions[0], heard, verdict=verdict)])
    target = next(cue for cue in result.cues if cue.index == 2)
    assert result.resolved_cue_ids == {2}
    assert (target.start_ms, target.end_ms, target.plain_text) == (42000, 42600, heard)
    assert result.spoken_spans == {2: (42005, 42545)}
    assert result.cues[0] == case[0][0] and result.cues[2] == case[0][2]
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert case == original


@pytest.mark.parametrize("extra_region", [(42.75, 42.765), (41.84, 41.99), (42.55, 42.8)])
def test_fragmented_cue_cannot_ignore_another_chain_or_region_crossing_an_anchor(extra_region):
    case = _fragmented_laugh_case()
    case[3].append(SpeechRegion(start=extra_region[0], end=extra_region[1]))
    questions = _questions(case, audio_duration_seconds=45)
    result = _reconcile(case, questions, [_decision(questions[0], "ははは")])
    assert result.cues == case[0] and result.resolved_cue_ids == set()


@pytest.mark.parametrize("gap_ms,accepted", [(199, True), (200, False), (201, False)])
def test_fragmented_cue_uses_strict_existing_two_hundred_ms_chain_gap(gap_ms, accepted):
    case = _case(bursts=((1.5, 1.6), ((1600 + gap_ms) / 1000, 1.95)))
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0])])
    assert bool(result.resolved_cue_ids) is accepted


@pytest.mark.parametrize("left_end,right_start,laugh_bursts,next_burst", [
    (90.98, 92.66, ((91.245, 91.425), (91.525, 91.665), (91.745, 91.855)), (92.385, 95.545)),
    (90.84, 92.94, ((91.205, 91.455), (91.535, 91.815), (91.945, 92.375)), (92.665, 96.715)),
])
def test_confirmed_laugh_uses_separate_chain_before_owned_neighbor_preroll(
    left_end, right_start, laugh_bursts, next_burst,
):
    cues, words, alignment, _ = _case(target="ははは")
    words = [word.model_copy(update={"start": start, "end": end}) for word, (start, end) in zip(
        words, ((left_end - 1, left_end - .8), (left_end - .7, left_end), (right_start, right_start + .3)),
    )]
    regions = [SpeechRegion(start=left_end - 1, end=left_end),
               *(SpeechRegion(start=start, end=end) for start, end in laugh_bursts),
               SpeechRegion(start=next_burst[0], end=next_burst[1])]
    case = (cues, words, alignment, regions)
    original = deepcopy(case)
    questions = _questions(case, audio_duration_seconds=100)
    result = _reconcile(case, questions, [_decision(questions[0], "ははは", verdict="keep_srt")])
    assert result.resolved_cue_ids == {2}
    assert result.spoken_spans == {2: (round(laugh_bursts[0][0] * 1000), round(laugh_bursts[-1][1] * 1000))}
    assert result.cues[0] == cues[0] and result.cues[2] == cues[2]
    assert result.alignment.cue_word_indices == alignment.cue_word_indices
    assert case == original


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_anchor_speech_chain_can_bound_an_independent_confirmed_utterance(side):
    case = _case(bursts=((1.6, 1.75),))
    if side in {"left", "both"}:
        case[3][0] = SpeechRegion(start=1, end=1.35)
    if side in {"right", "both"}:
        case[3][-1] = SpeechRegion(start=1.96, end=2.4)
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0])])
    assert result.resolved_cue_ids == {2}
    assert result.spoken_spans == {2: (1600, 1750)}


@pytest.mark.parametrize("fault", ["connected", "extra_chain", "both_anchors", "omission"])
def test_neighbor_chain_exclusion_never_discards_unassigned_or_connected_activity(fault):
    case = _case(bursts=((1.5, 1.6),))
    case[3][-1] = SpeechRegion(start=2.05, end=2.4)
    if fault == "connected":
        case[3].append(SpeechRegion(start=1.72, end=1.91))
    elif fault == "extra_chain":
        case[3].append(SpeechRegion(start=1.81, end=1.825))
    elif fault == "both_anchors":
        case[3][:] = [SpeechRegion(start=1, end=2.4)]
    questions = _questions(case)
    text = "" if fault == "omission" else "Tao"
    result = _reconcile(case, questions, [_decision(questions[0], text)])
    assert result.resolved_cue_ids == set()
    assert result.cues == case[0]


def test_neighbor_chain_ownership_does_not_count_silence_between_its_bursts():
    case = _case(bursts=((1.5, 1.6),))
    case[1][-1] = case[1][-1].model_copy(update={"end": 2.16})
    case[3][-1:] = [SpeechRegion(start=1.95, end=2.101), SpeechRegion(start=2.159, end=2.4)]
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0])])
    assert result.resolved_cue_ids == set()
    assert result.cues == case[0]


def _wide_gap_case(*, left_end, bursts, right_start=3.2):
    """A wider anchored gap: the right anchor word starts at 3.2 s."""
    cues, words, alignment, regions = _case(bursts=bursts)
    words[2] = words[2].model_copy(update={"start": 3.2, "end": 3.5})
    regions[0] = SpeechRegion(start=1, end=left_end)
    regions[-1] = SpeechRegion(start=right_start, end=3.5)
    return cues, words, alignment, regions


@pytest.mark.parametrize("left_end,bursts,right_start", [
    (2.5, ((2.8, 2.83),), 3.2),  # neighbour activity runs 1.2 s past its word, then a separate blip
    (1.35, ((1.5, 1.9), (2.3, 2.5)), 3.2),  # a 50 ms crossing joins a whole gap burst to the neighbour
    (1.3, ((1.6, 1.9),), 2.6),  # 0.6 s right pre-roll
    (1.3, ((1.6, 1.9), (3.0, 3.05)), 3.15),  # within the allowance, but a whole gap burst joins the right chain
])
def test_anchor_chain_beyond_the_neighbor_boundary_allowance_keeps_the_cue_held(left_end, bursts, right_start):
    case = _wide_gap_case(left_end=left_end, bursts=bursts, right_start=right_start)
    questions = _questions(case)
    assert questions
    result = _reconcile(case, questions, [_decision(questions[0], "Tao,", verdict="keep_srt")])
    assert result.outcomes[0]["outcome"] == "speech_burst_crosses_anchor"
    assert result.cues == case[0] and result.resolved_cue_ids == set()
    assert result.alignment == case[2]


def test_anchor_chain_within_the_neighbor_boundary_allowance_still_bounds_a_separate_utterance():
    case = _wide_gap_case(left_end=1.6, bursts=((1.9, 2.4),))
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], "Tao,", verdict="keep_srt")])
    assert result.outcomes[0]["outcome"] == "audio_confirmed_utterance"
    assert result.spoken_spans == {2: (1900, 2400)}


def _captured_laugh_tail_case(final_pulse):
    """Captured 1B cue-11 geometry: the left region runs 0.345 s past its ASR anchor."""
    cues, words, alignment, _ = _case(bursts=(), target="ははは")
    cues = [cues[0].with_timing(40000, 41800), cues[1].with_timing(42800, 43130),
            cues[2].with_timing(43500, 44200)]
    words = [word.model_copy(update={"start": start, "end": end}) for word, (start, end) in zip(
        words, ((41.36, 41.6), (41.61, 41.84), (42.905, 43.7)),
    )]
    parent = alignment.divergence_spans[0].model_copy(update={
        "start": 41.84, "end": 42.905, "left_anchor_end": 41.84, "right_anchor_start": 42.905,
    })
    alignment = alignment.model_copy(update={"divergence_spans": [parent]})
    regions = [SpeechRegion(start=39.565, end=42.185), SpeechRegion(start=final_pulse[0], end=final_pulse[1]),
               SpeechRegion(start=42.905, end=43.7)]
    return cues, words, alignment, regions


@pytest.mark.parametrize("final_pulse", [(42.265, 42.465), (42.385, 42.585), (42.4, 42.6)])
def test_laugh_is_not_timed_on_its_last_pulse_after_an_overrunning_neighbor_region(final_pulse):
    case = _captured_laugh_tail_case(final_pulse)
    questions = _questions(case, audio_duration_seconds=45)
    result = _reconcile(case, questions, [_decision(questions[0], "ははは", verdict="keep_srt")])
    assert result.outcomes[0]["outcome"] == "speech_burst_crosses_anchor"
    assert result.cues == case[0] and result.resolved_cue_ids == set()


@pytest.mark.parametrize("burst,side,crossing", [
    ((1.45, 1.74), "right", (1.96, 2.4)),  # 150 ms after the left region; 140 ms right pre-roll
    ((1.45, 1.74), "right", (2.09, 2.4)),  # 10 ms right pre-roll
    ((1.7, 1.95), "left", (1, 1.4)),  # 150 ms before the right region; 100 ms left post-roll
])
def test_burst_beside_one_anchor_does_not_depend_on_the_other_anchor_crossing(burst, side, crossing):
    results = []
    for crossed in (False, True):
        case = _case(bursts=(burst,))
        if crossed:
            case[3][0 if side == "left" else -1] = SpeechRegion(start=crossing[0], end=crossing[1])
        questions = _questions(case)
        result = _reconcile(case, questions, [_decision(questions[0], "Tao,", verdict="keep_srt")])
        results.append((result.outcomes[0]["outcome"], result.spoken_spans,
                        next(cue for cue in result.cues if cue.index == 2)))
    assert results[0] == results[1]
    assert results[0][0] == "audio_confirmed_utterance"
    assert results[0][1] == {2: (round(burst[0] * 1000), round(burst[1] * 1000))}


@pytest.mark.parametrize("burst", [(1.8, 1.83), (1.8, 1.84), (1.8, 1.88), (1.7, 1.85)])
@pytest.mark.parametrize("verdict", ["keep_srt", "use_audio"])
def test_short_burst_cannot_carry_a_multi_word_line(burst, verdict):
    line = "Tao, please hurry along right now today."
    case = _case(bursts=(burst,), target=line if verdict == "keep_srt" else "Tao,")
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], line, verdict=verdict)])
    assert result.outcomes[0]["outcome"] == "speech_burst_too_short"
    assert result.cues == case[0] and result.resolved_cue_ids == set()
    assert result.alignment == case[2]


def test_multi_word_line_on_a_plausible_burst_is_still_recovered():
    case = _case(bursts=((1.45, 1.9),))
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], "Tao, hurry up")])
    assert result.outcomes[0]["outcome"] == "audio_confirmed_utterance"
    assert next(cue for cue in result.cues if cue.index == 2).plain_text == "Tao, hurry up"
    assert result.spoken_spans == {2: (1450, 1900)}


@pytest.mark.parametrize("fault", ["missing_region_coverage", "no_anchor", "shared_anchor", "ambiguous_anchor",
                                      "unresolved", "partial_target", "song", "annotation", "owned_target",
                                      "duplicate_target_case", "foreign_gap_word", "short_audio"])
def test_question_requires_whole_dialogue_and_independent_acoustic_anchors(fault):
    cues, words, alignment, regions = _case()
    uncertain = set()
    duration = 4.0
    if fault == "missing_region_coverage":
        regions = None
    elif fault == "no_anchor":
        alignment.token_matches = []
    elif fault == "shared_anchor":
        alignment.cue_word_indices[9] = [1]
    elif fault == "ambiguous_anchor":
        uncertain = {1}
    elif fault == "unresolved":
        alignment.diagnostics.unresolved = True
    elif fault == "partial_target":
        cues[1] = cues[1].with_lines(["Tao wait"])
    elif fault == "song":
        cues[1] = cues[1].with_lines(["♪Tao"])
    elif fault == "annotation":
        cues[1] = cues[1].with_lines(["[name] Tao"])
    elif fault == "owned_target":
        alignment.cue_word_indices[2] = [1]
    elif fault == "duplicate_target_case":
        alignment.divergence_spans.append(alignment.divergence_spans[0].model_copy(update={"case_id": "other"}))
    elif fault == "foreign_gap_word":
        words.append(Word(text="Other", start=1.7, end=1.8))
    elif fault == "short_audio":
        duration = 2.2
    assert build_missing_dialogue_questions(cues, alignment, words, regions,
                                           uncertain_word_indices=uncertain, audio_duration_seconds=duration) == []


@pytest.mark.parametrize("bursts", [((1.6, 1.9),), ((1.5, 1.6), (1.8, 1.9))])
def test_empty_native_hearing_does_not_delete_detected_untranscribed_activity(bursts):
    case = _case(bursts=bursts)
    questions = _questions(case)
    result = _reconcile(case, questions, [_decision(questions[0], "")])
    assert result.cues == case[0] and result.resolved_cue_ids == set()


_RAW_PRIMARY = (("Wait", 1.0, 1.15), ("here", 1.16, 1.3), ("closer", 2.1, 2.4))


@pytest.mark.parametrize("left_end, secondary, outcome", [
    (1.3, _RAW_PRIMARY, "audio_confirmed_omission"),
    (1.3, (*_RAW_PRIMARY, ("now", 2.45, 2.6)), "audio_confirmed_omission"),
    (1.3, (*_RAW_PRIMARY[:2], ("Tao", 1.7, 1.9), _RAW_PRIMARY[2]), "secondary_evidence_in_gap"),
    # Edge repair extended the primary anchor word over the secondary's word.
    (1.55, (*_RAW_PRIMARY[:2], ("Tao", 1.33, 1.53), _RAW_PRIMARY[2]), "secondary_evidence_in_gap"),
    # Secondary timestamps drift past the anchor; its word order still places it in the gap.
    (1.3, (*_RAW_PRIMARY[:2], ("Tao", 2.41, 2.44), ("closer", 2.45, 2.75)), "secondary_evidence_in_gap"),
])
def test_empty_native_hearing_does_not_delete_a_cue_the_secondary_asr_heard_in_the_gap(left_end, secondary, outcome):
    cues, words, alignment, regions = case = _case(bursts=())
    words[1] = words[1].model_copy(update={"end": left_end})
    regions[0] = SpeechRegion(start=1, end=left_end)
    questions = _questions(case)
    assert questions[0].span.left_anchor_end == left_end
    result = reconcile_missing_dialogue(
        cues, cues, alignment, words, regions, questions, [_decision(questions[0], "")], StyleProfile(fps=30),
        flags=list(alignment.flags), secondary_words=[Word(text=text, start=start, end=end) for text, start, end in secondary],
    )
    assert result.outcomes[0]["outcome"] == outcome
    if outcome == "audio_confirmed_omission":
        assert result.cues == [cues[0], cues[2]] and result.resolved_cue_ids == {2}
    else:
        assert result.cues == cues and result.resolved_cue_ids == set()
        assert result.alignment.diagnostics.missing_audio_cue_ids == [2]
        assert not any(flag.kind == "missing_dialogue_audio_reconciled" for flag in result.flags)


@pytest.mark.parametrize("tamper", ["audio", "words", "regions", "questions", "policy", "decision_case"])
def test_reconciliation_resume_rejects_stale_or_tampered_bindings(tamper):
    case = _case()
    questions = _questions(case)
    context = reconciliation_context(questions, case[0], case[2], case[1], case[3], audio_sha256="audio-hash")
    artifact = MissingDialogueEvidence(questions, deepcopy(context), [_decision(questions[0])], []).artifact()
    if tamper == "decision_case":
        artifact["decisions"][0]["case_id"] = "case-mixed"
    elif tamper == "policy":
        artifact["context"]["policy_version"] = -1
    else:
        key = {"audio": "audio_sha256", "words": "words_sha256", "regions": "regions_sha256",
               "questions": "questions_sha256"}[tamper]
        artifact["context"][key] = "changed"
    with pytest.raises(ValueError, match="resume from adjudicate"):
        validate_reconciliation_artifact(artifact, context, questions)


def test_reconciliation_artifact_round_trip_keeps_native_evidence_and_exact_target():
    case = _case()
    questions = _questions(case)
    context = reconciliation_context(questions, case[0], case[2], case[1], case[3], audio_sha256="audio-hash")
    decisions = [_decision(questions[0])]
    artifact = json.loads(json.dumps(MissingDialogueEvidence(questions, context, decisions, []).artifact()))
    assert validate_reconciliation_artifact(artifact, context, questions) == (decisions, [])


def test_missing_parent_no_longer_preempts_its_nonmissing_exact_indexed_fragment():
    case = _case()
    questions = _questions(case)
    alignment = with_missing_dialogue_residual_questions(case[2], questions, case[0])
    assert alignment.divergence_spans[0] == case[2].divergence_spans[0]
    residual = alignment.divergence_spans[1]
    assert residual.cue_ids == [3] and residual.srt_text == "come"
    assert residual.srt_token_indices == [3] and residual.asr_word_indices == []
    assert [cue.cue_id for cue in residual.context_before] == [1, 2]
    assert not residual.context_after
    assert (residual.start, residual.end) == (1.0, 2.4)
    provider, held, _ = pipeline._hold_incomplete_source_insertions(
        alignment.divergence_spans, {}, missing_audio_cue_ids={2},
    )
    assert provider == [residual] and [decision.case_id for decision in held] == ["case-mixed"]
    assert alignment.cue_word_indices == case[2].cue_word_indices
    assert with_missing_dialogue_residual_questions(alignment, questions, case[0]) == alignment


@pytest.mark.parametrize("fault", ["missing", "truncated", "text_only_adapter"])
def test_audio_question_never_falls_back_to_text_only_or_partial_audio(fault):
    case = _case(bursts=())
    question = _questions(case)[0]
    class NoCall:
        def adjudicate(self, spans):
            pytest.fail("Missing dialogue cannot be approved from text alone.")
    class AudioNoCall(NoCall):
        def adjudicate_with_audio(self, spans, snippets):
            pytest.fail("The complete anchored window must be supplied before hearing it.")
    snippet = AudioSnippet(case_id=question.span.case_id, path="fixture.wav", start=1, end=2.4)
    snippets = {question.span.case_id: snippet}
    if fault == "missing":
        snippets = {}
    elif fault == "truncated":
        snippets[question.span.case_id] = snippet.model_copy(update={"start": 1.3, "end": 2.1})
    adapter = NoCall() if fault == "text_only_adapter" else AudioNoCall()
    decisions, flags = AdjudicationEngine(adapter, audio_snippets=snippets,
                                         required_audio_case_ids={question.span.case_id}).adjudicate([question.span])
    assert decisions[0].verdict == "keep_srt" and decisions[0].evidence != "heard_clearly"
    assert flags[0].kind == "adjudication_audio_unavailable"
    assert _reconcile(case, [question], decisions, flags=flags).cues == case[0]


def _write_wave(path, seconds=4.0):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * round(seconds * 16000))


def _pipeline_case(tmp_path, monkeypatch, *, bursts=(), heard="", evidence="heard_clearly", snippets=True,
                   residual_heard=None, residual_evidence="heard_clearly", neighbor_lines=None, case_override=None):
    case = deepcopy(case_override) if case_override is not None else _case(bursts=bursts)
    cues, words, alignment, regions = case
    if neighbor_lines is not None:
        cues[2] = cues[2].with_lines(neighbor_lines)
    source, audio, config, fixture = [tmp_path / name for name in ("source.srt", "audio.wav", "provider.yaml", "asr-fixture.json")]
    source.write_text(write_srt(cues), encoding="utf-8")
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    audio_duration = max(4.0, max(cue.end_ms for cue in cues) / 1000 + .5)
    _write_wave(audio, audio_duration)
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)},
                                    "llm": {"provider": "fixture", "audio_snippet_double_check": True}}), encoding="utf-8")
    class NativeAudioFixture:
        def __init__(self):
            self.seen = []
        def adjudicate(self, spans):
            pytest.fail("A missing-dialogue question requires complete native audio.")
        def adjudicate_with_audio(self, spans, clips):
            self.seen.extend(spans)
            assert all(clips[span.case_id].start <= span.start and clips[span.case_id].end >= span.end for span in spans)
            decisions = []
            for span in spans:
                if span.case_id.startswith("missing-dialogue-residual-"):
                    text = span.srt_text if residual_heard is None else residual_heard
                    verdict = "keep_srt" if text == span.srt_text else "use_audio"
                    decision = _decision(type("Question", (), {"span": span})(), text,
                                         evidence=residual_evidence, verdict=verdict)
                else:
                    decision = _decision(type("Question", (), {"span": span})(), heard, evidence=evidence)
                decisions.append(decision.model_dump())
            return decisions
    adapter = NativeAudioFixture()
    def extract(_audio, spans, output, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        if not snippets:
            return []
        result = []
        for span in spans:
            clip = output / f"{span.case_id}.wav"
            _write_wave(clip, span.end - span.start)
            result.append(AudioSnippet(case_id=span.case_id, path=str(clip), start=span.start, end=span.end))
        return result
    monkeypatch.setattr(pipeline, "align_cues_to_words", lambda *_a, **_k: alignment.model_copy(deep=True))
    monkeypatch.setattr(pipeline, "speech_evidence_for_words", lambda *_a, **_k: SpeechEvidence(
        words=deepcopy(words), regions=deepcopy(regions), detected=True,
    ))
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *_a, **_k: adapter)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, language="en", **kwargs)
    return case, adapter, run


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify", "legacy_case_cache", "legacy_batch_cache"])
def test_whole_timing_recovery_requires_native_audio_even_when_wording_is_identical(tmp_path, monkeypatch, mode):
    cues = [Cue(index=1, start_ms=900, end_ms=1300, lines=["Wait here."]),
            Cue(index=2, start_ms=2000, end_ms=2300, lines=["Tao."]),
            Cue(index=3, start_ms=2300, end_ms=2800, lines=["Come."])]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Wait", 1.0, 1.15), ("here", 1.16, 1.3), ("Tao", 1.701, 1.702), ("Come", 2.1, 2.4),
    ]]
    tokens = tokenize_cues(cues)
    ownership = {1: [0, 1], 2: [2], 3: [3]}
    alignment = AlignmentResult(
        cue_word_indices=ownership,
        token_matches=[TokenMatch(cue_id=token.cue_id, srt_token_index=token.token_index,
                                 asr_word_index=token.token_index, score=1) for token in tokens],
        divergence_spans=[], unmatched_cue_ids=[],
        diagnostics={"missing_audio_cue_ids": [], "missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION},
    )
    regions = [SpeechRegion(start=1, end=1.3), SpeechRegion(start=1.6, end=1.9), SpeechRegion(start=2.1, end=2.4)]
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=(cues, words, alignment, regions), heard="Tao.")
    if mode.startswith("legacy_"):
        # Reproduce the prior cache writer: a typographic shortcut with no
        # hearing-specific key context. No native evidence is fabricated.
        with monkeypatch.context() as legacy:
            legacy.setattr(pipeline, "_required_hearing_cache_context", lambda _span: {})
            legacy.setattr(AdjudicationEngine, "adjudicate", lambda engine, spans: (
                [engine.deterministic_policy.decide(span) for span in spans], [],
            ))
            old_result = run()
        assert adapter.seen == []
        old_proof = json.loads((old_result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
        assert old_proof["decisions"][0]["evidence"] is None
        if mode == "legacy_case_cache":
            monkeypatch.setattr(pipeline, "_load_cached_adjudication", lambda *_a, **_k: None)
    result = run()
    assert len(adapter.seen) == 1
    assert adapter.seen[0].case_id.startswith("whole-utterance-timing-")
    assert adapter.seen[0].asr_text == "Tao"
    if mode in {"cache", "rebuild", "verify"}:
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    artifact = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in artifact["cues"] if cue["index"] == 2)
    assert target["start_ms"] <= 1600 and target["end_ms"] >= 1900
    assert target["end_ms"] <= 2100
    assert artifact["alignment"]["cue_word_indices"] == {str(k): v for k, v in ownership.items()}
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["decisions"][0]["heard_text"] == "Tao."
    assert proof["decisions"][0]["evidence"] == "heard_clearly"
    assert any(outcome["outcome"] == "audio_confirmed_utterance" for outcome in proof["outcomes"])


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_pipeline_reconciles_complete_native_omission_in_fresh_cached_and_resumed_runs(tmp_path, monkeypatch, mode):
    case, adapter, run = _pipeline_case(tmp_path, monkeypatch)
    result = run()
    assert len(adapter.seen) == 2
    whole = next(span for span in adapter.seen if span.cue_ids == [2])
    assert whole.srt_text == "Tao,"
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert [cue["index"] for cue in payload["cues"]] == [1, 3]
    assert payload["cues"][-1]["lines"] == case[0][-1].lines
    assert payload["alignment"]["cue_word_indices"] == {"1": [0, 1], "2": [], "3": [2]}
    assert payload["alignment"]["diagnostics"]["missing_audio_cue_ids"] == []
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["outcomes"][0]["outcome"] == "audio_confirmed_omission"
    assert proof["decisions"][0]["evidence"] == "heard_clearly"
    assert payload["missing_dialogue_receipt_sha256"] == proof["receipt_sha256"]
    assert any(flag["kind"] == "missing_dialogue_audio_reconciled" for flag in result.report["flags"])
    assert not any(flag["kind"] in {"missing_audio_source_cue_held", "missing_audio_timing_held"}
                   and 2 in flag["cue_ids"] for flag in result.report["flags"])


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
@pytest.mark.parametrize("fault", ["decision", "source_binding", "questions", "missing_artifact"])
def test_pipeline_does_not_reuse_changed_native_evidence_on_resume(tmp_path, monkeypatch, mode, fault):
    _, _, run = _pipeline_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "missing_dialogue_reconciliation.json"
    proof = json.loads(path.read_text(encoding="utf-8"))
    if fault == "missing_artifact":
        path.unlink()
    else:
        if fault == "decision":
            proof["decisions"][0]["heard_text"] = "Changed"
        elif fault == "source_binding":
            proof["context"]["source_sha256"] = "changed"
        elif fault == "questions":
            proof["questions"][0]["span"]["cue_ids"] = [3]
        path.write_text(json.dumps(proof), encoding="utf-8")
    if fault == "missing_artifact" and mode == "rebuild":
        result = run(resume=mode)
        assert "Tao," in result.output_srt.read_text(encoding="utf-8")
    else:
        with pytest.raises(ValueError, match="resume from adjudicate"):
            run(resume=mode)


@pytest.mark.parametrize("case_kwargs", [{"snippets": False}, {"evidence": "not_audible"},
                                         {"bursts": ((1.5, 1.6), (1.85, 1.9)), "heard": "Tao"}])
def test_pipeline_uncertain_missing_dialogue_retains_source_and_review(tmp_path, monkeypatch, case_kwargs):
    case, _, run = _pipeline_case(tmp_path, monkeypatch, **case_kwargs)
    result = run()
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 2)
    assert target["lines"] == case[0][1].lines
    assert payload["alignment"]["diagnostics"]["missing_audio_cue_ids"] == [2]
    assert not any(flag["kind"] == "missing_dialogue_audio_reconciled" for flag in result.report["flags"])


@pytest.mark.parametrize("mode", ["fresh", "rebuild"])
def test_pipeline_does_not_time_a_heard_cue_on_a_breath_beyond_overrunning_neighbor_activity(
    tmp_path, monkeypatch, mode,
):
    # The cue is spoken straight after "here", so the neighbour's region runs
    # 0.35 s past its word; a separate breath is the only other gap activity.
    case = _case(bursts=((1.87, 1.89),))
    case[3][0] = SpeechRegion(start=1, end=1.65)
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=case, heard="Tao,")
    result = run()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(resume=mode)
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 2)
    assert (target["start_ms"], target["end_ms"], target["lines"]) == (2000, 2300, ["Tao,"])
    assert payload["alignment"]["diagnostics"]["missing_audio_cue_ids"] == [2]
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["outcomes"][0]["outcome"] == "speech_burst_crosses_anchor"
    assert not any(flag["kind"] == "missing_dialogue_audio_reconciled" for flag in result.report["flags"])
    assert any(flag["kind"] == "missing_audio_source_cue_held" and 2 in flag["cue_ids"]
               for flag in result.report["flags"])


@pytest.mark.parametrize("mode", ["fresh", "rebuild", "verify"])
def test_pipeline_keeps_unique_burst_timing_and_exact_neighbor_lines_on_resume(tmp_path, monkeypatch, mode):
    case, adapter, run = _pipeline_case(tmp_path, monkeypatch, bursts=((1.6, 1.9),), heard="Hey!")
    result = run()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(resume=mode)
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 2)
    assert (target["start_ms"], target["end_ms"], target["lines"]) == (1600, 1967, ["Hey!"])
    assert payload["cues"][0]["lines"] == case[0][0].lines
    assert payload["cues"][-1]["lines"] == case[0][-1].lines
    assert not any(flag["kind"] == "missing_audio_source_cue_restored" and 2 in flag["cue_ids"]
                   for flag in result.report["flags"])


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_pipeline_reuses_hearing_when_anchor_chain_bounds_are_refined(tmp_path, monkeypatch, mode):
    case = _case(bursts=((1.6, 1.75),))
    case[3][0] = SpeechRegion(start=1, end=1.35)
    case[3][-1] = SpeechRegion(start=1.96, end=2.4)
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=case, heard="Tao,")
    result = run()
    assert len(adapter.seen) == 2
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 2)
    assert (target["start_ms"], target["end_ms"], target["lines"]) == (1600, 1800, ["Tao,"])
    assert payload["alignment"]["cue_word_indices"] == {"1": [0, 1], "2": [], "3": [2]}
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["outcomes"][0]["outcome"] == "audio_confirmed_utterance"


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_pipeline_keeps_fragmented_laugh_envelope_and_ownership_across_modes(tmp_path, monkeypatch, mode):
    case, adapter, run = _pipeline_case(
        tmp_path, monkeypatch, case_override=_fragmented_laugh_case(), heard="ははは",
    )
    result = run()
    assert len(adapter.seen) == 2
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 2)
    assert (target["start_ms"], target["end_ms"], target["lines"]) == (42000, 42600, ["ははは"])
    assert not any(flag["kind"] == "output_order_inversion" and 2 in flag["cue_ids"]
                   for flag in result.report["flags"])
    assert payload["alignment"]["cue_word_indices"] == {"1": [0, 1], "2": [], "3": [2]}
    assert payload["cues"][0]["lines"] == case[0][0].lines
    assert payload["cues"][-1]["lines"] == case[0][-1].lines
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["outcomes"][0]["outcome"] == "audio_confirmed_utterance"
    assert not any(flag["kind"] == "missing_audio_source_cue_restored" and 2 in flag["cue_ids"]
                   for flag in result.report["flags"])


def test_no_llm_keeps_missing_dialogue_and_records_pending_question_without_model_call(tmp_path, monkeypatch):
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch)
    result = run(no_llm=True)
    assert adapter.seen == []
    assert "Tao," in result.output_srt.read_text(encoding="utf-8")
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["decisions"] == [] and proof["outcomes"][0]["outcome"] == "pending_audio_question"


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_no_llm_keeps_missing_dialogue_without_requesting_new_residual_audio(tmp_path, monkeypatch, mode):
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch)
    result = run(no_llm=True)
    if mode != "fresh":
        result = run(no_llm=True, **({} if mode == "cache" else {"resume": mode}))
    assert adapter.seen == []
    alignment = json.loads((result.episode_workdir / "align.json").read_text(encoding="utf-8"))
    assert not [span for span in alignment["divergence_spans"] if span["case_id"].startswith("missing-dialogue-residual-")]
    assert not [flag for flag in result.report["flags"] if flag["kind"] == "adjudication_audio_unavailable"]
    assert any(flag["kind"] == "missing_audio_source_cue_held" for flag in result.report["flags"])
    assert "Tao," in result.output_srt.read_text(encoding="utf-8")
    proof = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert proof["decisions"] == [] and proof["outcomes"][0]["outcome"] == "pending_audio_question"


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_native_residual_omission_is_independent_of_parent_hold_and_missing_target(tmp_path, monkeypatch, mode):
    case, adapter, run = _pipeline_case(tmp_path, monkeypatch, evidence="not_audible", residual_heard="")
    result = run()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    cue_by_id = {cue["index"]: cue for cue in payload["cues"]}
    assert cue_by_id[2]["lines"] == case[0][1].lines
    assert cue_by_id[3]["lines"] == ["closer."]
    assert payload["alignment"]["diagnostics"]["missing_audio_cue_ids"] == [2]
    assert payload["alignment"]["cue_word_indices"] == {"1": [0, 1], "2": [], "3": [2]}
    decisions = json.loads((result.episode_workdir / "adjudicate.json").read_text(encoding="utf-8"))["decisions"]
    parent = next(decision for decision in decisions if decision["case_id"] == "case-mixed")
    residual = next(decision for decision in decisions if decision["case_id"].startswith("missing-dialogue-residual-"))
    assert parent["verdict"] == "keep_srt" and parent["confidence"] == 0
    assert residual["evidence"] == "heard_clearly" and residual["final_text"] == ""
    assert not any(flag["kind"] == "missing_audio_source_cue_held" and flag["cue_ids"] == [3]
                   for flag in result.report["flags"])


def test_confirmed_unchanged_residual_keeps_authored_line_structure(tmp_path, monkeypatch):
    _, _, run = _pipeline_case(tmp_path, monkeypatch, neighbor_lines=["come", "closer."])
    result = run()
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert next(cue for cue in payload["cues"] if cue["index"] == 3)["lines"] == ["come", "closer."]


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
@pytest.mark.parametrize("residual_heard", ["Please", "Wait here"])
def test_novel_residual_hearing_cannot_borrow_retained_neighbor_word_timing(
    tmp_path, monkeypatch, mode, residual_heard,
):
    case, adapter, run = _pipeline_case(
        tmp_path, monkeypatch, bursts=((1.6, 1.9),), evidence="not_audible",
        residual_heard=residual_heard,
    )
    result = run()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    cue_by_id = {cue["index"]: cue for cue in payload["cues"]}
    assert cue_by_id[3]["lines"] == case[0][2].lines
    assert (cue_by_id[3]["start_ms"], cue_by_id[3]["end_ms"]) == (2300, 2800)
    assert payload["alignment"]["cue_word_indices"] == {"1": [0, 1], "2": [], "3": [2]}
    held = [flag for flag in result.report["flags"] if flag["kind"] == "adjudication_word_mapping_held"
            and flag["cue_ids"] == [3]]
    assert held and any(flag["new_text"] == residual_heard for flag in held)
    decisions = json.loads((result.episode_workdir / "adjudicate.json").read_text(encoding="utf-8"))["decisions"]
    residual = next(decision for decision in decisions if decision["case_id"].startswith("missing-dialogue-residual-"))
    assert residual["heard_text"] == residual_heard and residual["evidence"] == "heard_clearly"


def test_verify_rejects_policy_20_residual_output_and_rebuild_repairs_without_model_call(tmp_path, monkeypatch):
    case, adapter, run = _pipeline_case(
        tmp_path, monkeypatch, bursts=((1.6, 1.9),), evidence="not_audible", residual_heard="Please",
    )
    result = run()
    path = result.episode_workdir / "rebuild.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy_version"] = 20
    stale = next(cue for cue in payload["cues"] if cue["index"] == 3)
    stale.update(lines=["Please closer."], start_ms=2100, end_ms=2467)
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()

    with pytest.raises(ValueError, match="resume from rebuild"):
        run(resume="verify")

    for mode in ("rebuild", "verify"):
        result = run(resume=mode)
        payload = json.loads(path.read_text(encoding="utf-8"))
        repaired = next(cue for cue in payload["cues"] if cue["index"] == 3)
        assert payload["policy_version"] == pipeline._REBUILD_POLICY_VERSION
        assert repaired["lines"] == case[0][2].lines
        assert (repaired["start_ms"], repaired["end_ms"]) == (2300, 2800)
        assert any(flag["kind"] == "adjudication_word_mapping_held" and flag["cue_ids"] == [3]
                   and flag["new_text"] == "Please" for flag in result.report["flags"])
    assert adapter.seen == []
