from __future__ import annotations

import json
import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.adjudication_regions import join_isolated_anchor_regions
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, Word
from dubsync.pipeline import (
    _adlib_cue_ids_by_case,
    _alignment_with_decision_words,
    _confidence_gate_decisions,
    _timing_evidence_held_cue_ids,
    _hold_incomplete_source_insertions,
    _validate_alignment_screen_text_provenance,
    _validate_rebuild_policy,
)
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature, tokenize_cues


def actual_w02():
    # Original source204–207 and actual full-episode MAI words806–819.
    # Provider confidence and speaker fields were absent and remain absent.
    cues = [
        Cue(index=204, start_ms=833710, end_ms=834880, lines=["que com a secretária anterior dele"]),
        Cue(index=205, start_ms=834880, end_ms=835950, lines=["também é assim que eles se relacionam."]),
        Cue(index=206, start_ms=835950, end_ms=837590, lines=["aí já sabia que que ele já fez isso antes."]),
        Cue(index=207, start_ms=838760, end_ms=839880, lines=["Até pensei"]),
    ]
    values = [
        ("que", 834.28, 834.4), ("se", 834.56, 834.659),
        ("relacionava", 834.72, 835.24), ("com", 835.28, 835.36),
        ("a", 835.4, 835.419), ("antiga", 835.46, 835.6990000000001),
        ("secretária.", 835.8, 836.279), ("Você", 836.36, 836.539),
        ("acredita", 836.6, 837.12), ("nisso?", 837.24, 837.639),
        ("E", 838.84, 838.919), ("eu", 839.0, 839.0989999999999),
        ("até", 839.14, 839.3), ("pensei", 839.4, 839.779),
    ]
    return cues, [Word(text=text, start=start, end=end, confidence=None, speaker_id=None)
                  for text, start, end in values]


def joint_span(alignment):
    joined = [span for span in alignment.divergence_spans if span.case_id.startswith("joint-")]
    assert len(joined) == 1
    return joined[0]


def decision(span, **updates):
    return AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=0.98, reason="fresh diagnostic approval of the complete joint region",
    ).model_copy(update=updates)


def test_joint_case_adds_the_isolated_bridge_to_fresh_adjudication_without_moving_any_match():
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)

    assert span.case_id == "joint-case-3--case-4"
    assert [span.case_id for span in alignment.divergence_spans] == ["case-1", "case-2", "joint-case-3--case-4"]
    assert span.cue_ids == [204, 205, 206]
    assert span.srt_token_indices == list(range(4, 23))
    assert span.asr_word_indices == [7, 8, 9, 10, 11]
    assert span.asr_text == "Você acredita nisso? E eu"
    assert (span.start, span.end) == (words[7].start, words[11].end)
    assert (span.left_anchor_cue_id, span.right_anchor_cue_id) == (204, 207)
    assert [(match.cue_id, match.srt_token_index, match.asr_word_index) for match in alignment.token_matches] == [
        (204, 0, 0), (204, 1, 3), (204, 2, 4), (204, 3, 6),
        (205, 7, 10), (207, 23, 12), (207, 24, 13),
    ]
    assert alignment.cue_word_indices == {204: [0, 3, 4, 6], 205: [10], 207: [12, 13]}


def test_fresh_joint_approval_keeps_both_acoustic_groups_whole_and_conserves_every_word():
    cues, words = actual_w02()
    original = [cue.model_dump() for cue in cues]
    alignment = align_cues_to_words(cues, words)
    joint_span(alignment)
    spans = alignment.divergence_spans
    decisions = [decision(span) for span in spans]
    adlib_ids, _ = _adlib_cue_ids_by_case(cues, spans, decisions, alignment.unmatched_cue_ids)
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, flags = apply_adjudication_decisions(cues, spans, decisions, profile, adlib_ids, words=words)
    mapped = _alignment_with_decision_words(
        alignment, decisions, spans, adlib_ids, source_cues=cues, words=words,
    )
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile)

    assert [alphanumeric_signature(cue.plain_text) for cue in changed] == [
        ["que", "se", "relacionava", "com", "a", "antiga", "secretaria"],
        ["voce", "acredita", "nisso"], ["e", "eu", "ate", "pensei"],
    ]
    assert mapped.cue_word_indices == {204: list(range(7)), 205: [7, 8, 9], 206: [], 207: [10, 11, 12, 13]}
    assert [index for cue in changed for index in mapped.cue_word_indices[cue.index]] == list(range(14))
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (profile.snap_floor(words[0].start * 1000), profile.snap_ceil(words[6].end * 1000)),
        (profile.snap_floor(words[7].start * 1000), profile.snap_ceil(words[9].end * 1000)),
        (profile.snap_floor(words[10].start * 1000), profile.snap_ceil(words[13].end * 1000)),
    ]
    assert [cue.model_dump() for cue in cues] == original
    assert not any("held" in flag.kind for flag in [*flags, *mapped.flags])


@pytest.mark.parametrize("old_case_ids", [[], ["case-3"], ["case-3", "case-4"]])
def test_separate_case_decisions_do_not_authorize_the_new_joint_region(old_case_ids):
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)
    old_decisions = [decision(span, case_id=case_id) for case_id in old_case_ids]

    changed, _ = apply_adjudication_decisions(cues, [span], old_decisions, StyleProfile(), words=words)
    mapped = _alignment_with_decision_words(
        alignment, old_decisions, [span], source_cues=cues, words=words,
    )

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    held = _timing_evidence_held_cue_ids(mapped.flags)
    assert held == {204, 205, 206}
    rebuilt, _ = rebuild_cues(changed, words, mapped, StyleProfile(), protected_cue_ids=held)
    assert [(cue.index, cue.start_ms, cue.end_ms, cue.text) for cue in rebuilt if cue.index in held] == [
        (cue.index, cue.start_ms, cue.end_ms, cue.text) for cue in cues if cue.index in held
    ]


@pytest.mark.parametrize("verdict, confidence", [("keep_srt", 0.98), ("use_audio", 0.3)])
def test_unapproved_joint_region_preserves_its_source_text_timing_and_anchor_ownership(verdict, confidence):
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)
    decisions, _ = _confidence_gate_decisions([span], [decision(span, verdict=verdict, confidence=confidence)], {}, [])
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, _ = apply_adjudication_decisions(cues, [span], decisions, profile, words=words)
    mapped = _alignment_with_decision_words(alignment, decisions, [span], source_cues=cues, words=words)
    protected = _timing_evidence_held_cue_ids(mapped.flags)
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile, protected_cue_ids=protected)

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert {204, 205, 206} <= protected
    assert [(cue.index, cue.start_ms, cue.end_ms, cue.text) for cue in rebuilt if cue.index in span.cue_ids] == [
        (cue.index, cue.start_ms, cue.end_ms, cue.text) for cue in cues if cue.index in span.cue_ids
    ]


def ungrouped(cues, words, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr("dubsync.aligner.join_isolated_anchor_regions", lambda spans, *_args, **_kwargs: spans)
        return align_cues_to_words(cues, words)


@pytest.mark.parametrize("problem", [
    "literal_bridge", "another_retained_word", "incomplete_next_cue", "nonconsecutive_next_cue",
    "screen_source", "screen_target", "music", "protected_source", "protected_target",
    "mixed_speakers", "conflicting_anchor_speakers", "conflicting_span_speakers", "low_confidence",
    "source_indices", "word_indices", "punctuation_word", "invalid_time", "multiple_gaps",
    "no_bridge_gap", "distant_next_anchor", "missing_next_words",
])
def test_joint_creation_preserves_original_regions_when_evidence_is_ambiguous(problem, monkeypatch):
    cues, words = actual_w02()
    alignment = ungrouped(cues, words, monkeypatch)
    spans, matches = list(alignment.divergence_spans), list(alignment.token_matches)
    protected = set()
    if problem == "literal_bridge":
        cues[1] = cues[1].with_lines(["também E assim que eles se relacionam."])
    elif problem == "another_retained_word":
        matches.append(matches[4].model_copy(update={"srt_token_index": 6, "asr_word_index": 9}))
    elif problem == "incomplete_next_cue":
        matches.pop()
    elif problem == "nonconsecutive_next_cue":
        matches[-1] = matches[-1].model_copy(update={"asr_word_index": 11})
    elif problem in {"screen_source", "screen_target"}:
        index = 1 if problem == "screen_source" else 3
        cues[index] = cues[index].with_lines(["[ON SCREEN]", *cues[index].lines])
    elif problem == "music":
        cues[0] = cues[0].with_lines(["♪ " + cues[0].plain_text])
    elif problem in {"protected_source", "protected_target"}:
        protected = {205 if problem == "protected_source" else 207}
    elif problem in {"mixed_speakers", "conflicting_anchor_speakers"}:
        first, last = (10, 11) if problem == "mixed_speakers" else (6, 12)
        words[first] = words[first].model_copy(update={"speaker_id": "A"})
        words[last] = words[last].model_copy(update={"speaker_id": "B"})
    elif problem == "conflicting_span_speakers":
        spans[2] = spans[2].model_copy(update={"speaker_ids": ["A"]})
        spans[3] = spans[3].model_copy(update={"speaker_ids": ["B"]})
    elif problem == "low_confidence":
        words[10] = words[10].model_copy(update={"confidence": 0.7})
    elif problem == "source_indices":
        spans[2] = spans[2].model_copy(update={"srt_token_indices": [4, 6]})
    elif problem == "word_indices":
        spans[2] = spans[2].model_copy(update={"asr_word_indices": [7, 9]})
    elif problem == "punctuation_word":
        words[9] = words[9].model_copy(update={"text": "?"})
        spans[2] = spans[2].model_copy(update={"asr_text": "Você acredita ?"})
    elif problem == "invalid_time":
        words[10] = words[10].model_copy(update={"end": float("inf")})
    elif problem == "multiple_gaps":
        words[8] = words[8].model_copy(update={"start": 837.5, "end": 837.55})
        words[9] = words[9].model_copy(update={"start": 837.6})
    elif problem == "no_bridge_gap":
        words[9] = words[9].model_copy(update={"end": 838.4})
    elif problem == "missing_next_words":
        words = words[:12]
    else:
        words[12] = words[12].model_copy(update={"start": 839.7, "end": 839.75})
    before = [span.model_dump() for span in spans]

    result = join_isolated_anchor_regions(
        spans, matches, cues, tokenize_cues(cues), words, protected_cue_ids=protected,
    )

    assert [span.model_dump() for span in result] == before
    assert [span.model_dump() for span in spans] == before


def test_joint_ids_and_other_regions_are_stable_and_grouping_is_idempotent(monkeypatch):
    cues, words = actual_w02()
    alignment = ungrouped(cues, words, monkeypatch)
    tail = DivergenceSpan(case_id="case-99", cue_ids=[], srt_text="", asr_text="later")
    spans = [*alignment.divergence_spans, tail]
    result = join_isolated_anchor_regions(spans, alignment.token_matches, cues, tokenize_cues(cues), words,
                                          protected_cue_ids=set())

    assert [span.case_id for span in result] == ["case-1", "case-2", "joint-case-3--case-4", "case-99"]
    assert result[:2] == spans[:2]
    assert result[-1] == tail
    assert join_isolated_anchor_regions(result, alignment.token_matches, cues, tokenize_cues(cues), words,
                                        protected_cue_ids=set()) == result


@pytest.mark.parametrize("problem", ["nonexact_text", "empty_text", "missing_words", "speaker_conflict", "multiple_gaps", "no_gap"])
def test_joint_application_holds_the_complete_edit_when_approved_partition_is_not_provable(problem):
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)
    proposal = decision(span)
    if problem == "nonexact_text":
        proposal = proposal.model_copy(update={"final_text": "Você acredita nisso? E eu também"})
    elif problem == "empty_text":
        proposal = proposal.model_copy(update={"final_text": ""})
    elif problem == "missing_words":
        words = None
    elif problem == "speaker_conflict":
        words[10] = words[10].model_copy(update={"speaker_id": "A"})
        words[11] = words[11].model_copy(update={"speaker_id": "B"})
    elif problem == "multiple_gaps":
        words[8] = words[8].model_copy(update={"start": 837.5, "end": 837.55})
        words[9] = words[9].model_copy(update={"start": 837.6})
    else:
        words[9] = words[9].model_copy(update={"end": 838.4})

    changed, flags = apply_adjudication_decisions(cues, [span], [proposal], StyleProfile(), words=words)
    mapped = _alignment_with_decision_words(alignment, [proposal], [span], source_cues=cues, words=words)

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in flags] == ["adjudication_replacement_ownership_held"]
    held_flags = [flag for flag in mapped.flags if flag.kind == "adjudication_word_mapping_held"]
    assert len(held_flags) == 1
    # No transfer was proved or applied, so the read-only continuation keeps
    # its own valid timing instead of inheriting the source-region hold.
    assert held_flags[0].cue_ids == span.cue_ids


@pytest.mark.parametrize("protected_id", [205, 206, 207])
def test_joint_application_does_not_take_or_transfer_words_in_a_protected_interior_or_target(protected_id):
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision(span)], StyleProfile(), protected_cue_ids={protected_id}, words=words,
    )
    mapped = _alignment_with_decision_words(
        alignment, [decision(span)], [span], source_cues=cues, words=words, protected_cue_ids={protected_id},
    )

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in flags] == ["adjudication_replacement_ownership_held"]
    assert any(flag.kind == "adjudication_word_mapping_held" for flag in mapped.flags)


def test_independent_held_prefix_does_not_block_the_approved_joint_tail():
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision(span)], StyleProfile(), protected_cue_ids={204}, words=words,
    )
    mapped = _alignment_with_decision_words(
        alignment, [decision(span)], [span], source_cues=cues, words=words, protected_cue_ids={204},
    )

    assert [cue.plain_text for cue in changed] == ["que com a secretária", "Você acredita nisso?", "E eu Até pensei"]
    assert mapped.cue_word_indices[207] == [10, 11, 12, 13]
    assert not any("held" in flag.kind for flag in [*flags, *mapped.flags])


def test_missing_audio_in_any_joint_source_still_blocks_fresh_adjudication():
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    span = joint_span(alignment)

    selected, decisions, flags = _hold_incomplete_source_insertions([span], {}, missing_audio_cue_ids={204})

    assert selected == []
    assert decisions[0].case_id == span.case_id
    assert decisions[0].verdict == "keep_srt"
    assert flags[0].kind == "missing_audio_source_cue_held"


def test_frozen_v2_alignment_provenance_cannot_resume_with_stale_separate_decisions():
    cues, words = actual_w02()
    alignment = align_cues_to_words(cues, words)
    old = alignment.model_copy(update={
        "diagnostics": alignment.diagnostics.model_copy(update={"missing_audio_guard_version": 4}),
    })

    with pytest.raises(RuntimeError, match="resume from align"):
        _validate_alignment_screen_text_provenance(old, cues)
    _validate_alignment_screen_text_provenance(alignment, cues)


def test_frozen_v2_rebuild_policy_cannot_bypass_the_new_joint_decision(tmp_path):
    path = tmp_path / "rebuild.json"
    path.write_text(json.dumps({"policy_version": 5, "cues": []}), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="resume from rebuild"):
        _validate_rebuild_policy(path)
    assert path.read_bytes() == original


def test_joint_region_detection_uses_measured_groups_without_sentence_punctuation():
    cues, words = actual_w02()
    words = [word.model_copy(update={"text": word.text.rstrip(".?")}) for word in words]

    span = joint_span(align_cues_to_words(cues, words))

    assert span.asr_text == "Você acredita nisso E eu"
    assert span.asr_word_indices == [7, 8, 9, 10, 11]
