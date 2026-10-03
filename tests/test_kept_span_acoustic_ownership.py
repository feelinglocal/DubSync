from __future__ import annotations

import pytest

from dubsync.adjudication import confidence_gated_decision
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, Word
from dubsync.pipeline import _alignment_with_decision_words, _confidence_held_source_cue_ids
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile


def _keep(span):
    return AdjudicationDecision(
        case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
        confidence=0.95, reason="The source wording is retained.",
    )


def test_scribe_episode_11_keep_does_not_lend_a_cue_the_interjection_21_seconds_later():
    # Saved Scribe case-142. Word and token indices are rebased to these two cues.
    cues = [
        Cue(index=403, start_ms=1183990, end_ms=1185210, lines=["na nossa viagem anual", "deste ano?"]),
        Cue(index=406, start_ms=1207920, end_ms=1209200, lines=["Essa vista é linda."]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id=speaker) for text, start, end, speaker in [
        ("na", 1184.538, 1184.618, "speaker_4"),
        ("nossa", 1184.638, 1184.818, "speaker_4"),
        ("viagem", 1184.898, 1185.218, "speaker_4"),
        ("anual?", 1185.258, 1185.578, "speaker_4"),
        ("Ah,", 1206.398, 1208.158, "speaker_5"),
        ("que", 1208.168, 1208.318, "speaker_5"),
        ("vista", 1208.378, 1208.738, "speaker_5"),
        ("linda.", 1208.758, 1209.078, "speaker_5"),
    ]]
    span = DivergenceSpan(
        case_id="case-142", cue_ids=[403, 406], srt_text="deste ano Essa", asr_text="Ah, que",
        start=1206.398, end=1208.318, confidence=1.0, srt_token_indices=[4, 5, 6],
        asr_word_indices=[4, 5], left_anchor_cue_id=403, right_anchor_cue_id=406,
        left_anchor_end=1185.578, right_anchor_start=1208.378,
        left_anchor_speaker_id="speaker_4", right_anchor_speaker_id="speaker_5", speaker_ids=["speaker_5"],
    )
    alignment = AlignmentResult(cue_word_indices={403: [0, 1, 2, 3], 406: [6, 7]})
    decision = _keep(span)

    updated = _alignment_with_decision_words(alignment, [decision], [span], source_cues=cues, words=words)
    unchanged, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    _, flags = rebuild_cues(unchanged, words, updated, StyleProfile())

    assert updated.cue_word_indices == {403: [0, 1, 2, 3], 406: [4, 5, 6, 7]}
    assert alignment.cue_word_indices == {403: [0, 1, 2, 3], 406: [6, 7]}
    assert unchanged == cues
    assert not any(flag.kind == "timing_outlier_trimmed" and 403 in flag.cue_ids for flag in flags)


def test_continuous_kept_words_follow_unequal_source_lexical_contributions():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["Alpha one two three."]),
        Cue(index=2, start_ms=2000, end_ms=3000, lines=["Four omega."]),
    ]
    words = [Word(text=text, start=1 + i * 0.2, end=1.15 + i * 0.2) for i, text in enumerate(["Alpha", "one", "two", "three", "four", "omega."])]
    span = DivergenceSpan(
        case_id="unequal", cue_ids=[1, 2], srt_text="one two three Four", asr_text="one two three four",
        srt_token_indices=[1, 2, 3, 4], asr_word_indices=[1, 2, 3, 4],
        start=words[1].start, end=words[4].end,
        left_anchor_cue_id=1, left_anchor_end=words[0].end,
        right_anchor_cue_id=2, right_anchor_start=words[5].start,
    )
    updated = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={1: [0], 2: [5]}), [_keep(span)], [span], source_cues=cues, words=words,
    )

    assert updated.cue_word_indices == {1: [0, 1, 2, 3], 2: [4, 5]}
    assert not updated.flags


def test_kept_words_use_separate_acoustic_groups_when_the_text_does_not_match():
    cues = [Cue(index=1, start_ms=1000, end_ms=2200, lines=["Alpha old."]), Cue(index=2, start_ms=9000, end_ms=11000, lines=["Other omega."])]
    words = [Word(text=text, start=start, end=start + 0.15) for text, start in [
        ("Alpha", 1.0), ("one", 1.2), ("two", 1.4), ("three.", 1.6), ("four", 10.0), ("omega.", 10.2),
    ]]
    span = DivergenceSpan(
        case_id="two-groups", cue_ids=[1, 2], srt_text="old Other", asr_text="one two three. four",
        srt_token_indices=[1, 2], asr_word_indices=[1, 2, 3, 4], start=1.2, end=10.15,
        left_anchor_cue_id=1, left_anchor_end=1.15, right_anchor_cue_id=2, right_anchor_start=10.2,
    )
    updated = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={1: [0], 2: [5]}), [_keep(span)], [span], source_cues=cues, words=words,
    )

    assert updated.cue_word_indices == {1: [0, 1, 2, 3], 2: [4, 5]}


def test_ambiguous_kept_boundary_preserves_anchors_without_assigning_uncertain_words():
    cues = [Cue(index=1, start_ms=1000, end_ms=1600, lines=["Alpha old"]), Cue(index=2, start_ms=1600, end_ms=2200, lines=["other omega."])]
    words = [Word(text=text, start=1 + i * 0.2, end=1.15 + i * 0.2) for i, text in enumerate(["Alpha", "yes", "yes", "yes", "omega."])]
    span = DivergenceSpan(
        case_id="ambiguous", cue_ids=[1, 2], srt_text="old other", asr_text="yes yes yes",
        srt_token_indices=[1, 2], asr_word_indices=[1, 2, 3], start=1.2, end=1.75,
        left_anchor_cue_id=1, left_anchor_end=1.15, right_anchor_cue_id=2, right_anchor_start=1.8,
    )
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [4]})
    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == alignment.cue_word_indices
    assert not updated.flags


def test_one_provider_word_cannot_be_divided_between_kept_cues():
    cues = [Cue(index=1, start_ms=1000, end_ms=1600, lines=["Alpha one"]), Cue(index=2, start_ms=1600, end_ms=2200, lines=["two omega."])]
    words = [Word(text="Alpha", start=1, end=1.15), Word(text="one two", start=1.2, end=1.7), Word(text="omega", start=1.8, end=2)]
    span = DivergenceSpan(
        case_id="shared-word", cue_ids=[1, 2], srt_text="one two", asr_text="one two",
        srt_token_indices=[1, 2], asr_word_indices=[1], start=1.2, end=1.7,
        left_anchor_cue_id=1, left_anchor_end=1.15, right_anchor_cue_id=2, right_anchor_start=1.8,
    )
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [2]})
    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == alignment.cue_word_indices
    assert not updated.flags


def test_ambiguous_kept_span_does_not_freeze_its_existing_matched_word_timing():
    cues = [
        Cue(index=1, start_ms=10000, end_ms=11000, lines=["Alpha beta gamma old"]),
        Cue(index=2, start_ms=11000, end_ms=12000, lines=["other delta epsilon omega."]),
    ]
    words = [Word(text=text, start=1 + i * 0.2, end=1.15 + i * 0.2) for i, text in enumerate([
        "Alpha", "beta", "gamma", "yes", "yes", "yes", "delta", "epsilon", "omega.",
    ])]
    span = DivergenceSpan(
        case_id="ambiguous-existing", cue_ids=[1, 2], srt_text="old other", asr_text="yes yes yes",
        srt_token_indices=[3, 4], asr_word_indices=[3, 4, 5], start=1.6, end=2.15,
        left_anchor_cue_id=1, left_anchor_end=1.55, right_anchor_cue_id=2, right_anchor_start=2.2,
    )
    alignment = AlignmentResult(cue_word_indices={1: [0, 1, 2], 2: [6, 7, 8]})

    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)
    rebuilt, _ = rebuild_cues(cues, words, updated, StyleProfile(), protected_cue_ids={
        cue_id for flag in updated.flags if flag.kind == "adjudication_word_mapping_held" for cue_id in flag.cue_ids
    })

    assert updated.cue_word_indices == alignment.cue_word_indices
    assert all(cue.start_ms < 3000 for cue in rebuilt)


def test_unmatched_neighbor_does_not_block_a_kept_cues_measured_own_words():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["Today,"]),
        Cue(index=2, start_ms=2500, end_ms=4500, lines=["I really thank Luke."]),
    ]
    words = [Word(text=text, start=start, end=start + 0.2) for text, start in [
        ("Tonight,", 1.0), ("I", 2.5), ("would", 2.75), ("like", 3.0), ("to", 3.25),
        ("thank", 3.5), ("Luke.", 3.75),
    ]]
    span = DivergenceSpan(
        case_id="whole-and-prefix", cue_ids=[1, 2], srt_text="Today I really", asr_text="Tonight, I would like to",
        srt_token_indices=[0, 1, 2], asr_word_indices=[0, 1, 2, 3, 4], start=1.0, end=3.45,
        right_anchor_cue_id=2, right_anchor_start=3.5,
    )
    alignment = AlignmentResult(cue_word_indices={1: [], 2: [5, 6]})

    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == {1: [], 2: [1, 2, 3, 4, 5, 6]}
    assert not updated.flags


def test_unmatched_canonical_names_keep_their_normal_acoustic_fallback():
    cues = [
        Cue(index=244, start_ms=1012470, end_ms=1013360, lines=["Luan Nian."]),
        Cue(index=245, start_ms=1017790, end_ms=1018880, lines=["Luan Nian..."]),
    ]
    words = [Word(text="Luanyan.", start=1012.705, end=1013.205),
             Word(text="Luanyan.", start=1018.115, end=1018.615)]
    span = DivergenceSpan(
        case_id="repeated-name", cue_ids=[244, 245], srt_text="Luan Nian Luan Nian", asr_text="Luanyan. Luanyan.",
        srt_token_indices=[0, 1, 2, 3], asr_word_indices=[0, 1], start=1012.705, end=1018.615,
    )
    alignment = AlignmentResult(cue_word_indices={244: [], 245: []})

    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == alignment.cue_word_indices
    assert not updated.flags


@pytest.mark.parametrize("first_name_speaker", ["speaker_0", "speaker_5"])
def test_testlong_kept_name_uses_only_its_own_continuous_speaker_group(first_name_speaker):
    # Saved testlong case-83; the next speaker repeats the name, followed by
    # an acknowledgement absent from the source. Ratio partitioning lent
    # the second speaker's first word to cue251; a whole-span hold cut its
    # own name off. The existing Me/chamo anchors distinguish the first name.
    cues = [
        Cue(index=251, start_ms=765190, end_ms=766480, lines=["Me chamo Shang Zhitao."]),
        Cue(index=252, start_ms=766710, end_ms=768000, lines=["Shang Zhitao..."]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id=speaker) for text, start, end, speaker in [
        ("Me", 765.24, 765.34, "speaker_0"),
        ("chamo", 765.36, 765.58, "speaker_0"),
        ("Zhang", 765.62, 765.8, first_name_speaker),
        ("Zitao.", 765.86, 766.235, first_name_speaker),
        ("Zhang", 766.615, 766.9, "speaker_5"),
        ("Zitao?", 766.98, 767.38, "speaker_5"),
        ("Uhum.", 767.48, 767.755, "speaker_0"),
    ]]
    span = DivergenceSpan(
        case_id="repeated-name-and-reply", cue_ids=[251, 252], srt_text="Shang Zhitao Shang Zhitao",
        asr_text="Zhang Zitao. Zhang Zitao? Uhum.", srt_token_indices=[2, 3, 4, 5],
        asr_word_indices=[2, 3, 4, 5, 6], start=765.62, end=767.755,
        left_anchor_cue_id=251, left_anchor_end=765.58, left_anchor_speaker_id="speaker_0",
        right_anchor_cue_id=253, right_anchor_start=771.125, right_anchor_speaker_id="speaker_5",
        speaker_ids=["speaker_0", "speaker_5"],
    )
    alignment = AlignmentResult(cue_word_indices={251: [0, 1], 252: []})

    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == {251: [0, 1, 2, 3] if first_name_speaker == "speaker_0" else [0, 1], 252: []}
    assert not updated.flags
    assert alignment.cue_word_indices == {251: [0, 1], 252: []}


def test_composite_stutter_is_not_lent_to_the_only_anchored_neighbor():
    cues = [Cue(index=309, start_ms=934710, end_ms=935360, lines=["Conseguiu..."]),
            Cue(index=310, start_ms=935360, end_ms=936640, lines=["Conseguiu um emprego!"])]
    words = [Word(text="Consegui-conseguiu", start=934.795, end=935.71),
             Word(text="um", start=935.72, end=935.74), Word(text="emprego?", start=935.78, end=936.3)]
    span = DivergenceSpan(
        case_id="indivisible-stutter", cue_ids=[309, 310], srt_text="Conseguiu Conseguiu",
        asr_text="Consegui-conseguiu", srt_token_indices=[0, 1], asr_word_indices=[0],
        start=934.795, end=935.71, right_anchor_cue_id=310, right_anchor_start=935.72,
    )
    alignment = AlignmentResult(cue_word_indices={309: [], 310: [1, 2]})

    updated = _alignment_with_decision_words(alignment, [_keep(span)], [span], source_cues=cues, words=words)

    assert updated.cue_word_indices == alignment.cue_word_indices
    assert not updated.flags


def _episode_11_interjection_case(rest, *, interjection_end=1206.925):
    # Delivered EP11 MAI case-145 / Scribe case-142 with the effective
    # (repaired) word times; indices are rebased to these two cues.
    cues = [
        Cue(index=403, start_ms=1183990, end_ms=1185210, lines=["na nossa viagem anual deste ano?"]),
        Cue(index=406, start_ms=1207920, end_ms=1209200, lines=["Essa vista é linda."]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id=speaker) for text, start, end, speaker in [
        ("na", 1184.56, 1184.639, "chunk_4:3"),
        ("nossa", 1184.68, 1184.839, "chunk_4:3"),
        ("viagem", 1184.92, 1185.179, "chunk_4:3"),
        ("anual?", 1185.24, 1185.735, "chunk_4:3"),
        ("Ah,", interjection_end - 0.55, interjection_end, "chunk_5:0"),
        *[(text, start, end, "chunk_5:0") for text, (start, end) in zip(["que", "vista", "linda."], rest)],
    ]]
    span = DivergenceSpan(
        case_id="case-145", cue_ids=[403, 406], srt_text="deste ano Essa", asr_text="Ah, que",
        start=words[4].start, end=words[5].end, confidence=0.0, srt_token_indices=[4, 5, 6],
        asr_word_indices=[4, 5], left_anchor_cue_id=403, right_anchor_cue_id=406,
        left_anchor_end=1185.735, right_anchor_start=words[6].start,
        left_anchor_speaker_id="chunk_4:3", right_anchor_speaker_id="chunk_5:0", speaker_ids=["chunk_5:0"],
    )
    return cues, words, span, AlignmentResult(cue_word_indices={403: [0, 1, 2, 3], 406: [6, 7]})


_MAI_REST = [(1208.175, 1208.239), (1208.44, 1208.68), (1208.84, 1209.105)]
_SCRIBE_REST = [(1208.175, 1208.338), (1208.378, 1208.718), (1208.798, 1209.105)]


def _answer(span, *, evidence, confidence):
    return AdjudicationDecision(
        case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text, confidence=confidence,
        reason="The audio window does not contain cue 403 dialogue, so the complete divergent span cannot be evaluated.",
        evidence=evidence, heard_text="" if evidence == "heard_unclear" else span.srt_text,
    )


def _replay_keep(cues, words, span, alignment, answer, hold_flags=()):
    # The pipeline order: engine gate, held source cues, word ownership, text, rebuild.
    decision, gate_flag = confidence_gated_decision(span, answer, 0.7)
    flags = [*hold_flags, *([gate_flag] if gate_flag is not None else [])]
    held = _confidence_held_source_cue_ids(flags)
    updated = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words, protected_cue_ids=held,
    )
    kept, change_flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(), protected_cue_ids=held, words=words,
    )
    rebuilt, timing_flags = rebuild_cues(kept, words, updated, StyleProfile())
    return updated, kept, rebuilt, [*flags, *updated.flags, *change_flags, *timing_flags]


@pytest.mark.parametrize("rest", [_MAI_REST, _SCRIBE_REST], ids=["mai-case-145", "scribe-case-142"])
def test_episode_11_held_keep_does_not_start_its_cue_on_an_unconfirmed_interjection(rest):
    # "Ah," is not shown and no answer heard it; 1.25 s of pause separate it
    # from "que vista linda.". The caption started 1.8 s early on it.
    cues, words, span, alignment = _episode_11_interjection_case(rest)
    answer = _answer(span, evidence="heard_unclear", confidence=0.0)

    updated, kept, rebuilt, flags = _replay_keep(cues, words, span, alignment, answer)

    assert updated.cue_word_indices == {403: [0, 1, 2, 3], 406: [5, 6, 7]}
    assert not updated.flags
    assert kept == cues
    cue = next(cue for cue in rebuilt if cue.index == 406)
    assert cue.start_ms == StyleProfile().snap_floor(words[5].start * 1000)
    # The hold stays, and its case still lists the detached word for review.
    holds = [flag for flag in flags if flag.kind == "low_confidence_adjudication"]
    assert [flag.cue_ids for flag in holds] == [[403, 406]]
    assert holds[0].start <= words[4].start
    assert span.asr_word_indices == [4, 5]
    assert alignment.cue_word_indices == {403: [0, 1, 2, 3], 406: [6, 7]}


@pytest.mark.parametrize("evidence,confidence,pause", [
    ("heard_clearly", 0.95, 1.25),
    ("heard_clearly", 0.95, 0.4),
    ("heard_unclear", 0.0, 0.6),
])
def test_confident_or_continuous_keep_words_still_time_their_cue(evidence, confidence, pause):
    cues, words, span, alignment = _episode_11_interjection_case(_MAI_REST, interjection_end=1208.175 - pause)
    answer = _answer(span, evidence=evidence, confidence=confidence)

    updated, _, rebuilt, _ = _replay_keep(cues, words, span, alignment, answer)

    assert updated.cue_word_indices == {403: [0, 1, 2, 3], 406: [4, 5, 6, 7]}
    cue = next(cue for cue in rebuilt if cue.index == 406)
    assert cue.start_ms == StyleProfile().snap_floor(words[4].start * 1000)


def test_german_held_keep_keeps_a_first_name_said_before_a_short_pause():
    # Delivered German Scribe case-8: the invalid answer held the keep, and
    # "Damien," is spoken 0.59 s before the rest of its own cue.
    cues = [
        Cue(index=41, start_ms=90883, end_ms=91970, lines=["als mitzukommen."]),
        Cue(index=42, start_ms=93930, end_ms=96400, lines=["Damien, aber ich kann", "gar nicht reiten!"]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id="speaker_0") for text, start, end in [
        ("als", 90.9, 91.08), ("mitzukommen.", 91.12, 91.775), ("„Damian,", 93.705, 94.155),
        ("ich", 94.745, 94.94), ("kann", 95.0, 95.16), ("gar", 95.2, 95.3), ("nicht", 95.34, 95.5),
        ("reiten.\"", 95.56, 96.035),
    ]]
    span = DivergenceSpan(
        case_id="case-8", cue_ids=[42], srt_text="Damien aber", asr_text="„Damian,", start=93.705, end=94.155,
        srt_token_indices=[2, 3], asr_word_indices=[2], left_anchor_cue_id=41, right_anchor_cue_id=42,
        left_anchor_end=91.775, right_anchor_start=94.745, speaker_ids=["speaker_0"],
    )
    alignment = AlignmentResult(cue_word_indices={41: [0, 1], 42: [3, 4, 5, 6, 7]})
    answer = AdjudicationDecision(
        case_id="case-8", verdict="keep_srt", final_text="Damien aber", confidence=0.0,
        reason="Invalid LLM response; preserved source SRT.",
    )
    invalid = QCFlag(kind="invalid_llm_response", cue_ids=[42], severity="error",
                     message="LLM response failed schema validation.")

    updated, _, rebuilt, _ = _replay_keep(cues, words, span, alignment, answer, hold_flags=[invalid])

    assert updated.cue_word_indices == {41: [0, 1], 42: [2, 3, 4, 5, 6, 7]}
    assert next(cue for cue in rebuilt if cue.index == 42).start_ms == StyleProfile().snap_floor(93705)


def test_held_keep_word_between_the_cues_own_words_stays_owned():
    # Only edges are trimmed: dropping an inner word would open a pause
    # longer than the cue's maximum and split its own speech.
    cues = [Cue(index=1, start_ms=1000, end_ms=5000, lines=["Alpha old omega."])]
    words = [Word(text=text, start=start, end=start + 0.3) for text, start in [
        ("Alpha", 1.0), ("new", 2.4), ("omega.", 3.8),
    ]]
    span = DivergenceSpan(
        case_id="inner", cue_ids=[1], srt_text="old", asr_text="new", start=2.4, end=2.7,
        srt_token_indices=[1], asr_word_indices=[1], left_anchor_cue_id=1, right_anchor_cue_id=1,
        left_anchor_end=1.3, right_anchor_start=3.8,
    )
    alignment = AlignmentResult(cue_word_indices={1: [0, 2]})
    answer = AdjudicationDecision(
        case_id="inner", verdict="keep_srt", final_text="old", confidence=0.0,
        reason="unclear", evidence="heard_unclear", heard_text="",
    )

    updated, _, _, _ = _replay_keep(cues, words, span, alignment, answer)

    assert updated.cue_word_indices == {1: [0, 1, 2]}


def _german_name_case(pause):
    # Delivered German Scribe case-8 geometry with the pause after the
    # spoken name lengthened: "„Damian," is the only hearing of the cue's own
    # first source words "Damien aber".
    name_end = 94.745 - pause
    cues = [
        Cue(index=41, start_ms=90883, end_ms=91970, lines=["als mitzukommen."]),
        Cue(index=42, start_ms=93000, end_ms=96400, lines=["Damien, aber ich kann", "gar nicht reiten!"]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id="speaker_0") for text, start, end in [
        ("als", 90.9, 91.08), ("mitzukommen.", 91.12, 91.775), ("„Damian,", name_end - 0.45, name_end),
        ("ich", 94.745, 94.94), ("kann", 95.0, 95.16), ("gar", 95.2, 95.3), ("nicht", 95.34, 95.5),
        ("reiten.\"", 95.56, 96.035),
    ]]
    span = DivergenceSpan(
        case_id="case-8", cue_ids=[42], srt_text="Damien aber", asr_text="„Damian,", start=words[2].start,
        end=name_end, srt_token_indices=[2, 3], asr_word_indices=[2], left_anchor_cue_id=41,
        right_anchor_cue_id=42, left_anchor_end=91.775, right_anchor_start=94.745, speaker_ids=["speaker_0"],
    )
    return cues, words, span, AlignmentResult(cue_word_indices={41: [0, 1], 42: [3, 4, 5, 6, 7]})


_INVALID_HOLD = (
    AdjudicationDecision(case_id="case-8", verdict="keep_srt", final_text="Damien aber", confidence=0.0,
                         reason="Invalid LLM response; preserved source SRT."),
    [QCFlag(kind="invalid_llm_response", cue_ids=[42], severity="error",
            message="LLM response failed schema validation.")],
)
_UNCLEAR_HOLD = (
    AdjudicationDecision(case_id="case-8", verdict="keep_srt", final_text="Damien aber", confidence=0.0,
                         reason="unclear", evidence="heard_unclear", heard_text=""),
    [],
)


@pytest.mark.parametrize("hold", [_INVALID_HOLD, _UNCLEAR_HOLD], ids=["invalid", "heard-unclear"])
@pytest.mark.parametrize("pause", [1.1, 1.3])
def test_held_keep_keeps_the_only_hearing_of_its_own_first_words_across_a_longer_pause(pause, hold):
    # W3R-1: the trim cannot tell an inserted word from the hearing of the
    # cue's own edge source words. "Damien aber" has no other counterpart,
    # so dropping "„Damian," started the cue after its spoken first word.
    cues, words, span, alignment = _german_name_case(pause)
    answer, hold_flags = hold

    updated, kept, rebuilt, flags = _replay_keep(cues, words, span, alignment, answer, hold_flags=hold_flags)

    assert updated.cue_word_indices == {41: [0, 1], 42: [2, 3, 4, 5, 6, 7]}
    assert kept == cues
    assert next(cue for cue in rebuilt if cue.index == 42).start_ms == StyleProfile().snap_floor(words[2].start * 1000)
    # The case stays held for review.
    assert any(flag.kind in {"invalid_llm_response", "low_confidence_adjudication"} and 42 in flag.cue_ids
               for flag in flags)


def test_held_keep_keeps_the_only_hearing_of_its_own_last_word_across_a_longer_pause():
    # The same rule at the cue's end: "agora" is the only hearing of the
    # trailing source word, said 1.2 s after the rest of the cue.
    cues = [
        Cue(index=1, start_ms=1000, end_ms=4000, lines=["Vamos embora já."]),
        Cue(index=2, start_ms=6000, end_ms=7000, lines=["Certo."]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Vamos", 1.0, 1.3), ("embora", 1.35, 1.8), ("agora.", 3.0, 3.4), ("Certo.", 6.0, 6.4),
    ]]
    span = DivergenceSpan(
        case_id="tail", cue_ids=[1], srt_text="já", asr_text="agora.", start=3.0, end=3.4,
        srt_token_indices=[2], asr_word_indices=[2], left_anchor_cue_id=1, right_anchor_cue_id=2,
        left_anchor_end=1.8, right_anchor_start=6.0,
    )
    alignment = AlignmentResult(cue_word_indices={1: [0, 1], 2: [3]})
    answer = AdjudicationDecision(
        case_id="tail", verdict="keep_srt", final_text="já", confidence=0.0,
        reason="unclear", evidence="heard_unclear", heard_text="",
    )

    updated, _, rebuilt, _ = _replay_keep(cues, words, span, alignment, answer)

    assert updated.cue_word_indices == {1: [0, 1, 2], 2: [3]}
    assert next(cue for cue in rebuilt if cue.index == 1).end_ms >= 3400


def test_held_inserted_word_before_a_longer_pause_is_still_not_lent_to_the_cue():
    # A span with no source tokens has no edge word to hear: the trim stays.
    cues, words, span, alignment = _episode_11_interjection_case(_MAI_REST)
    span = span.model_copy(update={
        "case_id": "insert", "cue_ids": [406], "srt_text": "", "asr_text": "Ah,", "srt_token_indices": [],
        "asr_word_indices": [4], "start": words[4].start, "end": words[4].end, "left_anchor_cue_id": 403,
        "right_anchor_cue_id": 406, "right_anchor_start": words[5].start,
    })
    alignment = AlignmentResult(cue_word_indices={403: [0, 1, 2, 3], 406: [5, 6, 7]})
    answer = AdjudicationDecision(
        case_id="insert", verdict="keep_srt", final_text="", confidence=0.0,
        reason="unclear", evidence="heard_unclear", heard_text="",
    )

    updated, _, _, _ = _replay_keep(cues, words, span, alignment, answer)

    assert updated.cue_word_indices == {403: [0, 1, 2, 3], 406: [5, 6, 7]}
