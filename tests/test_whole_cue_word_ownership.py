from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, TokenMatch, Word
from dubsync.pipeline import _alignment_with_decision_words, _timing_evidence_held_cue_ids
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile


def actual_name_case():
    # Episode11 source114 and full MAI words462–464. Token/word indices are
    # local to this fixture; all provider times and absent speaker fields remain exact.
    cues = [Cue(index=114, start_ms=389960, end_ms=390880, lines=["Luan Nian!"])]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Hã?", 377.44, 377.72), ("Luanian!", 389.96, 390.719), ("Ei,", 395.2, 395.579),
    ]]
    span = DivergenceSpan(
        case_id="case-32", cue_ids=[114], srt_token_indices=[0, 1],
        srt_text="Luan Nian", asr_text="Hã? Luanian! Ei,", asr_word_indices=[0, 1, 2],
        start=377.44, end=395.579, left_anchor_cue_id=113, left_anchor_end=376.96,
        right_anchor_cue_id=115, right_anchor_start=396.0,
    )
    return cues, words, span


def replay(cues, words, span, final_text, *, verdict="use_audio", existing=None, max_gap=1.5):
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict=verdict, final_text=final_text,
        confidence=0.95, reason="saved model wording or explicit deterministic variation",
    )
    alignment = AlignmentResult(divergence_spans=[span], cue_word_indices=existing or {})
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)
    options = {} if max_gap == 1.5 else {"max_intra_cue_gap": max_gap}
    changed, flags = apply_adjudication_decisions(cues, [span], [decision], profile, words=words, **options)
    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words, **options,
    )
    held = _timing_evidence_held_cue_ids([*flags, *mapped.flags])
    rebuilt, timing_flags = rebuild_cues(
        changed, words, mapped, profile, protected_cue_ids=held, max_intra_cue_gap=max_gap,
    )
    return changed, mapped, rebuilt, [*flags, *mapped.flags, *timing_flags], held, profile


@pytest.mark.parametrize("verdict,final_text", [
    ("use_audio", "Luan Nian!"),  # actual Lite response
    ("hybrid", "Hã? Luan Nian! Ei,"),  # actual Flash response
    ("use_audio", "Hã? Luanian! Ei,"),  # exact text still spans three speech groups
])
def test_whole_cue_multigroup_response_without_one_exact_window_holds_source_atomically(verdict, final_text):
    cues, words, span = actual_name_case()
    changed, mapped, rebuilt, flags, held, _ = replay(cues, words, span, final_text, verdict=verdict)

    assert changed == cues
    assert rebuilt == cues
    assert mapped.cue_word_indices.get(114, []) == []
    assert held == {114}
    ownership = [flag for flag in flags if flag.kind in {
        "adjudication_replacement_ownership_held", "adjudication_word_mapping_held",
    }]
    assert len(ownership) == 2
    assert all(flag.cue_ids == [114] and flag.new_text == final_text for flag in ownership)
    assert not any("missing_audio" in flag.kind for flag in flags)


def test_unique_exact_whole_word_window_uses_only_its_own_actual_endpoints():
    cues, words, span = actual_name_case()
    changed, mapped, rebuilt, flags, held, profile = replay(
        cues, words, span, "Luanian!", existing={114: [0, 1, 2], 115: [9]},
    )

    assert changed[0].plain_text == "Luanian!"
    assert mapped.cue_word_indices == {114: [1], 115: [9]}
    assert (rebuilt[0].start_ms, rebuilt[0].end_ms) == (
        profile.snap_floor(words[1].start * 1000), profile.snap_ceil(words[1].end * 1000),
    )
    assert held == set()
    assert not any(flag.kind == "timing_outlier_trimmed" for flag in flags)


def test_duplicate_exact_windows_do_not_use_source_time_to_pick_the_nearer_one():
    cues, words, span = actual_name_case()
    words[0] = words[0].model_copy(update={"text": "Luanian!"})
    span = span.model_copy(update={"asr_text": "Luanian! Luanian! Ei,"})
    changed, mapped, rebuilt, _, held, _ = replay(cues, words, span, "Luanian!")

    assert changed == rebuilt == cues
    assert mapped.cue_word_indices.get(114, []) == []
    assert held == {114}


@pytest.mark.parametrize("gap,max_gap,expect_hold", [(1.5, 1.5, False), (1.6, 1.5, True), (1.6, 2.0, False)])
def test_whole_cue_guard_uses_the_same_configured_strict_speech_gap_as_recue(gap, max_gap, expect_hold):
    cues, words, span = actual_name_case()
    words = [Word(text="Hello", start=10, end=10.5), Word(text="world", start=10.5 + gap, end=11 + gap)]
    span = span.model_copy(update={"asr_text": "Hello world", "asr_word_indices": [0, 1], "start": 10, "end": 11 + gap})
    changed, _, _, _, held, _ = replay(cues, words, span, "Hello world", max_gap=max_gap)

    assert bool(held) is expect_hold
    assert changed[0].plain_text == (cues[0].plain_text if expect_hold else "Hello world!")


def test_partial_cue_replacement_does_not_enter_whole_cue_guard():
    cues, words, span = actual_name_case()
    cues[0] = cues[0].with_lines(["Say Luan Nian!"])
    span = span.model_copy(update={"srt_token_indices": [1, 2]})
    changed, _, _, flags, _, _ = replay(cues, words, span, "Luan Nian!")

    assert changed[0].plain_text == "Say Luan Nian!"
    assert not any(flag.kind.startswith("adjudication_") and flag.kind.endswith("ownership_held") for flag in flags)


def actual_continuation_case():
    # Episode11 source330/331, actual MAI1371–1380, and exact retained matches.
    cues = [
        Cue(index=330, start_ms=1002750, end_ms=1003920, lines=["Lá em cima não tem banheiro."]),
        Cue(index=331, start_ms=1009230, end_ms=1010080, lines=["Vou segurar mais um pouco."]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("Tá", 1002.92, 1003.019), ("bom,", 1003.039, 1003.179),
        ("então", 1003.22, 1003.379), ("vamos", 1003.42, 1003.56),
        ("lá.", 1003.6, 1003.76), ("Eu", 1009.4, 1009.46),
        ("vou", 1009.48, 1009.559), ("esperar", 1009.6, 1009.819),
        ("um", 1009.86, 1009.9), ("pouco,", 1009.92, 1010.12),
    ]]
    spans = [
        DivergenceSpan(case_id="case-121", cue_ids=[330], srt_token_indices=list(range(6)),
                       asr_word_indices=list(range(6)), srt_text="Lá em cima não tem banheiro",
                       asr_text="Tá bom, então vamos lá. Eu", start=1002.92, end=1009.46,
                       left_anchor_cue_id=329, left_anchor_end=1002.48,
                       right_anchor_cue_id=331, right_anchor_start=1009.48),
        DivergenceSpan(case_id="case-122", cue_ids=[331], srt_token_indices=[7, 8],
                       asr_word_indices=[7], srt_text="segurar mais", asr_text="esperar",
                       start=1009.6, end=1009.819, left_anchor_cue_id=331, left_anchor_end=1009.559,
                       right_anchor_cue_id=331, right_anchor_start=1009.86),
    ]
    matches = [TokenMatch(cue_id=331, srt_token_index=source, asr_word_index=word, score=1.0)
               for source, word in [(6, 6), (9, 8), (10, 9)]]
    return cues, words, spans, matches


def continuation_replay(cues, words, spans, matches, *, protected=None, final_text=None):
    decisions = [AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=final_text if position == 0 and final_text else span.asr_text,
        confidence=0.98, reason="saved matching model approvals",
    ) for position, span in enumerate(spans)]
    alignment = AlignmentResult(divergence_spans=spans, token_matches=matches, cue_word_indices={331: [6, 8, 9]})
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)
    changed, flags = apply_adjudication_decisions(
        cues, spans, decisions, profile, words=words, token_matches=matches, protected_cue_ids=protected,
    )
    mapped = _alignment_with_decision_words(
        alignment, decisions, spans, source_cues=cues, words=words, protected_cue_ids=protected,
    )
    held = _timing_evidence_held_cue_ids([*flags, *mapped.flags])
    rebuilt, timing_flags = rebuild_cues(changed, words, mapped, profile, protected_cue_ids=held)
    return changed, mapped, rebuilt, [*flags, *mapped.flags, *timing_flags], held, profile


def test_actual_whole_cue_improvisation_keeps_its_phrase_and_moves_only_the_proved_next_prefix():
    from dubsync.tokenize import alphanumeric_signature

    cues, words, spans, matches = actual_continuation_case()
    before = [cue.model_dump() for cue in cues]
    changed, mapped, rebuilt, flags, held, profile = continuation_replay(cues, words, spans, matches)

    assert [cue.plain_text for cue in changed] == ["Tá bom, então vamos lá.", "Eu Vou esperar um pouco."]
    assert mapped.cue_word_indices == {330: list(range(5)), 331: list(range(5, 10))}
    assert [index for cue in changed for index in mapped.cue_word_indices[cue.index]] == list(range(10))
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in changed)) == alphanumeric_signature(
        " ".join(word.text for word in words)
    )
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (profile.snap_floor(words[0].start * 1000), profile.snap_ceil(words[4].end * 1000)),
        (profile.snap_floor(words[5].start * 1000), profile.snap_ceil(words[9].end * 1000)),
    ]
    assert held == set()
    assert not any("held" in flag.kind or flag.kind == "timing_outlier_trimmed" for flag in flags)
    assert [cue.model_dump() for cue in cues] == before


@pytest.mark.parametrize("variant", [
    "no_matches", "missing_first_anchor", "no_consecutive_pair", "pair_in_later_group",
    "forged_match", "fuzzy_match", "protected_target", "conflicting_word_speakers",
    "conflicting_anchor_speakers", "wrong_target", "distant_right_join", "several_gaps",
    "two_word_prefix", "nonexact_approval", "screen_target",
])
def test_whole_cue_continuation_needs_the_complete_bounded_evidence(variant):
    cues, words, spans, matches = actual_continuation_case()
    protected = None
    final_text = None
    if variant == "no_matches":
        matches = []
    elif variant == "missing_first_anchor":
        matches = matches[1:]
    elif variant == "no_consecutive_pair":
        matches = matches[:2]
    elif variant == "pair_in_later_group":
        for index in (8, 9):
            words[index] = words[index].model_copy(update={"start": words[index].start + 3, "end": words[index].end + 3})
    elif variant == "forged_match":
        words[8] = words[8].model_copy(update={"text": "diferente"})
    elif variant == "fuzzy_match":
        matches[-1] = matches[-1].model_copy(update={"score": 0.9})
    elif variant == "protected_target":
        protected = {331}
    elif variant == "conflicting_word_speakers":
        words[0] = words[0].model_copy(update={"speaker_id": "A"})
        words[-1] = words[-1].model_copy(update={"speaker_id": "B"})
    elif variant == "conflicting_anchor_speakers":
        spans[0] = spans[0].model_copy(update={"left_anchor_speaker_id": "A", "right_anchor_speaker_id": "B"})
    elif variant == "wrong_target":
        spans[0] = spans[0].model_copy(update={"right_anchor_cue_id": 332})
    elif variant == "distant_right_join":
        spans[0] = spans[0].model_copy(update={"right_anchor_start": 1009.8})
    elif variant == "several_gaps":
        words[3] = words[3].model_copy(update={"start": 1005.4, "end": 1005.56})
        words[4] = words[4].model_copy(update={"start": 1005.6, "end": 1005.76})
    elif variant == "two_word_prefix":
        spans[0] = spans[0].model_copy(update={"asr_word_indices": list(range(7)),
                                             "asr_text": "Tá bom, então vamos lá. Eu vou", "end": words[6].end})
    elif variant == "nonexact_approval":
        final_text = "Tá bom, vamos lá. Eu"
    elif variant == "screen_target":
        cues[1] = cues[1].with_lines(["[ON SCREEN]", "Vou segurar mais um pouco."])
    changed, mapped, rebuilt, flags, held, _ = continuation_replay(
        cues, words, spans, matches, protected=protected, final_text=final_text,
    )

    assert changed[0] == rebuilt[0] == cues[0]
    assert mapped.cue_word_indices.get(330, []) == []
    assert 5 not in mapped.cue_word_indices[331]
    assert 330 in held
    assert any(flag.kind == "adjudication_replacement_ownership_held" and 330 in flag.cue_ids for flag in flags)
    if variant == "protected_target":
        # A held transfer cannot erase a different accepted edit in its target.
        assert changed[1].plain_text == "Vou esperar um pouco."
        assert (rebuilt[1].start_ms, rebuilt[1].end_ms) == (cues[1].start_ms, cues[1].end_ms)


def test_whole_cue_continuation_does_not_require_the_model_to_copy_asr_punctuation():
    cues, words, spans, matches = actual_continuation_case()
    changed, mapped, _, _, held, _ = continuation_replay(
        cues, words, spans, matches, final_text="Tá bom então vamos lá Eu",
    )

    assert changed[0].plain_text == "Tá bom então vamos lá."
    assert changed[1].plain_text == "Eu Vou esperar um pouco."
    assert mapped.cue_word_indices == {330: list(range(5)), 331: list(range(5, 10))}
    assert held == set()
