from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions, indexed_multi_cue_replacements
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, Word
from dubsync.pipeline import (
    _alignment_with_decision_words,
    _confidence_gate_decisions,
    _confidence_held_source_cue_ids,
    _hold_incomplete_source_insertions,
)
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


def _episode17_boundary(case: str) -> tuple[list[Cue], list[Word], list[str], dict[int, list[int]]]:
    # Actual MAI probes from 2026-09-14, rebased to episode time:
    # mai-e17-w05-18s-v1/words.json and mai-e17-w03-v1/words.json.
    if case in {"sem", "sem_single"}:
        cues = [
            Cue(index=527, start_ms=2074400, end_ms=2075680,
                lines=["Estou aqui embaixo do seu prédio."]),
            Cue(index=528, start_ms=2086690, end_ms=2087590,
                lines=["Não tenho pressa."]),
        ]
        timed_tokens = [
            ("Estou", 2074.56, 2074.78), ("aqui", 2074.84, 2075.00),
            ("embaixo.", 2075.04, 2075.479), ("Sem", 2086.84, 2087.019),
            ("pressa,", 2087.12, 2087.439),
        ]
        expected = ["Estou aqui embaixo.", "Sem pressa."]
        ownership = {527: [0, 1, 2], 528: [3, 4]}
        if case == "sem_single":
            # Structural variant, using the same actual provider words.
            cues[1] = cues[1].with_lines(["pressa."])
    else:
        cues = [
            Cue(index=214, start_ms=853670, end_ms=854830,
                lines=["As fotos ainda estão com ele,"]),
            Cue(index=215, start_ms=854830, end_ms=855670, lines=["não pode denunciar."]),
            Cue(index=216, start_ms=855880, end_ms=857640,
                lines=["Se ele divulgar essas fotos,"]),
        ]
        timed_tokens = [
            ("as", 855.08, 855.14), ("fotos", 855.20, 855.48),
            ("ainda.", 855.60, 855.879), ("E", 856.36, 856.42),
            ("se", 856.44, 856.54), ("ele", 856.58, 856.679),
            ("divulgar", 856.76, 857.06), ("essas", 857.08, 857.32),
            ("fotos,", 857.40, 857.779),
        ]
        expected = ["As fotos ainda,", "E Se ele divulgar essas fotos,"]
        ownership = {214: [0, 1, 2], 215: [], 216: [3, 4, 5, 6, 7, 8]}
    words = [
        Word(
            text=text, start=start, end=end,
            confidence=None,
            speaker_id=f"chunk_1:{int(position >= 3)}" if case.startswith("sem") else "chunk_1:1",
        )
        for position, (text, start, end) in enumerate(timed_tokens)
    ]
    return cues, words, expected, ownership


@pytest.mark.parametrize("case", ["sem", "sem_single", "e"])
def test_replaced_word_follows_the_next_retained_phrase_without_moving_its_audio(case):
    cues, words, expected, ownership = _episode17_boundary(case)
    source_snapshot = [cue.model_dump() for cue in cues]
    alignment = align_cues_to_words(cues, words)
    decisions = [
        AdjudicationDecision(
            case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
            confidence=1.0, reason="deterministic spoken-word fixture",
        )
        for span in alignment.divergence_spans
    ]
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, _ = apply_adjudication_decisions(
        cues, alignment.divergence_spans, decisions, profile,
    )
    mapped = _alignment_with_decision_words(
        alignment, decisions, alignment.divergence_spans, source_cues=cues, words=words,
    )

    assert [cue.plain_text for cue in changed] == expected
    assert mapped.cue_word_indices == ownership
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in changed)) == (
        alphanumeric_signature(" ".join(word.text for word in words))
    )
    assert [cue.model_dump() for cue in cues] == source_snapshot
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile)
    assert rebuilt[0].end_ms == profile.snap_ceil(words[2].end * 1000)
    assert rebuilt[-1].start_ms == profile.snap_floor(words[3].start * 1000)
    assert not any(flag.kind == "adjudication_word_mapping_held" for flag in mapped.flags)


@pytest.mark.parametrize("case, expected_ids", [
    ("sem", (527, 528)), ("sem_single", (527, 528)), ("e", (214, 216)),
])
def test_replacement_keeps_matched_neighbor_evidence_without_becoming_an_insertion(case, expected_ids):
    cues, words, _, _ = _episode17_boundary(case)
    span, = align_cues_to_words(cues, words).divergence_spans

    assert (span.left_anchor_cue_id, span.right_anchor_cue_id) == expected_ids
    assert (span.left_anchor_end, span.right_anchor_start) == (words[2].end, words[4].start)
    assert span.srt_token_indices
    assert span.insertion_token_offset is None


@pytest.mark.parametrize("updates, final_text", [
    ({"left_anchor_end": None}, "E"),
    ({"right_anchor_start": None}, "E"),
    ({"right_anchor_start": 857.0}, "E"),
    ({"left_anchor_end": 856.25}, "E"),
    ({"start": float("inf")}, "E"),
    ({"right_anchor_speaker_id": "another_actor"}, "E"),
    ({"speaker_ids": ["actor", "another_actor"]}, "E"),
    ({"asr_word_indices": []}, "E"),
    ({}, "E."),
    ({}, "Outro"),
])
def test_next_phrase_transfer_requires_complete_unambiguous_word_evidence(updates, final_text):
    cues, words, _, _ = _episode17_boundary("e")
    span, = align_cues_to_words(cues, words).divergence_spans
    span = span.model_copy(update=updates)

    edits = indexed_multi_cue_replacements(cues, span, final_text)

    assert edits is not None
    assert 216 not in edits
    assert " ".join(edit[2] for edit in edits.values()).strip() == final_text


@pytest.mark.parametrize("guard", ["missing_audio", "confidence"])
@pytest.mark.parametrize("case", ["e", "sem_single"])
def test_protected_external_target_holds_the_complete_replacement_and_its_word(guard, case):
    cues, words, _, _ = _episode17_boundary(case)
    alignment = align_cues_to_words(cues, words)
    span, = alignment.divergence_spans
    target_id = cues[-1].index
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=1.0, reason="verified word with protected neighboring cue",
    )
    if guard == "missing_audio":
        alignment = alignment.model_copy(update={
            "diagnostics": alignment.diagnostics.model_copy(update={"missing_audio_cue_ids": [target_id]}),
        })
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], profile, protected_cue_ids={target_id},
    )
    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words,
        protected_cue_ids={target_id} if guard == "confidence" else None,
    )

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in flags] == ["adjudication_replacement_ownership_held"]
    assert flags[0].new_text == span.asr_text
    assert flags[0].cue_ids == [cue.index for cue in cues]
    ownership_flags = [flag for flag in mapped.flags if flag.kind == "adjudication_word_mapping_held"]
    assert len(ownership_flags) == 1
    assert ownership_flags[0].cue_ids == [cue.index for cue in cues]
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile, protected_cue_ids={cue.index for cue in cues})
    assert [(cue.index, cue.text, cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (cue.index, cue.text, cue.start_ms, cue.end_ms) for cue in cues
    ]


@pytest.mark.parametrize("uncertain_cue_id, source_index, word_index", [(527, 0, 0), (528, 8, 4)])
def test_independent_confident_replacement_survives_an_uncertain_span_in_a_shared_cue(
    uncertain_cue_id, source_index, word_index,
):
    cues, words, expected, ownership = _episode17_boundary("sem")
    alignment = align_cues_to_words(cues, words)
    accepted_span, = alignment.divergence_spans
    uncertain = DivergenceSpan(
        case_id="uncertain", cue_ids=[uncertain_cue_id],
        srt_text=words[word_index].text, asr_text="Uncertain",
        srt_token_indices=[source_index], asr_word_indices=[word_index],
        start=words[word_index].start, end=words[word_index].end,
    )
    spans = [uncertain, accepted_span]
    proposals = [
        AdjudicationDecision(
            case_id="uncertain", verdict="use_audio", final_text="Uncertain",
            confidence=0.3, reason="uncertain separate source token",
        ),
        AdjudicationDecision(
            case_id=accepted_span.case_id, verdict="use_audio", final_text="Sem",
            confidence=1.0, reason="independently verified word before retained continuation",
        ),
    ]
    decisions, confidence_flags = _confidence_gate_decisions(spans, proposals, {}, [])
    protected = _confidence_held_source_cue_ids(confidence_flags)
    assert protected == {uncertain_cue_id}

    changed, flags = apply_adjudication_decisions(
        cues, spans, decisions, StyleProfile(), protected_cue_ids=protected,
    )
    mapped = _alignment_with_decision_words(
        alignment, decisions, spans, source_cues=cues, words=words,
        protected_cue_ids=protected,
    )

    assert [cue.plain_text for cue in changed] == expected
    assert mapped.cue_word_indices == ownership
    assert not any(flag.kind == "adjudication_replacement_ownership_held" for flag in flags)
    assert not any(flag.kind == "adjudication_word_mapping_held" for flag in mapped.flags)


@pytest.mark.parametrize("protected_ids, held", [({1, 3}, False), ({2}, True)])
def test_replacement_context_does_not_expand_missing_audio_source_ownership(protected_ids, held):
    span = DivergenceSpan(
        case_id="replacement", cue_ids=[2], srt_text="old", asr_text="new",
        srt_token_indices=[1], asr_word_indices=[1], start=2.0, end=2.3,
        left_anchor_cue_id=1, right_anchor_cue_id=3,
        left_anchor_end=1.2, right_anchor_start=3.0,
    )

    selected, decisions, flags = _hold_incomplete_source_insertions(
        [span], {}, missing_audio_cue_ids=protected_ids,
    )

    assert bool(decisions) is held
    assert selected == ([] if held else [span])
    if held:
        assert decisions[0].verdict == "keep_srt"
        assert decisions[0].final_text == "old"
        assert flags[0].cue_ids == [2]
    else:
        assert flags == []


@pytest.mark.parametrize("anchor", ["left_anchor_cue_id", "right_anchor_cue_id"])
def test_pure_insertion_still_holds_when_a_neighboring_source_cue_is_protected(anchor):
    span = DivergenceSpan(
        case_id="insertion", cue_ids=[], srt_text="", asr_text="E",
        asr_word_indices=[1], start=2.0, end=2.3,
        **{anchor: 2},
    )

    selected, decisions, flags = _hold_incomplete_source_insertions(
        [span], {}, missing_audio_cue_ids={2},
    )

    assert selected == []
    assert decisions[0].verdict == "keep_srt"
    assert flags[0].kind == "missing_audio_source_cue_held"


def _actual_w01_tail_replacements():
    # The exact 13s w01 probe succeeded with diarization disabled. These are
    # actual returned words; missing speaker/confidence values remain missing.
    cues = [
        Cue(index=190, start_ms=798830, end_ms=800190,
            lines=["Depois de acordar, eu descobri"]),
        Cue(index=191, start_ms=802590, end_ms=803920,
            lines=["que eu estava sem roupa,"]),
        Cue(index=192, start_ms=805280, end_ms=806760,
            lines=["ele estava tirando fotos minhas."]),
    ]
    values = [
        ("E", 799.04, 799.12), ("depois", 799.24, 799.439),
        ("que", 799.48, 799.56), ("eu", 799.58, 799.66),
        ("acordei...", 799.72, 800.24), ("percebi", 802.84, 803.2),
        ("que", 803.24, 803.299), ("tava", 803.4, 803.579),
        ("pelada.", 803.68, 804.12), ("E", 805.44, 805.52),
        ("que", 805.56, 805.659), ("ele", 805.7, 805.839),
        ("tinha", 805.88, 806.04), ("tirado", 806.08, 806.359),
        ("foto.", 806.48, 806.839),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in values]
    return cues, words


def test_actual_mai_tail_replacements_split_before_percebi_and_e_que():
    from dubsync.pipeline import _adlib_cue_ids_by_case

    cues, words = _actual_w01_tail_replacements()
    alignment = align_cues_to_words(cues, words)
    spans = alignment.divergence_spans
    decisions = [AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=1.0, reason="diagnostic replay of exact provider words",
    ) for span in spans]
    adlib_ids, _ = _adlib_cue_ids_by_case(cues, spans, decisions, alignment.unmatched_cue_ids)
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, _ = apply_adjudication_decisions(cues, spans, decisions, profile, adlib_ids, words=words)
    mapped = _alignment_with_decision_words(
        alignment, decisions, spans, adlib_ids, source_cues=cues, words=words,
    )

    assert [alphanumeric_signature(cue.plain_text) for cue in changed] == [
        ["e", "depois", "que", "eu", "acordei"],
        ["percebi", "que", "tava", "pelada"],
        ["e", "que", "ele", "tinha", "tirado", "foto"],
    ]
    assert mapped.cue_word_indices == {
        190: [0, 1, 2, 3, 4], 191: [5, 6, 7, 8], 192: [9, 10, 11, 12, 13, 14],
    }
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile)
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (profile.snap_floor(799040), profile.snap_ceil(800240)),
        (profile.snap_floor(802840), profile.snap_ceil(804120)),
        (profile.snap_floor(805440), profile.snap_ceil(806839)),
    ]


@pytest.mark.parametrize("drop_policy", ["keep_flagged", "remove"])
def test_actual_w03_empty_cross_cue_edit_preserves_both_matched_residues(drop_policy):
    cues = [
        Cue(index=212, start_ms=851240, end_ms=851830, lines=["Não pode!"]),
        Cue(index=213, start_ms=852070, end_ms=853190,
            lines=["Não pode de jeito nenhum fazer denúncia."]),
    ]
    span = DivergenceSpan(
        case_id="w03-omitted-prefixes", cue_ids=[212, 213],
        srt_text="pode Não pode", asr_text="", srt_token_indices=[1, 2, 3],
        start=851.559, end=852.16,
        left_anchor_cue_id=212, right_anchor_cue_id=213,
        left_anchor_end=851.559, right_anchor_start=852.16,
    )
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="", confidence=1.0,
        reason="exact MAI words retain Não and de jeito nenhum, not the indexed source prefixes",
    )

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(drop_policy=drop_policy),
    )

    assert [cue.plain_text for cue in changed] == ["Não!", "de jeito nenhum fazer denúncia."]
    assert [(cue.start_ms, cue.end_ms) for cue in changed] == [(cue.start_ms, cue.end_ms) for cue in cues]
    assert not any(flag.kind in {"dropped_adjudicated_cue", "dropped_line_candidate"} for flag in flags)


@pytest.mark.parametrize("evidence", ["punctuation_word", "conflicting_anchors", "mixed_speakers", "multiple_gaps"])
def test_ambiguous_acoustic_tail_split_holds_text_and_word_mapping(evidence):
    if evidence == "punctuation_word":
        # A nonempty punctuation-only Word survives provider parsing and is
        # included by the aligner in the exact replacement interval.
        cues = [
            Cue(index=10, start_ms=0, end_ms=600, lines=["eu descobri"]),
            Cue(index=11, start_ms=2200, end_ms=2800, lines=["que aconteceu"]),
        ]
        words = [Word(text=text, start=start, end=end) for text, start, end in [
            ("eu", 0.0, 0.1), ("acordei", 0.2, 0.3), ("percebi", 0.4, 0.6),
            ("...", 2.0, 2.1), ("que", 2.2, 2.3), ("aconteceu", 2.4, 2.7),
        ]]
        source_text = "descobri"
    else:
        cues, words = _actual_w01_tail_replacements()
        source_text = "eu estava sem roupa"
        if evidence == "conflicting_anchors":
            words[6] = words[6].model_copy(update={"speaker_id": "left_actor"})
            words[11] = words[11].model_copy(update={"speaker_id": "right_actor"})
        elif evidence == "mixed_speakers":
            words[7] = words[7].model_copy(update={"speaker_id": "left_actor"})
            words[9] = words[9].model_copy(update={"speaker_id": "right_actor"})
        else:
            words[8] = words[8].model_copy(update={"start": 804.5, "end": 804.6})
    alignment = align_cues_to_words(cues, words)
    span, = [span for span in alignment.divergence_spans if span.srt_text == source_text]
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=1.0, reason="accepted words with ambiguous acoustic ownership",
    )

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(), words=words,
    )
    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words,
    )

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in flags] == ["adjudication_replacement_ownership_held"]
    assert flags[0].new_text == span.asr_text
    assert any(flag.kind == "adjudication_word_mapping_held" for flag in mapped.flags)


@pytest.mark.parametrize("guard", ["missing_audio", "confidence"])
def test_acoustic_split_holds_both_parts_when_the_new_destination_is_protected(guard):
    cues, words = _actual_w01_tail_replacements()
    alignment = align_cues_to_words(cues, words)
    span, = [span for span in alignment.divergence_spans if span.srt_text == "descobri"]
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=1.0, reason="diagnostic split with a protected destination",
    )
    if guard == "missing_audio":
        alignment = alignment.model_copy(update={
            "diagnostics": alignment.diagnostics.model_copy(update={"missing_audio_cue_ids": [191]}),
        })

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(), protected_cue_ids={191}, words=words,
    )
    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words,
        protected_cue_ids={191} if guard == "confidence" else None,
    )

    assert changed == cues
    assert mapped.cue_word_indices == alignment.cue_word_indices
    assert [flag.kind for flag in flags] == ["adjudication_replacement_ownership_held"]
    assert flags[0].new_text == "acordei... percebi"
    assert any(flag.kind == "adjudication_word_mapping_held" for flag in mapped.flags)


@pytest.mark.parametrize("drop_policy", ["keep_flagged", "remove"])
@pytest.mark.parametrize("partial", [False, True])
def test_indexed_empty_multicue_edit_keeps_the_full_cue_drop_policy(drop_policy, partial):
    if partial:
        cues = [
            Cue(index=1, start_ms=0, end_ms=500, lines=["one old"]),
            Cue(index=2, start_ms=600, end_ms=1100, lines=["delete all"]),
            Cue(index=3, start_ms=1200, end_ms=1700, lines=["old three"]),
        ]
        indices, source_text, full_ids = [1, 2, 3, 4], "old delete all old", [2]
        expected = ["one", "delete all", "three"] if drop_policy == "keep_flagged" else ["one", "three"]
    else:
        cues = [
            Cue(index=1, start_ms=0, end_ms=500, lines=["delete one"]),
            Cue(index=2, start_ms=600, end_ms=1100, lines=["delete two"]),
        ]
        indices, source_text, full_ids = [0, 1, 2, 3], "delete one delete two", [1, 2]
        expected = ["delete one", "delete two"] if drop_policy == "keep_flagged" else []
    span = DivergenceSpan(
        case_id="empty", cue_ids=[cue.index for cue in cues], srt_text=source_text,
        asr_text="", srt_token_indices=indices, start=0.1, end=1.4,
    )
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="", confidence=1.0,
        reason="exact indexed source omission",
    )

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(drop_policy=drop_policy),
    )

    assert [cue.plain_text for cue in changed] == expected
    dropped, = [flag for flag in flags if flag.kind.startswith("dropped_")]
    assert dropped.cue_ids == full_ids
    assert dropped.kind == ("dropped_line_candidate" if drop_policy == "keep_flagged" else "dropped_adjudicated_cue")


@pytest.mark.parametrize("verdict", ["use_audio", "hybrid"])
@pytest.mark.parametrize("already_mapped", [False, True])
def test_empty_multicue_decision_does_not_time_residue_with_rejected_words(verdict, already_mapped):
    cues = [
        Cue(index=1, start_ms=0, end_ms=800, lines=["Hello extra."]),
        Cue(index=2, start_ms=1200, end_ms=2300, lines=["extra world."]),
    ]
    words = [
        Word(text="Hello", start=0.1, end=0.4),
        Word(text="noise", start=1.0, end=1.2),
        Word(text="world", start=2.0, end=2.3),
    ]
    alignment = align_cues_to_words(cues, words)
    span, = alignment.divergence_spans
    assert span.srt_text == "extra extra"
    if already_mapped:
        alignment = alignment.model_copy(update={"cue_word_indices": {1: [0, 1], 2: [2]}})
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict=verdict, final_text="", confidence=1.0,
        reason="the divergent sound is not spoken dialogue",
    )
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], profile, words=words)
    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], source_cues=cues, words=words,
    )
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile)

    assert [cue.plain_text for cue in changed] == ["Hello.", "world."]
    assert mapped.cue_word_indices == {1: [0], 2: [2]}
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (profile.snap_floor(100), profile.snap_ceil(400)),
        (profile.snap_floor(2000), profile.snap_ceil(2300)),
    ]


@pytest.mark.parametrize("screen_text", [False, True])
def test_empty_multicue_edit_holds_when_its_source_cannot_be_safely_reconstructed(screen_text):
    cues = [
        Cue(index=1, start_ms=0, end_ms=800, lines=["Hello extra."]),
        Cue(index=2, start_ms=1200, end_ms=2300,
            lines=["[ON SCREEN]", "extra world."] if screen_text else ["extra world."]),
    ]
    span = DivergenceSpan(
        case_id="unsafe-empty", cue_ids=[1, 2], asr_text="noise",
        srt_text="extra extra" if screen_text else "wrong source",
        srt_token_indices=[1, 2], asr_word_indices=[1], start=1.0, end=1.2,
    )
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="", confidence=1.0,
        reason="source edit requires review",
    )

    changed, flags = apply_adjudication_decisions(
        cues, [span], [decision], StyleProfile(drop_policy="remove"),
    )

    assert changed == cues
    assert [flag.kind for flag in flags] == [
        "screen_text_adjudication_held" if screen_text else "adjudication_span_edit_held",
    ]


def test_actual_case84_approved_local_rewrite_does_not_hold_its_unrelated_next_anchor():
    from dubsync.pipeline import _adlib_cue_ids_by_case, _timing_evidence_held_cue_ids

    cues = [
        Cue(index=197, start_ms=818710, end_ms=820880, lines=["Eu lutei com todas as minhas forças"]),
        Cue(index=198, start_ms=821830, end_ms=823120, lines=["e não deixei que ele conseguisse"]),
        Cue(index=199, start_ms=823950, end_ms=825240, lines=["Ele ainda me acalmou,"]),
    ]
    # Full MAI words775–790, with the actual hesitation inside cue197.
    values = [
        ("E", 818.96, 819.039), ("eu", 819.12, 819.219),
        ("tentei", 819.3199999999999, 819.639), ("com", 819.76, 819.92),
        ("toda", 820.3199999999999, 820.52), ("a", 820.54, 820.58),
        ("minha...", 820.64, 820.94), ("força", 822.0, 822.28),
        ("pra", 822.3199999999999, 822.42), ("impedir.", 822.52, 822.999),
        ("E", 824.0, 824.08), ("ele", 824.12, 824.279),
        ("ainda", 824.3199999999999, 824.479), ("tentou", 824.52, 824.6990000000001),
        ("me", 824.74, 824.8), ("acalmar...", 824.84, 825.319),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None, speaker_id=None)
             for text, start, end in values]
    alignment = align_cues_to_words(cues, words)
    spans = alignment.divergence_spans
    decisions = [AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=0.98, reason="fresh diagnostic replay of actual approved wording",
    ) for span in spans]
    rewrite, = [span for span in spans if span.srt_text == "todas as minhas forças"]
    assert rewrite.start - rewrite.left_anchor_end == pytest.approx(0.4)
    assert rewrite.right_anchor_start - rewrite.end == pytest.approx(1.001)
    # Both saved models approved this exact lexical wording without the ASR ellipsis.
    decisions = [item.model_copy(update={"final_text": "toda a minha força pra impedir"})
                 if item.case_id == rewrite.case_id else item for item in decisions]
    adlib_ids, _ = _adlib_cue_ids_by_case(cues, spans, decisions, alignment.unmatched_cue_ids)
    profile = StyleProfile(min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)

    changed, flags = apply_adjudication_decisions(cues, spans, decisions, profile, adlib_ids, words=words)
    mapped = _alignment_with_decision_words(alignment, decisions, spans, adlib_ids, source_cues=cues, words=words)
    held = _timing_evidence_held_cue_ids(mapped.flags)
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile, protected_cue_ids=held)

    assert [alphanumeric_signature(cue.plain_text) for cue in changed] == [
        ["e", "eu", "tentei", "com", "toda", "a", "minha", "forca", "pra", "impedir"],
        ["e"], ["ele", "ainda", "tentou", "me", "acalmar"],
    ]
    assert mapped.cue_word_indices == {197: list(range(10)), 198: [10], 199: list(range(11, 16))}
    assert held == set()
    assert not any("held" in flag.kind for flag in [*flags, *mapped.flags])
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (profile.snap_floor(words[0].start * 1000), profile.snap_ceil(words[9].end * 1000)),
        (profile.snap_floor(words[10].start * 1000), profile.snap_ceil(words[10].end * 1000)),
        (profile.snap_floor(words[11].start * 1000), profile.snap_ceil(words[15].end * 1000)),
    ]
