from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import wave

import pytest

from dubsync.missing_dialogue_reconciliation import (
    MissingDialogueEvidence, reconciliation_context, validate_reconciliation_artifact,
)
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SourcePairEvidence, SpeechRegion, TokenMatch, Word
from dubsync.source_pair_timing import (
    SOURCE_PAIR_TIMING_POLICY_VERSION, SOURCE_PAIR_TIMING_PREFIX,
    build_source_pair_timing_questions, reconcile_source_pair_timing,
)
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature, tokenize_cues


def _case(provider="mai"):
    source = [
        Cue(index=9, start_ms=40400, end_ms=41900, lines=["一席くらい"]),
        Cue(index=10, start_ms=42030, end_ms=42790, lines=["いいでしょう？"]),
        Cue(index=11, start_ms=42800, end_ms=43130, lines=["ははは"]),
        Cue(index=12, start_ms=43200, end_ms=44100, lines=["お嬢さんよ"]),
    ]
    speaker, other = ("chunk_1:0", "chunk_1:1") if provider == "mai" else ("speaker_0", "speaker_1")
    current = [source[0].with_timing(40700, 41334),
               source[1].with_lines(["いいでしょう？" if provider == "mai" else "いいでしょ？"]).with_timing(41334, 41900),
               source[2], source[3].with_timing(42900, 43567)]
    spoken = [("一席くらい", 40.72, 41.36 if provider == "scribe" else 41.299, speaker),
              ("い", 41.36, 41.38 if provider == "scribe" else 41.399, speaker),
              ("い", 41.40, 41.48, speaker), ("で", 41.48, 41.559, speaker),
              ("し", 41.60, 41.639, speaker), ("ょ", 41.64, 41.84 if provider == "scribe" else 41.679, speaker)]
    if provider == "mai":
        spoken.append(("う", 41.68, 41.76, speaker))
    first_indices = list(range(1, len(spoken)))
    punctuation_index = len(spoken)
    spoken.append(("？" if provider == "mai" else "。", 41.80 if provider == "mai" else 41.84,
                   41.84 if provider == "mai" else 41.841, speaker))
    next_index = len(spoken)
    spoken.extend([(text, start, end, other) for text, start, end in [
        ("お", 42.905, 43.039), ("嬢", 43.08, 43.159), ("さ", 43.28, 43.34),
        ("ん", 43.34, 43.419), ("よ", 43.44, 43.519),
    ]])
    words = [Word(text=text, start=start, end=end, speaker_id=actor) for text, start, end, actor in spoken]
    ownership = {9: [0], 10: first_indices, 11: [], 12: list(range(next_index, len(words)))}
    tokens = tokenize_cues(source)
    matches = []
    for cue_id in (9, 10, 12):
        cursor = 0
        for word_index in ownership[cue_id]:
            for normalized in alphanumeric_signature(words[word_index].text):
                own_tokens = [token for token in tokens if token.cue_id == cue_id]
                token = next(token for token in own_tokens[cursor:] if token.normalized == normalized)
                cursor = own_tokens.index(token) + 1
                matches.append(TokenMatch(cue_id=cue_id, srt_token_index=token.token_index,
                                          asr_word_index=word_index, score=1))
    target_tokens = [token.token_index for token in tokens if token.cue_id == 11]
    parent = DivergenceSpan(case_id="parent", cue_ids=[11] if provider == "mai" else [10, 11],
                            srt_text="ははは" if provider == "mai" else "うははは",
                            srt_token_indices=target_tokens if provider == "mai" else [target_tokens[0] - 1, *target_tokens],
                            asr_text="？" if provider == "mai" else "",
                            asr_word_indices=[punctuation_index] if provider == "mai" else [],
                            start=41.8 if provider == "mai" else 41.84, end=41.84 if provider == "mai" else 42.905)
    alignment = AlignmentResult(cue_word_indices=ownership, token_matches=matches, divergence_spans=[parent],
                                unmatched_cue_ids=[11], diagnostics={"missing_audio_cue_ids": [11]})
    regions = [SpeechRegion(start=39.565, end=42.185), SpeechRegion(start=42.265, end=42.465),
               SpeechRegion(start=42.905, end=43.605)]
    return current, source, alignment, words, regions


def _questions(case, **kwargs):
    return build_source_pair_timing_questions(*case, audio_duration_seconds=50, **kwargs)


def _pair_evidence(question, **updates):
    payload = dict(first_text=question.accepted_pair_texts[0], second_text=question.accepted_pair_texts[1],
                   sequence="first_then_second", voice_relation="same", intervening_speech=False,
                   candidate_complete=True, candidate_start_clipped=False, candidate_end_clipped=False,
                   laugh_outside_candidate=False, candidate_audio_id=question.span.case_id + "-candidate")
    payload.update(updates)
    return SourcePairEvidence(**payload)


def _decision(question, **updates):
    payload = dict(case_id=question.span.case_id, verdict="keep_srt", final_text=question.span.srt_text,
                   heard_text=question.span.srt_text, evidence="heard_clearly", confidence=1,
                   speaker=question.anchor_speaker_id, reason="Unit fixture: both parts are one voice in order.",
                   source_pair_evidence=_pair_evidence(question))
    payload.update(updates)
    return AdjudicationDecision(**payload)


def _resolve(case, questions, *, decisions=None, flags=(), profile=None):
    return reconcile_source_pair_timing(*case, questions,
                                       [_decision(q) for q in questions] if decisions is None else decisions,
                                       profile or StyleProfile(fps=30), flags=list(flags))


@pytest.mark.parametrize("provider", ["mai", "scribe"])
def test_complete_native_pair_uses_first_owned_onset_and_whole_raw_chain(provider):
    case = _case(provider)
    before = deepcopy(case)
    questions = _questions(case)
    assert len(questions) == 1
    q = questions[0]
    assert q.span.cue_ids == [10, 11]
    assert q.span.srt_text == case[0][1].text + "\n" + case[0][2].text
    assert q.original_pair_texts == ("いいでしょう？", "ははは")
    assert q.accepted_pair_texts == (case[0][1].text, "ははは")
    assert (q.utterance_start_seconds, q.utterance_end_seconds) == (41.36, 42.465)
    assert q.span.start <= 40.72 and q.span.end >= 43.519
    result = _resolve(case, questions, flags=[
        QCFlag(kind="missing_audio_source_cue_held", cue_ids=[11], message="Previously unanchored."),
        QCFlag(kind="timing_evidence_held", cue_ids=[11], message="No word interval."),
    ])
    assert [cue.index for cue in result.cues] == [9, 10, 12]
    assert result.cues[1].lines == [case[0][1].text, "ははは"]
    assert result.cues[1].start_ms == 41334 and result.cues[1].end_ms == 42534
    assert result.cues[1].end_ms < result.cues[2].start_ms
    assert result.spoken_spans == {10: (41360, 42465)}
    assert result.resolved_cue_ids == {10, 11}
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert result.alignment.unmatched_cue_ids == [] and result.alignment.diagnostics.missing_audio_cue_ids == []
    assert result.outcomes[0]["source_cue_ids"] == [10, 11]
    assert result.outcomes[0]["merged_cue_id"] == 10
    assert result.outcomes[0]["outcome"] == "audio_confirmed_source_pair"
    assert [(f.kind, f.cue_ids) for f in result.flags] == [("source_pair_audio_reconciled", [10, 11])]
    assert result.cues[0] == case[0][0] and result.cues[2] == case[0][3]
    assert case == before


@pytest.mark.parametrize("provider", ["mai", "scribe"])
def test_complete_two_voice_exchange_retains_separate_lines_and_no_global_speaker(provider):
    case = _case(provider)
    case[0][1].speaker_id, case[0][1].character = "original-speaker", "Original character"
    before = deepcopy(case)
    questions = _questions(case)
    q = questions[0]
    decision = _decision(q, speaker=None, character="Unverified model character",
                         source_pair_evidence=_pair_evidence(q, voice_relation="different"))
    result = _resolve(case, questions, decisions=[decision])
    assert SOURCE_PAIR_TIMING_POLICY_VERSION == 2 and SOURCE_PAIR_TIMING_PREFIX == "source-pair-timing-v2-"
    assert q.span.case_id == "source-pair-timing-v2-10-11"
    assert [cue.index for cue in result.cues] == [9, 10, 12]
    assert result.cues[1].lines == ["- " + case[0][1].text, "- ははは"]
    assert (result.cues[1].start_ms, result.cues[1].end_ms) == (41334, 42534)
    assert result.cues[1].speaker_id is None and result.cues[1].character is None
    assert result.spoken_spans == {10: (41360, 42465)}
    assert result.alignment.cue_word_indices == case[2].cue_word_indices
    assert result.resolved_cue_ids == {10, 11}
    assert [(flag.kind, flag.cue_ids, flag.severity) for flag in result.flags] == [
        ("source_exchange_audio_reconciled", [10, 11], "info")]
    outcome = result.outcomes[0]
    assert outcome["outcome"] == "audio_confirmed_source_exchange" and outcome["merged_cue_id"] == 10
    assert outcome["voice_relation"] == "different"
    assert outcome["line_provenance"] == [
        {"line": 1, "source_cue_id": 10, "voice": "first"},
        {"line": 2, "source_cue_id": 11, "voice": "second"},
    ]
    assert case == before


@pytest.mark.parametrize("relation", ["same", "different"])
@pytest.mark.parametrize("field,value", [
    ("first_text", "いいでしょ？"), ("second_text", "ふふふ"),
    ("sequence", "second_then_first"), ("sequence", "overlapping"), ("sequence", "unclear"),
    ("voice_relation", "unclear"), ("intervening_speech", True), ("intervening_speech", None),
    ("candidate_complete", False), ("candidate_complete", None),
    ("candidate_start_clipped", True), ("candidate_start_clipped", None),
    ("candidate_end_clipped", True), ("candidate_end_clipped", None),
    ("laugh_outside_candidate", True), ("laugh_outside_candidate", None),
    ("candidate_audio_id", "source-pair-timing-v1-10-11-candidate"),
    ("candidate_audio_id", "source-pair-timing-v2-10-11"), ("candidate_audio_id", ""),
])
def test_each_typed_candidate_finding_is_required_before_display_union(field, value, relation):
    case = _case()
    q = _questions(case)[0]
    updates = {"voice_relation": relation, field: value}
    evidence = _pair_evidence(q, **updates)
    result = _resolve(case, [q], decisions=[_decision(q, source_pair_evidence=evidence,
                      speaker=q.anchor_speaker_id if relation == "same" else None)])
    assert result.cues == case[0] and result.resolved_cue_ids == set() and result.spoken_spans == {}
    assert result.outcomes[0]["outcome"] == "pair_candidate_evidence_unconfirmed"


@pytest.mark.parametrize("speaker", ["chunk_1:0", None])
def test_legacy_v1_hearing_without_typed_candidate_evidence_cannot_approve_v2(speaker):
    case = _case()
    q = _questions(case)[0]
    result = _resolve(case, [q], decisions=[_decision(q, speaker=speaker, source_pair_evidence=None)])
    assert result.cues == case[0] and result.resolved_cue_ids == set()
    assert result.outcomes[0]["outcome"] == "pair_candidate_evidence_unconfirmed"


@pytest.mark.parametrize("fault", ["duplicate_answer", "conflicting_answer", "duplicate_question", "hybrid_outage"])
def test_duplicate_or_failed_native_evidence_cannot_authorize_pair_union(fault):
    case = _case()
    q = _questions(case)[0]
    questions, decisions, flags = [q], [_decision(q)], []
    if fault == "duplicate_answer": decisions.append(_decision(q))
    elif fault == "conflicting_answer": decisions.append(_decision(q, evidence="heard_unclear"))
    elif fault == "duplicate_question": questions.append(q)
    elif fault == "hybrid_outage":
        flags = [QCFlag(kind="adjudication_review_unavailable", cue_ids=[10, 11],
                        old_text=q.span.srt_text, start=q.span.start, end=q.span.end,
                        message="The configured native reviewer is unavailable.")]
    result = _resolve(case, questions, decisions=decisions, flags=flags)
    assert result.cues == case[0] and result.resolved_cue_ids == set() and result.spoken_spans == {}


@pytest.mark.parametrize("max_lines,max_width", [(1, 26), (2, 14)])
def test_exchange_line_limit_is_checked_after_adding_both_voice_dashes(max_lines, max_width):
    case = _case()
    q = _questions(case)[0]
    result = _resolve(case, [q], decisions=[_decision(q, speaker=None,
                      source_pair_evidence=_pair_evidence(q, voice_relation="different"))],
                      profile=StyleProfile(fps=30, max_lines_per_cue=max_lines, max_chars_per_line=max_width))
    assert result.cues == case[0] and result.resolved_cue_ids == set()
    assert result.outcomes[0]["outcome"] == "no_safe_display_union"


@pytest.mark.parametrize("relation,speaker", [("same", None), ("same", "chunk_1:1"), ("different", "chunk_1:0")])
def test_generic_speaker_must_be_consistent_with_typed_voice_relation(relation, speaker):
    case = _case()
    q = _questions(case)[0]
    result = _resolve(case, [q], decisions=[_decision(q, speaker=speaker,
                      source_pair_evidence=_pair_evidence(q, voice_relation=relation))])
    assert result.cues == case[0] and result.resolved_cue_ids == set()


def test_bad_customer_times_do_not_select_or_limit_the_pair_envelope():
    case = _case()
    shifted = (case[0], [cue.with_timing(cue.start_ms + 20000, cue.end_ms + 20000) for cue in case[1]], *case[2:])
    left, right = _questions(case)[0], _questions(shifted)[0]
    assert (left.utterance_start_seconds, left.utterance_end_seconds) == (right.utterance_start_seconds, right.utterance_end_seconds)
    assert _resolve(case, [left]).spoken_spans == _resolve(shifted, [right]).spoken_spans


@pytest.mark.parametrize("fault", [
    "unknown_speaker", "mixed_speaker", "unknown_next", "same_next", "different_chunk_next", "shared_first", "owned_laugh",
    "foreign_word", "foreign_same_speaker", "overlapping_first_words", "uncertain_first", "collapsed_first", "low_confidence",
    "first_wording_mismatch", "target_wording_changed", "ordinary_short_word", "target_markup", "first_markup", "target_song",
    "target_annotation", "nonadjacent_source", "missing_target", "no_parent", "lexical_parent", "shared_punctuation",
    "protected_first", "missing_regions", "separate_later_burst", "next_in_same_region", "no_onset_region", "long_tail",
    "nonfinite_word", "nonfinite_region", "unresolved",
])
def test_source_pair_requires_exclusive_complete_bounded_acoustic_evidence(fault):
    case = _case()
    current, source, alignment, words, regions = case
    kwargs = {}
    if fault == "unknown_speaker": words[1].speaker_id = None
    elif fault == "mixed_speaker": words[2].speaker_id = "chunk_1:2"
    elif fault in {"unknown_next", "same_next", "different_chunk_next"}:
        label = None if fault == "unknown_next" else "chunk_1:0" if fault == "same_next" else "chunk_2:1"
        for index in alignment.cue_word_indices[12]: words[index].speaker_id = label
    elif fault == "shared_first": alignment.cue_word_indices[99] = [1]
    elif fault == "owned_laugh": alignment.cue_word_indices[11] = [1]
    elif fault in {"foreign_word", "foreign_same_speaker"}:
        words.append(Word(text="Other", start=42, end=42.1, speaker_id="chunk_1:0" if fault.endswith("same_speaker") else "chunk_1:2"))
    elif fault == "overlapping_first_words": words[2].start = words[1].start + .01
    elif fault == "uncertain_first": kwargs["uncertain_word_indices"] = {1}
    elif fault == "collapsed_first":
        for position, index in enumerate(alignment.cue_word_indices[10]):
            words[index].start, words[index].end = 41.36 + position * .001, 41.361 + position * .001
    elif fault == "low_confidence": words[1].confidence = .5
    elif fault == "first_wording_mismatch": current[1] = current[1].with_lines(["いいでしょ？"])
    elif fault == "target_wording_changed": current[2] = current[2].with_lines(["ふふふ"])
    elif fault == "ordinary_short_word": current[2] = source[2] = source[2].with_lines(["Tao"])
    elif fault == "target_markup": current[2] = source[2] = source[2].with_lines(["<i>ははは</i>"])
    elif fault == "first_markup": current[1] = current[1].with_lines(["<i>いいでしょう？</i>"])
    elif fault == "target_song": current[2] = source[2] = source[2].with_lines(["♪ははは"])
    elif fault == "target_annotation": current[2] = source[2] = source[2].with_lines(["[ははは]"])
    elif fault == "nonadjacent_source": source.insert(2, Cue(index=77, start_ms=42100, end_ms=42200, lines=["Other"]))
    elif fault == "missing_target": current.pop(2)
    elif fault == "no_parent": alignment.divergence_spans.clear()
    elif fault == "lexical_parent": words[alignment.divergence_spans[0].asr_word_indices[0]].text = "Hah"
    elif fault == "shared_punctuation": alignment.cue_word_indices[99] = alignment.divergence_spans[0].asr_word_indices
    elif fault == "protected_first": kwargs["protected_cue_ids"] = {10}
    elif fault == "missing_regions": regions.clear()
    elif fault == "separate_later_burst": regions.insert(2, SpeechRegion(start=42.7, end=42.75))
    elif fault == "next_in_same_region": regions[:] = [SpeechRegion(start=39.565, end=43.605)]
    elif fault == "no_onset_region": regions[0] = SpeechRegion(start=41.4, end=42.185)
    elif fault == "long_tail":
        for index in alignment.cue_word_indices[12]: words[index].start += 3; words[index].end += 3
        regions[:] = [SpeechRegion(start=39.565, end=44.6), SpeechRegion(start=45.905, end=46.605)]
    elif fault == "nonfinite_word": words[1].end = float("inf")
    elif fault == "nonfinite_region": regions[1].end = float("inf")
    elif fault == "unresolved": alignment.diagnostics.unresolved = True
    assert _questions(case, **kwargs) == []


@pytest.mark.parametrize("gap_ms,accepted", [(199, True), (200, False), (201, False)])
def test_pair_uses_existing_strict_speech_chain_gap(gap_ms, accepted):
    case = _case()
    case[4][1] = SpeechRegion(start=42.185 + gap_ms / 1000, end=42.465)
    assert bool(_questions(case)) is accepted


def test_next_actor_preroll_cannot_conceal_an_additional_untranscribed_chuckle():
    case = _case()
    # Native wording/voice confirmation cannot prove whether this 205 ms of
    # activity is the next actor's onset or another part of the missing laugh.
    case[4][-1] = SpeechRegion(start=42.700, end=43.605)
    assert _questions(case) == []


@pytest.mark.parametrize("fault", ["unclear", "null_speaker", "different_speaker", "partial_hearing", "reordered_hearing", "omission",
    "changed_current", "changed_source", "changed_word", "changed_ownership", "changed_region", "changed_parent", "unavailable",
    "changed_receipt", "too_wide"])
def test_pair_merge_requires_fresh_exact_scoped_hearing_and_unchanged_proof(fault):
    case = _case()
    questions = _questions(case)
    q = questions[0]
    changes, flags, profile = {}, [], None
    if fault == "unclear": changes["evidence"] = "heard_unclear"
    elif fault == "null_speaker": changes["speaker"] = None
    elif fault == "different_speaker": changes["speaker"] = "chunk_1:1"
    elif fault == "partial_hearing": changes["heard_text"] = "ははは"
    elif fault == "reordered_hearing": changes["heard_text"] = "ははは いいでしょう？"
    elif fault == "omission": changes.update(verdict="use_audio", final_text="", heard_text="")
    elif fault == "changed_current": case[0][1] = case[0][1].with_lines(["違います"])
    elif fault == "changed_source": case[1][2] = case[1][2].with_lines(["ふふふ"])
    elif fault == "changed_word": case[3][1].end += .01
    elif fault == "changed_ownership": case[2].cue_word_indices[99] = [1]
    elif fault == "changed_region": case[4][1] = SpeechRegion(start=42.265, end=42.5)
    elif fault == "changed_parent": case[2].divergence_spans[0] = case[2].divergence_spans[0].model_copy(update={"asr_text": "Other"})
    elif fault == "unavailable": flags = [QCFlag(kind="adjudication_audio_unavailable", cue_ids=[10, 11], message="No scoped audio.")]
    elif fault == "changed_receipt": questions = [replace(q, utterance_end_seconds=42.5)]
    elif fault == "too_wide": profile = StyleProfile(max_chars_per_line=3)
    result = _resolve(case, questions, decisions=[_decision(q, **changes)], flags=flags, profile=profile)
    assert result.cues == case[0] and result.resolved_cue_ids == set() and result.spoken_spans == {}


@pytest.mark.parametrize("tamper", ["anchor_speaker", "native_relation", "candidate_audio", "native_first_text"])
def test_pair_receipt_binds_both_source_ids_accepted_text_speaker_and_raw_regions(tamper):
    case = _case("scribe")
    questions = _questions(case)
    context = reconciliation_context(questions, case[1], case[2], case[3], case[4], audio_sha256="unit-audio")
    evidence = MissingDialogueEvidence(questions, context, [_decision(questions[0])], [])
    artifact = evidence.artifact()
    restored, flags = validate_reconciliation_artifact(artifact, context, questions)
    assert restored == evidence.decisions and flags == []
    assert artifact["questions"][0]["source_pair"]["accepted_pair_texts"] == ["いいでしょ？", "ははは"]
    changed = deepcopy(artifact)
    if tamper == "anchor_speaker": changed["questions"][0]["source_pair"]["anchor_speaker_id"] = "speaker_1"
    elif tamper == "native_relation": changed["decisions"][0]["source_pair_evidence"]["voice_relation"] = "different"
    elif tamper == "candidate_audio": changed["decisions"][0]["source_pair_evidence"]["candidate_audio_id"] = "another-candidate"
    elif tamper == "native_first_text": changed["decisions"][0]["source_pair_evidence"]["first_text"] = "いいでしょう？"
    with pytest.raises(ValueError, match="stale or invalid"):
        validate_reconciliation_artifact(changed, context, questions)


def test_an_unowned_missing_target_can_be_asked_but_an_independently_resolved_target_cannot():
    case = _case()
    assert len(_questions(case, protected_cue_ids={11})) == 1
    assert _questions(case, protected_cue_ids={11}, resolved_cue_ids={11}) == []
    questions = _questions(case)
    result = reconcile_source_pair_timing(*case, questions, [_decision(questions[0])], StyleProfile(),
                                         flags=[], resolved_cue_ids={11})
    assert result.cues == case[0] and result.resolved_cue_ids == set()


def test_unrelated_question_failure_does_not_veto_fresh_complete_pair_hearing():
    case = _case()
    questions = _questions(case)
    old_failure = QCFlag(kind="low_confidence_adjudication", cue_ids=[11], old_text="ははは",
                         start=41.8, end=41.84, message="The earlier single-fragment question was unresolved.")
    result = _resolve(case, questions, flags=[old_failure])
    assert result.resolved_cue_ids == {10, 11}
    assert old_failure in result.flags  # The scoped matcher does not erase history.


@pytest.mark.parametrize("relation,returned_speaker,merged", [
    ("same", "speaker_0", True), ("same", None, False), ("same", "speaker_1", False),
    ("different", None, True), ("different", "speaker_0", False), (None, "speaker_0", False),
])
def test_production_native_schema_and_required_audio_route_preserve_scoped_candidate_gate(
    tmp_path, monkeypatch, relation, returned_speaker, merged,
):
    from dubsync import llm_providers
    from dubsync.adjudication import AdjudicationEngine
    from dubsync.audio_snippets import extract_audio_snippets
    from dubsync.llm_providers import GeminiLLMAdapter
    from dubsync.source_pair_audio import SourcePairAudioAdapter

    case = _case("scribe")
    questions = _questions(case)
    q = questions[0]
    source_path = tmp_path / "source.wav"
    with wave.open(str(source_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000 * 50)
    snippet = extract_audio_snippets(source_path, [q.span], tmp_path / "wide", pad_seconds=.1)[0]
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        evidence = _pair_evidence(q, voice_relation=relation) if relation else None
        payload = _decision(q, speaker=returned_speaker, source_pair_evidence=evidence).model_dump(exclude={"confidence"})
        return SimpleNamespace(text=json.dumps({"decisions": [payload]}), usage_metadata={})

    monkeypatch.setattr(llm_providers, "_gemini_generate_json", transport)
    adapter = GeminiLLMAdapter(api_key="unit-only", model="gemini-3.8-flash")
    adapter.set_episode_context(case[0])
    adapter.set_episode_words(case[3])
    hearing = SourcePairAudioAdapter(adapter, questions, source_path, tmp_path / "candidate")
    decisions, flags = AdjudicationEngine(hearing, audio_snippets={q.span.case_id: snippet},
                                         required_audio_case_ids={q.span.case_id}, source_cues=case[0], language="ja").adjudicate([q.span])
    candidate_id = q.span.case_id + "-candidate"
    assert len(calls) == 1, flags
    assert set(calls[0]["audio_snippets"]) == {q.span.case_id, candidate_id}
    assert calls[0]["audio_snippets"][q.span.case_id] == snippet
    candidate = calls[0]["audio_snippets"][candidate_id]
    assert (candidate.start, candidate.end) == (41.36, 42.465)
    with wave.open(candidate.path, "rb") as audio:
        assert (audio.getnframes(), audio.getframerate(), audio.getnchannels()) == (17680, 16000, 1)
    assert hearing.manifest()[0]["candidate_audio_id"] == candidate_id
    sent = json.loads(calls[0]["prompt"])["spans"][0]
    assert sent["srt_text"] == "いいでしょ？\nははは"
    assert sent["asr_text"] == q.span.asr_text
    assert sent["source_pair_hearing_policy"]["anchor_speaker_id"] == "speaker_0"
    assert sent["source_pair_hearing_policy"]["version"] == 2
    assert candidate_id in json.dumps(sent["source_pair_hearing_policy"])
    result = _resolve(case, questions, decisions=decisions, flags=flags)
    assert bool(result.resolved_cue_ids) is merged
    assert decisions[0].speaker == returned_speaker
    assert decisions[0].source_pair_evidence == (_pair_evidence(q, voice_relation=relation) if relation else None)


def test_required_pair_cannot_use_a_text_only_or_missing_audio_answer():
    from dubsync.adjudication import AdjudicationEngine

    class TextOnly:
        def adjudicate(self, spans):
            raise AssertionError("A source pair must reach native audio.")

    case = _case()
    questions = _questions(case)
    decisions, flags = AdjudicationEngine(TextOnly(), required_audio_case_ids={questions[0].span.case_id}).adjudicate([questions[0].span])
    result = _resolve(case, questions, decisions=decisions, flags=flags)
    assert result.cues == case[0] and result.resolved_cue_ids == set()
