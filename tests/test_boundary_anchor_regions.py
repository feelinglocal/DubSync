"""A folded accent match becomes editable only in a newly scoped audio case."""
from copy import deepcopy

import pytest

from dubsync.adjudication_case_cache import case_cache_key
from dubsync.models import Cue, DivergenceSpan, TokenMatch, Word
from dubsync.tokenize import tokenize_cues


def _case(side="left"):
    if side == "left":
        cues = [Cue(index=i, start_ms=a, end_ms=b, lines=[text]) for i, a, b, text in [
            (221, 0, 500, "Antes."), (222, 750, 1830, "E aí, o que você está sentindo agora?"),
            (223, 2320, 3590, "Que cheiro?"), (224, 4100, 4500, "Fim."),
        ]]
        words = [Word(text=text, start=a, end=b, speaker_id="speaker") for text, a, b in [
            ("Antes.", 0, .5), ("É", 1.120, 1.199), ("sério", 1.240, 1.519),
            ("isso.", 1.6, 1.84), ("Esse", 2.920, 3.120), ("cheiro,", 3.240, 3.599), ("Fim.", 4.1, 4.5),
        ]]
        matches = [TokenMatch(cue_id=cue, srt_token_index=source, asr_word_index=audio, score=1)
                   for cue, source, audio in [(221, 0, 0), (222, 1, 1), (223, 10, 5), (224, 11, 6)]]
        span = DivergenceSpan(case_id="case-77", cue_ids=[222, 223],
            srt_text="aí o que você está sentindo agora Que", asr_text="sério isso. Esse",
            srt_token_indices=list(range(2, 10)), asr_word_indices=[2, 3, 4], start=1.240, end=3.120,
            left_anchor_cue_id=222, left_anchor_end=1.199, left_anchor_speaker_id="speaker",
            right_anchor_cue_id=223, right_anchor_start=3.240, right_anchor_speaker_id="speaker")
        ownership = {221: [0], 222: [1], 223: [5], 224: [6]}
    else:
        cues = [Cue(index=i, start_ms=a, end_ms=b, lines=[text]) for i, a, b, text in [
            (221, 0, 500, "Antes."), (222, 1100, 1350, "algo é"), (223, 1800, 2200, "Fim."),
        ]]
        words = [Word(text=text, start=a, end=b, speaker_id="speaker") for text, a, b in [
            ("Antes.", 0, .5), ("sim", 1.120, 1.199), ("E", 1.240, 1.319), ("Fim.", 1.8, 2.2),
        ]]
        matches = [TokenMatch(cue_id=cue, srt_token_index=source, asr_word_index=audio, score=1)
                   for cue, source, audio in [(221, 0, 0), (222, 2, 2), (223, 3, 3)]]
        span = DivergenceSpan(case_id="case-right", cue_ids=[222], srt_text="algo", asr_text="sim",
            srt_token_indices=[1], asr_word_indices=[1], start=1.120, end=1.199,
            left_anchor_cue_id=221, left_anchor_end=.5, left_anchor_speaker_id="speaker",
            right_anchor_cue_id=222, right_anchor_start=1.240, right_anchor_speaker_id="speaker")
        ownership = {221: [0], 222: [2], 223: [3]}
    return {"spans": [span], "matches": matches, "cues": cues, "tokens": tokenize_cues(cues),
            "words": words, "protected_cue_ids": set(), "cue_word_indices": ownership}


def _extend(case):
    from dubsync.adjudication_regions import extend_boundary_anchor_regions
    return extend_boundary_anchor_regions(**case)


@pytest.mark.parametrize("side", ["left", "right"])
def test_adjacent_accent_collision_explicitly_enters_fresh_scope_without_changing_evidence(side):
    case = _case(side)
    before = deepcopy(case)
    old = case["spans"][0]
    result = _extend(case)
    assert case == before
    assert len(result) == 1 and result[0].case_id == old.case_id
    changed = result[0]
    if side == "left":
        assert changed.srt_token_indices == list(range(1, 10))
        assert changed.asr_word_indices == [1, 2, 3, 4]
        assert changed.srt_text == "E aí o que você está sentindo agora Que"
        assert changed.asr_text == "É sério isso. Esse"
        assert changed.start == 1.120 and changed.end == old.end
        assert changed.left_anchor_cue_id is None and changed.left_anchor_end is None
        assert changed.left_anchor_speaker_id is None
        assert changed.right_anchor_cue_id == old.right_anchor_cue_id
    else:
        assert changed.srt_token_indices == [1, 2] and changed.asr_word_indices == [1, 2]
        assert changed.srt_text == "algo é" and changed.asr_text == "sim E"
        assert changed.start == old.start and changed.end == 1.319
        assert changed.right_anchor_cue_id is None and changed.right_anchor_start is None
        assert changed.right_anchor_speaker_id is None
        assert changed.left_anchor_cue_id == old.left_anchor_cue_id
    assert changed.context_before == old.context_before and changed.context_after == old.context_after
    cache = dict(model="test", params={}, policy_context={}, source_cues=case["cues"], source_words=case["words"])
    assert case_cache_key(old, **cache) != case_cache_key(changed, **cache)
    assert _extend({**case, "spans": result}) == result


@pytest.mark.parametrize("fault", [
    "literal_same", "case_only", "different_non_accent", "multitoken_anchor", "shared_owner", "unowned_anchor",
    "wrong_owner", "missing_match", "duplicate_match", "shared_match", "wrong_match_cue", "weak_match",
    "stale_tokens", "wrong_source_text", "wrong_asr_text", "noncontiguous_source", "noncontiguous_audio",
    "missing_anchor_cue", "wrong_anchor_time", "wrong_anchor_speaker", "overlap", "large_gap", "collapsed_anchor",
    "low_confidence", "unknown_speaker", "different_speaker", "same_word_repetition", "protected_cue",
    "protected_other_cue", "song", "bracketed_text", "derived", "anchor_claimed_source", "anchor_claimed_audio",
    "duplicate_cases", "competing_scope", "extra_retained_source", "incorrect_span_start", "incorrect_span_end",
])
def test_uncertain_or_readonly_boundary_evidence_does_not_expand(fault):
    case = _case()
    span = case["spans"][0]
    if fault in {"literal_same", "case_only", "different_non_accent", "multitoken_anchor"}:
        text = {"literal_same": "E", "case_only": "e", "different_non_accent": "A", "multitoken_anchor": "É É"}[fault]
        case["words"][1] = case["words"][1].model_copy(update={"text": text})
    elif fault == "shared_owner": case["cue_word_indices"][999] = [1]
    elif fault == "unowned_anchor": case["cue_word_indices"].pop(222)
    elif fault == "wrong_owner": case["cue_word_indices"] = {999: [1]}
    elif fault == "missing_match": case["matches"].pop(1)
    elif fault in {"duplicate_match", "shared_match"}:
        case["matches"].append(case["matches"][1].model_copy(update={"srt_token_index": 2} if fault == "shared_match" else {}))
    elif fault in {"wrong_match_cue", "weak_match"}:
        case["matches"][1] = case["matches"][1].model_copy(update={"cue_id": 221} if fault == "wrong_match_cue" else {"score": .99})
    elif fault == "stale_tokens": case["tokens"] = case["tokens"][1:]
    elif fault == "wrong_source_text": span.srt_text = "outro texto"
    elif fault == "wrong_asr_text": span.asr_text = "outra coisa"
    elif fault == "noncontiguous_source": span.srt_token_indices.remove(4)
    elif fault == "noncontiguous_audio": span.asr_word_indices.remove(3)
    elif fault == "missing_anchor_cue": span.left_anchor_cue_id = None
    elif fault == "wrong_anchor_time": span.left_anchor_end = 1.190
    elif fault == "wrong_anchor_speaker": span.left_anchor_speaker_id = "other"
    elif fault in {"overlap", "large_gap", "collapsed_anchor", "low_confidence", "unknown_speaker", "different_speaker"}:
        update = {"overlap": {"end": 1.250}, "large_gap": {"start": .620, "end": .799},
                  "collapsed_anchor": {"start": 1.198}, "low_confidence": {"confidence": .6},
                  "unknown_speaker": {"speaker_id": None}, "different_speaker": {"speaker_id": "other"}}[fault]
        case["words"][1] = case["words"][1].model_copy(update=update)
        if "end" in update: span.left_anchor_end = update["end"]
    elif fault == "same_word_repetition":
        case["words"][2] = case["words"][2].model_copy(update={"text": "É"})
        span.asr_text = "É isso. Esse"
    elif fault == "protected_cue": case["protected_cue_ids"] = {222}
    elif fault == "protected_other_cue": case["protected_cue_ids"] = {223}
    elif fault in {"song", "bracketed_text"}:
        text = case["cues"][1].plain_text
        case["cues"][1] = case["cues"][1].with_lines([f"♪ {text} ♪" if fault == "song" else f"[SINAL] {text}"])
        case["tokens"] = tokenize_cues(case["cues"])
    elif fault == "derived": span.case_id = "joint-case-77"
    elif fault in {"anchor_claimed_source", "anchor_claimed_audio"}:
        case["spans"].append(DivergenceSpan(case_id="other", cue_ids=[222], srt_text="E", asr_text="É",
            srt_token_indices=[1] if fault == "anchor_claimed_source" else [],
            asr_word_indices=[1] if fault == "anchor_claimed_audio" else []))
    elif fault in {"duplicate_cases", "competing_scope"}:
        other = span.model_copy(deep=True)
        if fault == "competing_scope": other.case_id = "other-case"
        case["spans"].append(other)
    elif fault == "extra_retained_source":
        span.srt_token_indices.remove(2)
        span.srt_text = "o que você está sentindo agora Que"
    elif fault == "incorrect_span_start": span.start = 1.250
    elif fault == "incorrect_span_end": span.end = 3.130
    before = deepcopy(case)
    assert _extend(case) == case["spans"]
    assert case == before


def test_an_adjacent_case_cannot_borrow_the_other_cues_accent_anchor():
    case = _case()
    case["spans"] = [DivergenceSpan(case_id="previous", cue_ids=[221], srt_text="Antes", asr_text="Antes.",
        srt_token_indices=[0], asr_word_indices=[0], start=0, end=.5,
        right_anchor_cue_id=222, right_anchor_start=1.120, right_anchor_speaker_id="speaker")]
    assert _extend(case) == case["spans"]


def test_accent_candidate_shared_between_adjacent_partial_cases_is_not_consumed_twice():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["antes E depois"])]
    words = [Word(text=t, start=a, end=b, speaker_id="s") for t, a, b in [("antes", 0, .2), ("É", .24, .32), ("depois", .36, .55)]]
    spans = [DivergenceSpan(case_id="a", cue_ids=[1], srt_text="antes", asr_text="antes", srt_token_indices=[0], asr_word_indices=[0],
                start=0, end=.2, right_anchor_cue_id=1, right_anchor_start=.24, right_anchor_speaker_id="s"),
             DivergenceSpan(case_id="b", cue_ids=[1], srt_text="depois", asr_text="depois", srt_token_indices=[2], asr_word_indices=[2],
                start=.36, end=.55, left_anchor_cue_id=1, left_anchor_end=.32, left_anchor_speaker_id="s")]
    assert _extend(dict(spans=spans, matches=[TokenMatch(cue_id=1, srt_token_index=1, asr_word_index=1, score=1)],
        cues=cues, tokens=tokenize_cues(cues), words=words, protected_cue_ids=set(), cue_word_indices={1: [1]})) == spans
