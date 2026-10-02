from copy import deepcopy

import pytest

from dubsync import pipeline
from dubsync.adjudication_case_cache import case_cache_key, read_case
from dubsync.cache import JsonDiskCache
from dubsync.models import AdjudicationDecision, DivergenceSpan, QCFlag
from test_adjudication_case_cache import _inputs
from test_whole_utterance_secondary import _ambiguous_case, _ask
from test_whole_utterance_timing import _resolve


def _partial():
    return DivergenceSpan(case_id="case-7", cue_ids=[2], srt_text="跪", asr_text="ひざまず",
                          start=22.26, end=22.76)


def _warning(text="跪", start=22.26, end=22.76):
    return QCFlag(kind="low_confidence_adjudication", cue_ids=[2], confidence=0,
                  old_text=text, new_text="ひざまず", start=start, end=end,
                  message="Partial orthography equivalence is unresolved.")


def test_legacy_partial_cache_cannot_carry_a_different_whole_case_failure(tmp_path):
    span = _partial()
    cache = JsonDiskCache(tmp_path)
    key = case_cache_key(span, **_inputs())
    decision = AdjudicationDecision(case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
                                   confidence=1, evidence="heard_clearly", heard_text="ひざまず", reason="Legacy partial hearing.")
    own = _warning()
    other = _warning("ここで跪いて", 17.16, 25.42)
    cache.write(key, {"decision": decision.model_dump(), "flags": [own.model_dump(), other.model_dump()]})
    hit = read_case(cache, key, span)
    assert hit is not None and hit[0] == decision and hit[1] == [own]


def test_complete_hearing_recovers_timing_while_retaining_the_partial_wording_warning():
    case, secondary, uncertain = _ambiguous_case()
    original = deepcopy(case)
    questions = _ask(case, secondary, uncertain)
    warning = _warning()
    result = _resolve(case, questions, flags=[warning])
    assert result.resolved_cue_ids == {2}
    assert result.spoken_spans[2][0] == 21715
    assert 23075 <= result.spoken_spans[2][1] <= 23076
    assert warning in result.flags
    assert result.alignment.cue_word_indices == case[2].cue_word_indices and case == original


@pytest.mark.parametrize("kind", ["low_confidence_adjudication", "invalid_llm_response", "adjudication_audio_unavailable"])
@pytest.mark.parametrize("scope", ["same_case", "unscoped", "incomplete"])
def test_same_case_and_unscoped_failures_still_block_complete_hearing(kind, scope):
    case, secondary, uncertain = _ambiguous_case()
    questions = _ask(case, secondary, uncertain)
    span = questions[0].span
    data = {"kind": kind, "cue_ids": [2], "message": "Unresolved failure."}
    if scope == "same_case":
        data.update(old_text=span.srt_text, start=span.start, end=span.end)
    elif scope == "unscoped":
        data["cue_ids"] = []
    flag = QCFlag(**data)
    result = _resolve(case, questions, flags=[flag])
    assert result.resolved_cue_ids == set() and result.cues == case[0]
    assert flag in result.flags


def test_case_flag_selection_preserves_partial_failure_only_on_its_own_case():
    case, secondary, uncertain = _ambiguous_case()
    whole = _ask(case, secondary, uncertain)[0].span
    warning = _warning()
    assert pipeline._flags_for_adjudication_case(_partial(), [warning]) == [warning]
    assert pipeline._flags_for_adjudication_case(whole, [warning]) == []


def test_unscoped_wording_failure_survives_case_cache_selection():
    flag = QCFlag(kind="low_confidence_adjudication", message="No case could be identified.")
    assert pipeline._flags_for_adjudication_case(_partial(), [flag]) == [flag]
