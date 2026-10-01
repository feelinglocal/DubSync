import pytest

from dubsync.cache import JsonDiskCache
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag, Word


def _inputs():
    return dict(
        model="test-model", params={"thinking_level": "medium"},
        policy_context={"prompt_version": 12, "audio_sha256": "abc", "register_policy": "script"},
        source_cues=[Cue(index=i, start_ms=i*1000, end_ms=i*1000+800, lines=[f"cue {i}"]) for i in range(1, 9)],
        source_words=[Word(text="spoken", start=1, end=1.5)],
    )


def _span(case_id="case-1", text="script"):
    return DivergenceSpan(case_id=case_id, cue_ids=[1], srt_text=text, asr_text="spoken", start=1, end=1.5)


def test_changed_or_renumbered_case_does_not_invalidate_other_cases(tmp_path):
    from dubsync.adjudication_case_cache import case_cache_key, read_case, write_case
    cache = JsonDiskCache(tmp_path)
    span = _span()
    decision = AdjudicationDecision(case_id=span.case_id, verdict="use_audio", final_text="spoken", confidence=1, reason="heard")
    key = case_cache_key(span, **_inputs())
    write_case(cache, key, span, decision, [])
    renamed = _span("case-99")
    hit = read_case(cache, case_cache_key(renamed, **_inputs()), renamed)
    assert hit[0].case_id == "case-99" and hit[0].final_text == "spoken"
    assert read_case(cache, case_cache_key(_span(text="new source"), **_inputs()), _span()) is None
    assert read_case(cache, key, span)[0] == decision


@pytest.mark.parametrize("change", ["local_cue", "word", "model", "policy", "params"])
def test_changed_evidence_or_policy_invalidates_case(change):
    from dubsync.adjudication_case_cache import case_cache_key
    values = _inputs()
    before = case_cache_key(_span(), **values)
    if change == "local_cue":
        values["source_cues"][1] = values["source_cues"][1].model_copy(update={"lines": ["changed context"]})
    elif change == "word":
        values["source_words"][0] = values["source_words"][0].model_copy(update={"end": 1.6})
    elif change == "model":
        values["model"] = "different"
    elif change == "policy":
        values["policy_context"]["prompt_version"] = 13
    else:
        values["params"]["thinking_level"] = "high"
    assert case_cache_key(_span(), **values).digest != before.digest


def test_unrelated_distant_cue_does_not_invalidate_case():
    from dubsync.adjudication_case_cache import case_cache_key
    values = _inputs()
    before = case_cache_key(_span(), **values)
    values["source_cues"][-1] = values["source_cues"][-1].model_copy(update={"lines": ["unrelated wording"]})
    assert case_cache_key(_span(), **values).digest == before.digest


def test_failed_or_invalid_case_is_never_reused(tmp_path):
    from dubsync.adjudication_case_cache import case_cache_key, read_case, write_case
    cache = JsonDiskCache(tmp_path)
    span = _span()
    key = case_cache_key(span, **_inputs())
    decision = AdjudicationDecision(case_id=span.case_id, verdict="keep_srt", final_text="script", confidence=0, reason="unavailable")
    write_case(cache, key, span, decision, [QCFlag(kind="llm_provider_unavailable", cue_ids=[1], message="unavailable")])
    assert read_case(cache, key, span) is None
    cache.write(key, {"decision": {"bad": True}, "flags": []})
    assert read_case(cache, key, span) is None
