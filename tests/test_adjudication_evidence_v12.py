"""Native evidence contracts and conservative routing, without provider calls."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dubsync import llm_providers
from dubsync.hybrid_adjudication import HybridAdjudicationAdapter, triage_decisions
from dubsync.models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan


def _span(source="original wording", asr="spoken words", **kwargs):
    return DivergenceSpan(case_id="case-1", cue_ids=[15], srt_text=source, asr_text=asr,
                          start=14, end=15, **kwargs)


def _native(span, evidence="heard_clearly", **kwargs):
    return dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
                heard_text=span.asr_text, evidence=evidence, reason="Acoustic observation", **kwargs)


def _response(decisions):
    return SimpleNamespace(text=json.dumps({"decisions": decisions}), usage_metadata={})


def test_v12_supplies_local_context_language_register_and_global_edit_markers():
    cues = [Cue(index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 800,
                lines=[f"context{i} stay tomorrow"]) for i in range(30)]
    span = _span(source="tomorrow", asr="next week", srt_token_indices=[44])
    payload = json.loads(llm_providers._adjudication_prompt(
        [span], episode_context=cues, language="pt-BR", register_policy="script"))
    assert payload["prompt_version"].startswith("adjudication-v12-")
    assert payload["language"] == "pt-BR" and payload["register_policy"] == "script"
    assert [cue["cue_id"] for cue in payload["episode_context"]] == [13, 14, 15, 16, 17]
    source = payload["spans"][0]["source_cue_ownership"][0]
    assert source["source_text"] == "context14 stay tomorrow"
    assert source["marked_text"] == "context14 stay <editable>tomorrow</editable>"
    assert [(token["token_index"], token["editable_here"]) for token in source["tokens"]] == [
        (42, False), (43, False), (44, True)]
    instructions = " ".join(payload["instructions"])
    assert "heard_text" in instructions and "heard_unclear" in instructions
    assert "free confidence" in instructions and "Never return timestamps" in instructions
    assert "abbreviation" in instructions and "register_policy" in instructions
    assert "context0 " not in json.dumps(payload)


def test_native_schema_requests_evidence_not_provider_confidence(monkeypatch):
    captured = []
    span = _span()
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", lambda **kwargs:
                        captured.append(kwargs) or _response([_native(span)]))
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    result = adapter.adjudicate([span])
    schema = captured[0]["response_schema"].model_json_schema()
    decision = next(value for value in schema["$defs"].values() if "case_id" in value.get("properties", {}))
    assert {"evidence", "heard_text"} <= set(decision["required"])
    assert "confidence" not in decision["properties"]
    assert result[0]["confidence"] == 1.0 and result[0]["evidence"] == "heard_clearly"


@pytest.mark.parametrize("evidence,heard,expected", [
    ("heard_clearly", "spoken words", 1.0),
    ("heard_unclear", "spoken", 0.0),
    ("not_audible", "", 0.0),
])
def test_internal_evidence_deterministically_replaces_provider_confidence(evidence, heard, expected):
    data = _native(_span(), evidence=evidence, confidence=.99)
    data["heard_text"] = heard
    result = AdjudicationDecision.model_validate(data)
    assert result.confidence == expected
    assert result.evidence == evidence and result.heard_text == heard


def test_legacy_saved_decision_keeps_its_original_confidence():
    data = dict(case_id="saved", verdict="use_audio", final_text="stored wording",
                confidence=.83, reason="Saved v11 fixture")
    assert AdjudicationDecision.model_validate(data).confidence == .83


@pytest.mark.parametrize("changes", [
    {"evidence": None}, {"evidence": "certain"}, {"heard_text": None},
    {"evidence": "not_audible", "heard_text": "spoken words"},
    {"heard_text": "entirely different dialogue"},
    {"verdict": "keep_srt", "final_text": "original wording"},
    {"start": 100, "end": 200},
])
def test_malformed_native_evidence_cannot_authorize_words(monkeypatch, changes):
    span = _span()
    payload = {**_native(span), **changes}
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", lambda **_kwargs: _response([payload]))
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    # Per-case invalid results are left invalid for the engine's retry/hold path.
    raw = adapter.adjudicate([span])
    assert span.case_id in triage_decisions([span], raw)


def test_new_native_missing_evidence_never_uses_legacy_confidence(monkeypatch):
    span = _span()
    old_response = dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
                        confidence=1, reason="Old native response")
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", lambda **_kwargs: _response([old_response]))
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    adapter.defer_adjudication_validation = True
    assert span.case_id in triage_decisions([span], adapter.adjudicate([span]))


@pytest.mark.parametrize("source,asr,risk", [
    ("buy 2 apples", "buy 3 apples", "risky_number"),
    ("buy two apples", "buy three apples", "risky_number"),
    ("二つください", "三つください", "risky_number"),
    ("eu posso ir", "eu não posso", "risky_negation"),
    ("ich gehe hin", "ich gehe nicht", "risky_negation"),
    ("vou sair agora", "nunca vou sair", "risky_negation"),
    ("come", "stay", "risky_single_word_substitution"),
    ("call Alice now", "call Alex now", "risky_name"),
])
def test_asr_agreement_still_reviews_risky_changes(source, asr, risk):
    span = _span(source=source, asr=asr)
    assert risk in triage_decisions([span], [_native(span, confidence=1)])[span.case_id]


def test_clear_ordinary_asr_agreement_avoids_review():
    span = _span()
    assert triage_decisions([span], [_native(span, confidence=1)]) == {}


def test_native_source_typography_can_differ_from_heard_abbreviation(monkeypatch):
    span = _span(source="Sr. Silva espera", asr="senhor Silva fica")
    payload = _native(span)
    payload.update(final_text="Sr. Silva fica", heard_text="senhor Silva fica")
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", lambda **_kwargs: _response([payload]))
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    adapter.set_adjudication_context(language="pt", register_policy="script")
    result = adapter.adjudicate([span])[0]
    assert result["final_text"] == "Sr. Silva fica" and result["confidence"] == 1.0


def test_source_cue_markers_preserve_annotated_screen_text():
    cue = Cue(index=15, start_ms=14000, end_ms=15000, lines=["[AMANHÃ] Eu volto amanhã."])
    span = _span(source="amanhã", asr="depois", srt_token_indices=[2])
    payload = json.loads(llm_providers._adjudication_prompt([span], episode_context=[cue], language="pt"))
    marked = payload["spans"][0]["source_cue_ownership"][0]["marked_text"]
    assert marked == "[AMANHÃ] Eu volto <editable>amanhã</editable>."


def test_review_uses_v12_evidence_with_global_source_ownership():
    cues = [Cue(index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 800,
                lines=[f"context{i} stay tomorrow"]) for i in range(30)]
    span = _span(source="tomorrow", asr="next week", srt_token_indices=[44])
    payload = json.loads(llm_providers._adjudication_review_prompt(
        spans=[span], audio_snippets={span.case_id: AudioSnippet(
            case_id=span.case_id, path="prompt-only.wav", start=13, end=16)},
        reasons={span.case_id: ["review"]}, primary_decisions={}, batch_spans=[span],
        episode_context=cues, episode_words=[], confidence_gate=.7,
        language="en", register_policy="spoken"))
    assert payload["prompt_version"].startswith("adjudication-review-v2-")
    assert payload["language"] == "en" and payload["register_policy"] == "spoken"
    assert payload["spans"][0]["source_cue_ownership"][0]["tokens"][-1] == {
        "token_index": 44, "text": "tomorrow", "editable_here": True}
    assert "heard_unclear" in " ".join(payload["instructions"])


def test_unclear_evidence_cannot_bypass_gate_even_with_perfect_confidence():
    span = _span()
    reasons = triage_decisions([span], [_native(span, evidence="heard_unclear", confidence=1)])
    assert "primary_audio_unclear" in reasons[span.case_id]


def test_context_setter_is_bound_to_primary_and_frozen_review_context(tmp_path):
    span = _span(source="call Alice now", asr="call Alex now")
    context_calls = []
    reviews = []

    class Primary:
        def set_adjudication_context(self, **kwargs):
            context_calls.append(kwargs)

        def adjudicate_with_audio(self, spans, clips):
            return [_native(span, confidence=1)]

    def review(**kwargs):
        reviews.append(kwargs)
        return [_native(span, confidence=1)], []

    adapter = HybridAdjudicationAdapter(Primary(), review)
    adapter.set_adjudication_context(language="pt-BR", register_policy="spoken")
    path = tmp_path / "synthetic.wav"
    path.write_bytes(b"clip")
    output = adapter.adjudicate_with_audio([span], {span.case_id: AudioSnippet(
        case_id=span.case_id, path=str(path), start=13, end=16)})
    assert context_calls == [{"language": "pt-BR", "register_policy": "spoken"}]
    assert reviews[0]["language"] == "pt-BR" and reviews[0]["register_policy"] == "spoken"
    assert output[0]["reason"].startswith("[hybrid:fallback]")


@pytest.mark.parametrize("policy", ["natural", "", None, True])
def test_invalid_register_policy_is_rejected_before_provider_call(policy):
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    with pytest.raises(ValueError):
        adapter.set_adjudication_context(language="pt", register_policy=policy)
