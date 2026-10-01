"""Source spelling keeps are valid syntax but can still need semantic review."""
from __future__ import annotations

import json
import socket
import wave
from types import SimpleNamespace

import pytest
import yaml

from dubsync import llm_providers, pipeline
from dubsync.adjudication import AdjudicationEngine, confidence_gated_decision
from dubsync.adjudication_policy import DeterministicAdjudicationPolicy
from dubsync.hybrid_adjudication import HybridAdjudicationAdapter, _indexed_decisions, triage_decisions
from dubsync.models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan
from dubsync.srt_io import read_srt


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Editorial keep tests must not call external providers")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _span(source="Damien", heard="Damian"):
    return DivergenceSpan(case_id="name", cue_ids=[2], srt_text=source, asr_text=heard,
                          start=2.1, end=2.4, asr_word_indices=[1])


def _native(span, *, final=None, heard=None, verdict="keep_srt"):
    return dict(case_id=span.case_id, verdict=verdict,
                final_text=span.srt_text if final is None else final,
                heard_text=span.asr_text if heard is None else heard, evidence="heard_clearly",
                reason="The audible name is Damian; preserve the authored name spelling.")


def _adapter(monkeypatch, make_payload):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"decisions": [make_payload()]}), usage_metadata={})
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", generate)
    adapter = llm_providers.GeminiLLMAdapter(api_key="synthetic")
    adapter.set_adjudication_context(language="de", register_policy="script")
    return adapter, calls


@pytest.mark.parametrize("source,heard", [("Damien", "Damian"), ("Dr. Klemensen", "Dr. Clemensen")])
def test_valid_native_name_keep_is_called_once_and_gets_semantic_review(monkeypatch, source, heard):
    span = _span(source, heard)
    adapter, calls = _adapter(monkeypatch, lambda: _native(span))
    decisions, flags = AdjudicationEngine(adapter, language="de").adjudicate([span])
    assert len(calls) == 1
    assert decisions[0].verdict == "keep_srt" and decisions[0].final_text == source
    assert decisions[0].evidence == "heard_clearly" and decisions[0].heard_text == heard
    assert decisions[0].confidence == 1  # hearing confidence is not spelling equivalence
    assert len(flags) == 1 and flags[0].kind == "low_confidence_adjudication"
    assert flags[0].confidence == 0
    assert flags[0].old_text == source and flags[0].new_text == heard
    assert "equivalence" in flags[0].message
    assert "schema" not in flags[0].message.lower()


@pytest.mark.parametrize("gate", [0, .7, 1])
def test_semantic_keep_hold_survives_serialization_and_zero_numeric_gate(gate):
    span = _span()
    decision = AdjudicationDecision.model_validate(_native(span))
    policy = DeterministicAdjudicationPolicy(language="de")
    first, first_flag = confidence_gated_decision(span, decision, gate, policy=policy)
    reloaded = AdjudicationDecision.model_validate_json(first.model_dump_json())
    second, second_flag = confidence_gated_decision(span, reloaded, gate, policy=policy)
    assert first == second == decision
    assert first_flag == second_flag
    assert first_flag is not None and first_flag.kind == "low_confidence_adjudication"


@pytest.mark.parametrize("source,heard,language", [
    ("Ich gehe.", "Ich gehe.", "de"),
    ("Ich habe Zeit.", "Ich hab Zeit.", "de"),
    ("Eu estou aqui.", "Eu tô aqui.", "pt"),
])
def test_matching_and_proven_editorial_keeps_under_script_policy_do_not_create_semantic_flags(source, heard, language):
    span = _span(source, heard)
    decision = AdjudicationDecision.model_validate(_native(span))
    selected, flag = confidence_gated_decision(
        span, decision, .7, policy=DeterministicAdjudicationPolicy(language=language, register_policy="script"),
    )
    assert selected == decision and flag is None


def test_engine_passes_its_language_policy_for_a_proven_register_keep(monkeypatch):
    # Source versus ASR is not deterministic; the review hears a source register
    # form instead. Only the engine's German policy can prove that equivalence.
    span = _span("Ich habe Zeit.", "Er geht weg.")
    adapter, calls = _adapter(monkeypatch, lambda: _native(span, heard="Ich hab Zeit."))
    decisions, flags = AdjudicationEngine(adapter, language="de", register_policy="script").adjudicate([span])
    assert len(calls) == 1 and decisions[0].final_text == span.srt_text
    assert flags == []


@pytest.mark.parametrize("final", ["Damian", "Damien."])
def test_keep_must_preserve_exact_source_even_if_native_fields_are_valid(monkeypatch, final):
    span = _span()
    adapter, calls = _adapter(monkeypatch, lambda: _native(span, final=final, heard=final))
    decisions, flags = AdjudicationEngine(adapter, language="de").adjudicate([span])
    assert len(calls) == 2
    assert decisions[0].final_text == "Damien"
    assert [flag.kind for flag in flags] == ["invalid_llm_response"]


def test_unsupported_changed_wording_still_uses_invalid_response_hold(monkeypatch):
    span = _span("Müller", "Damian")
    adapter, calls = _adapter(monkeypatch, lambda: _native(span, final="Damien", heard="Damian", verdict="use_audio"))
    decisions, flags = AdjudicationEngine(adapter, language="de").adjudicate([span])
    assert len(calls) == 2
    assert decisions[0].final_text == "Müller"
    assert [flag.kind for flag in flags] == ["invalid_llm_response"]


def test_hybrid_indexes_valid_source_keep_and_triages_its_wording_disagreement():
    span = _span()
    raw = [{**_native(span), "confidence": 1}]
    policy = DeterministicAdjudicationPolicy(language="de")
    indexed, faults = _indexed_decisions([span], raw, prefix="primary", policy=policy)
    assert faults == {} and indexed[span.case_id]["verdict"] == "keep_srt"
    assert triage_decisions([span], raw, language="de", policy=policy) == {
        span.case_id: ["source_keep_hearing_unresolved", "source_differs_from_owned_asr"],
    }


def test_hybrid_reviews_unproved_hearing_even_when_source_and_asr_are_equivalent():
    span = _span("Ich gehe.", "ich gehe")
    raw = [{**_native(span, heard="Ich bleibe."), "confidence": 1}]
    assert triage_decisions([span], raw, language="de") == {
        span.case_id: ["source_keep_hearing_unresolved"],
    }


def test_hybrid_review_keeps_semantic_uncertainty_without_schema_retry(tmp_path):
    span = _span()
    counts = {"primary": 0, "review": 0}
    raw = {**_native(span), "confidence": 1}
    class Primary:
        def adjudicate_with_audio(self, spans, snippets):
            counts["primary"] += 1
            return [raw]
    def review(**kwargs):
        counts["review"] += 1
        return [raw], []
    adapter = HybridAdjudicationAdapter(Primary(), review)
    adapter.set_adjudication_context(language="de", register_policy="script")
    clip = tmp_path / "snippet.wav"
    clip.write_bytes(b"Local test fixture only")
    engine = AdjudicationEngine(adapter, language="de", audio_snippets={span.case_id: AudioSnippet(
        case_id=span.case_id, path=str(clip), start=2, end=2.5,
    )})
    decisions, flags = engine.adjudicate([span])
    assert counts == {"primary": 1, "review": 1}
    assert decisions[0].final_text == "Damien" and decisions[0].heard_text == "Damian"
    assert [flag.kind for flag in flags] == ["low_confidence_adjudication"]
    assert "invalid" not in " ".join(adapter.route_report()["decisions"][0]["reasons"])


def test_pipeline_preserves_semantic_keep_review_through_cache_and_resume(tmp_path, monkeypatch):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:01,700\nHallo.\n\n"
                      "2\n00:00:02,000 --> 00:00:02,700\nDamien\n\n"
                      "3\n00:00:03,000 --> 00:00:04,000\nist hier.\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 5)
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": "Hallo", "start": 1.1, "end": 1.4}, {"text": "Damian", "start": 2.1, "end": 2.4},
        {"text": "ist", "start": 3.1, "end": 3.3}, {"text": "hier", "start": 3.4, "end": 3.7},
    ]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture"},
                                     "timing": {"phrase_edge_snap": False}}), encoding="utf-8")
    span = _span()
    span = span.model_copy(update={"case_id": "case-1"})
    adapter, calls = _adapter(monkeypatch, lambda: _native(span))
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **kw: adapter)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *a, **kw: None)
    options = dict(srt_path=source, audio_path=audio, output_path=tmp_path / "out.srt",
                   workdir=tmp_path / "work", providers_path=config, language="de")
    first = pipeline.sync_episode(**options)
    assert len(calls) == 1
    cue = next(c for c in read_srt(first.output_srt) if c.plain_text == "Damien")
    assert (cue.start_ms, cue.end_ms) == (2000, 2700)
    expected = first.output_srt.read_bytes()
    for resume in (None, "rebuild", "verify"):
        current = pipeline.sync_episode(**options, resume=resume)
        assert len(calls) == 1
        assert current.output_srt.read_bytes() == expected
        stage = json.loads((current.episode_workdir / "adjudicate.json").read_text(encoding="utf-8"))
        review = [flag for flag in stage["flags"] if flag["kind"] == "low_confidence_adjudication"]
        assert len(review) == 1 and review[0]["cue_ids"] == [2]
        assert "equivalence" in review[0]["message"]
        assert not any(flag["kind"] == "invalid_llm_response" for flag in stage["flags"])
