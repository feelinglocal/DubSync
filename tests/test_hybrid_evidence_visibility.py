"""Hybrid evidence holds remain actionable in the delivered customer reports."""
from __future__ import annotations

import json
import socket
import wave
from types import SimpleNamespace

import pytest
import yaml

from dubsync import llm_providers, pipeline
from dubsync.adjudication import AdjudicationEngine
from dubsync.hybrid_adjudication import HybridAdjudicationAdapter
from dubsync.models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan
from dubsync.providers import ProviderError
from dubsync.reports import write_qc_report


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Hybrid evidence regressions must not call external providers")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _native(span, evidence="heard_clearly", verdict="keep_srt"):
    return {"case_id": span.case_id, "verdict": verdict,
            "final_text": span.srt_text if verdict == "keep_srt" else span.asr_text,
            "evidence": evidence,
            "heard_text": "" if evidence == "not_audible" else "Dam..." if evidence == "heard_unclear" else span.asr_text,
            "reason": "Synthetic local audio evidence."}


def _engine(tmp_path, *, evidence=None, verdict="keep_srt", failure=None, gate=.7):
    span = DivergenceSpan(case_id="name", cue_ids=[17], srt_text="Damien", asr_text="Damian",
                          start=2.1, end=2.8, asr_word_indices=[1])
    class Primary:
        def adjudicate_with_audio(self, spans, snippets):
            return [_native(span)]
    def review(**kwargs):
        if failure == "provider":
            raise ProviderError("Synthetic provider failure")
        if failure == "missing":
            return [], []
        if failure == "legacy_zero":
            return [{"case_id": span.case_id, "verdict": "keep_srt", "final_text": span.srt_text,
                     "confidence": 0, "reason": "No model proposal was made."}], []
        return [_native(span, evidence, verdict)], []
    clip = tmp_path / "snippet.wav"
    clip.write_bytes(b"Local fixture only")
    adapter = HybridAdjudicationAdapter(Primary(), review)
    adapter.set_adjudication_context(language="de")
    snippets = {} if failure == "audio" else {span.case_id: AudioSnippet(
        case_id=span.case_id, path=str(clip), start=2, end=3)}
    engine = AdjudicationEngine(adapter, language="de", confidence_gate=gate, audio_snippets=snippets)
    return engine.adjudicate([span])


@pytest.mark.parametrize("evidence", ["heard_unclear", "not_audible"])
@pytest.mark.parametrize("verdict", ["keep_srt", "use_audio"])
@pytest.mark.parametrize("gate", [0, .7])
def test_hybrid_uncertain_evidence_survives_customer_serialization(tmp_path, evidence, verdict, gate):
    decisions, flags = _engine(tmp_path, evidence=evidence, verdict=verdict, gate=gate)
    cue = Cue(index=17, start_ms=1000, end_ms=5000, lines=["Hallo Damien ist hier."])
    report_path, html_path = tmp_path / "qc.json", tmp_path / "qc.html"
    write_qc_report(report_path, html_path, [cue], flags, [], source_cues=[cue])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["verdict"] == "check"
    assert report["summary"]["review_item_count"] == 1
    item, = report["review"]
    assert item["kind"] == "low_confidence_adjudication" and item["severity"] == "warning"
    assert item["cue_ids"] == [17] and item["srt_numbers"] == [1] and item["raw_flags"] == [0]
    assert "Listen" in item["action"]
    assert "Ready: nothing needs review" not in html_path.read_text(encoding="utf-8")
    assert flags[0].confidence == 0 and flags[0].severity == "warning"
    decision = AdjudicationDecision.model_validate_json(decisions[0].model_dump_json())
    assert decision.verdict == "keep_srt" and decision.final_text == "Damien"
    assert decision.evidence == evidence and decision.confidence == 0
    assert decision.heard_text == ("" if evidence == "not_audible" else "Dam...")


@pytest.mark.parametrize("failure", ["audio", "missing", "provider", "legacy_zero"])
def test_synthetic_hybrid_holds_do_not_invent_audio_evidence(tmp_path, failure):
    decisions, flags = _engine(tmp_path, failure=failure)
    assert decisions[0].verdict == "keep_srt" and decisions[0].final_text == "Damien"
    assert decisions[0].evidence is None and decisions[0].heard_text is None
    assert decisions[0].confidence == 0
    assert len(flags) == 1 and flags[0].kind == "low_confidence_adjudication"
    assert flags[0].message.startswith("Adjudication confidence is below the configured gate")
    if failure == "legacy_zero":
        cue = Cue(index=17, start_ms=1000, end_ms=5000, lines=["Hallo Damien ist hier."])
        report_path = tmp_path / "qc.json"
        write_qc_report(report_path, tmp_path / "qc.html", [cue], flags, [], source_cues=[cue])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["review"] == []
        assert report["diagnostics"][0]["raw_flags"] == [0]


@pytest.mark.parametrize("evidence", ["heard_unclear", "not_audible"])
def test_hybrid_customer_review_survives_cache_rebuild_and_verify(tmp_path, monkeypatch, evidence):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:05,000\nHallo Damien ist hier.\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 6)
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": index + 1.1, "end": index + 1.8}
        for index, text in enumerate(["Hallo", "Damian", "ist", "hier"])
    ]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}, "llm": {"adjudication": {
        "provider": "gemini", "model": "gemini-3.5-flash-lite", "api_key": "synthetic-key",
        "audio_context": {"enabled": False}, "audio_snippet_double_check": {"enabled": True},
        "fallback": {"enabled": True, "provider": "gemini", "model": "gemini-3.8-flash"},
    }}, "timing": {"phrase_edge_snap": False}}), encoding="utf-8")
    calls = []
    def generate(**kwargs):
        calls.append(kwargs["model"])
        selected = "heard_clearly" if kwargs["model"] == "gemini-3.5-flash-lite" else evidence
        spans = [DivergenceSpan.model_validate(span) for span in json.loads(kwargs["prompt"])["spans"]]
        return SimpleNamespace(text=json.dumps({"decisions": [_native(span, selected) for span in spans]}),
                               usage_metadata={})
    monkeypatch.setattr(llm_providers, "_gemini_generate_json", generate)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *args, **kwargs: None)
    options = dict(srt_path=source, audio_path=audio, output_path=tmp_path / "out.srt",
                   workdir=tmp_path / "work", providers_path=config, language="de")
    expected = None
    for resume in (None, None, "rebuild", "verify"):
        result = pipeline.sync_episode(**options, resume=resume)
        if expected is None:
            expected = result.output_srt.read_bytes()
        assert result.output_srt.read_bytes() == expected
        assert calls == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        report = json.loads((result.episode_workdir / "qc_report.json").read_text(encoding="utf-8"))
        assert report["summary"]["verdict"] == "check" and report["summary"]["review_item_count"] == 1
        assert report["review"][0]["kind"] == "low_confidence_adjudication"
        assert report["review"][0]["severity"] == "warning" and report["review"][0]["srt_numbers"] == [1]
        assert not any(flag["kind"] == "invalid_llm_response" for flag in report["flags"])
        html = (result.episode_workdir / "qc_report.html").read_text(encoding="utf-8")
        assert "Ready: nothing needs review" not in html
        stage = json.loads((result.episode_workdir / "adjudicate.json").read_text(encoding="utf-8"))
        assert stage["decisions"][0]["evidence"] == evidence
        assert stage["decisions"][0]["confidence"] == 0
