from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from dubsync.cost import CostMeter
from dubsync.llm_providers import (
    _adjudication_review_prompt,
    adjudication_fallback_config,
    llm_adapter_from_config,
)
from dubsync.models import AudioSnippet, Cue, DivergenceSpan, Word
from dubsync.pipeline import _record_llm_usage_events
from dubsync.providers import ProviderError


def hybrid_config(**fallback):
    return {"llm": {"adjudication": {
        "provider": "gemini", "model": "gemini-3.5-flash-lite",
        "thinking_level": "high", "api_key": "test-key",
        "audio_snippet_double_check": {"enabled": True},
        "audio_context": {"enabled": False},
        "fallback": {"enabled": True, "provider": "gemini",
                     "model": "gemini-3.8-flash", "thinking_level": "medium", **fallback},
    }}}


def test_review_prompt_limits_context_and_marks_every_edit_boundary():
    cues = [Cue(index=i + 1, start_ms=i * 10000, end_ms=i * 10000 + 5000,
                lines=[f"context{i} stay tomorrow"]) for i in range(30)]
    words = [Word(text="stay", start=140.1, end=140.4),
             Word(text="next", start=140.5, end=140.7),
             Word(text="week", start=140.8, end=141.0),
             Word(text="neighbor", start=141.1, end=141.5),
             Word(text="remote", start=250, end=251)]
    span = DivergenceSpan(case_id="changed", cue_ids=[15], srt_text="tomorrow",
                          asr_text="next week", start=140.5, end=141,
                          srt_token_indices=[44], asr_word_indices=[1, 2])
    neighbor = DivergenceSpan(case_id="neighbor", cue_ids=[15], srt_text="stay",
                              asr_text="neighbor", start=141.1, end=141.5,
                              srt_token_indices=[43], asr_word_indices=[3])
    snippet = AudioSnippet(case_id="changed", path="clip.wav", start=138.5, end=143)
    payload = json.loads(_adjudication_review_prompt(
        spans=[span], audio_snippets={span.case_id: snippet},
        reasons={span.case_id: ["audio_text_disagreement"]},
        primary_decisions={}, batch_spans=[span, neighbor],
        episode_context=cues, episode_words=words, confidence_gate=.7,
    ))
    assert payload["adjudication_route"] == "fallback"
    assert [s["case_id"] for s in payload["spans"]] == ["changed"]
    assert {c["cue_id"] for c in payload["episode_context"]} == {13, 14, 15, 16, 17}
    case = payload["review_cases"][0]
    assert [w["word_index"] for w in case["local_asr_words"]] == [0, 1, 2, 3]
    assert case["local_asr_words"][3]["editable_here"] is False
    tokens = case["source_token_ownership"]
    assert [(t["token_index"], t["editable_here"]) for t in tokens] == [(42, False), (43, False), (44, True)]
    assert [c["case_id"] for c in case["other_cases_read_only"]] == ["neighbor"]
    assert "context0 " not in json.dumps(payload)
    assert "remote" not in json.dumps(payload)


@pytest.mark.parametrize("fallback", [
    {"provider": "openai"}, {"model": "gemini-3.5-flash-lite"},
    {"cached_content": "cachedContents/episode"},
    {"audio_context": {"enabled": True}}, {"enabled": "false"},
])
def test_hybrid_rejects_unsafe_or_ambiguous_fallback_config(fallback):
    with pytest.raises(ProviderError):
        llm_adapter_from_config(hybrid_config(**fallback), pass_name="adjudication")


def test_fallback_prices_do_not_inherit_primary_tariff():
    config = hybrid_config()
    primary = config["llm"]["adjudication"]
    primary.update(input_per_million=.123, output_per_million=.456)
    fallback = adjudication_fallback_config(primary)
    assert fallback["model"] == "gemini-3.8-flash"
    assert "input_per_million" not in fallback
    assert "output_per_million" not in fallback
    assert "api_key" in fallback


def test_route_usage_is_charged_to_each_configured_model(monkeypatch):
    monkeypatch.setattr("dubsync.cost._utc_today", lambda: date(2026, 9, 14))
    adapter = SimpleNamespace(usage_events=[
        {"adjudication_route": "primary", "usage_metadata": {"prompt_token_count": 1000, "candidates_token_count": 100, "thoughts_token_count": 50}},
        {"adjudication_route": "fallback", "usage_metadata": {"prompt_token_count": 2000, "candidates_token_count": 200, "thoughts_token_count": 100}},
    ])
    meter = CostMeter()
    flags = _record_llm_usage_events(meter, adapter, hybrid_config(), "adjudication")
    assert flags == []
    assert [item.provider for item in meter.items] == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert meter.items[0].usd == pytest.approx(.000675)
    assert meter.items[1].usd == pytest.approx(.002625)


def test_unconfigured_fallback_usage_cannot_be_silently_priced_as_lite():
    config = hybrid_config(enabled=False)
    adapter = SimpleNamespace(usage_events=[{
        "adjudication_route": "fallback",
        "usage_metadata": {"prompt_token_count": 1000, "candidates_token_count": 100},
    }])
    meter = CostMeter()
    flags = _record_llm_usage_events(meter, adapter, config, "adjudication")
    assert meter.items == []
    assert any(flag.kind == "cost_unmetered" for flag in flags)


def test_real_factory_sends_only_flagged_clip_to_fallback(monkeypatch, tmp_path):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "gemini-3.5-flash-lite":
            decisions = [
                {"case_id": "safe", "verdict": "use_audio", "final_text": "clear spoken words",
                 "heard_text": "clear spoken words", "evidence": "heard_clearly", "reason": "Audible"},
                {"case_id": "review", "verdict": "keep_srt", "final_text": "Go",
                 "heard_text": "Go", "evidence": "heard_clearly", "reason": "Source"},
            ]
        else:
            decisions = [{"case_id": "review", "verdict": "use_audio", "final_text": "Stay",
                          "heard_text": "Stay", "evidence": "heard_clearly", "reason": "Stay is audible"}]
        return SimpleNamespace(text=json.dumps({"decisions": decisions}), usage_metadata={
            "prompt_token_count": 1000, "candidates_token_count": 100,
        })

    monkeypatch.setattr("dubsync.llm_providers._gemini_generate_json", generate)
    adapter = llm_adapter_from_config(hybrid_config(), "adjudication")
    spans = [
        DivergenceSpan(case_id="safe", cue_ids=[1], srt_text="old source wording", asr_text="clear spoken words",
                       start=1, end=2, asr_word_indices=[0, 1, 2]),
        DivergenceSpan(case_id="review", cue_ids=[2], srt_text="Go", asr_text="Stay", start=4, end=5, asr_word_indices=[3]),
    ]
    snippets = {}
    for span in spans:
        clip = tmp_path / f"{span.case_id}.wav"
        clip.write_bytes(b"clip")
        snippets[span.case_id] = AudioSnippet(case_id=span.case_id, path=str(clip), start=span.start - .2, end=span.end + .2)
    adapter.set_episode_context([Cue(index=1, start_ms=1000, end_ms=2000, lines=["old source wording"]),
                                 Cue(index=2, start_ms=4000, end_ms=5000, lines=["Go"])])
    adapter.set_episode_words([Word(text="clear", start=1, end=1.3), Word(text="spoken", start=1.3, end=1.7),
                               Word(text="words", start=1.7, end=2), Word(text="Stay", start=4, end=5)])
    result = adapter.adjudicate_with_audio(spans, snippets)
    assert [item["final_text"] for item in result] == ["clear spoken words", "Stay"]
    assert len(calls) == 2
    assert calls[0]["thinking_level"] == "high"
    assert calls[1]["thinking_level"] == "medium"
    assert calls[1]["cached_content"] is None
    assert calls[1]["audio_context"] is None
    assert set(calls[1]["audio_snippets"]) == {"review"}
    prompt = json.loads(calls[1]["prompt"])
    assert prompt["editable_case_ids"] == ["review"]
    events = adapter.drain_usage_events()
    assert sorted(event["adjudication_route"] for event in events) == ["fallback", "primary"]


def test_invalid_review_envelope_still_records_its_usage(monkeypatch):
    from dubsync.llm_providers import _gemini_adjudication_reviewer

    monkeypatch.setattr("dubsync.llm_providers._gemini_generate_json", lambda **kwargs:
                        SimpleNamespace(text="invalid JSON", usage_metadata={"prompt_token_count": 42}))
    config = adjudication_fallback_config(hybrid_config()["llm"]["adjudication"])
    review = _gemini_adjudication_reviewer(config, .7)
    decisions, events = review(spans=[], audio_snippets={}, reasons={}, primary_decisions={},
                              batch_spans=[], episode_context=[], episode_words=[])
    assert decisions == []
    assert events == [{"usage_metadata": {"prompt_token_count": 42}}]


def test_focused_session_persists_review_trace_and_usage_on_later_failure(tmp_path):
    from dubsync.pipeline import _adjudication_audio_session

    report = {"policy_version": 1, "counts": {"primary": 1, "fallback": 2, "held": 1, "review_requested": 3}, "decisions": []}
    adapter = SimpleNamespace(
        route_report=lambda: report,
        usage_events=[{"adjudication_route": "fallback", "usage_metadata": {
            "prompt_token_count": 1000, "candidates_token_count": 100,
        }}],
    )
    flags = []
    meter = CostMeter()
    with pytest.raises(RuntimeError, match="later failure"):
        with _adjudication_audio_session(adapter, tmp_path / "source.wav", tmp_path / "normalized.wav",
                                         hybrid_config(), tmp_path, meter, flags):
            raise RuntimeError("later failure")
    assert json.loads((tmp_path / "hybrid_adjudication.json").read_text()) == report
    assert json.loads((tmp_path / "cost.json").read_text())["items"][0]["provider"] == "gemini-3.8-flash"
    assert adapter.usage_events == []
    assert flags[0].kind == "hybrid_adjudication_summary"


def test_hybrid_cache_tracks_read_only_neighbor_asr_word_evidence():
    from dubsync.pipeline import _adjudication_cache_key

    span = DivergenceSpan(case_id="change", cue_ids=[1], srt_text="Go", asr_text="Stay",
                          start=1, end=2, asr_word_indices=[0])
    original = [Word(text="Stay", start=1, end=2), Word(text="please", start=2.1, end=2.5)]
    changed = [original[0], Word(text="now", start=2.1, end=2.5)]
    assert _adjudication_cache_key([span], hybrid_config(), source_words=original) != _adjudication_cache_key(
        [span], hybrid_config(), source_words=changed)
