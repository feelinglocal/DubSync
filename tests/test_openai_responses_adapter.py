from __future__ import annotations

import json
import sys
import types

import pytest

from dubsync.llm_providers import (
    AdjudicationBatch,
    OpenAILLMAdapter,
    PunctuationBatch,
    SpeakerMappingBatch,
    _openai_parsed_response,
)
from dubsync.models import Cue, DivergenceSpan
from dubsync.providers import ProviderError


def test_openai_adapter_uses_responses_parse_for_structured_calls(monkeypatch):
    calls: list[dict[str, object]] = []
    parsed = [
        AdjudicationBatch.model_validate(
            {
                "decisions": [
                    {
                        "case_id": "case-1",
                        "verdict": "keep_srt",
                        "final_text": "Hallo",
                        "confidence": 0.91,
                        "speaker": "A",
                        "character": "unknown",
                        "reason": "source is correct",
                    }
                ]
            }
        ),
        PunctuationBatch.model_validate({"cues": [{"cue_id": 1, "text": "Hallo."}]}),
        SpeakerMappingBatch.model_validate({"mappings": [{"speaker_id": "A", "character": "Luna"}]}),
    ]

    class FakeResponse:
        status = "completed"

        def __init__(self, output_parsed):
            self.output_parsed = output_parsed
            self.usage = {"input_tokens": 10, "output_tokens": 5}
            self.output = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse(parsed[len(calls) - 1])

    class FakeClient:
        def __init__(self, *, api_key, timeout, max_retries):
            self.api_key = api_key
            self.timeout = timeout
            self.max_retries = max_retries
            self.responses = FakeResponses()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    adapter = OpenAILLMAdapter(
        api_key="test-key",
        model="gpt-5.6-luna",
        reasoning_effort="high",
        timeout_seconds=12,
        max_retries=1,
    )
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="Hallo",
        asr_text="Hallo",
        confidence=0.9,
        speaker_ids=["A"],
    )

    decisions = adapter.adjudicate([span])
    punctuation = adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["Hallo"])])
    mapping = adapter.map_speakers([Cue(index=1, start_ms=0, end_ms=500, lines=["Hallo"], speaker_id="A")])

    assert decisions[0]["case_id"] == "case-1"
    assert punctuation == {1: "Hallo."}
    assert mapping == {"A": "Luna"}
    assert [call["model"] for call in calls] == ["gpt-5.6-luna"] * 3
    assert [call["text_format"] for call in calls] == [AdjudicationBatch, PunctuationBatch, SpeakerMappingBatch]
    assert all(call["reasoning"] == {"effort": "high"} for call in calls)
    assert all(call["store"] is False for call in calls)
    assert all("input" in call for call in calls)
    assert len(adapter.usage_events) == 3


def test_openai_adapter_parses_json_output_text_fallback():
    class FakeResponse:
        status = "completed"
        output = []
        output_parsed = None
        output_text = json.dumps({"cues": [{"cue_id": 1, "text": "Hallo."}]})

    parsed = _openai_parsed_response(FakeResponse(), PunctuationBatch)

    assert parsed.cues[0].text == "Hallo."


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("incomplete", "incomplete"),
        ("failed", "failed"),
    ],
)
def test_openai_adapter_raises_on_incomplete_or_failed_response(status, message):
    class FakeResponse:
        output = []
        output_parsed = None
        output_text = ""
        incomplete_details = {"reason": "max_output_tokens"}
        error = {"message": "provider error"}

    response = FakeResponse()
    response.status = status

    with pytest.raises(ProviderError, match=message):
        _openai_parsed_response(response, PunctuationBatch)


def test_openai_adapter_raises_on_refusal_content():
    class FakeResponse:
        status = "completed"
        output_parsed = None
        output_text = ""
        output = [{"type": "message", "content": [{"type": "refusal", "refusal": "blocked"}]}]

    with pytest.raises(ProviderError, match="refused"):
        _openai_parsed_response(FakeResponse(), PunctuationBatch)
