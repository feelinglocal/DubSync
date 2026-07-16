from __future__ import annotations

import json
import sys
import types

from dubsync.llm_providers import GeminiLLMAdapter, _adjudication_prompt, _punctuation_prompt, llm_adapter_from_config
from dubsync.models import AudioSnippet, Cue, DivergenceSpan


def test_google_genai_sdk_supports_medium_thinking_level():
    from google.genai import types as google_types

    assert google_types.ThinkingLevel.MEDIUM.value == "MEDIUM"


def test_gemini_adapter_uses_models_generate_content_for_structured_calls(monkeypatch):
    calls: list[dict[str, object]] = []
    responses = [
        {
            "decisions": [
                {
                    "case_id": "case-1",
                    "verdict": "keep_srt",
                    "final_text": "hello there",
                    "confidence": 0.91,
                    "speaker": "A",
                    "character": "unknown",
                    "reason": "ASR noise",
                }
            ]
        },
        {"cues": [{"cue_id": 1, "text": "Hello, there."}]},
        {"mappings": [{"speaker_id": "A", "character": "Luna"}]},
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.text = json.dumps(payload)
            self.usage_metadata = {"input_token_count": 10, "output_token_count": 5}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse(responses[len(calls) - 1])

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="hello there",
        asr_text="hello their",
        confidence=0.8,
        speaker_ids=["A"],
    )

    decisions = adapter.adjudicate([span])
    punctuation = adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])
    mapping = adapter.map_speakers([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"], speaker_id="A")])

    assert decisions[0]["case_id"] == "case-1"
    assert punctuation == {1: "Hello, there."}
    assert mapping == {"A": "Luna"}
    assert [call["model"] for call in calls] == ["gemini-3.5-flash"] * 3
    assert calls[0]["config"]["response_mime_type"] == "application/json"
    assert "response_schema" in calls[0]["config"]
    assert len(adapter.usage_events) == 3


def test_gemini_adapter_passes_thinking_level_to_generate_content(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash", thinking_level="low")

    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert calls[0]["config"]["thinking_config"] == {"thinking_level": "low"}


def test_gemini_adapter_passes_cached_content_to_generate_content(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(
        api_key="test-key",
        model="gemini-3.5-flash",
        cached_content="cachedContents/episode-context",
    )

    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert calls[0]["config"]["cached_content"] == "cachedContents/episode-context"


def test_gemini_adjudication_can_include_inline_audio_snippet(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    snippet_path = tmp_path / "case-1.wav"
    snippet_path.write_bytes(b"RIFFsnippetWAVEfmt ")

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"inline_data": data, "mime_type": mime_type}

    class FakeResponse:
        text = json.dumps(
            {
                "decisions": [
                    {
                        "case_id": "case-1",
                        "verdict": "use_audio",
                        "final_text": "new line",
                        "confidence": 0.91,
                        "speaker": "A",
                        "character": "unknown",
                        "reason": "audio snippet confirms the spoken line",
                    }
                ]
            }
        )

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_types = types.ModuleType("types")
    fake_types.Part = FakePart
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    fake_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="old line",
        asr_text="new line",
        confidence=0.8,
        speaker_ids=["A"],
    )
    snippet = AudioSnippet(
        case_id="case-1",
        path=str(snippet_path),
        mime_type="audio/wav",
        start=0.0,
        end=3.0,
    )

    decisions = adapter.adjudicate_with_audio([span], {"case-1": snippet})

    prompt = json.loads(calls[0]["contents"][0])
    assert prompt["audio_snippets"][0]["case_id"] == "case-1"
    assert prompt["task"].startswith("Adjudicate")
    assert calls[0]["contents"][1] == {"inline_data": b"RIFFsnippetWAVEfmt ", "mime_type": "audio/wav"}
    assert decisions[0]["reason"] == "audio snippet confirms the spoken line"


def test_adjudication_prompt_instructs_audio_literal_check_and_no_word_drops(tmp_path):
    snippet_path = tmp_path / "case-1.wav"
    snippet_path.write_bytes(b"RIFFsnippetWAVEfmt ")
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[11],
        srt_text="Drachen Evolutionssystem",
        asr_text="Drachenevolutionssystem",
        context_after=[],
    )
    snippet = AudioSnippet(
        case_id="case-1",
        path=str(snippet_path),
        mime_type="audio/wav",
        start=22.0,
        end=27.0,
    )

    prompt = json.loads(_adjudication_prompt([span], confidence_gate=0.9, audio_snippets={"case-1": snippet}))
    instructions = "\n".join(prompt["instructions"])

    assert "Listen to each attached audio snippet" in instructions
    assert "final_text is the replacement for only the divergent span" in instructions
    assert "Do not drop matched cue words outside the divergent span" in instructions
    assert "Drachenevolutionssystem besitze" in instructions
    assert prompt["spans"][0]["cue_ids"] == [11]
    assert prompt["audio_snippets"][0]["duration_seconds"] == 5.0


def test_punctuation_prompt_includes_speaker_and_character_labels():
    prompt = json.loads(
        _punctuation_prompt(
            [
                Cue(
                    index=1,
                    start_ms=0,
                    end_ms=500,
                    lines=["hello there"],
                    speaker_id="SPEAKER_00",
                    character="Luna",
                )
            ]
        )
    )

    assert prompt["cues"][0]["speaker_id"] == "SPEAKER_00"
    assert prompt["cues"][0]["character"] == "Luna"


def test_gemini_adjudication_prompt_uses_configured_confidence_gate(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        text = json.dumps(
            {
                "decisions": [
                    {
                        "case_id": "case-1",
                        "verdict": "keep_srt",
                        "final_text": "hello there",
                        "confidence": 0.91,
                        "speaker": "A",
                        "character": "unknown",
                        "reason": "ASR noise",
                    }
                ]
            }
        )

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = llm_adapter_from_config(
        {
            "llm": {
                "provider": "gemini",
                "api_key": "test-key",
                "model": "gemini-3.5-flash",
                "adjudication": {"confidence_gate": 0.95},
            }
        },
        pass_name="adjudication",
    )
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="hello there",
        asr_text="hello their",
        confidence=0.8,
        speaker_ids=["A"],
    )

    adapter.adjudicate([span])

    prompt = json.loads(calls[0]["contents"])
    assert prompt["confidence_gate"] == 0.95
