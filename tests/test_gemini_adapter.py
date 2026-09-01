from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dubsync.llm_providers import GeminiLLMAdapter, _adjudication_prompt, _punctuation_prompt, llm_adapter_from_config
from dubsync.models import AudioSnippet, Cue, CueContext, DivergenceSpan
from dubsync.providers import ProviderError


def test_google_genai_sdk_supports_medium_thinking_level():
    google_types = pytest.importorskip("google.genai.types", reason="requires the optional cloud dependencies")

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
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
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


def test_gemini_adapter_passes_configured_timeout_and_retry_options(monkeypatch):
    client_options: list[dict[str, object] | None] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            client_options.append(http_options)
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
                "timeout_seconds": 12.5,
                "max_retries": 3,
            }
        },
        pass_name="punctuation",
    )

    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert client_options == [
        {"timeout": 12_500, "retry_options": {"attempts": 4}}
    ]


def test_gemini_adapter_closes_each_request_client(monkeypatch):
    closed_clients: list[str] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()

        def close(self):
            closed_clients.append(self.api_key)

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")

    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert closed_clients == ["test-key"]


def test_gemini_adapter_keeps_valid_response_when_client_close_fails(monkeypatch, caplog):
    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()

        def close(self):
            raise OSError("transport cleanup failed")

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")

    punctuation = adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert punctuation == {1: "Hello, there."}
    assert "Gemini client cleanup failed" in caplog.text


def test_gemini_adapter_records_usage_without_retaining_response(monkeypatch):
    response_objects: list[object] = []

    class FakeUsage:
        def __init__(self):
            self.input_token_count = 10
            self.output_token_count = 5

    class FakeResponse:
        def __init__(self):
            self.text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})
            self.usage_metadata = FakeUsage()
            response_objects.append(self)

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()

        def close(self):
            pass

    fake_genai = types.ModuleType("genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")

    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])])

    assert adapter.usage_events == [
        {"usage_metadata": {"input_token_count": 10, "output_token_count": 5}}
    ]
    assert adapter.usage_events[0] is not response_objects[0]


def test_gemini_adapter_passes_thinking_level_to_generate_content(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
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


def test_direct_gemini_37_adapter_rejects_unsupported_minimal_thinking():
    with pytest.raises(RuntimeError, match="gemini-3.7-flash thinking_level"):
        GeminiLLMAdapter(
            api_key="test-key",
            model="gemini-3.7-flash",
            thinking_level="minimal",
        )


def test_gemini_adapter_passes_cached_content_to_generate_content(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        text = json.dumps({"cues": [{"cue_id": 1, "text": "Hello, there."}]})

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
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
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
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


def test_gemini_audio_read_oserror_is_wrapped_and_client_is_closed(monkeypatch, tmp_path):
    closed_clients: list[str] = []
    snippet_path = tmp_path / "case-1.wav"
    snippet_path.write_bytes(b"RIFFsnippetWAVEfmt ")

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            raise AssertionError("audio bytes should fail before constructing a Gemini part")

    class FakeModels:
        def generate_content(self, **_kwargs):
            raise AssertionError("Gemini should not be called when evidence cannot be read")

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()

        def close(self):
            closed_clients.append(self.api_key)

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

    original_read_bytes = Path.read_bytes

    def fail_snippet_read(path: Path) -> bytes:
        if path == snippet_path:
            raise OSError(1455, "The paging file is too small")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_snippet_read)
    adapter = GeminiLLMAdapter(api_key="test-key", model="gemini-3.5-flash")
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="old line",
        asr_text="new line",
    )
    snippet = AudioSnippet(
        case_id="case-1",
        path=str(snippet_path),
        mime_type="audio/wav",
        start=0.0,
        end=3.0,
    )

    with pytest.raises(ProviderError, match="Gemini request failed"):
        adapter.adjudicate_with_audio([span], {"case-1": snippet})

    assert closed_clients == ["test-key"]


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


def test_adjudication_prompt_marks_neighbor_context_read_only_and_preserves_editorial_marks():
    span = DivergenceSpan(
        case_id="case-context",
        cue_ids=[11],
        srt_text="Verdammt, „bleib hier“.",
        asr_text="Verdammt bleib hier",
        context_before=[CueContext(cue_id=10, text="Komm zurück!", start=4.0, end=5.0)],
        context_after=[CueContext(cue_id=12, text="Ich gehe nicht.", start=7.0, end=8.0)],
    )

    prompt = json.loads(_adjudication_prompt([span]))
    instructions = "\n".join(prompt["instructions"])

    assert prompt["spans"][0]["context_before"][0]["text"] == "Komm zurück!"
    assert prompt["spans"][0]["context_after"][0]["text"] == "Ich gehe nicht."
    assert "read-only context" in instructions
    assert "cannot prove unheard words" in instructions
    assert "partial audio window" in instructions
    assert "compress, omit, or absorb dialogue" in instructions
    assert "hard scene boundary" in instructions
    assert "line breaks" in instructions
    assert "quotation marks" in instructions


def test_adjudication_prompt_includes_complete_ordered_episode_context():
    span = DivergenceSpan(case_id="case-2", cue_ids=[2], srt_text="Bleib.", asr_text="bleib")
    episode = [
        Cue(index=1, start_ms=0, end_ms=800, lines=["Vorher."]),
        Cue(index=2, start_ms=900, end_ms=1_600, lines=["Bleib."], speaker_id="S1"),
        Cue(index=3, start_ms=1_700, end_ms=2_400, lines=["Nachher."], character="Mara"),
    ]

    prompt = json.loads(_adjudication_prompt([span], episode_context=episode))

    assert [item["cue_id"] for item in prompt["episode_context"]] == [1, 2, 3]
    assert prompt["episode_context"][1]["source_lines"] == ["Bleib."]
    assert prompt["episode_context"][1]["speaker_id"] == "S1"
    assert prompt["episode_context"][2]["character"] == "Mara"
    assert "read_only" in prompt["episode_context_role"]


def test_adjudication_prompt_keeps_shared_context_before_batch_specific_payload():
    raw_prompt = _adjudication_prompt(
        [
            DivergenceSpan(
                case_id="case-2",
                cue_ids=[2],
                srt_text="Bleib.",
                asr_text="bleib",
                prompt_scene_id=7,
                prompt_scene_position=2,
            )
        ],
        episode_context=[Cue(index=2, start_ms=900, end_ms=1_600, lines=["Bleib."])],
    )
    prompt = json.loads(raw_prompt)
    keys = list(prompt)

    assert prompt["prompt_version"] == "adjudication-v9-explicit-scene-isolation"
    assert prompt["spans"][0]["scene_id"] == 7
    assert prompt["spans"][0]["scene_position"] == 2
    assert keys.index("episode_context") < keys.index("spans")
    assert keys.index("episode_context") < keys.index("audio_snippets")


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
                    prompt_scene_id=3,
                    prompt_scene_position=1,
                )
            ]
        )
    )

    assert prompt["cues"][0]["speaker_id"] == "SPEAKER_00"
    assert prompt["cues"][0]["character"] == "Luna"
    assert prompt["cues"][0]["scene_id"] == 3
    assert prompt["cues"][0]["scene_position"] == 1
    assert prompt["prompt_version"].startswith("punctuation-")
    instructions = "\n".join(prompt["instructions"])
    assert "Preserve every cue ID" in instructions
    assert "Preserve the source line-break positions" in instructions
    assert "Do not add, remove, or restyle quotation marks" in instructions
    assert "hard scene boundaries" in instructions


def test_punctuation_prompt_supplies_full_ordered_text_context_without_audio_or_reflow_authority():
    prompt = json.loads(
        _punctuation_prompt(
            [
                Cue(
                    index=7,
                    start_ms=1200,
                    end_ms=2400,
                    lines=["Er sagte", "„Bleib hier.“"],
                    speaker_id="SPEAKER_00",
                    character="Luna",
                ),
                Cue(index=8, start_ms=2500, end_ms=3100, lines=["Ich bleibe"]),
            ]
        )
    )
    instructions = "\n".join(prompt["instructions"])

    assert prompt["modality"] == "text_only"
    assert prompt["cues"][0] == {
        "cue_id": 7,
        "sequence_position": 1,
        "start_ms": 1200,
        "end_ms": 2400,
        "duration_ms": 1200,
        "source_lines": ["Er sagte", "„Bleib hier.“"],
        "text": "Er sagte\n„Bleib hier.“",
        "speaker_id": "SPEAKER_00",
        "character": "Luna",
    }
    assert prompt["cues"][1]["sequence_position"] == 2
    assert "full ordered batch" in instructions
    assert "„ “" in instructions
    assert "source_lines" in instructions
    assert "alphanumeric token sequence" in instructions
    assert "quotation-mark sequence" in instructions


def test_punctuation_prompt_separates_editable_batch_from_complete_episode_context():
    episode = [
        Cue(index=1, start_ms=0, end_ms=800, lines=["Vorher."]),
        Cue(index=2, start_ms=900, end_ms=1_600, lines=["bleib hier"]),
        Cue(index=3, start_ms=1_700, end_ms=2_400, lines=["Nachher."]),
    ]

    prompt = json.loads(_punctuation_prompt([episode[1]], episode_context=episode))

    assert prompt["editable_cue_ids"] == [2]
    assert [item["cue_id"] for item in prompt["episode_context"]] == [1, 2, 3]
    assert prompt["cues"][0]["cue_id"] == 2


def test_punctuation_prompt_keeps_shared_context_before_editable_batch_payload():
    episode = [
        Cue(index=1, start_ms=0, end_ms=800, lines=["Vorher."]),
        Cue(index=2, start_ms=900, end_ms=1_600, lines=["bleib hier"]),
    ]

    prompt = json.loads(_punctuation_prompt([episode[1]], episode_context=episode))
    keys = list(prompt)

    assert prompt["prompt_version"] == "punctuation-v8-explicit-scene-isolation"
    assert keys.index("episode_context") < keys.index("editable_cue_ids")
    assert keys.index("episode_context") < keys.index("cues")


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
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
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
