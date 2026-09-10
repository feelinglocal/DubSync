from __future__ import annotations

import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from dubsync.llm_providers import GeminiLLMAdapter, _adjudication_prompt, _punctuation_prompt, llm_adapter_from_config
from dubsync.models import AudioSnippet, Cue, CueContext, DivergenceSpan
from dubsync.providers import ProviderError


@pytest.fixture
def gemini_audio_sdk(monkeypatch):
    clients = []
    calls = []
    uploads = []
    deleted = []
    cache_creates = []
    cache_updates = []
    failures = {"generation": False}

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.options = http_options
            self.closed = False
            self.models = types.SimpleNamespace(generate_content=self.generate)
            self.files = types.SimpleNamespace(upload=self.upload, delete=lambda **kw: deleted.append(kw["name"]))
            self.caches = types.SimpleNamespace(create=self.cache, delete=lambda **kw: deleted.append(kw["name"]),
                                               update=lambda **kw: cache_updates.append(kw))
            clients.append(self)

        def generate(self, **kwargs):
            calls.append(kwargs)
            if failures.get("barrier"):
                failures["barrier"].wait(timeout=5)
            if failures["generation"]:
                raise RuntimeError("private failure")
            return types.SimpleNamespace(text='{"decisions": [], "cues": []}', usage_metadata={"prompt_token_count": 100})

        def upload(self, **kwargs):
            uploads.append(kwargs)
            return types.SimpleNamespace(name=f"files/{len(uploads)}", uri=f"https://example.test/audio/{len(uploads)}", state="ACTIVE")

        def cache(self, **kwargs):
            cache_creates.append(kwargs)
            return types.SimpleNamespace(name="cachedContents/owned", usage_metadata={"total_token_count": 10000})

        def close(self):
            self.closed = True

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = types.ModuleType("google.genai.types")
    fake_genai.types.Part = types.SimpleNamespace(
        from_bytes=lambda **kwargs: {"bytes": kwargs}, from_uri=lambda **kwargs: {"uri": kwargs})
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai.types)
    return types.SimpleNamespace(clients=clients, calls=calls, uploads=uploads,
                                 deleted=deleted, cache_creates=cache_creates, cache_updates=cache_updates,
                                 failures=failures)


def test_full_audio_uri_is_reused_with_case_offsets_and_no_automatic_paid_retries(gemini_audio_sdk, tmp_path):
    sdk = gemini_audio_sdk
    original = tmp_path / "episode.wav"
    original.write_bytes(b"original WAV")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"focused WAV")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash", thinking_level="medium")
    adapter.set_audio_context(original, duration_seconds=60)
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    snippet = AudioSnippet(case_id="case-1", path=str(clip), start=12.0, end=14.0)
    adapter.adjudicate_with_audio([span], {"case-1": snippet})
    adapter.adjudicate_with_audio([span], {"case-1": snippet})
    assert len(sdk.uploads) == 1
    assert sdk.uploads[0]["file"] == original
    assert not sdk.cache_creates
    contents = sdk.calls[0]["contents"]
    assert json.loads(contents[1])["audio_role"] == "full_episode_read_only_context"
    assert contents[2]["uri"]["file_uri"] == "https://example.test/audio/1"
    assert json.loads(contents[3])["case_id"] == "case-1"
    assert json.loads(contents[3])["local_time_zero_is_episode_seconds"] == 12.0
    assert contents[4]["bytes"]["data"] == b"focused WAV"
    assert sdk.calls[0]["config"]["thinking_config"] == {"thinking_level": "medium"}
    assert all(client.options["retry_options"]["attempts"] == 1 for client in sdk.clients)
    adapter.close()
    assert sdk.deleted == ["files/1"]
    assert all(client.closed for client in sdk.clients)


def test_owned_audio_cache_applies_only_to_adjudication_and_cleans_after_generation_failure(gemini_audio_sdk, tmp_path):
    sdk = gemini_audio_sdk
    original = tmp_path / "episode.mp3"
    original.write_bytes(b"original MP3")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash")
    adapter.set_audio_context(original, duration_seconds=300)
    adapter.set_episode_context([Cue(index=1, start_ms=0, end_ms=1000, lines=["source"] )])
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    adapter.adjudicate([span])
    assert sdk.calls[0]["config"]["cached_content"] == "cachedContents/owned"
    assert not any(isinstance(part, dict) and "uri" in part for part in sdk.calls[0]["contents"])
    assert "source" in sdk.cache_creates[0]["config"]["contents"][0]["parts"][0]["text"]
    assert json.loads(sdk.calls[0]["contents"][0])["episode_context"] == []
    adapter.punctuate([Cue(index=1, start_ms=0, end_ms=1000, lines=["source"])])
    assert "cached_content" not in sdk.calls[1]["config"]
    sdk.failures["generation"] = True
    with pytest.raises(ProviderError):
        adapter.adjudicate([span])
    adapter.close()
    assert sdk.deleted == ["cachedContents/owned", "files/1"]
    assert all(client.closed for client in sdk.clients)


def test_failed_uncached_generation_reserves_unreported_audio_cost(gemini_audio_sdk, tmp_path):
    sdk = gemini_audio_sdk
    path = tmp_path / "episode.wav"
    path.write_bytes(b"original")
    adapter = GeminiLLMAdapter(api_key="test")
    adapter.set_audio_context(path, duration_seconds=60)
    sdk.failures["generation"] = True
    with pytest.raises(ProviderError):
        adapter.adjudicate([DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)])
    assert adapter.audio_context_report()["unreported_uncached_audio_tokens_reserved"] == 1920
    adapter.close()


@pytest.mark.parametrize("generation_fails", [False, True])
def test_large_aggregate_clips_upload_instead_of_loading_inline_and_always_delete(gemini_audio_sdk, monkeypatch, tmp_path, generation_fails):
    import dubsync.llm_providers as module

    sdk = gemini_audio_sdk
    sdk.failures["generation"] = generation_fails
    path = tmp_path / "case.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(module, "_GEMINI_INLINE_REQUEST_BYTES", 1)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("oversized clips must upload from path"))
    adapter = GeminiLLMAdapter(api_key="test")
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    snippet = AudioSnippet(case_id="case-1", path=str(path), start=10, end=12)
    if generation_fails:
        with pytest.raises(ProviderError):
            adapter.adjudicate_with_audio([span], {"case-1": snippet})
    else:
        adapter.adjudicate_with_audio([span], {"case-1": snippet})
    assert sdk.uploads[0]["file"] == path
    assert sdk.deleted == ["files/1"]
    assert all(client.closed for client in sdk.clients)
    assert json.loads(sdk.calls[0]["contents"][1])["episode_start_seconds"] == 10
    assert sdk.calls[0]["contents"][2]["uri"]["file_uri"] == "https://example.test/audio/1"


def test_audio_context_disabled_or_external_cache_does_not_create_owned_resources(tmp_path):
    path = tmp_path / "episode.wav"
    adapter = GeminiLLMAdapter(api_key="test", cached_content="cachedContents/user-owned")
    adapter.set_audio_context(path, duration_seconds=60, config={"enabled": False})
    assert adapter.audio_context_report() == {"enabled": False}
    with pytest.raises(ProviderError, match="user-supplied"):
        adapter.set_audio_context(path, duration_seconds=60)
    adapter.close()


def test_cached_audio_requests_do_not_retry_and_repeated_failures_close_the_job_circuit(gemini_audio_sdk, tmp_path):
    sdk = gemini_audio_sdk
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"original")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash", max_retries=2)
    adapter.set_audio_context(path, duration_seconds=300)
    sdk.failures["generation"] = True
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    for _ in range(3):
        with pytest.raises(ProviderError):
            adapter.adjudicate([span])
    assert len(sdk.calls) == 2
    assert all(client.options["retry_options"]["attempts"] == 1 for client in sdk.clients)
    report = adapter.audio_context_report()
    assert report["failed_requests"] == 2
    assert "context_request_circuit_open" in report["warnings"]
    assert report["unreported_cached_audio_tokens_reserved"] == 20000
    assert report["cleanup_status"] == "complete"
    assert sdk.deleted == ["cachedContents/owned", "files/1"]


def test_cache_lease_is_renewed_after_slow_focus_upload_before_generation(gemini_audio_sdk, monkeypatch, tmp_path):
    import dubsync.llm_providers as module

    sdk = gemini_audio_sdk
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"original")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"focus")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash")
    adapter.set_audio_context(path, duration_seconds=300)
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    adapter.adjudicate([span])
    upload = module.GeminiSnippetUploads.upload

    def slow_upload(self, *args):
        result = upload(self, *args)
        clock[0] = 1000.0
        return result

    monkeypatch.setattr(module.GeminiSnippetUploads, "upload", slow_upload)
    monkeypatch.setattr(module, "_GEMINI_INLINE_REQUEST_BYTES", 1)
    adapter.adjudicate_with_audio([span], {"case-1": AudioSnippet(case_id="case-1", path=str(clip), start=10, end=12)})
    assert len(sdk.cache_updates) == 1
    assert sdk.cache_updates[0]["name"] == "cachedContents/owned"
    assert set(sdk.cache_updates[0]["config"]) == {"expire_time"}
    assert sdk.calls[-1]["config"]["cached_content"] == "cachedContents/owned"
    adapter.close()


def test_failed_clip_preparation_never_reserves_full_audio_input(gemini_audio_sdk, monkeypatch, tmp_path):
    import dubsync.llm_providers as module

    sdk = gemini_audio_sdk
    path = tmp_path / "episode.wav"
    path.write_bytes(b"original")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"focus")
    adapter = GeminiLLMAdapter(api_key="test")
    adapter.set_audio_context(path, duration_seconds=60)
    monkeypatch.setattr(module, "_GEMINI_INLINE_REQUEST_BYTES", 1)

    def fail(*_args):
        raise ProviderError("focused clip failed")

    monkeypatch.setattr(module.GeminiSnippetUploads, "upload", fail)
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    with pytest.raises(ProviderError):
        adapter.adjudicate_with_audio([span], {"case-1": AudioSnippet(case_id="case-1", path=str(clip), start=10, end=12)})
    assert adapter.audio_context_report()["uncached_audio_tokens_reserved"] == 0
    assert sdk.calls == []
    adapter.close()


def test_closed_audio_context_fails_before_uploading_more_focused_clips(gemini_audio_sdk, monkeypatch, tmp_path):
    import dubsync.llm_providers as module

    path = tmp_path / "episode.wav"
    path.write_bytes(b"original")
    adapter = GeminiLLMAdapter(api_key="test")
    adapter.set_audio_context(path, duration_seconds=60)
    adapter.close()
    monkeypatch.setattr(module, "_GEMINI_INLINE_REQUEST_BYTES", 1)
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)
    with pytest.raises(ProviderError):
        adapter.adjudicate_with_audio([span], {"case-1": AudioSnippet(case_id="case-1", path=str(path), start=10, end=12)})
    assert gemini_audio_sdk.uploads == []


def test_parallel_adapter_calls_share_context_and_preserve_every_usage_event(gemini_audio_sdk, tmp_path):
    from dubsync.llm_providers import drain_usage_events

    sdk = gemini_audio_sdk
    sdk.failures["barrier"] = Barrier(4)
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"original")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash")
    adapter.set_audio_context(path, duration_seconds=300)
    spans = [DivergenceSpan(case_id=f"case-{i}", cue_ids=[i], srt_text="hello", asr_text="hi", confidence=.9) for i in range(1, 5)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda span: adapter.adjudicate([span]), spans))
    assert results == [[], [], [], []]
    assert len(sdk.calls) == 4
    assert len(sdk.uploads) == len(sdk.cache_creates) == 1
    assert adapter.audio_context_report()["in_flight_requests"] == 0
    assert len(drain_usage_events(adapter)) == 4
    assert drain_usage_events(adapter) == []
    audit = adapter.audio_context_report()["request_usage_events"]
    assert len(audit) == 4
    assert audit[0]["usage"]["usage_metadata"]["prompt_token_count"] == 100
    audit[0]["usage"]["usage_metadata"]["prompt_token_count"] = 999
    assert adapter.audio_context_report()["request_usage_events"][0]["usage"]["usage_metadata"]["prompt_token_count"] == 100
    adapter.close()
    assert sdk.deleted == ["cachedContents/owned", "files/1"]
    assert all(client.closed for client in sdk.clients)


def test_generation_latency_excludes_wait_for_shared_bookkeeping_lock(gemini_audio_sdk, monkeypatch, tmp_path):
    import dubsync.llm_providers as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    client_class = sys.modules["google.genai"].Client
    generate = client_class.generate

    def one_second_generate(self, **kwargs):
        clock[0] += 1
        return generate(self, **kwargs)

    monkeypatch.setattr(client_class, "generate", one_second_generate)
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"original")
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash")
    adapter.set_audio_context(path, duration_seconds=300)
    record = adapter.audio_context.record_generation_result

    def delayed_record(**kwargs):
        clock[0] += 5
        return record(**kwargs)

    monkeypatch.setattr(adapter.audio_context, "record_generation_result", delayed_record)
    adapter.adjudicate([DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="hello", asr_text="hi", confidence=.9)])
    assert adapter.audio_context_report()["total_generation_seconds"] == 1
    adapter.close()


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
    label = json.loads(calls[0]["contents"][1])
    assert label["case_id"] == "case-1"
    assert label["local_time_zero_is_episode_seconds"] == 0.0
    assert calls[0]["contents"][2] == {"inline_data": b"RIFFsnippetWAVEfmt ", "mime_type": "audio/wav"}
    assert decisions[0]["reason"] == "audio snippet confirms the spoken line"


def test_gemini_audio_read_oserror_is_wrapped_before_opening_a_request_client(monkeypatch, tmp_path):
    closed_clients: list[str] = []
    created_clients: list[str] = []
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
            created_clients.append(api_key)
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

    assert created_clients == closed_clients == []


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


def test_adjudication_prompt_preserves_audible_improvisation_and_reactions():
    prompt = json.loads(_adjudication_prompt([
        DivergenceSpan(case_id="eu", cue_ids=[], srt_text="", asr_text="Eu"),
        DivergenceSpan(case_id="reactions", cue_ids=[604], srt_text="Muito bonito",
                       asr_text="Hã? Uau, que lindo! Ah,"),
    ]))
    instructions = "\n".join(prompt["instructions"])
    assert "without changing timing or cue structure" not in prompt["task"]
    assert "Do not omit audible short reactions, pronouns, hesitations, or improvised words" in instructions
    assert "Cue allocation and speaker splitting happen downstream" in instructions
    assert "Do not choose keep_srt merely because" in instructions
    assert "every audible word inside the supplied ASR span" in instructions


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

    assert prompt["prompt_version"] == "adjudication-v10-audible-span-ownership"
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
