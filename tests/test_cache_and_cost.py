from __future__ import annotations

import json
import wave
from pathlib import Path

from dubsync.cache import CacheKey, JsonDiskCache
from dubsync.cost import (
    CostMeter,
    asr_dollars_per_hour,
    llm_token_prices,
    record_llm_usage,
    token_usage_from_response,
)
from dubsync.llm_providers import drain_usage_events
from dubsync.models import Word
from dubsync.providers import CachedASRAdapter


class CountingASRAdapter:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path: Path) -> list[Word]:
        self.calls += 1
        return [Word(text="hello", start=0.0, end=0.5, confidence=0.98)]


def test_json_disk_cache_keys_audio_model_and_params(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio-a")
    cache = JsonDiskCache(tmp_path / "cache")
    key = CacheKey.from_audio(audio, model="scribe_v2", params={"diarize": True})

    cache.write(key, {"words": [{"text": "hello"}]})

    assert cache.read(key) == {"words": [{"text": "hello"}]}
    changed = CacheKey.from_audio(audio, model="scribe_v2", params={"diarize": False})
    assert cache.read(changed) is None


def test_cache_key_excludes_secret_params(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio-a")

    key = CacheKey.from_audio(
        audio,
        model="scribe_v2",
        params={
            "provider": "elevenlabs",
            "api_key": "secret-key",
            "diarize": True,
            "nested": {"token": "secret-token", "model": "scribe_v2"},
        },
    )
    same_without_secret_change = CacheKey.from_audio(
        audio,
        model="scribe_v2",
        params={
            "provider": "elevenlabs",
            "api_key": "rotated-key",
            "diarize": True,
            "nested": {"token": "rotated-token", "model": "scribe_v2"},
        },
    )
    changed_output_param = CacheKey.from_audio(
        audio,
        model="scribe_v2",
        params={"provider": "elevenlabs", "api_key": "secret-key", "diarize": False},
    )

    assert key.params == {"provider": "elevenlabs", "diarize": True, "nested": {"model": "scribe_v2"}}
    assert "secret-key" not in json.dumps(key.model_dump())
    assert "secret-token" not in json.dumps(key.model_dump())
    assert same_without_secret_change.digest == key.digest
    assert changed_output_param.digest != key.digest


def test_payload_cache_key_hashes_content_model_and_safe_params():
    payload = {"spans": [{"case_id": "case-1", "srt_text": "old", "asr_text": "new"}]}

    key = CacheKey.from_payload(
        payload,
        model="gemini-3.5-flash",
        params={"api_key": "secret-key", "provider": "gemini", "thinking_level": "low"},
    )
    same_without_secret_change = CacheKey.from_payload(
        payload,
        model="gemini-3.5-flash",
        params={"api_key": "rotated-key", "provider": "gemini", "thinking_level": "low"},
    )
    changed_payload = CacheKey.from_payload(
        {"spans": [{"case_id": "case-1", "srt_text": "old", "asr_text": "different"}]},
        model="gemini-3.5-flash",
        params={"provider": "gemini", "thinking_level": "low"},
    )
    changed_param = CacheKey.from_payload(
        payload,
        model="gemini-3.5-flash",
        params={"provider": "gemini", "thinking_level": "dynamic"},
    )

    assert key.content_sha256 is not None
    assert key.audio_sha256 is None
    assert key.params == {"provider": "gemini", "thinking_level": "low"}
    assert "secret-key" not in json.dumps(key.model_dump())
    assert same_without_secret_change.digest == key.digest
    assert changed_payload.digest != key.digest
    assert changed_param.digest != key.digest


def test_cost_meter_reports_audio_seconds_tokens_and_total():
    meter = CostMeter()
    meter.add_audio("scribe_v2", seconds=3600, dollars_per_hour=0.22)
    meter.add_tokens("gemini-3.5-flash", input_tokens=1000, output_tokens=500, input_per_million=1.5, output_per_million=9)

    data = json.loads(meter.to_json())

    assert data["total_usd"] == 0.226
    assert len(data["items"]) == 2


def test_token_usage_from_common_provider_response_shapes():
    class UsageObject:
        prompt_tokens = 100
        completion_tokens = 25

    class ResponseObject:
        usage = UsageObject()

    gemini = {"usage_metadata": {"input_token_count": 1200, "output_token_count": 300}}
    anthropic = {"usage": {"input_tokens": 80, "output_tokens": 20}}

    assert token_usage_from_response(gemini).input_tokens == 1200
    assert token_usage_from_response(gemini).output_tokens == 300
    assert token_usage_from_response(ResponseObject()).input_tokens == 100
    assert token_usage_from_response(ResponseObject()).output_tokens == 25
    assert token_usage_from_response(anthropic).input_tokens == 80
    assert token_usage_from_response(anthropic).output_tokens == 20
    assert token_usage_from_response({"usage": {"input_tokens": 1}}) is None


def test_llm_token_prices_use_plan_defaults_and_config_overrides():
    assert llm_token_prices("gemini", "gemini-3.5-flash", {}) == (1.5, 9.0)
    assert llm_token_prices("gemini", "gemini-3.1-flash-lite", {}) == (0.25, 1.5)
    assert llm_token_prices("openai", "gpt-5.5", {}) is None
    assert llm_token_prices("openai", "gpt-5.5", {"input_per_million": 2, "output_per_million": 12}) == (
        2.0,
        12.0,
    )


def test_record_llm_usage_adds_cost_only_when_usage_and_price_are_available():
    meter = CostMeter()

    record_llm_usage(
        meter,
        provider="gemini",
        model="gemini-3.5-flash",
        config={},
        response={"usage_metadata": {"input_token_count": 1000, "output_token_count": 500}},
    )
    record_llm_usage(
        meter,
        provider="openai",
        model="gpt-5.5",
        config={},
        response={"usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
    )
    record_llm_usage(
        meter,
        provider="openai",
        model="gpt-5.5",
        config={"input_per_million": 2, "output_per_million": 10},
        response={"usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
    )

    assert meter.as_dict()["items"] == [
        {
            "provider": "gemini-3.5-flash",
            "kind": "tokens",
            "units": {"input_tokens": 1000.0, "output_tokens": 500.0},
            "usd": 0.006,
        },
        {
            "provider": "gpt-5.5",
            "kind": "tokens",
            "units": {"input_tokens": 1000.0, "output_tokens": 500.0},
            "usd": 0.007,
        },
    ]


def test_drain_usage_events_returns_and_clears_adapter_events():
    class AdapterWithUsage:
        def __init__(self):
            self.usage_events = [{"usage": {"input_tokens": 10, "output_tokens": 5}}]

    adapter = AdapterWithUsage()

    assert drain_usage_events(adapter) == [{"usage": {"input_tokens": 10, "output_tokens": 5}}]
    assert drain_usage_events(adapter) == []
    assert drain_usage_events(object()) == []


def test_default_asr_prices_match_plan_cost_table():
    assert asr_dollars_per_hour("elevenlabs", {}) == 0.22
    assert asr_dollars_per_hour("openai", {}) == 0.36
    assert asr_dollars_per_hour("assemblyai", {}) == 0.23
    assert asr_dollars_per_hour("whisperx", {}) == 0.0
    assert asr_dollars_per_hour("elevenlabs", {"dollars_per_hour": 0.5}) == 0.5


def test_assemblyai_asr_prices_follow_plan_model_and_diarization_rates():
    assert asr_dollars_per_hour("assemblyai", {"model": "universal-3-pro", "speaker_labels": False}) == 0.21
    assert asr_dollars_per_hour("assemblyai", {"model": "universal-2", "speaker_labels": False}) == 0.15
    assert asr_dollars_per_hour("assemblyai", {"model": "universal-3-pro", "speaker_labels": True}) == 0.23
    assert asr_dollars_per_hour("assemblyai", {"model": "universal-2", "speaker_labels": True}) == 0.17


def test_assemblyai_default_cost_matches_adapter_speaker_label_default():
    assert asr_dollars_per_hour("assemblyai", {}) == 0.23


def test_elevenlabs_keyterm_prompting_adds_plan_surcharge():
    assert asr_dollars_per_hour("elevenlabs", {"keyterms": ["Luna"]}) == 0.27
    assert asr_dollars_per_hour("elevenlabs", {"character_names": ["Matthew"]}) == 0.27
    assert asr_dollars_per_hour("elevenlabs", {"keyterms": [], "character_names": [" "]}) == 0.22


def test_cached_asr_records_audio_cost_only_on_uncached_provider_call(tmp_path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    inner = CountingASRAdapter()
    meter = CostMeter()
    adapter = CachedASRAdapter(
        inner,
        JsonDiskCache(tmp_path / "cache"),
        model="scribe_v2",
        params={"provider": "elevenlabs"},
        cost_meter=meter,
        cost_provider="scribe_v2",
        dollars_per_hour=0.22,
    )

    adapter.transcribe(audio)
    adapter.transcribe(audio)

    assert inner.calls == 1
    assert meter.as_dict()["items"] == [
        {"provider": "scribe_v2", "kind": "audio", "units": {"seconds": 1.0}, "usd": 6.1e-05}
    ]
