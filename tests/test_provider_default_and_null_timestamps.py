from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest

from dubsync.mai_transcribe import MAITranscribeAdapter
from dubsync.providers import (
    ElevenLabsScribeAdapter,
    FixtureASRAdapter,
    MAI_TRANSCRIBE_MODEL,
    OpenAIWhisperAdapter,
    ProviderError,
    adapter_from_config,
    apply_asr_language,
    apply_transcription_provider_config,
)


@pytest.mark.parametrize("config", [
    {},
    {"asr": {}},
    {"asr": {"diarize": False}},
    {"asr": {"model": MAI_TRANSCRIBE_MODEL}},
])
def test_providerless_config_uses_mai_without_mutating_settings(config):
    before = copy.deepcopy(config)

    configured = apply_asr_language(apply_transcription_provider_config(config, "default"), "pt-BR")
    adapter = adapter_from_config(configured)
    direct = adapter_from_config(config)

    assert isinstance(adapter, MAITranscribeAdapter)
    assert isinstance(direct, MAITranscribeAdapter)
    assert adapter.model == MAI_TRANSCRIBE_MODEL
    assert adapter.language_code == "pt"
    assert adapter.diarize is config.get("asr", {}).get("diarize", True)
    # Sync and generation use these fields for cache identities and cost labels.
    assert configured["asr"]["provider"] == "openrouter"
    assert configured["asr"]["model"] == MAI_TRANSCRIBE_MODEL
    assert config == before


@pytest.mark.parametrize("asr,adapter_type,model", [
    ({"provider": "elevenlabs"}, ElevenLabsScribeAdapter, "scribe_v2"),
    ({"provider": "elevenlabs", "model_id": "scribe_v1"}, ElevenLabsScribeAdapter, "scribe_v1"),
    ({"model_id": "scribe_v2"}, ElevenLabsScribeAdapter, "scribe_v2"),
    ({"model_id": "scribe_v1"}, ElevenLabsScribeAdapter, "scribe_v1"),
    ({"provider": "openai", "model": "whisper-1"}, OpenAIWhisperAdapter, "whisper-1"),
])
def test_default_retains_explicit_provider_or_legacy_scribe_model(asr, adapter_type, model):
    config = {"asr": asr}
    before = copy.deepcopy(config)
    configured = apply_transcription_provider_config(config, "default")

    adapter = adapter_from_config(configured)

    assert isinstance(adapter, adapter_type)
    assert getattr(adapter, "model_id", getattr(adapter, "model", None)) == model
    assert configured["asr"]["provider"] == asr.get("provider", "elevenlabs")
    assert all(configured["asr"][key] == value for key, value in asr.items())
    assert config == before


def test_fixture_asr_takes_precedence_over_default_provider(tmp_path):
    fixture = tmp_path / "words.json"
    config = {"asr": {"fixture_path": str(fixture)}}

    configured = apply_transcription_provider_config(config, "default")
    adapter = adapter_from_config(configured)

    assert configured == config
    assert isinstance(adapter, FixtureASRAdapter)
    assert adapter.fixture_path == fixture


@pytest.mark.parametrize("nullable_fields", [("start",), ("end",), ("start", "end")])
@pytest.mark.parametrize("object_response", [False, True], ids=["dict", "sdk-object"])
def test_scribe_null_word_timestamp_is_typed_failure_without_retry(
    monkeypatch, tmp_path, nullable_fields, object_response,
):
    bad_word = {"type": "word", "text": "private-response-text", "start": 0.3, "end": 0.8}
    bad_word.update(dict.fromkeys(nullable_fields))
    valid_word = {"type": "word", "text": "Hallo", "start": 0.0, "end": 0.2}
    records = [valid_word, bad_word]
    words = [SimpleNamespace(**record) for record in records] if object_response else records
    calls = []

    def convert(**kwargs):
        calls.append(kwargs["file"].read())
        return SimpleNamespace(words=words)

    monkeypatch.setitem(sys.modules, "elevenlabs", SimpleNamespace(
        ElevenLabs=lambda **kwargs: SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert)),
    ))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture-audio")
    adapter = ElevenLabsScribeAdapter(api_key="fixture-key")

    with pytest.raises(ProviderError, match="missing or invalid word timing") as caught:
        adapter.transcribe(audio)

    assert caught.value.code == "invalid_response"
    assert "private-response-text" not in str(caught.value)
    assert "fixture-key" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert calls == [b"fixture-audio"]
    assert adapter.last_usage == {"request_count": 1}


def test_scribe_null_audio_event_timestamp_keeps_valid_words(monkeypatch, tmp_path):
    response = SimpleNamespace(words=[
        {"type": "audio_event", "text": "(laughs)", "start": None, "end": None},
        {"type": "word", "text": "Hallo", "start": 0.1, "end": 0.3},
    ])
    monkeypatch.setitem(sys.modules, "elevenlabs", SimpleNamespace(
        ElevenLabs=lambda **kwargs: SimpleNamespace(
            speech_to_text=SimpleNamespace(convert=lambda **kwargs: response),
        ),
    ))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture-audio")
    adapter = ElevenLabsScribeAdapter(api_key="fixture-key")

    words = adapter.transcribe(audio)

    assert [(word.text, word.start, word.end) for word in words] == [("Hallo", 0.1, 0.3)]
    assert adapter.last_evidence["audio_events"] == [
        {"text": "(laughs)", "start": None, "end": None, "speaker_id": None},
    ]
