from __future__ import annotations

import copy
import wave

import pytest

from dubsync.cache import JsonDiskCache
from dubsync.cost import CostMeter, asr_dollars_per_hour
from dubsync.models import QCFlag, Word
from dubsync.providers import CachedASRAdapter, ProviderError, adapter_from_config, apply_asr_language, apply_transcription_provider_config

MAI = "microsoft/mai-transcribe-2"


@pytest.mark.parametrize("language", ["auto", "de", "pt"])
def test_factory_and_language_compose_for_explicit_mai(language, monkeypatch):
    from dubsync.mai_transcribe import MAITranscribeAdapter
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = apply_transcription_provider_config({}, MAI)
    configured = apply_asr_language(config, language)
    adapter = adapter_from_config(configured)
    assert isinstance(adapter, MAITranscribeAdapter)
    assert adapter.language_code == (None if language == "auto" else language)
    assert "language_code" not in config["asr"]


@pytest.mark.parametrize("language", ["auto", "de", "pt"])
def test_unconfigured_default_uses_scribe_with_language(language, monkeypatch):
    from dubsync.providers import ElevenLabsScribeAdapter

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    config = apply_transcription_provider_config({}, "default")
    adapter = adapter_from_config(apply_asr_language(config, language))
    assert isinstance(adapter, ElevenLabsScribeAdapter)
    assert adapter.model_id == "scribe_v2"
    assert adapter.diarize is True
    assert adapter.language_code == (None if language == "auto" else language)
    assert "language_code" not in config["asr"]
    assert isinstance(adapter_from_config({}), ElevenLabsScribeAdapter)


def test_factory_rejects_unexpected_openrouter_model():
    with pytest.raises(ProviderError, match="must be microsoft/mai-transcribe-2"):
        adapter_from_config({"asr": {"provider": "openrouter", "model": "other"}})


def test_explicit_selection_keeps_matching_alias_credentials():
    selected = apply_transcription_provider_config({"asr": {"provider": MAI, "api_key": "configured"}}, MAI)
    assert selected["asr"]["api_key"] == "configured"
    assert selected["asr"]["provider"] == "openrouter"


@pytest.mark.parametrize("mode", ["generate", "sync"])
def test_failed_paid_asr_preserves_usage_and_cost_artifact(tmp_path, monkeypatch, mode):
    import json
    import dubsync.pipeline as pipeline
    import dubsync.transcription as transcription

    audio = tmp_path / "episode.wav"
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)

    class PartiallyBilled:
        last_usage = {}
        def transcribe(self, path):
            self.last_usage = {"cost": None, "seconds": None, "reported_cost": 0.001, "reported_seconds": 36, "request_count": 2, "generation_ids": ["gen-paid"]}
            raise ProviderError("timed out")

    module = transcription if mode == "generate" else pipeline
    monkeypatch.setattr(module, "adapter_from_config", lambda *args, **kwargs: PartiallyBilled())
    monkeypatch.setattr(module, "normalize_audio", lambda path, *args, **kwargs: path)
    kwargs = dict(audio_path=audio, output_path=tmp_path / "output.srt", workdir=tmp_path / "work", no_llm=True, transcription_provider=MAI)
    with pytest.raises(ProviderError, match="timed out"):
        if mode == "generate":
            transcription.generate_srt_from_audio(**kwargs)
        else:
            pipeline.sync_episode(srt_path=source, **kwargs)
    failure = json.loads((tmp_path / "work/episode/asr_failure.json").read_text())
    assert failure["usage"]["cost"] is None
    assert failure["cost"]["total_usd"] == 0.001
    assert failure["cost"]["items"][0]["kind"] == "audio_billed_partial"
    assert not (tmp_path / "output.srt").exists()


def test_model_switch_preserves_other_passes_but_not_other_provider_secrets():
    original = {"asr": {"provider": "elevenlabs", "model_id": "scribe_v2", "api_key": "private", "dollars_per_hour": 0.22, "diarize": True, "keyterms": ["Luna"]}, "llm": {"model": "existing"}}
    snapshot = copy.deepcopy(original)
    selected = apply_transcription_provider_config(original, MAI)
    assert selected["asr"] == {"provider": "openrouter", "model": MAI, "diarize": True, "keyterms": ["Luna"]}
    assert selected["llm"] == original["llm"]
    assert original == snapshot
    selected["asr"]["keyterms"].append("new")
    assert original == snapshot


def test_scribe_is_default_and_explicit_selection_removes_mai_credentials():
    assert apply_transcription_provider_config({}, "default")["asr"] == {"provider": "elevenlabs", "model_id": "scribe_v2"}
    selected = apply_transcription_provider_config({"asr": {"provider": "openrouter", "api_key": "private", "chunk_seconds": 60}}, "scribe_v2")
    assert selected["asr"] == {"provider": "elevenlabs", "model_id": "scribe_v2"}
    assert asr_dollars_per_hour("openrouter", {"model": MAI}) == 0.1


def test_provider_selection_preserves_matching_configuration_and_fixtures():
    config = {"asr": {"provider": "openrouter", "model": MAI, "api_key": "configured", "chunk_seconds": 120}}
    assert apply_transcription_provider_config(config, MAI) == config
    assert apply_transcription_provider_config(config, "default") == config
    fixture = {"asr": {"fixture_path": "test.json"}}
    assert apply_transcription_provider_config(fixture, MAI)["asr"]["fixture_path"] == "test.json"
    assert apply_transcription_provider_config(fixture, "default") == fixture


def test_provider_timing_correction_remains_reviewable_on_cache_hit(tmp_path):
    flag = QCFlag(kind="asr_timestamp_rounding_clamped", cue_ids=[], severity="info", message="End corrected from 1.01 to 1.00 at audio boundary.")
    class RoundedAdapter:
        last_repair_flags = [flag]
        def transcribe(self, path):
            return [Word(text="end", start=0.8, end=1.0)]
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"example")
    cached = CachedASRAdapter(RoundedAdapter(), JsonDiskCache(tmp_path / "cache"), MAI, {})
    cached.transcribe(audio)
    assert cached.last_repair_flags == [flag]
    cached.transcribe(audio)
    assert cached.last_repair_flags == [flag]
    assert cached.last_cache_hit


@pytest.mark.parametrize("failure", [False, True])
def test_cache_records_reported_cost_once_including_failed_paid_calls(tmp_path, failure):
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)

    class BilledAdapter:
        last_usage = {}
        def transcribe(self, path):
            self.last_usage = {"seconds": 2.0, "cost": 0.001234, "request_count": 1, "generation_ids": ["gen-test"]}
            if failure:
                raise RuntimeError("bad timestamps")
            return [Word(text="hello", start=0, end=0.5)]

    meter = CostMeter()
    adapter = CachedASRAdapter(BilledAdapter(), JsonDiskCache(tmp_path / "cache"), MAI, {}, cost_meter=meter, dollars_per_hour=0.1)
    if failure:
        with pytest.raises(RuntimeError, match="bad timestamps"):
            adapter.transcribe(audio)
    else:
        adapter.transcribe(audio)
        adapter.transcribe(audio)
        assert adapter.last_usage["generation_ids"] == ["gen-test"]
        assert adapter.last_cache_hit is True
    assert meter.total_usd == 0.001234
    assert len(meter.items) == 1
    assert meter.items[0].kind == "audio_billed"
    assert meter.items[0].units["seconds"] == 2
