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


@pytest.mark.parametrize("language,expected", [
    ("deu", "de"), ("ger", "de"), ("de-DE", "de"), ("ja-JP", "ja"), ("jpn", "ja"), ("JA_jp", "ja"),
    ("pt-BR", "pt"), ("pt_br", "pt"), ("por", "pt"), ("spa", "es"), ("fra", "fr"), ("fre", "fr"),
    ("eng", "en"), ("en-US", "en"), ("ind", "id"), ("zh-Hant", "zh"), ("PT", "pt"), (" de ", "de"),
    ("auto", None), ("", None), (None, None), ("yue", "yue"),
])
def test_asr_language_codes_are_normalised_to_iso_639_1_for_providers(language, expected):
    from dubsync.providers import asr_language_code

    assert asr_language_code(language) == expected


@pytest.mark.parametrize("selection,env", [(MAI, "OPENROUTER_API_KEY"), ("scribe_v2", "ELEVENLABS_API_KEY")])
@pytest.mark.parametrize("language,expected", [("pt-BR", "pt"), ("deu", "de"), ("ja-JP", "ja")])
def test_region_and_iso_639_3_codes_reach_each_provider_as_iso_639_1(monkeypatch, selection, env, language, expected):
    monkeypatch.setenv(env, "test-key")
    config = apply_asr_language(apply_transcription_provider_config({}, selection), language)
    assert config["asr"]["language_code"] == expected
    assert adapter_from_config(config).language_code == expected
    provider = config["asr"]["provider"]
    assert adapter_from_config({"asr": {"provider": provider, "language_code": language}}).language_code == expected
    assert adapter_from_config({"asr": {"provider": provider, "language": language}}).language_code == expected


def test_mai_request_carries_the_normalised_language(monkeypatch, tmp_path):
    import io
    import json

    calls = []

    class Response(io.BytesIO):
        headers = {}

    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return Response(b'{"words": []}')

    monkeypatch.setattr("dubsync.mai_transcribe.urlopen", urlopen)
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)
    adapter_from_config({"asr": {"provider": "openrouter", "api_key": "test-key", "language_code": "por"}}).transcribe(audio)
    assert calls[0]["language"] == "pt"


def test_forced_alignment_language_keeps_its_iso_639_3_code():
    from dubsync.forced_alignment import MMSForcedAlignmentAdapter

    assert MMSForcedAlignmentAdapter(language="deu").language == "deu"
    assert MMSForcedAlignmentAdapter(language="ja-JP").language == "jpn"


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


def test_provider_evidence_is_saved_with_the_cached_transcription_and_restored_on_hit(tmp_path):
    evidence = {"provider": "elevenlabs", "word_logprobs": [{"text": "hi", "start": 0.0, "end": 0.2, "logprob": -0.1}]}

    class EvidenceAdapter:
        last_evidence = evidence
        calls = 0

        def transcribe(self, path):
            self.calls += 1
            return [Word(text="hi", start=0.0, end=0.2)]

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"example")
    inner = EvidenceAdapter()
    adapter = CachedASRAdapter(inner, JsonDiskCache(tmp_path / "cache"), "scribe_v2", {})

    adapter.transcribe(audio)
    assert adapter.last_evidence == evidence
    [entry] = (tmp_path / "cache").glob("*.json")
    assert '"provider_evidence"' in entry.read_text(encoding="utf-8")

    restarted = CachedASRAdapter(EvidenceAdapter(), JsonDiskCache(tmp_path / "cache"), "scribe_v2", {})
    restarted.inner.last_evidence = None
    restarted.transcribe(audio)
    assert restarted.last_cache_hit is True
    assert restarted.last_evidence == evidence


def test_unserializable_provider_evidence_is_ignored(tmp_path):
    class OddAdapter:
        last_evidence = {"value": float("nan")}

        def transcribe(self, path):
            return [Word(text="hi", start=0.0, end=0.2)]

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"example")
    adapter = CachedASRAdapter(OddAdapter(), JsonDiskCache(tmp_path / "cache"), "fixture", {})
    adapter.transcribe(audio)
    assert adapter.last_evidence is None


def test_generate_mode_asr_artifact_keeps_provider_evidence(tmp_path, monkeypatch):
    import json
    import dubsync.transcription as transcription

    audio = tmp_path / "episode.wav"
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)

    class EvidenceAdapter:
        last_evidence = {"provider": "elevenlabs", "audio_events": [{"text": "(lacht)", "start": 0.0, "end": 0.2, "speaker_id": None}]}

        def transcribe(self, path):
            return [Word(text="Hallo.", start=0.3, end=0.7)]

    monkeypatch.setattr(transcription, "adapter_from_config", lambda *args, **kwargs: EvidenceAdapter())
    monkeypatch.setattr(transcription, "normalize_audio", lambda path, *args, **kwargs: path)
    transcription.generate_srt_from_audio(
        audio_path=audio, output_path=tmp_path / "output.srt", workdir=tmp_path / "work", no_llm=True,
        transcription_provider="scribe_v2",
    )
    asr = json.loads((tmp_path / "work/episode/asr.json").read_text(encoding="utf-8"))
    assert asr["metadata"]["provider_evidence"] == EvidenceAdapter.last_evidence


def test_cached_words_from_an_older_adapter_version_are_not_reused(tmp_path):
    from dubsync.cache import CacheKey
    from dubsync.mai_transcribe import MAITranscribeAdapter

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"example")

    class VersionedAdapter:
        cache_version = MAITranscribeAdapter(api_key="test-key").cache_version
        calls = 0

        def transcribe(self, path):
            self.calls += 1
            return [Word(text="once", start=0.1, end=0.4)]

    cache = JsonDiskCache(tmp_path / "cache")
    # An entry written before the raw-order word filters existed still holds the duplicate copy.
    cache.write(CacheKey.from_audio(audio, MAI, {"diarize": False}), {"words": [
        {"text": "once", "start": 0.1, "end": 0.45}, {"text": "once", "start": 0.1, "end": 0.4},
    ]})
    inner = VersionedAdapter()
    adapter = CachedASRAdapter(inner, cache, MAI, {"diarize": False})

    assert [word.text for word in adapter.transcribe(audio)] == ["once"]
    assert inner.calls == 1
    assert adapter.last_cache_hit is False
    assert [word.text for word in adapter.transcribe(audio)] == ["once"]
    assert inner.calls == 1
    assert adapter.last_cache_hit is True
    assert VersionedAdapter.cache_version


@pytest.mark.parametrize("succeeded", [True, False])
def test_retried_requests_with_unknown_billing_are_metered_as_uncertain_cost(tmp_path, succeeded):
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)

    class RetriedAdapter:
        last_usage = {}

        def transcribe(self, path):
            # One timed-out attempt (billing unknown), then a reported paid response.
            self.last_usage = {
                "seconds": None, "cost": None, "reported_seconds": 3.0, "reported_cost": 0.0001,
                "request_count": 2, "uncertain_request_count": 1, "uncertain_seconds": 3.0, "generation_ids": [],
            }
            if not succeeded:
                raise ProviderError("timed out")
            return [Word(text="hello", start=0, end=0.5)]

    meter = CostMeter()
    adapter = CachedASRAdapter(RetriedAdapter(), JsonDiskCache(tmp_path / "cache"), MAI, {}, cost_meter=meter, dollars_per_hour=0.1)
    if succeeded:
        adapter.transcribe(audio)
    else:
        with pytest.raises(ProviderError):
            adapter.transcribe(audio)

    assert [item.kind for item in meter.items] == ["audio_billed_partial", "audio_uncertain_estimate"]
    assert meter.items[0].usd == 0.0001
    assert meter.items[1].units["seconds"] == 3.0
    assert meter.items[1].usd == round(3.0 / 3600 * 0.1, 6)
    assert adapter.last_usage["uncertain_request_count"] == 1
    assert adapter.last_usage["uncertain_seconds"] == 3.0


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
