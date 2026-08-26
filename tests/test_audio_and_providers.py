from __future__ import annotations

import subprocess
import sys
import wave
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubsync.audio import AudioNormalizeError, normalize_audio, probe_audio_duration
from dubsync.cache import JsonDiskCache
from dubsync.cost import CostMeter
from dubsync.models import Word
from dubsync.providers import (
    AssemblyAIAdapter,
    CachedASRAdapter,
    GeminiTranscribeAdapter,
    ProviderError,
    WhisperXAdapter,
    adapter_from_config,
    apply_local_asr_config,
    repair_word_stream,
)


class CountingAdapter:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path):
        self.calls += 1
        return [Word(text="cached", start=0.0, end=0.2, confidence=0.9, speaker_id="A")]


class StaticWordAdapter:
    def __init__(self, words: list[Word]):
        self.words = words

    def transcribe(self, audio_path):
        del audio_path
        return list(self.words)


class GeneratorWordAdapter:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path):
        del audio_path
        self.calls += 1
        return (
            word
            for word in [
                Word(text="generated", start=0.0, end=0.2, confidence=0.9, speaker_id="A")
            ]
        )


class InvalidThenValidAdapter:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path):
        del audio_path
        self.calls += 1
        if self.calls == 1:
            return "not-a-word-stream"
        return [Word(text="valid", start=0.0, end=0.2, confidence=0.9)]


def _write_valid_wav(path: Path, *, duration_seconds: float = 1.0) -> None:
    sample_rate = 16_000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(sample_rate * duration_seconds))


def test_normalize_audio_uses_ffmpeg_16khz_mono(tmp_path, monkeypatch):
    source = tmp_path / "in.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")
    calls = []

    timeouts = []

    def fake_run(cmd, check, capture_output, text, timeout=None):
        calls.append(cmd)
        timeouts.append(timeout)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "1.0", "")
        Path(cmd[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = normalize_audio(source, dest)

    assert result == dest
    ffmpeg_call = next(call for call in calls if call[0] == "ffmpeg")
    assert ffmpeg_call == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        "14401",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-fs",
        "1080576",
        "-f",
        "wav",
        str(dest.with_name(f"{dest.name}.partial")),
    ]
    assert all(isinstance(timeout, float) for timeout in timeouts)
    assert all(0 < timeout < float("inf") for timeout in timeouts)


def test_normalize_audio_uses_configured_ffmpeg_timeout(tmp_path, monkeypatch):
    source = tmp_path / "in.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")
    seen = {}

    def fake_run(cmd, check, capture_output, text, timeout=None):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "1.0", "")
        seen["timeout"] = timeout
        Path(cmd[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("DUBSYNC_FFMPEG_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(subprocess, "run", fake_run)

    normalize_audio(source, dest)

    assert seen["timeout"] == 12.5


def test_normalize_audio_reports_ffmpeg_timeout_clearly(tmp_path, monkeypatch):
    source = tmp_path / "in.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")

    def fake_run(cmd, check, capture_output, text, timeout=None):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "1.0", "")
        Path(cmd[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioNormalizeError, match=r"timed out after 3 seconds"):
        normalize_audio(source, dest, timeout_seconds=3)

    assert not dest.exists()
    assert not dest.with_name(f"{dest.name}.partial").exists()


def test_normalize_audio_rejects_overlong_input_before_decoding(tmp_path, monkeypatch):
    source = tmp_path / "too-long.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")
    commands: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text, timeout=None):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "7201\n", "")

    monkeypatch.setenv("DUBSYNC_MAX_AUDIO_DURATION_SECONDS", "7200")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioNormalizeError, match=r"longer than 7200 seconds"):
        normalize_audio(source, dest)

    assert [command[0] for command in commands] == ["ffprobe"]
    assert not dest.exists()


def test_normalize_audio_removes_output_that_reaches_the_byte_ceiling(tmp_path, monkeypatch):
    source = tmp_path / "compressed.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")

    def fake_run(cmd, check, capture_output, text, timeout=None):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "1.0\n", "")
        Path(cmd[-1]).write_bytes(b"x" * 32)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("DUBSYNC_MAX_NORMALIZED_AUDIO_BYTES", "32")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioNormalizeError, match=r"storage limit"):
        normalize_audio(source, dest)

    assert not dest.exists()


@pytest.mark.parametrize(
    "probe_output",
    ["", "N/A", '{"streams": []}', '{"format": {"duration": "nan"}}'],
)
def test_probe_audio_duration_fails_closed_without_a_finite_audio_duration(
    tmp_path,
    monkeypatch,
    probe_output: str,
):
    source = tmp_path / "invalid-audio.bin"
    source.write_bytes(b"audio")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, probe_output, ""),
    )

    with pytest.raises(AudioNormalizeError, match=r"duration could not be inspected"):
        probe_audio_duration(source, timeout_seconds=2)


def test_cached_asr_adapter_avoids_second_provider_call(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"same audio")
    inner = CountingAdapter()
    adapter = CachedASRAdapter(inner, JsonDiskCache(tmp_path / "cache"), model="fixture", params={"diarize": True})

    first = adapter.transcribe(audio)
    second = adapter.transcribe(audio)

    assert [word.text for word in first] == ["cached"]
    assert [word.text for word in second] == ["cached"]
    assert inner.calls == 1


def test_cached_asr_boundary_repairs_provider_word_stream_quirks(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    adapter = CachedASRAdapter(
        StaticWordAdapter(
            [
                Word.model_construct(text="   ", start=0.0, end=0.2),
                Word.model_construct(text="later", start=0.8, end=0.8),
                Word.model_construct(text="earlier", start=0.2, end=0.4),
            ]
        ),
        JsonDiskCache(cache_dir),
        model="fixture",
        params={},
    )

    words = adapter.transcribe(audio)

    assert [word.text for word in words] == ["earlier", "later"]
    assert words[1].end == pytest.approx(0.801)
    assert adapter.last_repair_flags
    assert adapter.last_repair_flags[0].kind == "word_stream_repaired"
    assert adapter.last_repair_flags[0].severity == "warning"
    assert list(cache_dir.glob("*.json"))

    cached_words = adapter.transcribe(audio)

    assert [word.text for word in cached_words] == ["earlier", "later"]
    assert [flag.kind for flag in adapter.last_repair_flags] == ["word_stream_repaired"]


def test_cached_asr_boundary_rejects_a_large_malformed_fraction(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    mostly_usable = [
        Word.model_construct(text=f"word-{index}", start=index * 0.2, end=index * 0.2 + 0.1)
        for index in range(18)
    ]
    mostly_usable.extend(
        [
            Word.model_construct(text=" ", start=4.0, end=4.1),
            Word.model_construct(text=" ", start=4.2, end=4.3),
        ]
    )
    adapter = CachedASRAdapter(
        StaticWordAdapter(mostly_usable),
        JsonDiskCache(tmp_path / "cache"),
        model="fixture",
        params={},
    )

    with pytest.raises(ProviderError, match="malformed fraction"):
        adapter.transcribe(audio)


def test_cached_asr_malformed_limit_scales_proportionally_for_long_stream(tmp_path):
    usable = [
        Word.model_construct(text=f"word-{index}", start=index * 0.2, end=index * 0.2 + 0.1)
        for index in range(1_000)
    ]
    malformed = [
        Word.model_construct(text=" ", start=300.0 + index * 0.2, end=300.1 + index * 0.2)
        for index in range(30)
    ]

    words, flags = repair_word_stream([*usable, *malformed], source="ASR provider")

    assert len(words) == 1_000
    assert flags[0].kind == "word_stream_repaired"


def test_cached_asr_records_cost_and_caches_raw_unusable_response_before_validation(tmp_path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    meter = CostMeter()
    adapter = CachedASRAdapter(
        StaticWordAdapter([Word.model_construct(text=" ", start=0.0, end=0.0)]),
        JsonDiskCache(tmp_path / "cache"),
        model="scribe_v2",
        params={},
        cost_meter=meter,
        cost_provider="scribe_v2",
        dollars_per_hour=0.22,
    )

    with pytest.raises(ProviderError, match="no usable words"):
        adapter.transcribe(audio)

    assert len(meter.items) == 1
    assert meter.items[0].kind == "audio"

    with pytest.raises(ProviderError, match="no usable words"):
        adapter.transcribe(audio)

    assert len(meter.items) == 1


def test_cached_asr_raw_cache_does_not_consume_generator_provider_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    inner = GeneratorWordAdapter()
    adapter = CachedASRAdapter(
        inner,
        JsonDiskCache(tmp_path / "cache"),
        model="fixture",
        params={},
    )

    first = adapter.transcribe(audio)
    second = adapter.transcribe(audio)

    assert [word.text for word in first] == ["generated"]
    assert [word.text for word in second] == ["generated"]
    assert inner.calls == 1


def test_cached_asr_invalid_stream_type_does_not_poison_cache(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    inner = InvalidThenValidAdapter()
    adapter = CachedASRAdapter(
        inner,
        JsonDiskCache(tmp_path / "cache"),
        model="fixture",
        params={},
    )

    with pytest.raises(ProviderError, match="invalid word stream"):
        adapter.transcribe(audio)

    words = adapter.transcribe(audio)

    assert [word.text for word in words] == ["valid"]
    assert inner.calls == 2


def test_cached_asr_boundary_still_rejects_empty_repaired_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    adapter = CachedASRAdapter(
        StaticWordAdapter([Word.model_construct(text="   ", start=0.0, end=0.2)]),
        JsonDiskCache(tmp_path / "cache"),
        model="fixture",
        params={},
    )

    with pytest.raises(ProviderError, match="no usable words"):
        adapter.transcribe(audio)


def test_assemblyai_adapter_raises_for_terminal_error_status(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    transcript = SimpleNamespace(
        status="error",
        error="provider rejected the transcription",
        words=[
            SimpleNamespace(
                text="must-not-be-returned",
                start=0,
                end=100,
                confidence=0.99,
                speaker="A",
            )
        ],
    )

    class FakeTranscriber:
        def transcribe(self, path, config):
            del path, config
            return transcript

    fake_assemblyai = SimpleNamespace(
        settings=SimpleNamespace(api_key=None),
        TranscriptStatus=SimpleNamespace(error="error"),
        TranscriptionConfig=lambda **kwargs: kwargs,
        Transcriber=FakeTranscriber,
    )
    monkeypatch.setitem(sys.modules, "assemblyai", fake_assemblyai)

    with pytest.raises(ProviderError):
        AssemblyAIAdapter(api_key="test-key").transcribe(audio)


def test_elevenlabs_adapter_passes_configured_keyterms_to_scribe_v2(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = {}

    class FakeSpeechToText:
        def convert(self, **kwargs):
            calls["convert"] = kwargs
            return SimpleNamespace(
                words=[
                    {
                        "type": "word",
                        "text": "Luna",
                        "start": 0.1,
                        "end": 0.4,
                        "confidence": 0.93,
                        "speaker_id": "SPEAKER_00",
                    }
                ]
            )

    class FakeElevenLabs:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.speech_to_text = FakeSpeechToText()

    monkeypatch.setitem(sys.modules, "elevenlabs", SimpleNamespace(ElevenLabs=FakeElevenLabs))

    adapter = adapter_from_config(
        {
            "asr": {
                "provider": "elevenlabs",
                "api_key": "test-key",
                "model_id": "scribe_v2",
                "diarize": True,
                "language_code": "de",
                "keyterms": ["Drachen-Evolutionssystem", "Luna"],
                "character_names": ["Luna", "Matthew", " "],
            }
        }
    )

    words = adapter.transcribe(audio)

    assert [word.text for word in words] == ["Luna"]
    assert calls["api_key"] == "test-key"
    assert calls["convert"]["model_id"] == "scribe_v2"
    assert calls["convert"]["timestamps_granularity"] == "word"
    assert calls["convert"]["diarize"] is True
    assert calls["convert"]["language_code"] == "de"
    assert calls["convert"]["keyterms"] == ["Drachen-Evolutionssystem", "Luna", "Matthew"]


def test_gemini_transcribe_adapter_maps_word_info_annotations(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_valid_wav(audio)
    calls: dict[str, object] = {}

    class FakeFiles:
        def upload(self, file):
            calls["upload"] = file
            return SimpleNamespace(uri="files/audio-1", mime_type="audio/wav")

    class FakeInteractions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            return SimpleNamespace(
                output_text="Hallo Welt",
                steps=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                annotations=[
                                    SimpleNamespace(
                                        type="word_info",
                                        text="Hallo",
                                        speaker="spk_1",
                                        start_offset="0.100s",
                                        end_offset="0.450s",
                                    ),
                                    {
                                        "type": "word_info",
                                        "text": "Welt",
                                        "speaker": "spk_1",
                                        "start_offset": "0.500s",
                                        "end_offset": "0.850s",
                                    },
                                ]
                            )
                        ]
                    )
                ],
            )

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    adapter = GeminiTranscribeAdapter(
        api_key="test-key",
        language_codes=["de-DE"],
        custom_vocabulary=["Drachen-Evolutionssystem", "Luna"],
    )

    words = adapter.transcribe(audio)

    assert [word.text for word in words] == ["Hallo", "Welt"]
    assert [word.speaker_id for word in words] == ["spk_1", "spk_1"]
    assert words[0].start == 0.1
    assert words[1].end == 0.85
    assert words[0].confidence is None
    assert calls["upload"] == str(audio)
    assert calls["api_key"] == "test-key"
    assert calls["create"] == {
        "model": "gemini-3.5-transcribe",
        "store": False,
        "input": [{"type": "audio", "uri": "files/audio-1", "mime_type": "audio/wav"}],
        "generation_config": {
            "transcription_config": {
                "language_codes": ["de-DE"],
                "custom_vocabulary": ["Drachen-Evolutionssystem", "Luna"],
                "mode": {
                    "type": "verbatim",
                    "diarization_mode": "speaker",
                    "timestamp_granularities": ["word"],
                },
            }
        },
    }


def test_google_genai_version_supports_transcription_interaction_fields():
    pytest.importorskip("google.genai")
    version = tuple(int(part) for part in metadata.version("google-genai").split(".")[:2])

    assert version >= (2, 20)


def test_gemini_transcribe_unwraps_sdk_unknown_annotations_and_deletes_upload(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_valid_wav(audio)
    calls: dict[str, object] = {}

    class FakeFiles:
        def upload(self, file):
            calls["upload"] = file
            return SimpleNamespace(name="files/audio-1", uri="files/audio-1", mime_type="audio/wav")

        def delete(self, *, name):
            calls["delete"] = name

    class FakeInteractions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            return SimpleNamespace(
                status="completed",
                steps=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                annotations=[
                                    SimpleNamespace(
                                        type="UNKNOWN",
                                        raw={
                                            "type": "word_info",
                                            "text": "Hallo",
                                            "speaker": "spk_1",
                                            "start_offset": "0.100s",
                                            "end_offset": "0.450s",
                                        },
                                    )
                                ]
                            )
                        ]
                    )
                ],
            )

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    words = GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)

    assert [word.text for word in words] == ["Hallo"]
    assert words[0].confidence is None
    assert calls["delete"] == "files/audio-1"


def test_gemini_transcribe_deletes_upload_when_interaction_fails(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_valid_wav(audio)
    calls: dict[str, object] = {}

    class FakeFiles:
        def upload(self, file):
            return SimpleNamespace(name="files/audio-1", uri="files/audio-1", mime_type="audio/wav")

        def delete(self, *, name):
            calls["delete"] = name

    class FakeInteractions:
        def create(self, **kwargs):
            raise RuntimeError("temporary provider failure")

    class FakeClient:
        def __init__(self, api_key):
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    with pytest.raises(ProviderError, match="Gemini 3.5 Transcribe failed"):
        GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)

    assert calls["delete"] == "files/audio-1"


def test_gemini_transcribe_redacts_api_key_from_provider_errors(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_valid_wav(audio)

    class FakeFiles:
        def upload(self, file):
            return SimpleNamespace(name="files/audio-1", uri="files/audio-1", mime_type="audio/wav")

        def delete(self, *, name):
            del name

    class FakeInteractions:
        def create(self, **kwargs):
            del kwargs
            raise RuntimeError("provider rejected secret test-key")

    class FakeClient:
        def __init__(self, api_key):
            del api_key
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=FakeClient)))

    with pytest.raises(ProviderError) as exc_info:
        GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)

    assert "test-key" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_gemini_transcribe_factory_accepts_deduplicated_custom_vocabulary():
    adapter = adapter_from_config(
        {
            "asr": {
                "provider": "gemini_transcribe",
                "api_key": "test-key",
                "custom_vocabulary": ["Luna", "Drachen-Evolutionssystem", "Luna"],
                "character_names": ["Matthew", "Luna"],
            }
        },
        local_mode=True,
    )

    assert isinstance(adapter, GeminiTranscribeAdapter)
    assert adapter.custom_vocabulary == ["Luna", "Drachen-Evolutionssystem", "Matthew"]


def test_gemini_transcribe_rejects_invalid_request_limit():
    with pytest.raises(ProviderError, match="max_audio_seconds"):
        adapter_from_config(
            {
                "asr": {
                    "provider": "gemini_transcribe",
                    "api_key": "test-key",
                    "max_audio_seconds": 1800.1,
                }
            },
            local_mode=True,
        )


def test_gemini_transcribe_rejects_a_different_model():
    with pytest.raises(ProviderError, match="gemini-3.5-transcribe"):
        GeminiTranscribeAdapter(api_key="test-key", model="gemini-2.5-flash")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"word_timestamps": False}, "word_timestamps"),
        ({"store": True}, "store"),
    ],
)
def test_gemini_transcribe_rejects_unsafe_local_contract_overrides(config, message):
    with pytest.raises(ProviderError, match=message):
        adapter_from_config(
            {
                "asr": {
                    "provider": "gemini_transcribe",
                    "api_key": "test-key",
                    **config,
                }
            },
            local_mode=True,
        )


def test_gemini_transcribe_requires_local_mode():
    with pytest.raises(ProviderError, match="only available in --local"):
        adapter_from_config({"asr": {"provider": "gemini_transcribe", "api_key": "test-key"}})

    adapter = adapter_from_config(
        {"asr": {"provider": "gemini_transcribe", "api_key": "test-key"}},
        local_mode=True,
    )

    assert isinstance(adapter, GeminiTranscribeAdapter)


def test_local_mode_uses_nested_gemini_transcribe_override():
    config = {
        "asr": {
            "provider": "elevenlabs",
            "model_id": "scribe_v2",
            "diarize": True,
            "local": {
                "provider": "gemini_transcribe",
                "model": "gemini-3.5-transcribe",
                "language_codes": ["de-DE"],
                "custom_vocabulary": ["Drachen-Evolutionssystem"],
            },
        }
    }

    local_config = apply_local_asr_config(config, local=True)

    assert local_config["asr"] == {
        "provider": "gemini_transcribe",
        "diarize": True,
        "model": "gemini-3.5-transcribe",
        "language_codes": ["de-DE"],
        "custom_vocabulary": ["Drachen-Evolutionssystem"],
    }
    assert config["asr"]["provider"] == "elevenlabs"


def test_gemini_transcribe_fails_clearly_without_word_annotations(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_valid_wav(audio)

    class FakeClient:
        def __init__(self, api_key):
            del api_key
            self.files = SimpleNamespace(upload=lambda file: SimpleNamespace(uri=file, mime_type="audio/wav"))
            self.interactions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(output_text="Hallo Welt", steps=[]))

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=FakeClient)))

    with pytest.raises(ProviderError, match="word annotations"):
        GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)


def test_gemini_transcribe_rejects_overlong_timestamp_request(tmp_path):
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1)
        wav.writeframes(b"\x00\x00" * 1801)

    with pytest.raises(ProviderError, match="30 minutes"):
        GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)


def test_gemini_transcribe_rejects_audio_with_unknown_duration(tmp_path):
    audio = tmp_path / "invalid.wav"
    audio.write_bytes(b"not-a-valid-wave-file")

    with pytest.raises(ProviderError, match="duration"):
        GeminiTranscribeAdapter(api_key="test-key").transcribe(audio)


def test_whisperx_adapter_transcribes_and_aligns_with_word_timestamps(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = []

    class FakeModel:
        def transcribe(self, loaded_audio, batch_size):
            calls.append(("transcribe", loaded_audio, batch_size))
            return {"language": "de", "segments": [{"text": "Hallo Welt"}]}

    def fake_load_model(model, device, compute_type):
        calls.append(("load_model", model, device, compute_type))
        return FakeModel()

    def fake_load_audio(path):
        calls.append(("load_audio", path))
        return "loaded-audio"

    def fake_load_align_model(language_code, device):
        calls.append(("load_align_model", language_code, device))
        return "align-model", {"meta": True}

    def fake_align(segments, model_a, metadata, audio_data, device, return_char_alignments):
        calls.append(("align", segments, model_a, metadata, audio_data, device, return_char_alignments))
        return {
            "word_segments": [
                {"word": "Hallo", "start": 0.1, "end": 0.4, "score": 0.91, "speaker": "SPEAKER_00"},
                {"word": "Welt", "start": 0.45, "end": 0.8, "score": 0.88, "speaker": "SPEAKER_00"},
            ]
        }

    fake_whisperx = SimpleNamespace(
        load_model=fake_load_model,
        load_audio=fake_load_audio,
        load_align_model=fake_load_align_model,
        align=fake_align,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    words = WhisperXAdapter(model="large-v3", device="cpu", compute_type="int8", batch_size=4).transcribe(audio)

    assert [word.text for word in words] == ["Hallo", "Welt"]
    assert words[0].start == 0.1
    assert words[0].confidence == 0.91
    assert words[0].speaker_id == "SPEAKER_00"
    assert ("load_model", "large-v3", "cpu", "int8") in calls
    assert any(call[0] == "align" for call in calls)


def test_whisperx_diarization_accepts_documented_huggingface_access_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_ACCESS_TOKEN", "hf-access-token")

    adapter = WhisperXAdapter(diarize=True)

    assert adapter.hf_token == "hf-access-token"
