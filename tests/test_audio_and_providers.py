from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from dubsync.audio import normalize_audio
from dubsync.cache import JsonDiskCache
from dubsync.models import Word
from dubsync.providers import CachedASRAdapter, ProviderError, WhisperXAdapter, adapter_from_config


class CountingAdapter:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path):
        self.calls += 1
        return [Word(text="cached", start=0.0, end=0.2, confidence=0.9, speaker_id="A")]


def test_normalize_audio_uses_ffmpeg_16khz_mono(tmp_path, monkeypatch):
    source = tmp_path / "in.mp3"
    dest = tmp_path / "out.wav"
    source.write_bytes(b"audio")
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        dest.write_bytes(b"wav")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = normalize_audio(source, dest)

    assert result == dest
    assert calls[0] == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dest),
    ]


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


@pytest.mark.parametrize("language,expected", [("jpn", "ja"), ("auto", None)])
def test_openai_whisper_forwards_language_to_transcription(tmp_path, monkeypatch, language, expected):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(words=[{"word": "日本語", "start": 0.1, "end": 0.8}])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))))
    adapter = adapter_from_config({"asr": {"provider": "openai", "api_key": "test-key", "language_code": language}})

    words = adapter.transcribe(audio)

    assert calls.get("language") == expected
    assert [word.text for word in words] == ["日本語"]


@pytest.mark.parametrize("language,expected_models", [("ja-JP", ["universal-3-pro", "universal-2"]), ("auto", ["universal-3-pro", "universal-2"]), ("de", ["universal-3-pro"])])
def test_assemblyai_forwards_language_and_available_japanese_fallback(tmp_path, monkeypatch, language, expected_models):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = {}

    def config(**kwargs):
        calls.update(kwargs)
        return kwargs

    monkeypatch.setitem(sys.modules, "assemblyai", SimpleNamespace(
        settings=SimpleNamespace(api_key=None), TranscriptionConfig=config,
        Transcriber=lambda: SimpleNamespace(transcribe=lambda *args, **kwargs: SimpleNamespace(words=[{"text": "日本語", "start": 100, "end": 800}], status="completed")),
    ))
    adapter = adapter_from_config({"asr": {"provider": "assemblyai", "api_key": "test-key", "language_code": language}})

    words = adapter.transcribe(audio)

    assert calls["speech_models"] == expected_models
    assert calls["language_detection"] is (language == "auto")
    assert calls.get("language_code") == (None if language == "auto" else "ja" if language == "ja-JP" else "de")
    assert words[0].start == pytest.approx(0.1)


def test_assemblyai_surfaces_transcription_errors(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setitem(sys.modules, "assemblyai", SimpleNamespace(
        settings=SimpleNamespace(api_key=None), TranscriptionConfig=lambda **kwargs: kwargs,
        Transcriber=lambda: SimpleNamespace(transcribe=lambda *args, **kwargs: SimpleNamespace(status="error", error="transcription failed", words=None)),
    ))
    adapter = adapter_from_config({"asr": {"provider": "assemblyai", "api_key": "test-key", "language_code": "ja"}})

    with pytest.raises(ProviderError, match="transcription failed"):
        adapter.transcribe(audio)


def test_whisperx_japanese_hint_reaches_transcription_and_alignment(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = {}

    def transcribe(loaded_audio, **kwargs):
        calls["transcribe"] = kwargs
        return {"language": "en", "segments": []}

    def load_align_model(**kwargs):
        calls["align"] = kwargs
        return "model", {}

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_audio=lambda path: "audio", load_model=lambda *args, **kwargs: SimpleNamespace(transcribe=transcribe),
        load_align_model=load_align_model, align=lambda *args, **kwargs: {"word_segments": [{"word": "日本語", "start": 0.1, "end": 0.8}]},
    ))

    words = WhisperXAdapter(language="ja-JP").transcribe(audio)

    assert calls["transcribe"]["language"] == "ja"
    assert calls["align"]["language_code"] == "ja"
    assert [word.text for word in words] == ["日本語"]
