from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubsync.audio import AudioNormalizeError, normalize_audio, probe_audio_duration
from dubsync.cache import JsonDiskCache
from dubsync.models import Word
from dubsync.providers import CachedASRAdapter, WhisperXAdapter, adapter_from_config


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
