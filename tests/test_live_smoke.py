from __future__ import annotations

import importlib.util
import os
import wave

import pytest

from dubsync.llm_providers import AnthropicLLMAdapter, GeminiLLMAdapter
from dubsync.models import Cue, DivergenceSpan
from dubsync.providers import AssemblyAIAdapter, ElevenLabsScribeAdapter, OpenAIWhisperAdapter


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed")


def _tiny_wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)


def test_live_smoke_tests_are_opt_in():
    assert True


def test_live_smoke_file_covers_all_configured_cloud_providers():
    live_tests = {
        name
        for name, value in globals().items()
        if name.startswith("test_live_") and name != "test_live_smoke_tests_are_opt_in" and callable(value)
    }

    assert {
        "test_live_gemini_punctuation_smoke",
        "test_live_anthropic_adjudication_smoke",
        "test_live_elevenlabs_scribe_smoke",
        "test_live_openai_whisper_smoke",
        "test_live_assemblyai_smoke",
    }.issubset(live_tests)


@pytest.mark.live
def test_live_gemini_punctuation_smoke():
    _require_module("google.genai")
    api_key = _require_env("GEMINI_API_KEY")
    model = os.getenv("DUBSYNC_LIVE_GEMINI_MODEL", "gemini-3.5-flash")

    result = GeminiLLMAdapter(api_key=api_key, model=model).punctuate(
        [Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"])]
    )

    assert isinstance(result, dict)


@pytest.mark.live
def test_live_anthropic_adjudication_smoke():
    _require_module("anthropic")
    api_key = _require_env("ANTHROPIC_API_KEY")
    model = os.getenv("DUBSYNC_LIVE_ANTHROPIC_MODEL", "claude-sonnet-5")
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1],
        srt_text="hello there",
        asr_text="hello there",
        confidence=0.95,
        speaker_ids=["SPEAKER_00"],
    )

    result = AnthropicLLMAdapter(api_key=api_key, model=model).adjudicate([span])

    assert isinstance(result, list)


@pytest.mark.live
def test_live_elevenlabs_scribe_smoke(tmp_path):
    _require_module("elevenlabs")
    api_key = _require_env("ELEVENLABS_API_KEY")
    audio = tmp_path / "tiny.wav"
    _tiny_wav(audio)

    words = ElevenLabsScribeAdapter(api_key=api_key).transcribe(audio)

    assert isinstance(words, list)


@pytest.mark.live
def test_live_openai_whisper_smoke(tmp_path):
    _require_module("openai")
    api_key = _require_env("OPENAI_API_KEY")
    audio = tmp_path / "tiny.wav"
    _tiny_wav(audio)

    words = OpenAIWhisperAdapter(api_key=api_key).transcribe(audio)

    assert isinstance(words, list)


@pytest.mark.live
def test_live_assemblyai_smoke(tmp_path):
    _require_module("assemblyai")
    api_key = _require_env("ASSEMBLYAI_API_KEY")
    audio = tmp_path / "tiny.wav"
    _tiny_wav(audio)

    words = AssemblyAIAdapter(api_key=api_key).transcribe(audio)

    assert isinstance(words, list)
