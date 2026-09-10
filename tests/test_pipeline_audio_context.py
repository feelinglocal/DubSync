from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from dubsync import pipeline
from dubsync.cost import CostMeter


CONFIG = {"llm": {"adjudication": {"provider": "gemini", "model": "gemini-3.8-flash", "audio_context": {"enabled": True}}}}


def wav(path: Path, seconds: int):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16 * seconds)
    return path


def test_full_audio_identity_invalidates_adjudication_cache_even_without_snippets(tmp_path):
    original = tmp_path / "episode.mp3"
    original.write_bytes(b"original-one")
    normalized = wav(tmp_path / "audio.wav", 181)
    first = pipeline._adjudication_audio_cache_context(original, normalized, CONFIG, None)
    original.write_bytes(b"original-two")
    second = pipeline._adjudication_audio_cache_context(original, normalized, CONFIG, None)
    assert first != second
    assert pipeline._adjudication_cache_key([], CONFIG, audio_snippet_context=first) != pipeline._adjudication_cache_key([], CONFIG, audio_snippet_context=second)


@pytest.mark.parametrize("seconds,use_original", [(180, False), (181, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_context_configured_once_closed_and_accounted_even_when_adjudication_raises(tmp_path, seconds, use_original, fail):
    original = tmp_path / "episode.mp3"
    original.write_bytes(b"source")
    normalized = wav(tmp_path / "audio.wav", seconds)
    events = []

    class Adapter:
        usage_events = []

        def set_audio_context(self, path, **kwargs):
            events.append(("set", path, kwargs))

        def close(self):
            events.append(("closed",))

        def audio_context_report(self):
            assert events[-1][0] == "closed"
            return {"enabled": True, "cleanup_status": "deleted", "cache_create_input_tokens_reserved": 5000}

    meter, flags = CostMeter(), []
    def execute():
        with pipeline._adjudication_audio_session(Adapter(), original, normalized, CONFIG, tmp_path, meter, flags):
            events.append(("adjudicate",))
            if fail:
                raise RuntimeError("test failure")
    if fail:
        with pytest.raises(RuntimeError, match="test failure"):
            execute()
    else:
        execute()
    assert [event[0] for event in events] == ["set", "adjudicate", "closed"]
    assert events[0][1] == (original if use_original else normalized)
    assert events[0][2]["duration_seconds"] == seconds
    assert len(meter.items) == 1
    report = json.loads((tmp_path / "gemini_audio_context.json").read_text())
    assert report["cleanup_status"] == "deleted"


def test_disabled_context_never_reads_audio_or_uploads(tmp_path):
    missing = tmp_path / "missing.wav"
    assert pipeline._adjudication_audio_cache_context(missing, missing, {}, {"focused": True}) == {"focused": True}
    with pipeline._adjudication_audio_session(object(), missing, missing, {}, tmp_path, CostMeter(), []):
        pass
    assert not (tmp_path / "gemini_audio_context.json").exists()


@pytest.mark.parametrize("options", [{"enabled": 0}, {"enabled": ""}, {"enabled": "false"}, 0])
def test_invalid_context_options_cannot_silently_disable_audio(options):
    config = {"llm": {"adjudication": {"provider": "gemini", "audio_context": options}}}
    with pytest.raises((ValueError, RuntimeError)):
        pipeline._episode_audio_options(config)


def test_context_cost_uses_the_same_default_model_as_generation(tmp_path, monkeypatch):
    models = []
    monkeypatch.setattr(pipeline, "record_gemini_context_cost", lambda meter, model, config, report: models.append(model))
    class Adapter:
        def set_audio_context(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def audio_context_report(self):
            return {}
    config = {"llm": {"adjudication": {"provider": "gemini", "audio_context": True}}}
    audio = wav(tmp_path / "audio.wav", 1)
    with pipeline._adjudication_audio_session(Adapter(), audio, audio, config, tmp_path, CostMeter(), []):
        pass
    assert models == [pipeline._default_llm_model("gemini")]


def test_adjudication_receives_source_context_and_acoustic_word_ownership():
    from dubsync.models import Cue, Word
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Hello"])]
    words = [Word(text="Hello", start=0.1, end=0.5, speaker_id="actor-A")]
    received = []
    class Adapter:
        def set_episode_context(self, value):
            received.append(("cues", value))
        def set_episode_words(self, value):
            received.append(("words", value))
    pipeline._set_adapter_episode_context(Adapter(), cues, words=words)
    assert received == [("cues", cues), ("words", words)]
    pipeline._set_adapter_episode_context(object(), cues, words=words)


@pytest.mark.parametrize("change", ["time", "speaker", "text"])
def test_adjudication_cache_changes_with_local_word_evidence(change):
    from dubsync.models import DivergenceSpan, Word
    span = DivergenceSpan(case_id="one", cue_ids=[1], srt_text="old", asr_text="new line",
                          asr_word_indices=[0, 1], start=0, end=1, speaker_ids=["A", "B"])
    words = [Word(text="new", start=0, end=.3, speaker_id="A"),
             Word(text="line", start=.4, end=1, speaker_id="B")]
    altered = list(words)
    update = {"time": {"end": .35}, "speaker": {"speaker_id": "B"}, "text": {"text": "New"}}[change]
    altered[0] = words[0].model_copy(update=update)
    assert pipeline._adjudication_cache_key([span], CONFIG, source_words=words) != pipeline._adjudication_cache_key([span], CONFIG, source_words=altered)
