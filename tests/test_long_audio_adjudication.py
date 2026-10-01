"""Feature-length audio and long divergence spans still reach the adjudicator."""
from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine, _snippet_covers_span
from dubsync.adjudication_snippets import BoundedAudioSnippetBatchSource
from dubsync.audio_snippets import _snippet_window, extract_audio_snippets
from dubsync.models import AudioSnippet, DivergenceSpan


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


class _AudioAdapter:
    def __init__(self, verdict="use_audio", final_text=None):
        self.calls: list[list[str]] = []
        self.clips: dict[str, AudioSnippet] = {}
        self._verdict = verdict
        self._final_text = final_text

    def adjudicate(self, spans):
        raise AssertionError("required case audio cannot fall back to a text-only decision")

    def adjudicate_with_audio(self, spans, snippets):
        self.calls.append([span.case_id for span in spans])
        self.clips.update(snippets)
        return [dict(
            case_id=span.case_id, verdict=self._verdict,
            final_text=span.asr_text if self._final_text is None else self._final_text,
            confidence=0.99, reason="heard in the supplied case audio",
        ) for span in spans]


@pytest.mark.parametrize("duration_seconds", [91 * 60.0, 4 * 60 * 60.0])
def test_feature_length_audio_still_adjudicates_with_case_audio(tmp_path, monkeypatch, duration_seconds):
    # pipeline.md P-4: above 90 minutes every case was held as audio-unavailable.
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:00,100 --> 00:00:00,800\nold source words\n", encoding="utf-8")
    audio = _wav(tmp_path / "episode.wav")
    wordstream = tmp_path / "words.json"
    wordstream.write_text(json.dumps({"words": [
        {"text": "different", "start": 0.1, "end": 0.3},
        {"text": "spoken", "start": 0.3, "end": 0.5},
        {"text": "wording", "start": 0.5, "end": 0.8},
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"provider": "gemini", "adjudication": {
            "model": "gemini-3.5-flash-lite", "audio_context": {"enabled": False},
            "audio_snippet_double_check": {"enabled": True},
        }},
    }), encoding="utf-8")
    adapter = _AudioAdapter()

    def extract(_audio, spans, directory, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        return [AudioSnippet(case_id=span.case_id, path=str(_wav(directory / f"{span.case_id}.wav")), start=0, end=1)
                for span in spans]

    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: duration_seconds)
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)

    result = pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work", providers_path=providers)

    assert adapter.calls == [["case-1"]]
    assert "different spoken wording" in result.output_srt.read_text(encoding="utf-8")
    kinds = [flag["kind"] for flag in result.report["flags"]]
    assert "adjudication_audio_unavailable" not in kinds
    assert "audio_snippet_unavailable" not in kinds


def test_default_snippet_options_have_no_episode_duration_cap():
    options = pipeline._adjudication_audio_snippet_options({"llm": {"adjudication": {"audio_snippet_double_check": True}}})

    assert options[0] is True
    assert options[4] is None


@pytest.mark.parametrize("span_start, span_end", [
    (683.71, 704.51),    # ep11 case-83: 20.8 s deletion gap before a song
    (1355.0, 1406.9),    # ep02 case-196: 51.9 s
    (2285.4, 2379.7),    # test-long case-265: 94.3 s, spoken words at both ends and in the middle
])
def test_span_longer_than_the_clip_limit_gets_one_covering_clip(span_start, span_end):
    start, end = _snippet_window(span_start, span_end, 2.0, 20.0)

    assert start <= span_start and end >= span_end
    assert end - start <= 120.0
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="source", asr_text="", start=span_start, end=span_end)
    assert _snippet_covers_span(AudioSnippet(case_id="case-1", path="x.wav", start=round(start, 3), end=round(end, 3)), span)


def test_ordinary_span_window_is_unchanged():
    assert _snippet_window(10.0, 12.5, 2.0, 20.0) == (8.0, 14.5)
    # Padding is traded for coverage before a clip may grow.
    start, end = _snippet_window(100.0, 118.0, 2.0, 20.0)
    assert (round(start, 3), round(end, 3)) == (99.0, 119.0)


def test_span_beyond_the_covering_limit_stays_uncovered_and_held():
    start, end = _snippet_window(100.0, 400.0, 2.0, 20.0)

    assert end - start <= 20.0
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="source", asr_text="", start=100.0, end=400.0)
    assert not _snippet_covers_span(AudioSnippet(case_id="case-1", path="x.wav", start=start, end=end), span)


def test_long_deletion_span_is_adjudicated_from_its_covering_clip(tmp_path, monkeypatch):
    # adjudication.md BUG-6: cue 228 kept unspoken words because its 20.8 s gap could never get audio.
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    commands: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text, timeout=None):
        commands.append([str(part) for part in cmd])
        cmd[-1].write_bytes(b"RIFFsnippetWAVEfmt ")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("dubsync.audio_snippets.subprocess.run", fake_run)
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 49 * 60.0)
    span = DivergenceSpan(
        case_id="case-83", cue_ids=[228], srt_text="sua cheirosa de pêssego", asr_text="",
        start=683.71, end=704.51, srt_token_indices=[4, 5, 6, 7],
    )
    source = BoundedAudioSnippetBatchSource(
        audio, tmp_path / "snippets", pad_seconds=2.0, max_duration_seconds=20.0,
        max_snippets_per_batch=25, max_audio_duration_seconds=None, extractor=extract_audio_snippets,
    )
    adapter = _AudioAdapter(final_text="")

    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=source.load, require_audio_snippets=True,
    ).adjudicate([span])

    assert adapter.calls == [["case-83"]]
    clip = adapter.clips["case-83"]
    assert clip.start <= span.start and clip.end >= span.end
    assert (decisions[0].verdict, decisions[0].final_text) == ("use_audio", "")
    assert flags == []
    # ffmpeg cut one continuous clip: -ss <start> ... -t <duration covering the span>
    assert float(commands[0][commands[0].index("-t") + 1]) >= 20.8
    assert source.manifest()["fallback_count"] == 0
