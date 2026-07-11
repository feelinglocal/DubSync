from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import Cue
from dubsync.punctuation import StaticPunctuationAdapter, apply_punctuation_pass
from dubsync.srt_io import parse_srt_text


class RecordingPunctuationAdapter:
    def __init__(self):
        self.batches: list[list[int]] = []

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        self.batches.append([cue.index for cue in cues])
        return {}


def test_punctuation_pass_batches_cues_by_scene_gap():
    adapter = RecordingPunctuationAdapter()
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["hello"]),
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["there"]),
        Cue(index=3, start_ms=7000, end_ms=7500, lines=["again"]),
    ]

    updated, flags = apply_punctuation_pass(cues, adapter, scene_gap_seconds=4.0)

    assert adapter.batches == [[1, 2], [3]]
    assert updated == cues
    assert flags == []


def test_punctuation_pass_preserves_valid_proposed_line_breaks():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello", "there"])]
    adapter = StaticPunctuationAdapter({1: "Hello,\nthere."})

    updated, flags = apply_punctuation_pass(cues, adapter)

    assert flags == []
    assert updated[0].lines == ["Hello,", "there."]


def test_cli_sync_applies_fixture_punctuation_pass(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.2}, {"text": "there", "start": 0.25, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "fixture", "punctuation": {"1": "Hello, there."}},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].text == "Hello, there."


def test_cli_sync_rejects_word_changing_punctuation_with_qc_flag(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.2}, {"text": "there", "start": 0.25, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "fixture", "punctuation": {"1": "Hello, world."}},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].text == "hello there"
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "invalid_punctuation_change" for flag in report["flags"])
