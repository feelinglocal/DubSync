from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.forced_alignment import MMSForcedAlignmentAdapter, apply_forced_alignment
from dubsync.models import Cue, ForcedAlignmentCue
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def test_cli_sync_applies_forced_alignment_fixture_refinement(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    forced_alignment_path = tmp_path / "forced-align.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nhello there\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 1.00, "end": 1.20, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "there", "start": 1.24, "end": 1.50, "confidence": 0.97, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    forced_alignment_path.write_text(
        json.dumps({"cues": [{"cue_id": 1, "start": 0.96, "end": 1.62, "score": 0.94}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "forced_alignment": {"fixture_path": str(forced_alignment_path)},
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
            "--fps",
            "30",
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].start_ms == 933
    assert synced[0].end_ms == 1633
    assert (workdir / "episode" / "forced_align.json").exists()
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "forced_alignment_refined" for flag in report["flags"])


def test_forced_alignment_clamps_negative_start_to_zero_and_qc_window():
    cues = [Cue(index=1, start_ms=100, end_ms=700, lines=["hello there"])]
    alignments = [ForcedAlignmentCue(cue_id=1, start=-0.04, end=0.24, score=0.91)]
    profile = StyleProfile(fps=30.0, min_cue_dur=0.5)

    updated, flags = apply_forced_alignment(cues, alignments, profile)

    assert updated[0].start_ms == 0
    assert updated[0].end_ms == 500
    assert flags[0].start == 0.0
    assert flags[0].end == pytest.approx(0.5)


def test_mms_forced_alignment_adapter_uses_ctc_python_api(tmp_path, monkeypatch):
    calls: dict[str, object] = {}

    class FakeModel:
        dtype = "float32"
        device = "cpu"

    def load_alignment_model(device, dtype):
        calls["load_alignment_model"] = (device, dtype)
        return FakeModel(), "tokenizer"

    def load_audio(path, dtype, device):
        calls["load_audio"] = (path, dtype, device)
        return "waveform"

    def generate_emissions(model, waveform, batch_size):
        calls["generate_emissions"] = (model, waveform, batch_size)
        return "emissions", 0.02

    def preprocess_text(text, romanize, language):
        calls["preprocess_text"] = (text, romanize, language)
        return ["hello", "there", "general", "kenobi"], text

    def get_alignments(emissions, tokens_starred, alignment_tokenizer):
        calls["get_alignments"] = (emissions, tokens_starred, alignment_tokenizer)
        return "segments", [0.9, 0.8, 0.7, 0.6], "blank"

    def get_spans(tokens_starred, segments, blank_token):
        calls["get_spans"] = (tokens_starred, segments, blank_token)
        return "spans"

    def postprocess_results(text_starred, spans, stride, scores):
        calls["postprocess_results"] = (text_starred, spans, stride, scores)
        return [
            {"text": "hello", "start": 0.10, "end": 0.20, "score": 0.9},
            {"text": "there", "start": 0.24, "end": 0.40, "score": 0.8},
            {"text": "general", "start": 1.00, "end": 1.20, "score": 0.7},
            {"text": "kenobi", "start": 1.24, "end": 1.50, "score": 0.6},
        ]

    fake_torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_ctc = SimpleNamespace(
        load_audio=load_audio,
        load_alignment_model=load_alignment_model,
        generate_emissions=generate_emissions,
        preprocess_text=preprocess_text,
        get_alignments=get_alignments,
        get_spans=get_spans,
        postprocess_results=postprocess_results,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ctc_forced_aligner", fake_ctc)

    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["hello there"]),
        Cue(index=2, start_ms=1000, end_ms=1600, lines=["general kenobi"]),
    ]

    alignments = MMSForcedAlignmentAdapter(language="deu", romanize=True, batch_size=2).align(audio_path, cues)

    assert calls["load_alignment_model"] == ("cpu", "float32")
    assert calls["load_audio"] == (str(audio_path), "float32", "cpu")
    assert calls["generate_emissions"][2] == 2
    assert calls["preprocess_text"] == ("hello there general kenobi", True, "deu")
    assert [(row.cue_id, row.start, row.end, row.score) for row in alignments] == [
        (1, 0.10, 0.40, pytest.approx(0.85)),
        (2, 1.00, 1.50, pytest.approx(0.65)),
    ]
