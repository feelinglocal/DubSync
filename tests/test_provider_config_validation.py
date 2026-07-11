from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dubsync.cli import app


def _write_fixture_sync_inputs(tmp_path: Path, provider_overrides: dict[str, object]) -> tuple[Path, Path, Path]:
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers = {"asr": {"fixture_path": str(wordstream_path)}, **provider_overrides}
    providers_path.write_text(yaml.safe_dump(providers), encoding="utf-8")
    return srt_path, audio_path, providers_path


def test_cli_sync_rejects_non_mapping_vad_config_section(tmp_path):
    srt_path, audio_path, providers_path = _write_fixture_sync_inputs(tmp_path, {"vad": []})

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "providers.yaml vad section must be a mapping" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_non_mapping_forced_alignment_config_section(tmp_path):
    srt_path, audio_path, providers_path = _write_fixture_sync_inputs(tmp_path, {"forced_alignment": []})

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "providers.yaml forced_alignment section must be a mapping" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_non_mapping_overlap_detection_config_section(tmp_path):
    srt_path, audio_path, providers_path = _write_fixture_sync_inputs(tmp_path, {"overlap_detection": []})

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "providers.yaml overlap_detection section must be a mapping" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_non_mapping_speaker_mapping_config_section(tmp_path):
    srt_path, audio_path, providers_path = _write_fixture_sync_inputs(tmp_path, {"speaker_mapping": []})

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "providers.yaml speaker_mapping section must be a mapping" in result.output
    assert "Traceback" not in result.output
