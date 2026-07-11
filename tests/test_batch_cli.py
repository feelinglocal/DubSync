from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.srt_io import parse_srt_text


def test_cli_batch_ignores_generated_srt_artifacts(tmp_path, shifted_srt_text, shifted_wordstream):
    folder = tmp_path / "episodes"
    folder.mkdir()
    providers_path = tmp_path / "providers.yaml"
    wordstream_path = tmp_path / "ep1.wordstream.json"

    source_srt = folder / "ep1.srt"
    source_srt.write_text(shifted_srt_text, encoding="utf-8")
    (folder / "ep1.wav").write_bytes(b"RIFF....WAVEfmt ")

    generated_candidates = [
        folder / "ep1.synced.srt",
        folder / "ep1.changes.diff.srt",
    ]
    for generated_srt in generated_candidates:
        generated_srt.write_text(shifted_srt_text, encoding="utf-8")
        generated_srt.with_suffix(".wav").write_bytes(b"RIFF....WAVEfmt ")

    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            str(folder),
            "--providers",
            str(providers_path),
            "--workdir",
            str(tmp_path / "work"),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("Cost meter") == 1
    assert "ep1.synced.synced.srt" not in result.output
    assert "ep1.changes.diff.synced.srt" not in result.output
    assert not (folder / "ep1.synced.synced.srt").exists()
    assert not (folder / "ep1.changes.diff.synced.srt").exists()
    synced = parse_srt_text((folder / "ep1.synced.srt").read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in synced] == ["hello there", "general kenobi"]


def test_cli_batch_fails_when_no_episodes_are_processed(tmp_path):
    folder = tmp_path / "episodes"
    folder.mkdir()
    (folder / "ep1.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["batch", str(folder), "--no-llm"])

    assert result.exit_code != 0
    assert "no episodes processed" in result.output
    assert "Traceback" not in result.output
