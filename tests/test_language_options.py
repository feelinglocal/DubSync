from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.forced_alignment import MMSForcedAlignmentAdapter
from dubsync.providers import adapter_from_config, apply_asr_language


@pytest.mark.parametrize("provider,field", [("elevenlabs", "language_code"), ("openai", "language"), ("assemblyai", "language_code"), ("whisperx", "language")])
@pytest.mark.parametrize("language", ["ja", "jpn", "ja-JP", " JA_jp "])
def test_japanese_language_aliases_reach_each_provider(provider, field, language):
    original = {"asr": {"provider": provider}, "forced_alignment": {"provider": "mms", "language": "deu"}}
    before = deepcopy(original)

    config = apply_asr_language(original, language)
    adapter = adapter_from_config(config)

    assert getattr(adapter, field) == "ja"
    assert config["forced_alignment"]["language"] == "jpn"
    assert original == before


@pytest.mark.parametrize("provider,field", [("elevenlabs", "language_code"), ("openai", "language"), ("assemblyai", "language_code"), ("whisperx", "language")])
def test_explicit_auto_clears_configured_asr_hint(provider, field):
    original = {"asr": {"provider": provider, "language": "ja", "language_code": "ja"}}

    config = apply_asr_language(original, "auto")

    assert getattr(adapter_from_config(config), field) is None
    assert original["asr"]["language"] == "ja"


def test_missing_language_preserves_config_and_unknown_language_is_not_gated():
    config = {"asr": {"provider": "whisperx", "language": "de"}}

    assert apply_asr_language(config, None) == config
    assert apply_asr_language(config, "") == config
    assert adapter_from_config(apply_asr_language(config, "new-language")).language == "new-language"


@pytest.mark.parametrize("language", ["ja", "jpn", "ja-JP"])
def test_mms_japanese_uses_iso_639_3(language):
    assert MMSForcedAlignmentAdapter(language=language).language == "jpn"


@pytest.mark.parametrize("command", ["sync", "batch", "generate"])
@pytest.mark.parametrize("language", ["ja-JP", "auto", "new-language"])
def test_cli_forwards_optional_language_without_allowlist(tmp_path, monkeypatch, command, language):
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"audio")
    srt = tmp_path / "episode.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nこんにちは。\n", encoding="utf-8")
    calls = []

    def fake_process(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_srt=tmp_path / "output.srt", episode_workdir=tmp_path, cost_meter=SimpleNamespace(to_json=lambda: "{}"))

    monkeypatch.setattr("dubsync.cli.sync_episode", fake_process)
    monkeypatch.setattr("dubsync.cli.generate_srt_from_audio", fake_process)
    arguments = {"sync": [str(srt), str(audio)], "batch": [str(tmp_path)], "generate": [str(audio)]}[command]

    result = CliRunner().invoke(app, [command, *arguments, "--language", language])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["language"] == language
