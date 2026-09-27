from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dubsync.audio import AudioNormalizeError, normalize_audio


@pytest.mark.parametrize("alias", [False, True], ids=["same-path", "relative-alias"])
def test_normalization_rejects_input_output_identity_before_touching_source(tmp_path, monkeypatch, alias):
    source = tmp_path / "source.wav"
    source.write_bytes(b"irreplaceable source recording")
    destination = source
    if alias:
        monkeypatch.chdir(tmp_path)
        destination = Path("source.wav")

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("source/output identity must be rejected before any processing")

    monkeypatch.setattr("dubsync.audio.probe_audio_duration", unexpected_probe)
    with pytest.raises(AudioNormalizeError, match="different.*source|source.*different"):
        normalize_audio(source, destination)
    assert source.read_bytes() == b"irreplaceable source recording"
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("failure", ["ffmpeg", "timeout", "invalid-output", "publication"])
def test_failed_normalization_preserves_previous_output_and_cleans_only_its_temporary_file(
    tmp_path, monkeypatch, failure,
):
    source = tmp_path / "source.mp3"
    destination = tmp_path / "audio.16k.wav"
    source.write_bytes(b"source recording")
    destination.write_bytes(b"previous complete normalized recording")
    unrelated_partial = tmp_path / "audio.16k.wav.partial"
    unrelated_partial.write_bytes(b"preserve this existing file")
    generated_partials: list[Path] = []

    def probe(path, **_kwargs):
        if path != source and failure == "invalid-output":
            raise AudioNormalizeError("normalized output is invalid")
        return 1.0

    def run(cmd, **kwargs):
        partial = Path(cmd[-1])
        generated_partials.append(partial)
        partial.write_bytes(b"new normalized recording")
        if failure == "ffmpeg":
            raise subprocess.CalledProcessError(1, cmd, stderr="simulated encoder failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def failed_replace(*_args, **_kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("dubsync.audio.probe_audio_duration", probe)
    monkeypatch.setattr("dubsync.audio.subprocess.run", run)
    if failure == "publication":
        monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises((AudioNormalizeError, OSError)):
        normalize_audio(source, destination)

    assert source.read_bytes() == b"source recording"
    assert destination.read_bytes() == b"previous complete normalized recording"
    assert unrelated_partial.read_bytes() == b"preserve this existing file"
    assert generated_partials and all(not path.exists() for path in generated_partials)
    assert set(tmp_path.iterdir()) == {source, destination, unrelated_partial}


def test_successful_normalization_replaces_output_only_after_validation_and_preserves_partial_named_input(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "audio.wav"
    source = tmp_path / "audio.wav.partial"
    source.write_bytes(b"source recording")
    destination.write_bytes(b"previous recording")
    generated: list[Path] = []

    def probe(path, **_kwargs):
        assert source.read_bytes() == b"source recording"
        assert destination.read_bytes() == b"previous recording"
        if path != source:
            assert path.read_bytes() == b"validated replacement"
        return 1.0

    def run(cmd, **_kwargs):
        generated.append(Path(cmd[-1]))
        assert source.read_bytes() == b"source recording"
        assert destination.read_bytes() == b"previous recording"
        generated[-1].write_bytes(b"validated replacement")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("dubsync.audio.probe_audio_duration", probe)
    monkeypatch.setattr("dubsync.audio.subprocess.run", run)

    assert normalize_audio(source, destination) == destination
    assert source.read_bytes() == b"source recording"
    assert destination.read_bytes() == b"validated replacement"
    assert generated and all(not path.exists() for path in generated)
    assert set(tmp_path.iterdir()) == {source, destination}
