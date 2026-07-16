from __future__ import annotations

import subprocess

import pytest

from dubsync.audio_snippets import AudioSnippetError, extract_audio_snippets
from dubsync.models import DivergenceSpan


def test_extract_audio_snippets_uses_padded_span_window(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    output_dir = tmp_path / "snippets"
    calls: list[tuple[list[str], float | None]] = []

    def fake_run(cmd, check, capture_output, text, timeout=None):
        calls.append(([str(part) for part in cmd], timeout))
        output_path = cmd[-1]
        output_path.write_bytes(b"RIFFsnippetWAVEfmt ")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("dubsync.audio_snippets.subprocess.run", fake_run)

    snippets = extract_audio_snippets(
        audio_path,
        [
            DivergenceSpan(
                case_id="case-1",
                cue_ids=[1],
                srt_text="old line",
                asr_text="new line",
                start=1.0,
                end=2.5,
            )
        ],
        output_dir,
        pad_seconds=0.5,
        max_duration_seconds=20.0,
    )

    assert snippets[0].case_id == "case-1"
    assert snippets[0].start == 0.5
    assert snippets[0].end == 3.0
    assert snippets[0].mime_type == "audio/wav"
    assert snippets[0].path.endswith("case-1.wav")
    command, timeout = calls[0]
    assert command[1:7] == ["-y", "-ss", "0.500", "-i", str(audio_path), "-t"]
    assert command[7] == "2.500"
    assert isinstance(timeout, float)
    assert 0 < timeout < float("inf")


def test_extract_audio_snippets_reports_ffmpeg_timeout_clearly(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    def fake_run(cmd, check, capture_output, text, timeout=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("dubsync.audio_snippets.subprocess.run", fake_run)

    with pytest.raises(AudioSnippetError, match=r"timed out after 4 seconds"):
        extract_audio_snippets(
            audio_path,
            [
                DivergenceSpan(
                    case_id="case-timeout",
                    cue_ids=[1],
                    srt_text="old line",
                    asr_text="new line",
                    start=1.0,
                    end=2.0,
                )
            ],
            tmp_path / "snippets",
            ffmpeg_timeout_seconds=4,
        )


def test_extract_audio_snippets_rejects_an_insufficient_work_budget_before_ffmpeg(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    calls: list[list[str]] = []

    def unexpected_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("DUBSYNC_MAX_AUDIO_SNIPPET_BYTES", "1024")
    monkeypatch.setattr("dubsync.audio_snippets.subprocess.run", unexpected_run)

    with pytest.raises(AudioSnippetError, match=r"storage budget"):
        extract_audio_snippets(
            audio_path,
            [
                DivergenceSpan(
                    case_id="case-budget",
                    cue_ids=[1],
                    srt_text="old line",
                    asr_text="new line",
                    start=1.0,
                    end=3.0,
                )
            ],
            tmp_path / "snippets",
        )

    assert calls == []
    assert list((tmp_path / "snippets").glob("*")) == []
