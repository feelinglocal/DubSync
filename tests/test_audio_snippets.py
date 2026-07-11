from __future__ import annotations

import subprocess

from dubsync.audio_snippets import extract_audio_snippets
from dubsync.models import DivergenceSpan


def test_extract_audio_snippets_uses_padded_span_window(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    output_dir = tmp_path / "snippets"
    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):
        calls.append([str(part) for part in cmd])
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
    assert calls[0][1:7] == ["-y", "-ss", "0.500", "-i", str(audio_path), "-t"]
    assert calls[0][7] == "2.500"

