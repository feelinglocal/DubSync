from __future__ import annotations

import json
import wave

import pytest
import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import Cue, SpeechRegion
from dubsync.vad import EnergySpeechActivityAdapter, SileroSpeechActivityAdapter, speech_activity_adapter_from_config, speech_activity_flags_for_cues


def test_speech_activity_flags_cues_with_low_region_coverage():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["spoken line"]),
        Cue(index=2, start_ms=2000, end_ms=3000, lines=["missing speech"]),
    ]
    regions = [SpeechRegion(start=0.10, end=0.90, confidence=0.88)]

    flags = speech_activity_flags_for_cues(cues, regions, min_coverage=0.5)

    assert [flag.kind for flag in flags] == ["cue_without_speech_activity"]
    assert flags[0].cue_ids == [2]
    assert flags[0].confidence == 0.0


def test_energy_speech_activity_adapter_detects_loud_wav_region(tmp_path):
    audio = tmp_path / "speech.wav"
    samples = [0] * 8000 + [10000] * 8000
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(sample.to_bytes(2, byteorder="little", signed=True) for sample in samples))

    regions = EnergySpeechActivityAdapter(threshold_dbfs=-45.0, window_ms=100, min_region_ms=100).detect(audio)

    assert len(regions) == 1
    assert regions[0].start == pytest.approx(0.5)
    assert regions[0].end == pytest.approx(1.0)


def test_silero_vad_provider_config_is_optional_adapter():
    adapter = speech_activity_adapter_from_config({"vad": {"provider": "silero"}})

    assert isinstance(adapter, SileroSpeechActivityAdapter)


def test_cli_sync_writes_vad_fixture_artifact_and_qc_flag(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    vad_path = tmp_path / "episode.vad.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nmissing speech\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "there", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "A"},
                    {"text": "missing", "start": 1.0, "end": 1.2, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "speech", "start": 1.25, "end": 1.55, "confidence": 0.97, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    vad_path.write_text(json.dumps({"regions": [{"start": 0.0, "end": 0.8, "confidence": 0.92}]}), encoding="utf-8")
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "vad": {"fixture_path": str(vad_path), "min_coverage": 0.5},
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
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (workdir / "episode" / "vad.json").exists()
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    flags = [flag for flag in report["flags"] if flag["kind"] == "cue_without_speech_activity"]
    assert flags[0]["cue_ids"] == [2]


def test_cli_sync_flags_unmatched_cue_over_silence_as_dropped_line_candidate(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    vad_path = tmp_path / "episode.vad.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\ndropped line\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "there", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    vad_path.write_text(json.dumps({"regions": [{"start": 0.0, "end": 0.8, "confidence": 0.92}]}), encoding="utf-8")
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "vad": {"fixture_path": str(vad_path), "min_coverage": 0.5},
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
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    dropped_flags = [flag for flag in report["flags"] if flag["kind"] == "dropped_line_candidate"]
    assert dropped_flags[0]["cue_ids"] == [2]
    assert dropped_flags[0]["old_text"] == "dropped line"
    assert dropped_flags[0]["confidence"] == 0.0
