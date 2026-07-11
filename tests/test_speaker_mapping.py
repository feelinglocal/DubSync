from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.pipeline import sync_episode
from dubsync import speaker_mapping as speaker_mapping_module


class MeteredSpeakerMappingAdapter:
    def __init__(self):
        self.usage_events: list[object] = []
        self.seen_text: list[str] = []
        self.calls = 0

    def map_speakers(self, cues):
        self.calls += 1
        self.usage_events.append({"usage_metadata": {"input_token_count": 400, "output_token_count": 80}})
        self.seen_text = [cue.plain_text for cue in cues]
        return {"SPEAKER_00": "Luna"}


class CapturingPunctuationAdapter:
    def __init__(self):
        self.labels: list[tuple[str | None, str | None]] = []

    def punctuate(self, cues):
        self.labels = [(cue.speaker_id, getattr(cue, "character", None)) for cue in cues]
        return {}


def test_cli_sync_writes_speaker_character_mapping_artifact_and_qc(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:00,800\nhello luna\n\n"
        "2\n00:00:01,000 --> 00:00:01,800\nhello matthew\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "SPEAKER_00"},
                    {"text": "luna", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "SPEAKER_00"},
                    {"text": "hello", "start": 1.0, "end": 1.2, "confidence": 0.98, "speaker_id": "SPEAKER_01"},
                    {"text": "matthew", "start": 1.25, "end": 1.65, "confidence": 0.97, "speaker_id": "SPEAKER_01"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "speaker_mapping": {
                    "fixture": {
                        "SPEAKER_00": "Luna",
                        "SPEAKER_01": "Matthew",
                    }
                },
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
    assert "Luna" not in out_path.read_text(encoding="utf-8")
    speaker_map = json.loads((workdir / "episode" / "speaker_map.json").read_text(encoding="utf-8"))
    assert speaker_map == {"SPEAKER_00": "Luna", "SPEAKER_01": "Matthew"}
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "speaker_character_mapped" and flag["new_text"] == "Luna" for flag in report["flags"])


def test_speaker_mapping_provider_llm_uses_configured_llm_adapter(monkeypatch):
    adapter = MeteredSpeakerMappingAdapter()
    monkeypatch.setattr(speaker_mapping_module, "llm_adapter_from_config", lambda _config, pass_name=None: adapter)

    resolved = speaker_mapping_module.speaker_mapping_adapter_from_config(
        {"llm": {"provider": "gemini"}, "speaker_mapping": {"provider": "llm"}}
    )

    assert resolved is adapter


def test_pipeline_records_llm_speaker_mapping_artifact_and_cost(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    adapter = MeteredSpeakerMappingAdapter()

    srt_path.write_text("1\n00:00:00,000 --> 00:00:00,800\nhello luna\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "SPEAKER_00"},
                    {"text": "luna", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "SPEAKER_00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
                "speaker_mapping": {"provider": "llm"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(speaker_mapping_module, "llm_adapter_from_config", lambda _config, pass_name=None: adapter)
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: None)

    result = sync_episode(srt_path, audio_path, out_path, workdir, providers_path=providers_path)

    assert result.cost_meter.total_usd == 0.00132
    assert adapter.seen_text == ["hello luna"]
    speaker_map = json.loads((workdir / "episode" / "speaker_map.json").read_text(encoding="utf-8"))
    assert speaker_map == {"SPEAKER_00": "Luna"}
    cost_data = json.loads((workdir / "episode" / "cost.json").read_text(encoding="utf-8"))
    assert [item for item in cost_data["items"] if item["kind"] == "tokens"] == [
        {
            "provider": "gemini-3.5-flash",
            "kind": "tokens",
            "units": {"input_tokens": 400.0, "output_tokens": 80.0},
            "usd": 0.00132,
        }
    ]


def test_pipeline_reuses_cached_llm_speaker_mapping_without_resume(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    first_out_path = tmp_path / "episode.first.srt"
    second_out_path = tmp_path / "episode.second.srt"
    workdir = tmp_path / "work"
    adapter = MeteredSpeakerMappingAdapter()

    srt_path.write_text("1\n00:00:00,000 --> 00:00:00,800\nhello luna\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "SPEAKER_00"},
                    {"text": "luna", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "SPEAKER_00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
                "speaker_mapping": {"provider": "llm"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(speaker_mapping_module, "llm_adapter_from_config", lambda _config, pass_name=None: adapter)
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: None)

    first = sync_episode(srt_path, audio_path, first_out_path, workdir, providers_path=providers_path)
    second = sync_episode(srt_path, audio_path, second_out_path, workdir, providers_path=providers_path)

    assert first.cost_meter.total_usd == 0.00132
    assert second.cost_meter.total_usd == 0.0
    assert adapter.calls == 1
    assert (workdir / "episode" / "llm-cache").exists()
    speaker_map = json.loads((workdir / "episode" / "speaker_map.json").read_text(encoding="utf-8"))
    assert speaker_map == {"SPEAKER_00": "Luna"}


def test_pipeline_attaches_speaker_mapping_labels_before_punctuation(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    punctuation = CapturingPunctuationAdapter()

    srt_path.write_text("1\n00:00:00,000 --> 00:00:00,800\nhello luna\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "SPEAKER_00"},
                    {"text": "luna", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "SPEAKER_00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "speaker_mapping": {"fixture": {"SPEAKER_00": "Luna"}},
                "llm": {"provider": "gemini"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: punctuation)

    sync_episode(srt_path, audio_path, out_path, workdir, providers_path=providers_path)

    assert punctuation.labels == [("SPEAKER_00", "Luna")]
