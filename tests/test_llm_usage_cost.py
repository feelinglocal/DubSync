from __future__ import annotations

import json
from pathlib import Path

import yaml

from dubsync.pipeline import sync_episode


class MeteredLLMAdapter:
    def __init__(self):
        self.usage_events: list[object] = []

    def adjudicate(self, spans):
        self.usage_events.append({"usage_metadata": {"input_token_count": 2000, "output_token_count": 1000}})
        span = spans[0]
        return [
            {
                "case_id": span.case_id,
                "verdict": "use_audio",
                "final_text": span.asr_text,
                "confidence": 0.91,
                "speaker": "A",
                "character": "unknown",
                "reason": "actor improvised",
            }
        ]


class MeteredPunctuationAdapter:
    def __init__(self):
        self.usage_events: list[object] = []

    def punctuate(self, cues):
        self.usage_events.append({"usage_metadata": {"input_token_count": 300, "output_token_count": 100}})
        return {}


def test_pipeline_records_live_llm_usage_events_in_cost_json(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    wordstream_path = tmp_path / "episode.wordstream.json"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello there\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "old line\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.00, "end": 0.20, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "there", "start": 0.23, "end": 0.45, "confidence": 0.97, "speaker_id": "A"},
                    {"text": "new", "start": 1.00, "end": 1.22, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "spoken", "start": 1.24, "end": 1.54, "confidence": 0.96, "speaker_id": "A"},
                    {"text": "line", "start": 1.56, "end": 1.80, "confidence": 0.99, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {
                    "provider": "gemini",
                    "model": "gemini-3.5-flash",
                    "punctuation": {"model": "gemini-3.5-flash-lite"},
                },
            }
        ),
        encoding="utf-8",
    )
    llm_adapter = MeteredLLMAdapter()
    punctuation_adapter = MeteredPunctuationAdapter()
    monkeypatch.setattr("dubsync.pipeline.llm_adapter_from_config", lambda _config, pass_name=None: llm_adapter)
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: punctuation_adapter)

    result = sync_episode(srt_path, audio_path, out_path, workdir, providers_path=providers_path)

    assert result.cost_meter.total_usd == 0.01234
    cost_data = json.loads((workdir / "episode" / "cost.json").read_text(encoding="utf-8"))
    assert [item for item in cost_data["items"] if item["kind"] == "tokens"] == [
        {
            "provider": "gemini-3.5-flash",
            "kind": "tokens",
            "units": {"input_tokens": 2000.0, "output_tokens": 1000.0},
            "usd": 0.012,
        },
        {
            "provider": "gemini-3.5-flash-lite",
            "kind": "tokens",
            "units": {"input_tokens": 300.0, "output_tokens": 100.0},
            "usd": 0.00034,
        },
    ]
