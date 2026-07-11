from __future__ import annotations

import json

import yaml

from dubsync.pipeline import sync_episode


class ContextRecordingLLMAdapter:
    def __init__(self):
        self.seen_spans = []

    def adjudicate(self, spans):
        self.seen_spans.extend(spans)
        return [
            {
                "case_id": span.case_id,
                "verdict": "use_audio",
                "final_text": span.asr_text,
                "confidence": 0.92,
                "speaker": None,
                "character": "unknown",
                "reason": "actor improvised",
            }
            for span in spans
        ]


def test_pipeline_passes_two_neighboring_cues_as_adjudication_context(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:00,800\none alpha\n\n"
        "2\n00:00:01,000 --> 00:00:01,800\ntwo beta\n\n"
        "3\n00:00:02,000 --> 00:00:02,800\nold middle\n\n"
        "4\n00:00:04,000 --> 00:00:04,800\nfour delta\n\n"
        "5\n00:00:05,000 --> 00:00:05,800\nfive echo\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "one", "start": 0.0, "end": 0.2, "confidence": 0.98},
                    {"text": "alpha", "start": 0.25, "end": 0.55, "confidence": 0.98},
                    {"text": "two", "start": 1.0, "end": 1.2, "confidence": 0.98},
                    {"text": "beta", "start": 1.25, "end": 1.55, "confidence": 0.98},
                    {"text": "new", "start": 2.0, "end": 2.2, "confidence": 0.98},
                    {"text": "spoken", "start": 2.25, "end": 2.55, "confidence": 0.98},
                    {"text": "middle", "start": 2.6, "end": 2.8, "confidence": 0.98},
                    {"text": "four", "start": 4.0, "end": 4.2, "confidence": 0.98},
                    {"text": "delta", "start": 4.25, "end": 4.55, "confidence": 0.98},
                    {"text": "five", "start": 5.0, "end": 5.2, "confidence": 0.98},
                    {"text": "echo", "start": 5.25, "end": 5.55, "confidence": 0.98},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}, "llm": {"provider": "gemini"}}),
        encoding="utf-8",
    )
    llm = ContextRecordingLLMAdapter()
    monkeypatch.setattr("dubsync.pipeline.llm_adapter_from_config", lambda _config, pass_name=None: llm)
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: None)

    sync_episode(srt_path, audio_path, out_path, workdir, providers_path=providers_path)

    assert len(llm.seen_spans) == 1
    span = llm.seen_spans[0]
    assert [item.text for item in span.context_before] == ["one alpha", "two beta"]
    assert [item.text for item in span.context_after] == ["four delta", "five echo"]
