import json

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import AudioSnippet
from dubsync.srt_io import parse_srt_text
from dubsync.tokenize import alphanumeric_signature


def test_long_asr_only_tail_is_fully_adjudicated_in_bounded_audio_parts(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    provider_batches: list[list[str]] = []
    extracted_spans: list[object] = []

    class CompletePartAdapter:
        usage_events: list[object] = []

        def adjudicate(self, spans):
            raise AssertionError("expected bounded audio adjudication")

        def adjudicate_with_audio(self, spans, audio_snippets):
            provider_batches.append([span.case_id for span in spans])
            assert set(audio_snippets) == {span.case_id for span in spans}
            return list(
                reversed(
                    [
                        {
                            "case_id": span.case_id,
                            "verdict": "use_audio",
                            "final_text": span.asr_text,
                            "confidence": 0.97,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "the complete target window is spoken",
                        }
                        for span in spans
                    ]
                )
            )

    def fake_extract_audio_snippets(
        audio_path_arg,
        spans,
        output_dir,
        pad_seconds,
        max_duration_seconds,
    ):
        del audio_path_arg
        snippets = []
        for span in spans:
            extracted_spans.append(span)
            assert span.start is not None and span.end is not None
            assert span.end - span.start + 2 * pad_seconds <= max_duration_seconds + 0.001
            snippet_path = output_dir / f"{span.case_id}.wav"
            snippet_path.parent.mkdir(parents=True, exist_ok=True)
            snippet_path.write_bytes(b"RIFFsnippetWAVEfmt ")
            snippets.append(
                AudioSnippet(
                    case_id=span.case_id,
                    path=str(snippet_path),
                    start=max(0.0, span.start - pad_seconds),
                    end=span.end + pad_seconds,
                )
            )
        return snippets

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nsource line.\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    tail_tokens = [
        "first",
        "missing",
        "sentence.",
        "second",
        "complete",
        "sentence.",
        "third",
        "spoken",
        "sentence.",
        "final",
        "dialogue",
        "line.",
    ]
    words = [
        {"text": "source", "start": 0.10, "end": 0.30, "confidence": 0.99, "speaker_id": "A"},
        {"text": "line.", "start": 0.35, "end": 0.60, "confidence": 0.99, "speaker_id": "A"},
    ]
    cursor = 1.05
    for token in tail_tokens:
        words.append(
            {
                "text": token,
                "start": round(cursor, 3),
                "end": round(cursor + 0.34, 3),
                "confidence": 0.98,
                "speaker_id": "A",
            }
        )
        cursor += 0.86
    wordstream_path.write_text(json.dumps({"words": words}), encoding="utf-8")
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {
                    "provider": "gemini",
                    "adjudication": {
                        "confidence_gate": 0.7,
                        "audio_snippet_double_check": {
                            "enabled": True,
                            "pad_seconds": 1.0,
                            "max_duration_seconds": 6.0,
                        },
                    },
                },
                "generation": {
                    "max_gap_seconds": 0.8,
                    "max_cue_duration_seconds": 5.0,
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = CompletePartAdapter()
    monkeypatch.setattr(
        "dubsync.pipeline.llm_adapter_from_config",
        lambda _config, pass_name=None: adapter,
    )
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: None)
    monkeypatch.setattr("dubsync.pipeline.extract_audio_snippets", fake_extract_audio_snippets)

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
            "--fps",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(extracted_spans) > 1
    assert len(provider_batches) == 1
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    tail_cues = synced[1:]
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in tail_cues)) == [
        token.rstrip(".") for token in tail_tokens
    ]
    assert tail_cues[0].start_ms == 1_033
    assert tail_cues[-1].end_ms >= 10_866
    assert all(cue.duration_ms <= 5_000 for cue in tail_cues)
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "adjudication_span_partitioned" for flag in report["flags"])
    assert not any(flag["kind"] == "adjudication_parts_incomplete" for flag in report["flags"])
