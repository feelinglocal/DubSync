from __future__ import annotations

import json
import os
import wave

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import AudioSnippet
from dubsync.srt_io import parse_srt_text


def test_cli_profile_writes_style_profile(tmp_path, sample_srt_path):
    result = CliRunner().invoke(app, ["profile", str(sample_srt_path), "-o", str(tmp_path / "style.yaml")])

    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "style.yaml").read_text(encoding="utf-8"))
    assert data["fps"] == 30.0
    assert data["max_chars_per_line"] == 26


def test_cli_profile_rejects_malformed_sample_with_clear_message(tmp_path):
    sample_path = tmp_path / "bad.srt"
    sample_path.write_text("not an srt", encoding="utf-8")

    result = CliRunner().invoke(app, ["profile", str(sample_path), "-o", str(tmp_path / "style.yaml")])

    assert result.exit_code != 0
    assert "invalid sample SRT" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "style.yaml").exists()


def test_cli_help_runs():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "sync" in result.output


def test_cli_sync_help_documents_resume_stages():
    result = CliRunner().invoke(app, ["sync", "--help"])
    rendered_help = " ".join(result.output.replace("│", " ").split())

    assert result.exit_code == 0, result.output
    assert "Resume from asr, align, adjudicate, rebuild, or verify" in rendered_help
    assert "Reserved resumable stage selector" not in rendered_help


def test_cli_sync_loads_dotenv_before_provider_resolution(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    seen_keys: list[str | None] = []

    class EnvCapturingLLMAdapter:
        def adjudicate(self, spans):
            seen_keys.append(os.getenv("GEMINI_API_KEY"))
            span = spans[0]
            return [
                {
                    "case_id": span.case_id,
                    "verdict": "use_audio",
                    "final_text": span.asr_text,
                    "confidence": 0.98,
                    "speaker": "A",
                    "character": "unknown",
                    "reason": "captured environment after dotenv load",
                }
            ]

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.setattr("dubsync.pipeline.llm_adapter_from_config", lambda _config, pass_name=None: EnvCapturingLLMAdapter())
    monkeypatch.setattr("dubsync.pipeline.punctuation_adapter_from_config", lambda _config: None)

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nold line\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "new", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "line", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "A"},
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen_keys == ["dotenv-key"]


def test_cli_sync_rejects_unsupported_resume_stage_with_clear_message(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--resume", "bogus"])

    assert result.exit_code != 0
    assert "Unsupported resume stage: bogus" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_reports_stage_config_errors_without_traceback(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "vad": {"provider": "madeup"},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "Unsupported VAD provider: madeup" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_malformed_providers_yaml_with_clear_message(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    providers_path.write_text("asr: [", encoding="utf-8")

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--providers", str(providers_path)])

    assert result.exit_code != 0
    assert "invalid YAML" in result.output
    assert providers_path.name in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_malformed_style_yaml_with_clear_message(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    style_path = tmp_path / "style_profile.yaml"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    style_path.write_text("fps: [", encoding="utf-8")

    result = CliRunner().invoke(app, ["sync", str(srt_path), str(audio_path), "--style", str(style_path)])

    assert result.exit_code != 0
    assert "invalid YAML" in result.output
    assert style_path.name in result.output
    assert "Traceback" not in result.output


def test_cli_sync_rejects_invalid_style_profile_values_with_clear_message(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    style_path = tmp_path / "style_profile.yaml"
    providers_path = tmp_path / "providers.yaml"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    style_path.write_text(yaml.safe_dump({"fps": 0}), encoding="utf-8")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "--providers",
            str(providers_path),
            "--style",
            str(style_path),
        ],
    )

    assert result.exit_code != 0
    assert "invalid style profile" in result.output
    assert style_path.name in result.output
    assert "fps" in result.output
    assert "Traceback" not in result.output


def test_cli_sync_offline_fixture_outputs_reports(tmp_path, shifted_srt_text, shifted_wordstream):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(
        yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}),
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
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert len(synced) == 2
    assert [cue.text for cue in synced] == ["hello there", "general kenobi"]
    assert synced[0].start_ms == 1000
    assert synced[0].end_ms == 1500
    assert synced[1].start_ms == 2000
    assert max(abs(cue.start_ms - expected) for cue, expected in zip(synced, [1000, 2000], strict=True)) <= 34
    assert (workdir / "episode" / "asr.json").exists()
    assert (workdir / "episode" / "align.json").exists()
    assert (workdir / "episode" / "qc_report.json").exists()
    assert (workdir / "episode" / "qc_report.html").exists()
    assert (workdir / "episode" / "changes.diff.srt").exists()
    assert (workdir / "episode" / "verify.json").exists()

    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    align = json.loads((workdir / "episode" / "align.json").read_text(encoding="utf-8"))
    verify = json.loads((workdir / "episode" / "verify.json").read_text(encoding="utf-8"))
    assert align["anchor_regions"][0]["cue_ids"] == [1, 2]
    assert align["anchor_regions"][0]["srt_token_indices"] == [0, 1, 2, 3]
    assert align["anchor_regions"][0]["asr_word_indices"] == [0, 1, 2, 3]
    assert report["summary"]["style_violations"] == 0
    assert [score["cue_id"] for score in report["cue_scores"]] == [1, 2]
    assert report["cue_scores"][0]["source"] == "asr_confidence"
    assert report["cue_scores"][0]["cps"] == 22.0
    assert verify["stage"] == "verify"
    assert verify["summary"] == report["summary"]
    assert verify["cue_scores"] == report["cue_scores"]
    assert verify["style_issues"] == report["style_issues"]
    assert "Cost meter" in result.output


def test_cli_sync_writes_overlap_detection_fixture_report(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    overlap_path = tmp_path / "episode.overlap.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "hello\n"
        "\n"
        "2\n"
        "00:00:00,700 --> 00:00:01,200\n"
        "there\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "hello", "start": 0.00, "end": 0.30, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "there", "start": 0.70, "end": 1.00, "confidence": 0.97, "speaker_id": "B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    overlap_path.write_text(
        json.dumps({"regions": [{"start": 0.25, "end": 0.75, "confidence": 0.88}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "overlap_detection": {"fixture_path": str(overlap_path)},
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
    assert (workdir / "episode" / "overlap.json").exists()
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    overlap_flags = [flag for flag in report["flags"] if flag["kind"] == "overlap_detected"]
    assert overlap_flags[0]["cue_ids"] == [1, 2]
    assert overlap_flags[0]["confidence"] == 0.88


def test_cli_sync_fixture_llm_replaces_improvised_span(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

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
                    "provider": "fixture",
                    "adjudication": {"confidence_gate": 0.95},
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "new spoken line",
                            "confidence": 0.93,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor improvised",
                        }
                    },
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
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[1].text == "new spoken line"
    assert synced[1].start_ms == 1000
    assert synced[1].end_ms == 1875
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    verify = json.loads((workdir / "episode" / "verify.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "text_changed" for flag in report["flags"])
    change_flag = next(flag for flag in report["flags"] if flag["kind"] == "text_changed")
    assert change_flag["old_text"] == "old line"
    assert change_flag["new_text"] == "new spoken line"
    assert change_flag["confidence"] == 0.93
    verify_change_flag = next(flag for flag in verify["flags"] if flag["kind"] == "text_changed")
    assert verify_change_flag == change_flag
    low_confidence_flag = next(flag for flag in report["flags"] if flag["kind"] == "low_confidence_adjudication")
    assert low_confidence_flag["confidence"] == 0.93
    assert low_confidence_flag["old_text"] == "old"
    assert low_confidence_flag["new_text"] == "new spoken line"
    assert not any(flag["kind"] == "unmatched_cue" for flag in report["flags"])


def test_cli_sync_empty_adjudication_text_preserves_parseable_cue(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello there\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "duplicate line\n"
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
                    {"text": "actual", "start": 1.00, "end": 1.22, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "audio", "start": 1.24, "end": 1.54, "confidence": 0.96, "speaker_id": "A"},
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
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "",
                            "confidence": 0.95,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor did not speak this duplicate source cue",
                        }
                    },
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
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert len(synced) == 2
    assert synced[1].text == "duplicate line"
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    dropped_flags = [flag for flag in report["flags"] if flag["kind"] == "dropped_line_candidate"]
    assert dropped_flags
    assert dropped_flags[0]["old_text"] == "duplicate line"
    assert dropped_flags[0]["new_text"] == ""
    assert not any(flag["kind"] == "text_changed" and flag["new_text"] == "" for flag in report["flags"])


def test_cli_sync_audio_snippet_double_check_passes_snippets_to_adjudication(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    snippet_calls: list[dict[str, object]] = []

    class SnippetAwareLLMAdapter:
        def __init__(self):
            self.snippets_by_case: dict[str, AudioSnippet] = {}

        def adjudicate(self, spans):
            raise AssertionError("expected audio snippet-aware adjudication path")

        def adjudicate_with_audio(self, spans, audio_snippets):
            self.snippets_by_case = dict(audio_snippets)
            span = spans[0]
            return [
                    {
                        "case_id": span.case_id,
                        "verdict": "use_audio",
                        "final_text": "new spoken line",
                        "confidence": 0.91,
                    "speaker": "A",
                    "character": "unknown",
                    "reason": "audio snippet confirms the spoken line",
                }
            ]

    snippet_adapter = SnippetAwareLLMAdapter()

    def fake_extract_audio_snippets(audio_path_arg, spans, output_dir, pad_seconds, max_duration_seconds):
        snippet_path = output_dir / f"{spans[0].case_id}.wav"
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet_path.write_bytes(b"RIFFsnippetWAVEfmt ")
        snippet_calls.append(
            {
                "audio_path": audio_path_arg,
                "case_id": spans[0].case_id,
                "pad_seconds": pad_seconds,
                "max_duration_seconds": max_duration_seconds,
            }
        )
        return [
            AudioSnippet(
                case_id=spans[0].case_id,
                path=str(snippet_path),
                mime_type="audio/wav",
                start=0.0,
                end=3.0,
            )
        ]

    monkeypatch.setattr("dubsync.pipeline.llm_adapter_from_config", lambda _config, pass_name=None: snippet_adapter)
    monkeypatch.setattr("dubsync.pipeline.extract_audio_snippets", fake_extract_audio_snippets, raising=False)
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nold line\n\n",
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
                    "adjudication": {
                        "audio_snippet_double_check": {
                            "enabled": True,
                            "pad_seconds": 1.5,
                            "max_duration_seconds": 8.0,
                        }
                    },
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert snippet_calls == [
        {
            "audio_path": audio_path,
            "case_id": "case-1",
            "pad_seconds": 1.5,
            "max_duration_seconds": 8.0,
        }
    ]
    assert snippet_adapter.snippets_by_case["case-1"].mime_type == "audio/wav"
    artifact = json.loads((workdir / "episode" / "audio_snippets.json").read_text(encoding="utf-8"))
    assert artifact["snippets"][0]["case_id"] == "case-1"
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["new_text"] == "new spoken line" for flag in report["flags"] if flag["kind"] == "text_changed")


def test_cli_sync_reuses_cached_llm_adjudication_without_resume(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    changed_providers_path = tmp_path / "providers-changed.yaml"
    first_out_path = tmp_path / "episode.first.srt"
    second_out_path = tmp_path / "episode.second.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

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
                    "provider": "fixture",
                    "model": "fixture-adjudicator",
                    "adjudication": {"confidence_gate": 0.95},
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "new spoken line",
                            "confidence": 0.93,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor improvised",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    changed_providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {
                    "provider": "fixture",
                    "model": "fixture-adjudicator",
                    "adjudication": {"confidence_gate": 0.95},
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "wrong cached bypass",
                            "confidence": 0.99,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "this response should not be used on a cache hit",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    first = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(first_out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )
    assert first.exit_code == 0, first.output

    second = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(second_out_path),
            "--providers",
            str(changed_providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert second.exit_code == 0, second.output
    synced = parse_srt_text(second_out_path.read_text(encoding="utf-8"))
    assert synced[1].text == "new spoken line"
    assert (workdir / "episode" / "llm-cache").exists()


def test_cli_sync_reuses_cached_llm_punctuation_without_resume(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    changed_providers_path = tmp_path / "providers-changed.yaml"
    first_out_path = tmp_path / "episode.first.srt"
    second_out_path = tmp_path / "episode.second.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello there\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "general kenobi\n"
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
                    {"text": "general", "start": 1.00, "end": 1.30, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "kenobi", "start": 1.35, "end": 1.80, "confidence": 0.99, "speaker_id": "A"},
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
                    "provider": "fixture",
                    "model": "fixture-punctuation",
                    "punctuation": {"1": "Hello there."},
                },
            }
        ),
        encoding="utf-8",
    )
    changed_providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {
                    "provider": "fixture",
                    "model": "fixture-punctuation",
                    "punctuation": {"1": "Hello there?"},
                },
            }
        ),
        encoding="utf-8",
    )

    first = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(first_out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )
    assert first.exit_code == 0, first.output

    second = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(second_out_path),
            "--providers",
            str(changed_providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert second.exit_code == 0, second.output
    synced = parse_srt_text(second_out_path.read_text(encoding="utf-8"))
    assert synced[0].text == "Hello there."
    assert (workdir / "episode" / "llm-cache").exists()


def test_cli_sync_fixture_llm_inserts_adlib_span(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello there\n"
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
                    {"text": "surprise", "start": 1.00, "end": 1.36, "confidence": 0.95, "speaker_id": "A"},
                    {"text": "line", "start": 1.40, "end": 1.72, "confidence": 0.94, "speaker_id": "A"},
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
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "surprise line",
                            "confidence": 0.88,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor added an unscripted line",
                        }
                    },
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
            "--fps",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in synced] == ["hello there", "surprise line"]
    assert synced[1].start_ms == 1000
    assert synced[1].end_ms == 1766
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    adlib_flag = next(flag for flag in report["flags"] if flag["kind"] == "adlib_inserted")
    assert adlib_flag["cue_ids"] == [2]
    assert adlib_flag["old_text"] is None
    assert adlib_flag["new_text"] == "surprise line"
    assert adlib_flag["confidence"] == 0.88


def test_cli_sync_adlib_inserted_between_cues_exports_sequential_srt_indices(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello there\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "goodbye now\n"
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
                    {"text": "surprise", "start": 1.00, "end": 1.36, "confidence": 0.95, "speaker_id": "A"},
                    {"text": "line", "start": 1.40, "end": 1.72, "confidence": 0.94, "speaker_id": "A"},
                    {"text": "goodbye", "start": 2.00, "end": 2.22, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "now", "start": 2.24, "end": 2.45, "confidence": 0.96, "speaker_id": "A"},
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
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "surprise line",
                            "confidence": 0.88,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor added an unscripted line",
                        }
                    },
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
            "--fps",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in synced] == ["hello there", "surprise line", "goodbye now"]
    assert [cue.index for cue in synced] == [1, 2, 3]


def test_cli_sync_resume_align_reuses_asr_artifact(tmp_path, shifted_srt_text, shifted_wordstream):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    broken_providers_path = tmp_path / "broken-providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.resumed.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")
    broken_providers_path.write_text(yaml.safe_dump({"asr": {"provider": "not-real"}}), encoding="utf-8")

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    assert (workdir / "episode" / "asr.json").exists()

    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(broken_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "align",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    assert parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))[0].start_ms == 1000


def test_cli_sync_resume_asr_uses_ingest_artifact(tmp_path, shifted_srt_text, shifted_wordstream):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.asr.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    assert (workdir / "episode" / "ingest.json").exists()

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nmutated source\n\n", encoding="utf-8")
    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "asr",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in resumed_cues] == ["hello there", "general kenobi"]
    assert resumed_cues[0].start_ms == 1000


def test_cli_sync_resume_verify_uses_rebuild_artifact(tmp_path, shifted_srt_text, shifted_wordstream):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    broken_providers_path = tmp_path / "broken-providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.verify.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")
    broken_providers_path.write_text(yaml.safe_dump({"asr": {"provider": "not-real"}}), encoding="utf-8")

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    assert (workdir / "episode" / "rebuild.json").exists()

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nmutated source\n\n", encoding="utf-8")
    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(broken_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "verify",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in resumed_cues] == ["hello there", "general kenobi"]
    assert resumed_cues[0].start_ms == 1000


def test_cli_sync_resume_rebuild_uses_prior_stage_artifacts(tmp_path, shifted_srt_text, shifted_wordstream):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    broken_providers_path = tmp_path / "broken-providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.rebuild.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(json.dumps({"words": shifted_wordstream}), encoding="utf-8")
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")
    broken_providers_path.write_text(yaml.safe_dump({"asr": {"provider": "not-real"}}), encoding="utf-8")

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    assert (workdir / "episode" / "align.json").exists()
    assert (workdir / "episode" / "adjudicate.json").exists()

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nmutated source\n\n", encoding="utf-8")
    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(broken_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "rebuild",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in resumed_cues] == ["hello there", "general kenobi"]
    assert resumed_cues[0].start_ms == 1000


def test_cli_sync_resume_rebuild_preserves_adjudication_qc_flags(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    resumed_providers_path = tmp_path / "providers-resume.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.rebuild.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

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
                    "provider": "fixture",
                    "adjudication": {"confidence_gate": 0.95},
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "new spoken line",
                            "confidence": 0.93,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor improvised",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    resumed_providers_path.write_text(yaml.safe_dump({"asr": {"provider": "not-real"}}), encoding="utf-8")

    first = CliRunner().invoke(
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
        ],
    )
    assert first.exit_code == 0, first.output
    first_report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "low_confidence_adjudication" for flag in first_report["flags"])

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nmutated source\n\n", encoding="utf-8")
    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(resumed_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "rebuild",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "low_confidence_adjudication" for flag in resumed_report["flags"])


def test_cli_sync_resume_adjudicate_uses_prior_stage_artifacts(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    resumed_providers_path = tmp_path / "providers-resume.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.adjudicate.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

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
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")
    resumed_providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"provider": "not-real"},
                "llm": {
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "new spoken line",
                            "confidence": 0.93,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor improvised",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    assert (workdir / "episode" / "ingest.json").exists()
    assert (workdir / "episode" / "align.json").exists()
    assert (workdir / "episode" / "asr.json").exists()

    srt_path.write_text("1\n00:00:10,000 --> 00:00:11,000\nmutated source\n\n", encoding="utf-8")
    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(resumed_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "adjudicate",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))
    assert resumed_cues[1].plain_text == "new spoken line"
    assert resumed_cues[1].start_ms == 1000
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "text_changed" for flag in report["flags"])


def test_cli_sync_resume_adjudicate_uses_normalized_audio_artifact_for_verify(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    resumed_providers_path = tmp_path / "providers-resume.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.adjudicate.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    silence_paths: list[str] = []

    def fake_silence_flags(audio_path_arg, cues, threshold_dbfs=-45.0):
        del cues, threshold_dbfs
        silence_paths.append(str(audio_path_arg))
        return []

    monkeypatch.setattr("dubsync.pipeline.silence_flags_for_cues", fake_silence_flags)
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nold line\n\n",
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
    providers_path.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}), encoding="utf-8")
    resumed_providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"provider": "not-real"},
                "llm": {
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "new spoken line",
                            "confidence": 0.93,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor improvised",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    first = CliRunner().invoke(
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
    assert first.exit_code == 0, first.output
    normalized_audio = workdir / "episode" / "audio.16k.wav"
    normalized_audio.write_bytes(b"RIFFnormalizedWAVEfmt ")

    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_out_path),
            "--providers",
            str(resumed_providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "adjudicate",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    assert silence_paths == [str(normalized_audio)]


def test_cli_local_mode_routes_to_whisperx_without_cloud_keys(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    out_path = tmp_path / "episode.synced.srt"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8")
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(out_path),
            "--workdir",
            str(tmp_path / "work"),
            "--local",
        ],
    )

    if result.exit_code == 0:
        assert out_path.exists()
    else:
        assert "WhisperX" in result.output
    assert "Traceback" not in result.output


def test_cli_batch_accepts_fps_flag(tmp_path, shifted_srt_text, shifted_wordstream):
    folder = tmp_path / "episodes"
    folder.mkdir()
    srt_path = folder / "ep1.srt"
    audio_path = folder / "ep1.wav"
    wordstream_path = tmp_path / "ep1.wordstream.json"
    providers_path = tmp_path / "providers.yaml"

    srt_path.write_text(shifted_srt_text, encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
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
            "--fps",
            "25",
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Cost meter" in result.output
    assert "ep1.synced.srt" in result.output
    synced = parse_srt_text((folder / "ep1.synced.srt").read_text(encoding="utf-8"))
    assert synced[0].start_ms == 1000
    assert synced[0].end_ms % 40 == 0
