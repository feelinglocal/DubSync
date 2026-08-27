from __future__ import annotations

import json
import os
import wave

import pytest
import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import AlignmentResult, AudioSnippet, Cue, DivergenceSpan
import dubsync.pipeline as pipeline_module
from dubsync.pipeline import (
    _alignment_health_flags,
    _alignment_summary_metadata,
    _punctuation_cache_key,
    _speaker_mapping_cache_key,
)
from dubsync.srt_io import parse_srt_text
from dubsync.text_metrics import display_width
from dubsync.tokenize import alphanumeric_signature


def test_cli_profile_writes_style_profile(tmp_path, sample_srt_path):
    result = CliRunner().invoke(app, ["profile", str(sample_srt_path), "-o", str(tmp_path / "style.yaml")])

    assert result.exit_code == 0, result.output
    data = yaml.safe_load((tmp_path / "style.yaml").read_text(encoding="utf-8"))
    assert data["fps"] == 30.0
    assert data["max_chars_per_line"] == 26


def test_punctuation_cache_key_includes_line_constraints():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello there"])]
    config = {"llm": {"provider": "fixture", "model": "fixture-punctuation", "punctuation": {}}}

    house_style = _punctuation_cache_key(cues, config, max_chars_per_line=26, max_lines_per_cue=2)
    wider_style = _punctuation_cache_key(cues, config, max_chars_per_line=42, max_lines_per_cue=2)

    assert house_style.digest != wider_style.digest


def test_incomplete_source_guard_leaves_source_backed_and_short_adlib_spans_unchanged():
    source_backed = DivergenceSpan(
        case_id="case-source",
        cue_ids=[1],
        srt_text="customer subtitle text",
        asr_text="spoken replacement text",
        start=10.0,
        end=70.0,
        srt_token_indices=[0, 1, 2],
        asr_word_indices=[0, 1, 2],
    )
    short_adlib = DivergenceSpan(
        case_id="case-adlib",
        cue_ids=[],
        srt_text="",
        asr_text="short spoken aside",
        start=71.0,
        end=74.0,
        asr_word_indices=[3, 4, 5],
    )
    source_free_without_asr_words = DivergenceSpan(
        case_id="case-empty",
        cue_ids=[],
        srt_text="",
        asr_text="",
        start=None,
        end=None,
        asr_word_indices=[],
    )

    provider_spans, held_decisions, flags = pipeline_module._hold_incomplete_source_insertions(
        [source_backed, short_adlib, source_free_without_asr_words],
        {"generation": {"max_generated_adlib_duration_seconds": 20.0}},
    )

    assert provider_spans == [source_backed, short_adlib, source_free_without_asr_words]
    assert held_decisions == []
    assert flags == []


def test_incomplete_source_guard_holds_oversized_source_backed_span():
    spans = [
        DivergenceSpan(
            case_id="case-small",
            cue_ids=[1, 2, 3, 4, 5],
            srt_text="small source span",
            asr_text="small replacement span",
            start=1.0,
            end=6.0,
            srt_token_indices=[0, 1, 2],
            asr_word_indices=[0, 1, 2],
        ),
        DivergenceSpan(
            case_id="case-huge",
            cue_ids=list(range(1, 9)),
            srt_text="huge source span",
            asr_text="huge replacement span",
            start=10.0,
            end=70.0,
            srt_token_indices=list(range(16)),
            asr_word_indices=list(range(20)),
        ),
    ]

    provider_spans, held_decisions, flags = pipeline_module._hold_incomplete_source_insertions(
        spans,
        {"generation": {"max_generated_adlib_duration_seconds": 20.0}},
        source_cue_count=30,
    )

    assert provider_spans == [spans[0]]
    assert [decision.case_id for decision in held_decisions] == ["case-huge"]
    assert held_decisions[0].verdict == "keep_srt"
    assert [flag.kind for flag in flags] == ["oversized_adjudication_span_held"]
    assert flags[0].severity == "error"
    assert flags[0].old_text is None
    assert flags[0].new_text is None


def test_alignment_summary_metadata_and_health_flags_surface_low_coverage():
    alignment = AlignmentResult(anchor_coverage=0.4, unmatched_cue_ids=[2, 3])

    metadata = _alignment_summary_metadata(alignment, source_cue_count=4)
    flags = _alignment_health_flags(alignment, source_cue_count=4)

    assert metadata["alignment_anchor_coverage"] == 0.4
    assert metadata["alignment_unmatched_cue_ratio"] == 0.5
    assert flags[0].kind == "alignment_anchor_coverage_low"
    assert flags[0].severity == "error"


def test_incomplete_source_guard_holds_source_free_spans_with_invalid_timing():
    missing_start = DivergenceSpan(
        case_id="case-missing-start",
        cue_ids=[],
        srt_text="",
        asr_text="unbounded spoken section",
        start=None,
        end=70.0,
        asr_word_indices=[0, 1, 2],
    )
    reversed_timing = DivergenceSpan(
        case_id="case-reversed",
        cue_ids=[],
        srt_text="",
        asr_text="invalid spoken section",
        start=70.0,
        end=10.0,
        asr_word_indices=[3, 4, 5],
    )
    negative_timing = DivergenceSpan(
        case_id="case-negative",
        cue_ids=[],
        srt_text="",
        asr_text="negative spoken section",
        start=-70.0,
        end=-10.0,
        asr_word_indices=[6, 7, 8],
    )
    missing_word_indices = DivergenceSpan(
        case_id="case-missing-indices",
        cue_ids=[],
        srt_text="",
        asr_text="long ASR evidence without recoverable word indices",
        start=10.0,
        end=70.0,
        asr_word_indices=[],
    )

    provider_spans, held_decisions, flags = pipeline_module._hold_incomplete_source_insertions(
        [missing_start, reversed_timing, negative_timing, missing_word_indices],
        {"generation": {"max_generated_adlib_duration_seconds": 20.0}},
    )

    assert provider_spans == []
    assert [decision.case_id for decision in held_decisions] == [
        "case-missing-start",
        "case-reversed",
        "case-negative",
        "case-missing-indices",
    ]
    assert [flag.kind for flag in flags] == [
        "generated_adlib_rejected_incomplete_source",
        "generated_adlib_rejected_incomplete_source",
        "generated_adlib_rejected_incomplete_source",
        "generated_adlib_rejected_incomplete_source",
    ]
    assert all("valid timing" in flag.message for flag in flags[:3])
    assert "above the 20s" in flags[3].message


@pytest.mark.parametrize("invalid_limit", [float("inf"), float("nan"), 0.0, -1.0])
def test_incomplete_source_guard_rejects_invalid_safety_limits(invalid_limit):
    with pytest.raises(ValueError, match="max_generated_adlib_duration_seconds"):
        pipeline_module._hold_incomplete_source_insertions(
            [],
            {"generation": {"max_generated_adlib_duration_seconds": invalid_limit}},
        )


def test_adapter_episode_context_hook_receives_complete_ordered_cues():
    seen: list[list[Cue]] = []

    class ContextAdapter:
        def set_episode_context(self, cues: list[Cue]) -> None:
            seen.append(list(cues))

    episode = [
        Cue(index=4, start_ms=0, end_ms=700, lines=["Vorher."]),
        Cue(index=5, start_ms=800, end_ms=1_500, lines=["Bleib hier."]),
        Cue(index=6, start_ms=1_600, end_ms=2_300, lines=["Nachher."]),
    ]

    pipeline_module._set_adapter_episode_context(ContextAdapter(), episode)

    assert seen == [episode]


def test_cli_sync_flags_explicit_fps_that_disagrees_with_source_grid(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    output_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    srt_path.write_text(
        "1\n00:00:05,250 --> 00:00:06,625\nalpha\n\n"
        "2\n00:00:06,833 --> 00:00:09,083\nbeta\n\n"
        "3\n00:00:09,416 --> 00:00:11,291\ngamma\n\n"
        "4\n00:00:11,625 --> 00:00:13,000\ndelta\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "alpha", "start": 5.25, "end": 6.625},
                    {"text": "beta", "start": 6.833, "end": 9.083},
                    {"text": "gamma", "start": 9.416, "end": 11.291},
                    {"text": "delta", "start": 11.625, "end": 13.0},
                ]
            }
        ),
        encoding="utf-8",
    )
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
            "--providers",
            str(providers_path),
            "--fps",
            "30",
            "--no-llm",
            "--output",
            str(output_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    mismatch = [flag for flag in report["flags"] if flag["kind"] == "fps_override_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "warning"
    assert "24" in mismatch[0]["message"]
    assert "30" in mismatch[0]["message"]


def test_punctuation_cache_key_changes_with_prompt_version(monkeypatch):
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello there"])]
    config = {"llm": {"provider": "fixture", "model": "fixture-punctuation", "punctuation": {}}}
    monkeypatch.setattr(pipeline_module, "_PUNCTUATION_PROMPT_VERSION", "punctuation-a", raising=False)
    first = _punctuation_cache_key(cues, config, max_chars_per_line=26, max_lines_per_cue=2)
    monkeypatch.setattr(pipeline_module, "_PUNCTUATION_PROMPT_VERSION", "punctuation-b", raising=False)
    second = _punctuation_cache_key(cues, config, max_chars_per_line=26, max_lines_per_cue=2)

    assert first.digest != second.digest


def test_cli_sync_splits_overlong_source_cue_when_style_limits_lines(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    style_path = tmp_path / "style.yaml"
    output_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    srt_path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Team Falcon hat eigenmächtig die Position verraten.\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Team", "start": 1.00, "end": 1.18, "speaker_id": "A"},
                    {"text": "Falcon", "start": 1.20, "end": 1.48, "speaker_id": "A"},
                    {"text": "hat", "start": 1.50, "end": 1.62, "speaker_id": "A"},
                    {"text": "eigenmächtig", "start": 1.64, "end": 2.05, "speaker_id": "A"},
                    {"text": "die", "start": 2.07, "end": 2.20, "speaker_id": "A"},
                    {"text": "Position", "start": 2.22, "end": 2.55, "speaker_id": "A"},
                    {"text": "verraten.", "start": 2.57, "end": 2.95, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump({"asr": {"fixture_path": str(wordstream_path)}}),
        encoding="utf-8",
    )
    style_path.write_text(
        "fps: 30\nmax_lines_per_cue: 2\nmax_chars_per_line: 18\nmin_cue_dur: 0.4\ntail_ms: 0\n",
        encoding="utf-8",
    )

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
            "--no-llm",
            "--output",
            str(output_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert len(synced) == 2
    assert " ".join(cue.plain_text for cue in synced) == "Team Falcon hat eigenmächtig die Position verraten."
    assert all(len(cue.lines) <= 2 for cue in synced)
    assert synced[0].end_ms <= synced[1].start_ms
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "sync_cue_line_limit_split" for flag in report["flags"])


def test_speaker_mapping_cache_key_changes_with_prompt_version(monkeypatch):
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello"], speaker_id="A")]
    config = {"llm": {"provider": "fixture", "model": "fixture-speakers"}}
    monkeypatch.setattr(pipeline_module, "_SPEAKER_MAPPING_PROMPT_VERSION", "speakers-a", raising=False)
    first = _speaker_mapping_cache_key(cues, config)
    monkeypatch.setattr(pipeline_module, "_SPEAKER_MAPPING_PROMPT_VERSION", "speakers-b", raising=False)
    second = _speaker_mapping_cache_key(cues, config)

    assert first.digest != second.digest


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
    rendered_help = " ".join(result.output.replace("\u2502", " ").split())

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
            "--fps",
            "24",
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


def test_cli_sync_keeps_german_profanity_mask_without_llm_rewrite(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Das ist verd*mmt knapp.\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Das", "start": 0.00, "end": 0.12, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "ist", "start": 0.14, "end": 0.25, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "verdammt", "start": 0.27, "end": 0.55, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "knapp", "start": 0.57, "end": 0.80, "confidence": 0.98, "speaker_id": "A"},
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
                            "final_text": "Das ist verdammt knapp.",
                            "confidence": 0.97,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "ASR expanded censored German profanity",
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
            "24",
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].plain_text == "Das ist verd*mmt knapp."
    assert synced[0].start_ms == 0
    assert synced[0].end_ms == 875
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert not any(flag["kind"] == "text_changed" for flag in report["flags"])
    assert "verdammt" not in json.dumps(report, ensure_ascii=False)


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


def test_cli_sync_merges_bracketed_asr_insertion_into_source_cue(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "nicht von Bestien zerfleischt.\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "nicht", "start": 0.10, "end": 0.32, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "von", "start": 0.35, "end": 0.52, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "den", "start": 0.55, "end": 0.68, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "Bestien", "start": 0.71, "end": 1.08, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "zerfleischt", "start": 1.11, "end": 1.58, "confidence": 0.99, "speaker_id": "A"},
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
                            "final_text": "den",
                            "confidence": 0.98,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "audio confirms the inserted article",
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
    assert [cue.plain_text for cue in synced] == ["nicht von den Bestien zerfleischt."]
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert not any(flag["kind"] == "adlib_inserted" for flag in report["flags"])


def test_cli_sync_merges_continuation_insertion_into_following_cue(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "bin kein Bolton mehr.\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:02,500\n"
        "diese kranke alte Frau\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "bin", "start": 0.10, "end": 0.22, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "kein", "start": 0.24, "end": 0.36, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "Bolton", "start": 0.38, "end": 0.55, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "mehr", "start": 0.57, "end": 0.70, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "Ich", "start": 0.90, "end": 1.02, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "diese", "start": 1.42, "end": 1.57, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "kranke", "start": 1.59, "end": 1.77, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "alte", "start": 1.79, "end": 1.93, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "Frau", "start": 1.95, "end": 2.15, "confidence": 0.99, "speaker_id": "A"},
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
                            "final_text": "Ich,",
                            "confidence": 0.98,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "audio confirms a continuation into the next cue",
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
    assert [cue.plain_text for cue in synced] == [
        "bin kein Bolton mehr.",
        "Ich, diese kranke alte Frau",
    ]
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert not any(flag["kind"] == "adlib_inserted" for flag in report["flags"])


def test_cli_sync_reports_impossible_cps_without_extending_acoustic_timing(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    text = "this deliberately long subtitle must follow speech"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"{text}\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {
                        "text": word,
                        "start": index * 0.05,
                        "end": index * 0.05 + 0.04,
                        "confidence": 0.99,
                        "speaker_id": "A",
                    }
                    for index, word in enumerate(text.split())
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "timing": {"max_cps": 10},
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
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].end_ms == 500
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "impossible_cps_fast" for flag in report["flags"])


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


def test_cli_sync_holds_episode_length_asr_tail_when_source_is_incomplete(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    resumed_out_path = tmp_path / "episode.resumed.srt"
    verify_out_path = tmp_path / "episode.verify.srt"
    partial_verify_out_path = tmp_path / "episode.partial-verify.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    tail_text = (
        "Verraten von Feinden. Unser Ziel ist ein abgelegenes Versteck. "
        "Damian packte alle Vorräte, Waffen und das Bargeld auf die Packpferde hinter uns. "
        "Hätte ich gewusst, worauf diese verzweifelte Flucht zu Pferd hinausläuft, "
        "wäre ich lieber gestorben, als mitzukommen."
    )

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "in die Familie.\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    words = [
        {"text": "in", "start": 0.10, "end": 0.22, "confidence": 0.99, "speaker_id": "A"},
        {"text": "die", "start": 0.25, "end": 0.37, "confidence": 0.99, "speaker_id": "A"},
        {"text": "Familie.", "start": 0.40, "end": 0.70, "confidence": 0.99, "speaker_id": "A"},
    ]
    cursor = 1.05
    for token in tail_text.split():
        words.append(
            {
                "text": token,
                "start": round(cursor, 3),
                "end": round(cursor + 0.22, 3),
                "confidence": 0.98,
                "speaker_id": "A",
            }
        )
        cursor += 0.30
        if token.endswith((".", "?", "!")):
            cursor += 0.85
    wordstream_path.write_text(json.dumps({"words": words}), encoding="utf-8")
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
                            "final_text": tail_text,
                            "confidence": 0.99,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "the source subtitle ends while spoken narration continues",
                        }
                    },
                },
                "generation": {
                    "max_gap_seconds": 0.8,
                    "max_cue_duration_seconds": 5.0,
                    "max_generated_adlib_duration_seconds": 8.0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dubsync.pipeline.llm_adapter_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an incomplete-source tail must be held before provider work")
        ),
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
    assert [cue.plain_text for cue in synced] == ["in die Familie."]

    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    incomplete_source_flag = next(
        flag
        for flag in report["flags"]
        if flag["kind"] == "generated_adlib_rejected_incomplete_source"
    )
    assert incomplete_source_flag["severity"] == "error"
    assert incomplete_source_flag["new_text"] == tail_text
    assert not any(flag["kind"] == "adlib_inserted" for flag in report["flags"])
    adjudication = json.loads((workdir / "episode" / "adjudicate.json").read_text(encoding="utf-8"))
    assert adjudication["decisions"][0]["verdict"] == "keep_srt"
    assert adjudication["decisions"][0]["final_text"] == ""

    stale_decision = {
        **adjudication["decisions"][0],
        "verdict": "use_audio",
        "final_text": tail_text,
        "confidence": 0.99,
        "reason": "stale pre-guard adjudication generated the missing episode section",
    }
    (workdir / "episode" / "adjudicate.json").write_text(
        json.dumps({"decisions": [stale_decision], "flags": []}),
        encoding="utf-8",
    )
    stale_tail = Cue(
        index=2,
        start_ms=1_033,
        end_ms=round(cursor * 1000),
        lines=[tail_text],
    )
    (workdir / "episode" / "rebuild.json").write_text(
        json.dumps({"cues": [synced[0].model_dump(), stale_tail.model_dump()]}),
        encoding="utf-8",
    )

    verify = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(verify_out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
            "--fps",
            "30",
            "--resume",
            "verify",
            "--no-llm",
        ],
    )

    assert verify.exit_code != 0
    assert "resume from rebuild" in verify.output.lower()
    assert not verify_out_path.exists()

    (workdir / "episode" / "adjudicate.json").write_text(
        json.dumps({"decisions": adjudication["decisions"], "flags": adjudication["flags"]}),
        encoding="utf-8",
    )
    partial_stale_tail = stale_tail.model_copy(update={"lines": ["Verraten von Feinden."]})
    (workdir / "episode" / "rebuild.json").write_text(
        json.dumps({"cues": [synced[0].model_dump(), partial_stale_tail.model_dump()]}),
        encoding="utf-8",
    )
    partial_verify = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(partial_verify_out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
            "--fps",
            "30",
            "--resume",
            "verify",
            "--no-llm",
        ],
    )

    assert partial_verify.exit_code != 0
    assert "resume from rebuild" in partial_verify.output.lower()
    assert not partial_verify_out_path.exists()
    (workdir / "episode" / "adjudicate.json").write_text(
        json.dumps({"decisions": [stale_decision], "flags": []}),
        encoding="utf-8",
    )

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
            "--fps",
            "30",
            "--resume",
            "rebuild",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_out_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in resumed_cues] == ["in die Familie."]
    resumed_adjudication = json.loads(
        (workdir / "episode" / "adjudicate.json").read_text(encoding="utf-8")
    )
    assert resumed_adjudication["decisions"][0]["verdict"] == "keep_srt"
    assert any(
        flag["kind"] == "generated_adlib_rejected_incomplete_source"
        for flag in resumed_adjudication["flags"]
    )


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


def test_cli_sync_removes_generated_adlib_without_speech_activity(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"
    wordstream_path = tmp_path / "episode.wordstream.json"
    vad_path = tmp_path / "episode.vad.json"

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Du bildest dir nur etwas ein, Nova.\n"
        "\n"
        "2\n"
        "00:00:19,166 --> 00:00:20,200\n"
        "Informiere alle,\n"
        "\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Du", "start": 15.80, "end": 15.90, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "bildest", "start": 15.92, "end": 16.10, "confidence": 0.97, "speaker_id": "A"},
                    {"text": "dir", "start": 16.12, "end": 16.20, "confidence": 0.97, "speaker_id": "A"},
                    {"text": "nur", "start": 16.22, "end": 16.34, "confidence": 0.96, "speaker_id": "A"},
                    {"text": "ein", "start": 16.36, "end": 16.50, "confidence": 0.96, "speaker_id": "A"},
                    {"text": "Nova", "start": 16.52, "end": 16.70, "confidence": 0.96, "speaker_id": "A"},
                    {"text": "Du", "start": 17.43, "end": 17.80, "confidence": 0.91, "speaker_id": "A"},
                    {"text": "Informiere", "start": 19.46, "end": 19.80, "confidence": 0.98, "speaker_id": "A"},
                    {"text": "alle", "start": 19.82, "end": 20.10, "confidence": 0.97, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    vad_path.write_text(
        json.dumps(
            {
                "regions": [
                    {"start": 15.80, "end": 16.74, "confidence": 0.92},
                    {"start": 19.46, "end": 20.12, "confidence": 0.93},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "vad": {"fixture_path": str(vad_path), "min_coverage": 0.2},
                "llm": {
                    "provider": "fixture",
                    "responses": {
                        "case-1": {
                            "case_id": "case-1",
                            "verdict": "use_audio",
                            "final_text": "ein",
                            "confidence": 0.94,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "actor omitted one word",
                        },
                        "case-2": {
                            "case_id": "case-2",
                            "verdict": "use_audio",
                            "final_text": "Du",
                            "confidence": 0.91,
                            "speaker": "A",
                            "character": "unknown",
                            "reason": "ASR-only adlib lacks speech-region support",
                        },
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
    assert [cue.plain_text for cue in synced] == [
        "Du bildest dir nur ein, Nova.",
        "Informiere alle,",
    ]
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "adlib_removed_without_speech_activity" for flag in report["flags"])
    assert not any(flag["kind"] == "adlib_inserted" and flag["new_text"] == "Du" for flag in report["flags"])


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


def test_cli_sync_resume_align_repairs_legacy_asr_artifact(tmp_path, shifted_srt_text, shifted_wordstream):
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

    asr_path = workdir / "episode" / "asr.json"
    legacy_words = [
        {"text": "hello", "start": 1.00, "end": 1.20, "confidence": 0.98, "speaker_id": "A"},
        {"text": "general", "start": 2.00, "end": 2.33, "confidence": 0.98, "speaker_id": "A"},
        {"text": "there", "start": 1.23, "end": 1.23, "confidence": 0.97, "speaker_id": "A"},
        {"text": "kenobi", "start": 2.36, "end": 2.80, "confidence": 0.99, "speaker_id": "A"},
    ]
    asr_path.write_text(json.dumps({"words": legacy_words}), encoding="utf-8")

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
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    repair_flags = [flag for flag in report["flags"] if flag["kind"] == "word_stream_repaired"]
    assert repair_flags
    assert "ASR resume artifact" in repair_flags[0]["message"]
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
        assert "whisperx" in result.output.casefold()
    assert "Traceback" not in result.output


def test_cli_sync_local_mode_rejects_nested_gemini_transcribe_override(tmp_path, monkeypatch):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.gemini.synced.srt"
    workdir = tmp_path / "work"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHallo Welt\n\n", encoding="utf-8")
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {
                    "provider": "elevenlabs",
                    "model_id": "scribe_v2",
                    "local": {
                        "provider": "gemini_transcribe",
                        "model": "gemini-3.5-transcribe",
                        "api_key": "test-key",
                        "language_codes": ["de-DE"],
                        "word_timestamps": True,
                        "diarize": True,
                        "store": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("dubsync.pipeline.normalize_audio", lambda source, _dest, **_kwargs: source)

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
            "--local",
        ],
    )

    assert result.exit_code != 0
    assert "Gemini 3.5 Transcribe ASR is disabled" in result.output
    assert not out_path.exists()


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
