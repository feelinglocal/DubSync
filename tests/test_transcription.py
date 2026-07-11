from __future__ import annotations

import json

from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import Word
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import GenerationConstraints, StyleProfile
from dubsync.transcription import build_cues_from_words, generate_srt_from_audio


def test_build_cues_from_words_splits_on_gap_and_known_speaker_change():
    words = [
        Word(text="Hello", start=0.10, end=0.30, confidence=0.99, speaker_id="A"),
        Word(text="there.", start=0.32, end=0.70, confidence=0.98, speaker_id="A"),
        Word(text="Next", start=1.80, end=2.00, confidence=0.97, speaker_id="A"),
        Word(text="line.", start=2.05, end=2.40, confidence=0.96, speaker_id="A"),
        Word(text="New", start=2.45, end=2.70, confidence=0.95, speaker_id="B"),
        Word(text="voice.", start=2.75, end=3.10, confidence=0.94, speaker_id="B"),
    ]

    cues = build_cues_from_words(words, StyleProfile(fps=30.0), max_gap_seconds=0.8)

    assert [cue.plain_text for cue in cues] == ["Hello there.", "Next line.", "New voice."]
    assert [cue.speaker_id for cue in cues] == ["A", "A", "B"]
    assert all(cue.start_ms >= 0 and cue.end_ms > cue.start_ms for cue in cues)
    assert all(line == line.strip() for cue in cues for line in cue.lines)
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert all(len(line) <= 26 for cue in cues for line in cue.lines)


def test_build_cues_from_words_treats_unicode_ellipsis_as_sentence_end():
    words = [
        Word(text="Wait\u2026", start=0.10, end=0.70, confidence=0.99, speaker_id="A"),
        Word(text="Continue", start=0.72, end=0.95, confidence=0.99, speaker_id="A"),
        Word(text="now.", start=0.97, end=1.25, confidence=0.99, speaker_id="A"),
    ]

    cues = build_cues_from_words(words, StyleProfile(fps=30.0))

    assert [cue.plain_text for cue in cues] == ["Wait\u2026", "Continue now."]


def test_generate_srt_from_audio_with_fixture_provider_writes_downloadable_artifacts(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Ready", "start": 0.1, "end": 0.35, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "now.", "start": 0.4, "end": 0.8, "confidence": 0.98, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    output_path = tmp_path / "dialogue.generated.srt"

    result = generate_srt_from_audio(
        audio_path=audio_path,
        output_path=output_path,
        workdir=tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
    )

    cues = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["Ready now."]
    assert result.output_srt == output_path
    assert result.report["summary"] == {"cue_count": 1, "flags": 0, "style_violations": 0}
    assert (result.episode_workdir / "asr.json").exists()
    assert (result.episode_workdir / "generate.json").exists()
    assert (result.episode_workdir / "qc_report.json").exists()
    assert (result.episode_workdir / "qc_report.html").exists()
    assert (result.episode_workdir / "cost.json").exists()


def test_generate_srt_reflows_punctuation_to_the_style_width(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": text, "start": index * 0.3, "end": index * 0.3 + 0.2, "confidence": 0.99}
                    for index, text in enumerate(["Hello", "this", "is", "the", "Dubsync", "Cloud", "test"])
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(
        "\n".join(
            [
                "asr:",
                f"  fixture_path: '{words_path.as_posix()}'",
                "llm:",
                "  provider: fixture",
                "  punctuation:",
                "    1: 'Hello, this is the Dubsync Cloud test.'",
            ]
        ),
        encoding="utf-8",
    )
    style_path = tmp_path / "style.yaml"
    style_path.write_text("max_chars_per_line: 26\nmax_lines_per_cue: 2\n", encoding="utf-8")
    output_path = tmp_path / "dialogue.generated.srt"

    result = generate_srt_from_audio(
        audio_path=audio_path,
        output_path=output_path,
        workdir=tmp_path / "work",
        providers_path=providers_path,
        style_path=style_path,
    )

    cues = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert cues[0].lines == ["Hello, this is the Dubsync", "Cloud test."]
    assert result.report["summary"]["style_violations"] == 0


def test_generate_srt_applies_per_job_profile_and_reading_speed_constraints(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Every", "start": 0.0, "end": 0.12, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "word", "start": 0.13, "end": 0.24, "confidence": 0.99, "speaker_id": "A"},
                    {"text": "fits.", "start": 0.25, "end": 0.4, "confidence": 0.99, "speaker_id": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    output_path = tmp_path / "dialogue.generated.srt"
    profile = StyleProfile(max_lines_per_cue=1, max_chars_per_line=18, min_cue_dur=0.4, tail_ms=0)
    constraints = GenerationConstraints(
        max_gap_seconds=0.5,
        max_cue_duration_seconds=3.0,
        min_cps=2.0,
        max_cps=10.0,
    )

    result = generate_srt_from_audio(
        audio_path=audio_path,
        output_path=output_path,
        workdir=tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
        style_profile=profile,
        generation_constraints=constraints,
    )

    cues = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 1 for cue in cues)
    assert all(len(line) <= 18 for cue in cues for line in cue.lines)
    assert cues[-1].end_ms >= 1600
    assert result.report["summary"]["style_violations"] == 0
    generated = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    assert generated["profile"]["max_chars_per_line"] == 18
    assert generated["constraints"]["max_cps"] == 10.0


def test_generate_srt_cps_extension_respects_the_maximum_cue_duration(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Deliberately", "start": 0.0, "end": 0.2, "confidence": 0.99},
                    {"text": "lengthy.", "start": 0.21, "end": 0.4, "confidence": 0.99},
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    output_path = tmp_path / "dialogue.generated.srt"

    result = generate_srt_from_audio(
        audio_path=audio_path,
        output_path=output_path,
        workdir=tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
        style_profile=StyleProfile(max_lines_per_cue=1, max_chars_per_line=40, min_cue_dur=0.2, tail_ms=0),
        generation_constraints=GenerationConstraints(
            max_gap_seconds=0.5,
            max_cue_duration_seconds=1.0,
            min_cps=0.0,
            max_cps=5.0,
        ),
    )

    cues = parse_srt_text(output_path.read_text(encoding="utf-8"))
    assert cues[0].duration_ms <= 1000
    assert any(flag["kind"] == "impossible_cps_fast" for flag in result.report["flags"])


def test_cli_generate_exposes_audio_only_workflow(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "words.json"
    words_path.write_text(
        json.dumps({"words": [{"text": "Hello.", "start": 0.1, "end": 0.7, "confidence": 0.99}]}),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(f"asr:\n  fixture_path: '{words_path.as_posix()}'\n", encoding="utf-8")
    output_path = tmp_path / "generated.srt"

    result = CliRunner().invoke(
        app,
        [
            "generate",
            str(audio_path),
            "-o",
            str(output_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(tmp_path / "work"),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "Wrote" in result.output
