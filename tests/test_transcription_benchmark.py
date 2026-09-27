from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from dubsync.models import Word


SPEC = importlib.util.spec_from_file_location(
    "transcription_benchmark", Path(__file__).parents[1] / "scripts" / "benchmark_transcription_models.py"
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_normalization_removes_markup_annotations_but_preserves_unicode_and_numbers():
    text = "<i>ANNA: Grüß dich!</i> [Tür klingelt]\nBOB: It's 42 — Café."
    assert benchmark.normalize_text(text) == "grüss dich its 42 café"


def test_wer_counts_substitutions_deletions_and_insertions():
    metric = benchmark.error_metrics("one two three four", "one too four five")
    assert metric["word_errors"] == 3
    assert metric["wer"] == 0.75
    assert sum(metric[name] for name in ("substitutions", "deletions", "insertions")) == 3
    assert metric["reference_words"] == 4
    assert benchmark.error_metrics("a b", "a b c d e")["wer"] == 1.5


def test_empty_reference_is_undefined_and_cer_excludes_whitespace():
    assert benchmark.error_metrics("[Music]", "hello")["wer"] is None
    assert benchmark.error_metrics("[Music]", "hello")["cer"] is None
    assert benchmark.error_metrics("a bc", "ab c")["cer"] == 0


def test_timestamp_stats_detect_bad_and_missing_information():
    words = [
        Word(text="good", start=0.1, end=0.3, speaker_id="A"),
        Word(text="backwards", start=0.0, end=0.2),
        Word(text="zero", start=0.4, end=0.4, speaker_id="A"),
        Word(text="outside", start=1.0, end=3.1, speaker_id="B"),
    ]
    result = benchmark.word_statistics(words, 3)
    assert result["word_count"] == 4
    assert result["invalid_timestamps"] == 2
    assert result["nonmonotonic_starts"] == 1
    assert result["valid_timestamp_ratio"] == 0.5
    assert result["distinct_speakers"] == 2
    assert result["speaker_label_coverage"] == 0.75


def test_manifest_paths_resolve_relative_to_manifest_and_reject_duplicate_names(tmp_path):
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"clips": [
        {"id": "sample", "audio": "audio.wav", "reference_srt": "reference.srt"}
    ]}))
    clips = benchmark.load_manifest(manifest)
    assert clips[0].audio == tmp_path / "audio.wav"
    assert clips[0].reference == tmp_path / "reference.srt"
    manifest.write_text(json.dumps({"clips": [
        {"id": "same", "audio": "a", "reference_srt": "b"},
        {"id": "same", "audio": "c", "reference_srt": "d"},
    ]}))
    with pytest.raises(ValueError, match="unique"):
        benchmark.load_manifest(manifest)


def test_reference_selection_flags_partial_boundary_cues(tmp_path):
    reference = tmp_path / "ref.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello there\n\n"
                         "2\n00:00:03,000 --> 00:00:04,000\nnext line\n", encoding="utf-8")
    text, meta = benchmark.read_reference(reference, 1, 3)
    assert text == "hello there"
    assert meta["partial_boundary_cues"] == 1
    assert meta["human_verified_verbatim"] is False


def test_paired_run_alternates_order_records_cost_and_does_not_reuse_adapters(tmp_path):
    sequence = []
    created = []

    class Adapter:
        def __init__(self, name):
            self.name = name
            self.last_usage = {"cost": 0.001} if name == "mai" else None
            created.append(self)

        def transcribe(self, path):
            sequence.append((self.name, path))
            return [Word(text="hello", start=0, end=0.5)]

    audio = tmp_path / "audio.wav"
    prepared = [{"id": "one", "normalized_audio": str(audio), "duration_seconds": 60,
                 "reference_text": "hello", "audio_sha256": "abc", "reference": {}}]
    result = benchmark.run_benchmark(
        prepared, tmp_path, 3, {name: lambda name=name: Adapter(name) for name in ("mai", "scribe")}
    )
    assert [name for name, path in sequence] == ["mai", "scribe", "scribe", "mai", "mai", "scribe"]
    assert all(path == audio for name, path in sequence)
    assert len(created) == 6
    assert result["models"]["mai"]["cost_usd"] == pytest.approx(0.003)
    assert result["models"]["scribe"]["cost_usd"] == pytest.approx(0.22 / 20)
    assert result["models"]["scribe"]["cost_basis"] == "estimate_at_configured_hourly_rate"
    assert result["models"]["mai"]["pooled_wer"] == 0
    assert len(list((tmp_path / "runs").glob("*.json"))) == 6


def test_failed_run_preserves_result_and_redacts_credentials(tmp_path, monkeypatch):
    key = "sk-or-v1-" + "a" * 64
    monkeypatch.setenv("OPENROUTER_API_KEY", key)

    class Broken:
        last_usage = None

        def transcribe(self, path):
            raise RuntimeError(f"request failed: {key}")

    prepared = [{"id": "one", "normalized_audio": "a.wav", "duration_seconds": 1,
                 "reference_text": "hello", "audio_sha256": "abc", "reference": {}}]
    result = benchmark.run_benchmark(prepared, tmp_path, 1, {"mai": Broken})
    assert result["models"]["mai"]["failed_runs"] == 1
    assert result["runs"][0]["status"] == "error"
    assert key not in json.dumps(result)
    assert key not in (tmp_path / "runs" / "one-r1-mai.json").read_text()


def test_resume_reuses_success_and_failure_evidence_without_paid_calls(tmp_path):
    calls = []

    class Adapter:
        last_usage = {"cost": 0.002}

        def transcribe(self, path):
            calls.append(path)
            return [Word(text="hello", start=0, end=0.2)]

    prepared = [{"id": "one", "normalized_audio": "a.wav", "duration_seconds": 1,
                 "reference_text": "hello", "audio_sha256": "abc", "reference": {}}]
    first = benchmark.run_benchmark(prepared, tmp_path, 2, {"mai": Adapter})
    second = benchmark.run_benchmark(prepared, tmp_path, 2, {"mai": Adapter})
    assert len(calls) == 2
    assert first["runs"] == second["runs"]
    benchmark.run_benchmark(prepared, tmp_path, 2, {"mai": Adapter}, force=True)
    assert len(calls) == 4


def test_resume_rejects_mismatched_audio_before_calling_provider(tmp_path):
    class Adapter:
        last_usage = {"cost": 0.002}

        def transcribe(self, path):
            return [Word(text="hello", start=0, end=0.2)]

    prepared = [{"id": "one", "normalized_audio": "a.wav", "duration_seconds": 1,
                 "reference_text": "hello", "audio_sha256": "abc", "reference": {}}]
    benchmark.run_benchmark(prepared, tmp_path, 1, {"mai": Adapter})
    prepared[0]["audio_sha256"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        benchmark.run_benchmark(prepared, tmp_path, 1, {"mai": Adapter})


def test_single_model_continuation_keeps_other_model_in_combined_report(tmp_path):
    class Adapter:
        last_usage = {"cost": 0.002}

        def transcribe(self, path):
            return [Word(text="hello", start=0, end=0.2)]

    prepared = [{"id": "one", "normalized_audio": "a.wav", "duration_seconds": 1,
                 "reference_text": "hello", "audio_sha256": "abc", "reference": {}}]
    benchmark.run_benchmark(prepared, tmp_path, 1, {"mai": Adapter})
    result = benchmark.run_benchmark(prepared, tmp_path, 1, {"scribe": Adapter})
    assert set(result["models"]) == {"mai", "scribe"}
    assert len(result["runs"]) == 2
