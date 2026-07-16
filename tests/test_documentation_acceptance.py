from __future__ import annotations

from pathlib import Path
import tomllib

import yaml


def test_readme_includes_final_acceptance_report_sections():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "## Readiness Report",
        "### Measured Timings And Costs",
        "### Top 3 Risks",
        "Production web `generate` smoke",
        "309 passed, 5 deselected",
        "A second paid run was not made",
        "Production dependency audit",
    ):
        assert expected in readme


def test_render_blueprint_disk_omits_unsupported_shutdown_delay():
    blueprint = yaml.safe_load(Path("render.yaml").read_text(encoding="utf-8"))
    service = blueprint["services"][0]

    assert service["disk"]["mountPath"] == "/var/data"
    assert "maxShutdownDelaySeconds" not in service


def test_production_deployment_requires_job_gate_and_patched_installer():
    blueprint = yaml.safe_load(Path("render.yaml").read_text(encoding="utf-8"))
    env = {item["key"]: item for item in blueprint["services"][0]["envVars"]}
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert env["DUBSYNC_REQUIRE_JOB_ACCESS_CODE"]["value"] == "1"
    assert env["DUBSYNC_JOB_ACCESS_CODE"]["sync"] is False
    assert 'python -m pip install --upgrade "pip>=26.1.2"' in dockerfile
    assert dockerfile.index('python -m pip install --upgrade "pip>=26.1.2"') < dockerfile.index('python -m pip install ".[cloud,web]"')


def test_runtime_blueprint_sets_bounded_job_and_ffmpeg_limits():
    blueprint = yaml.safe_load(Path("render.yaml").read_text(encoding="utf-8"))
    env = {item["key"]: item for item in blueprint["services"][0]["envVars"]}

    assert env["DUBSYNC_MAX_SUBMISSIONS_PER_HOUR"]["value"] == "5"
    assert env["DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS"]["value"] == "10"
    assert env["DUBSYNC_MAX_UPLOAD_BYTES"]["value"] == "536870912"
    assert env["DUBSYNC_MAX_BATCH_UPLOAD_BYTES"]["value"] == "536870912"
    assert env["DUBSYNC_MAX_SRT_BYTES"]["value"] == "2097152"
    assert env["DUBSYNC_MAX_SRT_LINES"]["value"] == "60000"
    assert env["DUBSYNC_MAX_SRT_CUES"]["value"] == "20000"
    assert env["DUBSYNC_MAX_SRT_LINE_BYTES"]["value"] == "16384"
    assert env["DUBSYNC_MAX_SRT_LINE_CHARS"]["value"] == "4096"
    assert env["DUBSYNC_MAX_RETAINED_STORAGE_BYTES"]["value"] == "4294967296"
    assert env["DUBSYNC_MAX_AUDIO_DURATION_SECONDS"]["value"] == "14400"
    assert env["DUBSYNC_FFPROBE_TIMEOUT_SECONDS"]["value"] == "15"
    assert env["DUBSYNC_MAX_NORMALIZED_AUDIO_BYTES"]["value"] == "536870912"
    assert env["DUBSYNC_MAX_JOB_WORK_BYTES"]["value"] == "67108864"
    assert env["DUBSYNC_MAX_JOB_STORAGE_BYTES"]["value"] == "1073741824"
    assert env["DUBSYNC_MIN_FREE_STORAGE_BYTES"]["value"] == "2147483648"
    assert env["DUBSYNC_MAX_AUDIO_SNIPPET_BYTES"]["value"] == "33554432"
    assert env["DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS"]["value"] == "24"
    assert env["DUBSYNC_FFMPEG_TIMEOUT_SECONDS"]["value"] == "1800"
    assert "DUBSYNC_MAX_JOBS_PER_HOUR" not in env


def test_readme_documents_production_disk_admission_limits():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "Single and batch request payloads are capped at 512 MiB",
        "retained job commitments are capped at 4 GiB",
        "only one request may pass upload intake at a time",
        "four-hour audio limit",
        "1 GiB per-job ceiling",
        "2 GiB of minimum free disk",
        "SRT files are capped at 2 MiB, 60,000 lines, and 20,000 cues",
    ):
        assert expected in readme


def test_readme_describes_forced_alignment_as_implemented_optional_path():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "optional forced alignment" in readme
    assert "future forced alignment" not in readme


def test_provider_example_includes_documented_adjudication_confidence_gate():
    config = yaml.safe_load(Path("providers.example.yaml").read_text(encoding="utf-8"))
    example_text = Path("providers.example.yaml").read_text(encoding="utf-8")

    assert config["llm"]["adjudication"]["confidence_gate"] == 0.7
    assert config["llm"]["adjudication"]["audio_snippet_double_check"]["enabled"] is False
    assert config["llm"]["punctuation"]["thinking_level"] == "medium"
    assert "cached_content: cachedContents/your-episode-context-cache" in example_text


def test_production_punctuation_uses_flash_lite_with_medium_thinking():
    config = yaml.safe_load(Path("provider.yaml").read_text(encoding="utf-8"))

    assert config["llm"]["punctuation"]["model"] == "gemini-3.1-flash-lite"
    assert config["llm"]["punctuation"]["thinking_level"] == "medium"


def test_readme_names_remaining_unimplemented_plan_provider_controls():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "Audio-snippet double-checks are implemented for Gemini inline audio",
        "Automatic Gemini context-cache creation/deletion remains unimplemented",
    ):
        assert expected in readme


def test_readme_documents_gemini_thinking_level_controls():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "Gemini thinking-level controls",
        "thinking_config.thinking_level",
        "punctuation defaults to `medium`",
    ):
        assert expected in readme


def test_readme_documents_gemini_context_cache_reuse():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "Gemini explicit context-cache reuse",
        "`cached_content`",
        "does not create or delete remote caches automatically",
    ):
        assert expected in readme


def test_readme_documents_adjudication_audio_snippet_double_check():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "audio_snippet_double_check",
        "types.Part.from_bytes",
        "audio_snippets.json",
    ):
        assert expected in readme


def test_readme_mentions_improv_precision_recall_metrics():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "improv precision/recall",
        "at least 0.9 precision",
        "0.85 recall",
    ):
        assert expected in readme


def test_cloud_dependencies_require_medium_thinking_compatible_google_genai():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "google-genai>=1.56,<3" in pyproject["project"]["optional-dependencies"]["cloud"]
