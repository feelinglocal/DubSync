from __future__ import annotations

from pathlib import Path

import yaml


def test_readme_includes_final_acceptance_report_sections():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "## Readiness Report",
        "### Measured Timings And Costs",
        "### Top 3 Risks",
        "Live API smoke tests were not run",
        "150 passed, 5 deselected",
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
    assert config["llm"]["punctuation"]["thinking_level"] == "low"
    assert "cached_content: cachedContents/your-episode-context-cache" in example_text


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
        "punctuation defaults to `low`",
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
