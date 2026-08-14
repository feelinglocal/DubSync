from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dubsync.llm_providers import (
    GeminiLLMAdapter,
    OpenAILLMAdapter,
    llm_adapter_from_config,
    punctuation_adapter_from_config,
)
from dubsync.pipeline import _adjudication_confidence_gate, _adjudication_scene_gap_seconds, _punctuation_scene_gap_seconds


def test_live_llm_adapters_can_be_configured_per_pass():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "api_key": "gemini-key",
            "punctuation": {"model": "gemini-3.5-flash-lite"},
            "speaker_mapping": {
                "provider": "openai",
                "model": "gpt-5.5",
                "api_key": "openai-key",
            },
        }
    }

    adjudication = llm_adapter_from_config(config, pass_name="adjudication")
    punctuation = punctuation_adapter_from_config(config)
    speaker_mapping = llm_adapter_from_config(config, pass_name="speaker_mapping")

    assert isinstance(adjudication, GeminiLLMAdapter)
    assert adjudication.model == "gemini-3.5-flash"
    assert adjudication.api_key == "gemini-key"
    assert isinstance(punctuation, GeminiLLMAdapter)
    assert punctuation.model == "gemini-3.5-flash-lite"
    assert punctuation.api_key == "gemini-key"
    assert punctuation.thinking_level == "medium"
    assert isinstance(speaker_mapping, OpenAILLMAdapter)
    assert speaker_mapping.model == "gpt-5.5"
    assert speaker_mapping.api_key == "openai-key"
    assert speaker_mapping.reasoning_effort == "medium"


def test_production_config_routes_audio_adjudication_and_punctuation_to_gemini_37(monkeypatch):
    config = yaml.safe_load(Path("provider.yaml").read_text(encoding="utf-8"))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    adjudication = llm_adapter_from_config(config, pass_name="adjudication")
    punctuation = punctuation_adapter_from_config(config)
    speaker_mapping = llm_adapter_from_config(config, pass_name="speaker_mapping")

    assert isinstance(adjudication, GeminiLLMAdapter)
    assert adjudication.model == "gemini-3.7-flash"
    assert adjudication.thinking_level == "high"
    assert config["llm"]["adjudication"]["audio_snippet_double_check"]["enabled"] is True
    assert isinstance(punctuation, GeminiLLMAdapter)
    assert punctuation.model == "gemini-3.7-flash"
    assert punctuation.thinking_level == "medium"
    assert isinstance(speaker_mapping, OpenAILLMAdapter)
    assert speaker_mapping.model == "gpt-5.6-luna"
    assert speaker_mapping.reasoning_effort == "medium"


def test_gemini_thinking_level_can_be_configured_per_pass():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "api_key": "gemini-key",
            "adjudication": {"thinking_level": "medium"},
            "punctuation": {"thinking_level": "low"},
            "speaker_mapping": {"thinking_level": "minimal"},
        }
    }

    adjudication = llm_adapter_from_config(config, pass_name="adjudication")
    punctuation = punctuation_adapter_from_config(config)
    speaker_mapping = llm_adapter_from_config(config, pass_name="speaker_mapping")

    assert isinstance(adjudication, GeminiLLMAdapter)
    assert adjudication.thinking_level == "medium"
    assert isinstance(punctuation, GeminiLLMAdapter)
    assert punctuation.thinking_level == "low"
    assert isinstance(speaker_mapping, GeminiLLMAdapter)
    assert speaker_mapping.thinking_level == "minimal"


def test_gemini_37_flash_rejects_minimal_thinking_level():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.7-flash",
            "api_key": "gemini-key",
            "punctuation": {"thinking_level": "minimal"},
        }
    }

    with pytest.raises(RuntimeError, match="gemini-3.7-flash thinking_level"):
        punctuation_adapter_from_config(config)


def test_gemini_cached_content_can_be_configured_per_pass():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "api_key": "gemini-key",
            "cached_content": "cachedContents/base",
            "punctuation": {"cached_content": "cachedContents/punctuation"},
        }
    }

    adjudication = llm_adapter_from_config(config, pass_name="adjudication")
    punctuation = punctuation_adapter_from_config(config)

    assert isinstance(adjudication, GeminiLLMAdapter)
    assert adjudication.cached_content == "cachedContents/base"
    assert isinstance(punctuation, GeminiLLMAdapter)
    assert punctuation.cached_content == "cachedContents/punctuation"


def test_openai_reasoning_effort_can_be_configured_per_pass():
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "api_key": "openai-key",
            "adjudication": {"reasoning_effort": "high"},
            "punctuation": {"reasoning_effort": "medium"},
            "speaker_mapping": {"reasoning_effort": "medium"},
        }
    }

    adjudication = llm_adapter_from_config(config, pass_name="adjudication")
    punctuation = punctuation_adapter_from_config(config)
    speaker_mapping = llm_adapter_from_config(config, pass_name="speaker_mapping")

    assert isinstance(adjudication, OpenAILLMAdapter)
    assert adjudication.model == "gpt-5.6-luna"
    assert adjudication.reasoning_effort == "high"
    assert isinstance(punctuation, OpenAILLMAdapter)
    assert punctuation.reasoning_effort == "medium"
    assert isinstance(speaker_mapping, OpenAILLMAdapter)
    assert speaker_mapping.reasoning_effort == "medium"


def test_openai_adapter_defaults_to_luna_medium_reasoning():
    adapter = llm_adapter_from_config({"llm": {"provider": "openai", "api_key": "openai-key"}})

    assert isinstance(adapter, OpenAILLMAdapter)
    assert adapter.model == "gpt-5.6-luna"
    assert adapter.reasoning_effort == "medium"


@pytest.mark.parametrize("reasoning_effort", ["minimal", "turbo", 3])
def test_openai_adapter_rejects_unsupported_reasoning_effort(reasoning_effort):
    config = {
        "llm": {
            "provider": "openai",
            "api_key": "openai-key",
            "reasoning_effort": reasoning_effort,
        }
    }

    with pytest.raises(RuntimeError, match="reasoning_effort"):
        llm_adapter_from_config(config)


def test_openai_adapter_rejects_audio_snippet_configuration_it_cannot_send():
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "api_key": "openai-key",
            "adjudication": {"audio_snippet_double_check": {"enabled": True}},
        }
    }

    with pytest.raises(RuntimeError, match="does not support audio snippets"):
        llm_adapter_from_config(config, pass_name="adjudication")


def test_adjudication_scene_gap_uses_llm_pass_override():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "adjudication": {"scene_gap_seconds": 2.5},
        }
    }

    assert _adjudication_scene_gap_seconds(config) == 2.5


def test_adjudication_confidence_gate_uses_llm_pass_override():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "adjudication": {"confidence_gate": 0.82},
        }
    }

    assert _adjudication_confidence_gate(config) == 0.82


def test_punctuation_scene_gap_uses_llm_pass_override():
    config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "punctuation": {"model": "gemini-3.5-flash-lite", "scene_gap_seconds": 3.0},
        }
    }

    assert _punctuation_scene_gap_seconds(config) == 3.0


def test_fixture_punctuation_scene_gap_can_be_configured_without_live_model_override():
    config = {
        "llm": {
            "provider": "fixture",
            "responses": {},
            "punctuation": {"scene_gap_seconds": 1.25},
        }
    }

    assert _punctuation_scene_gap_seconds(config) == 1.25


def test_fixture_punctuation_settings_without_responses_do_not_create_static_adapter():
    config = {
        "llm": {
            "provider": "fixture",
            "responses": {},
            "punctuation": {"scene_gap_seconds": 1.25},
        }
    }

    assert punctuation_adapter_from_config(config) is None
