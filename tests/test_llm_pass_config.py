from __future__ import annotations

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
