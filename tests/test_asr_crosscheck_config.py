from __future__ import annotations

import copy
import json

import pytest

from dubsync.asr_crosscheck_config import cross_check_context, resolve_cross_check_config
from dubsync.models import Word

MAI = "microsoft/mai-transcribe-2"


@pytest.mark.parametrize("section", [None, False])
def test_missing_or_false_cross_check_is_off(section):
    config = {"asr": {"provider": "openrouter"}}
    if section is not None:
        config["asr"]["cross_check"] = section
    assert resolve_cross_check_config(config) is None


def test_web_false_overrides_yaml_without_validation_or_spending():
    assert resolve_cross_check_config({"asr": {"cross_check": "broken"}}, enabled=False) is None


@pytest.mark.parametrize("primary,expected", [
    ({"provider": "openrouter", "model": MAI}, {"provider": "elevenlabs", "model_id": "scribe_v2"}),
    ({"provider": "elevenlabs", "model_id": "scribe_v2"}, {"provider": "openrouter", "model": MAI}),
])
def test_web_true_selects_the_other_provider(primary, expected):
    assert resolve_cross_check_config({"asr": primary}, enabled=True) == {"asr": expected}


def test_primary_credentials_fixture_prices_and_keyterms_never_leak_to_secondary():
    primary = {"provider": "openrouter", "model": MAI, "api_key": "primary-secret", "fixture_path": "primary.json",
               "dollars_per_hour": 999, "keyterms": ["private name"], "language_code": "por"}
    before = copy.deepcopy(primary)
    actual = resolve_cross_check_config({"asr": primary}, enabled=True)
    assert actual == {"asr": {"provider": "elevenlabs", "model_id": "scribe_v2", "language_code": "pt"}}
    assert primary == before


def test_configured_secondary_retains_its_own_key_and_fixture_only():
    config = {"asr": {"provider": "openrouter", "api_key": "primary", "cross_check": {
        "provider": "elevenlabs", "model_id": "scribe_v2", "api_key": "secondary", "fixture_path": "secondary.json",
    }}}
    before = copy.deepcopy(config)
    actual = resolve_cross_check_config(config)
    assert actual["asr"]["api_key"] == "secondary"
    assert actual["asr"]["fixture_path"] == "secondary.json"
    assert config == before


@pytest.mark.parametrize("bad", [True, 1, "scribe_v2", [], {"provider": "openai"}, {"provider": "elevenlabs", "model_id": "scribe_v1"}, {"provider": "elevenlabs", "typo": True}, {"provider": "elevenlabs", "cross_check": {}}])
def test_malformed_or_unsupported_secondary_fails_before_transcription(bad):
    with pytest.raises(ValueError):
        resolve_cross_check_config({"asr": {"provider": "openrouter", "cross_check": bad}})


def test_explicit_yaml_cannot_cross_check_a_provider_against_itself():
    with pytest.raises(ValueError, match="different"):
        resolve_cross_check_config({"asr": {"provider": "openrouter", "cross_check": {"provider": "openrouter"}}})


def test_web_model_switch_selects_opposite_without_reusing_wrong_nested_credentials():
    config = {"asr": {"provider": "elevenlabs", "cross_check": {"provider": "elevenlabs", "api_key": "scribe-secret"}}}
    assert resolve_cross_check_config(config, enabled=True) == {"asr": {"provider": "openrouter", "model": MAI}}


def test_original_yaml_section_can_be_supplied_after_primary_selection_filtered_it():
    result = resolve_cross_check_config({"asr": {"provider": "openrouter"}}, configured={"provider": "elevenlabs", "api_key": "secondary"})
    assert result["asr"]["api_key"] == "secondary"


def test_shared_language_overrides_secondary_hint_and_auto_removes_it():
    config = {"asr": {"provider": "openrouter", "cross_check": {"provider": "elevenlabs", "language_code": "en"}}}
    assert resolve_cross_check_config(config, language="ja-JP")["asr"]["language_code"] == "ja"
    assert "language_code" not in resolve_cross_check_config(config, language="auto")["asr"]


def test_context_is_sensitive_to_wording_timing_and_model_without_containing_secrets():
    config = {"asr": {"provider": "elevenlabs", "model_id": "scribe_v2", "api_key": "secondary-secret"}}
    words = [Word(text="Hallo", start=1, end=2)]
    base = cross_check_context(words, config)
    assert "secondary-secret" not in json.dumps(base)
    assert "api_key" not in json.dumps(base)
    for changed in [Word(text="Welt", start=1, end=2), Word(text="Hallo", start=1.1, end=2)]:
        assert cross_check_context([changed], config)["words_sha256"] != base["words_sha256"]
    assert base != cross_check_context(words, {"asr": {"provider": "openrouter", "model": MAI}})


@pytest.mark.parametrize("field,value", [
    ("model_id", []), ("model_id", {}), ("timeout_seconds", "bad"), ("timeout_seconds", 0),
    ("chunk_seconds", float("nan")), ("chunk_seconds", -1), ("dollars_per_hour", -1),
    ("dollars_per_hour", float("inf")), ("keyterms", "name"), ("character_names", [9]),
])
def test_invalid_secondary_settings_raise_value_error_during_preflight(field, value):
    config = {"asr": {"provider": "openrouter", "cross_check": {"provider": "elevenlabs", field: value}}}
    with pytest.raises(ValueError, match="ASR cross-check|asr.cross_check"):
        resolve_cross_check_config(config)
