"""Resolve an opt-in secondary ASR independently of primary provider settings."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Sequence

from .asr_crosscheck import CROSS_CHECK_POLICY_VERSION
from .cache import _cache_safe_params
from .models import Word
from .providers import MAI_TRANSCRIBE_MODEL, SCRIBE_TRANSCRIBE_MODEL, asr_language_code

_UNSET = object()
_ALLOWED_FIELDS = frozenset({
    "provider", "model", "model_id", "api_key", "fixture_path", "language", "language_code",
    "diarize", "keyterms", "character_names", "timeout_seconds", "chunk_seconds", "dollars_per_hour",
})


def _model(config: dict[str, object], *, default: str) -> str:
    provider = config.get("provider")
    if provider is not None and (not isinstance(provider, str) or not provider.strip()):
        raise ValueError("ASR cross-check provider must be a nonempty string")
    provider = str(provider or "").strip().lower()
    model = config.get("model", config.get("model_id"))
    if any(not isinstance(config[key], str) or not config[key].strip()
           for key in ("model", "model_id") if key in config):
        raise ValueError("ASR cross-check model must be a nonempty string")
    if provider in {"openrouter", MAI_TRANSCRIBE_MODEL}:
        expected = MAI_TRANSCRIBE_MODEL
    elif provider in {"elevenlabs", SCRIBE_TRANSCRIBE_MODEL}:
        expected = SCRIBE_TRANSCRIBE_MODEL
    elif provider in {"", "fixture"}:
        expected = str(model or default)
    else:
        raise ValueError("ASR cross-check supports only MAI-Transcribe 2 and Scribe v2")
    if expected not in {MAI_TRANSCRIBE_MODEL, SCRIBE_TRANSCRIBE_MODEL} or model not in {None, expected}:
        raise ValueError("ASR cross-check provider and model must identify MAI-Transcribe 2 or Scribe v2")
    # Conflicting model/model_id fields must not be silently ignored.
    if any(config[key] != expected for key in ("model", "model_id") if key in config):
        raise ValueError("ASR cross-check model settings disagree")
    return expected


def resolve_cross_check_config(
    config: dict[str, object], *, enabled: bool | None = None,
    configured: object = _UNSET, language: str | None = None,
) -> dict[str, object] | None:
    """Return a separate ``{'asr': ...}`` config, or None without any side effects.

    ``enabled=None`` follows the optional YAML mapping; absence/false is off.
    Explicit false (the web default) always disables it. Explicit true selects
    the opposite of the chosen primary and reuses configured secondary settings
    only if they belong to that opposite provider. This keeps a web model switch
    from transferring credentials, fixtures, or pricing to the wrong provider.

    ``configured`` can carry the original YAML cross_check value after primary
    provider selection has filtered provider-specific settings from the config.
    """
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("ASR cross-check override must be a boolean or None")
    if enabled is False:
        return None
    primary = config.get("asr", {})
    if not isinstance(primary, dict):
        raise ValueError("ASR configuration must be a mapping")
    raw = primary.get("cross_check") if configured is _UNSET else configured
    if enabled is None and (raw is None or raw is False):
        return None
    if raw is not None and raw is not False and not isinstance(raw, dict):
        raise ValueError("asr.cross_check must be a provider mapping or false")
    if isinstance(raw, dict) and set(raw) - _ALLOWED_FIELDS:
        raise ValueError("asr.cross_check contains unsupported settings")
    primary_model = _model(primary, default=MAI_TRANSCRIBE_MODEL)
    opposite = SCRIBE_TRANSCRIBE_MODEL if primary_model == MAI_TRANSCRIBE_MODEL else MAI_TRANSCRIBE_MODEL
    secondary = deepcopy(raw) if isinstance(raw, dict) else {}
    secondary_model = _model(secondary, default=opposite)
    if secondary_model == primary_model:
        if enabled is not True:
            raise ValueError("ASR cross-check requires a different model from the primary")
        secondary, secondary_model = {}, opposite
    for field in ("api_key", "fixture_path"):
        if field in secondary and (not isinstance(secondary[field], str) or not secondary[field].strip()):
            raise ValueError(f"asr.cross_check {field} must be a nonempty string")
    if "diarize" in secondary and not isinstance(secondary["diarize"], bool):
        raise ValueError("asr.cross_check diarize must be a boolean")
    for field in ("timeout_seconds", "chunk_seconds", "dollars_per_hour"):
        if field not in secondary:
            continue
        value = secondary[field]
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                or value < 0 or (field != "dollars_per_hour" and value == 0)):
            raise ValueError(f"asr.cross_check {field} must be a finite {'nonnegative' if field == 'dollars_per_hour' else 'positive'} number")
    for field in ("keyterms", "character_names"):
        if field in secondary and (not isinstance(secondary[field], list)
                                   or any(not isinstance(item, str) for item in secondary[field])):
            raise ValueError(f"asr.cross_check {field} must be a list of strings")
    secondary.pop("model", None)
    secondary.pop("model_id", None)
    secondary["provider"] = "openrouter" if secondary_model == MAI_TRANSCRIBE_MODEL else "elevenlabs"
    secondary["model" if secondary_model == MAI_TRANSCRIBE_MODEL else "model_id"] = secondary_model
    hint = language if language is not None else primary.get("language_code", primary.get("language"))
    if hint is not None and not isinstance(hint, str):
        raise ValueError("ASR cross-check language must be a string")
    for field in ("language", "language_code"):
        secondary.pop(field, None)
    normalized_language = asr_language_code(hint)
    if normalized_language is not None:
        secondary["language_code"] = normalized_language
    return {"asr": secondary}


def cross_check_context(words: Sequence[Word], config: dict[str, object]) -> dict[str, object]:
    """Cache/resume identity for the evidence actually used, without credentials."""
    asr = config.get("asr", {})
    if not isinstance(asr, dict):
        raise ValueError("ASR cross-check configuration must be a mapping")
    content = json.dumps([word.model_dump(mode="json") for word in words],
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "policy_version": CROSS_CHECK_POLICY_VERSION,
        "provider": asr.get("provider"),
        "model": _model(asr, default=SCRIBE_TRANSCRIBE_MODEL),
        "words_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "config": _cache_safe_params(asr),
    }
