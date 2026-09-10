from __future__ import annotations

import json
import math
import wave
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


_LEGACY_FLASH_LITE_VERSION = "3.1"
_GEMINI_FLASH_STANDARD_PRICE_CHANGE = date(2027, 1, 1)
_GEMINI_INTRODUCTORY_FLASH_MODELS = {"gemini-3.7-flash", "gemini-3.8-flash"}


class CostItem(BaseModel):
    provider: str
    kind: str
    units: dict[str, float]
    usd: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reported_cached_input_tokens: int | None = None
    cache_metadata_inconsistent: bool = False


class CostMeter(BaseModel):
    items: list[CostItem] = Field(default_factory=list)

    def add_audio_billed(self, provider: str, seconds: float, usd: float, *, partial: bool = False) -> None:
        """Record the provider's reported charge rather than a catalog estimate."""
        self.items.append(CostItem(provider=provider, kind="audio_billed_partial" if partial else "audio_billed", units={"seconds": seconds}, usd=round(usd, 6)))

    def add_audio(self, provider: str, seconds: float, dollars_per_hour: float) -> None:
        usd = seconds / 3600.0 * dollars_per_hour
        self.items.append(CostItem(provider=provider, kind="audio", units={"seconds": seconds}, usd=round(usd, 6)))

    def add_tokens(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        input_per_million: float,
        output_per_million: float,
        *,
        cached_input_tokens: int = 0,
        cached_input_per_million: float | None = None,
    ) -> None:
        units = {"input_tokens": float(input_tokens), "output_tokens": float(output_tokens)}
        uncached_input_tokens = input_tokens
        cached_usd = 0.0
        if cached_input_tokens and cached_input_per_million is not None:
            uncached_input_tokens -= cached_input_tokens
            cached_usd = cached_input_tokens / 1_000_000 * cached_input_per_million
            units.update({
                "cached_input_tokens": float(cached_input_tokens),
                "uncached_input_tokens": float(uncached_input_tokens),
            })
        usd = (uncached_input_tokens / 1_000_000 * input_per_million) + cached_usd + (output_tokens / 1_000_000 * output_per_million)
        self.items.append(
            CostItem(
                provider=provider,
                kind="tokens",
                units=units,
                usd=round(usd, 6),
            )
        )

    @property
    def total_usd(self) -> float:
        return round(sum(item.usd for item in self.items), 6)

    def as_dict(self) -> dict[str, object]:
        return {"total_usd": self.total_usd, "items": [item.model_dump() for item in self.items]}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)


def asr_dollars_per_hour(provider: str, config: dict[str, object]) -> float | None:
    override = config.get("dollars_per_hour")
    if override is not None:
        return float(override)

    normalized = provider.lower()
    if normalized == "microsoft/mai-transcribe-2" or (
        normalized == "openrouter" and config.get("model", "microsoft/mai-transcribe-2") == "microsoft/mai-transcribe-2"
    ):
        # Launch catalog estimate (2026-09-05); usage.cost takes precedence.
        return 0.10
    if normalized in {"elevenlabs", "scribe_v2"}:
        surcharge = 0.05 if _has_keyterm_prompting(config) else 0.0
        return round(0.22 + surcharge, 6)
    if normalized in {"openai", "whisper-1"}:
        return 0.36
    if normalized == "assemblyai":
        return _assemblyai_dollars_per_hour(config)
    if normalized in {
        "gemini",
        "gemini_transcribe",
        "gemini-transcribe",
        "gemini-3.5-transcribe",
        "gemini_3.5_transcribe",
        "gemini_3_5_transcribe",
    }:
        return 0.3
    if normalized in {"whisperx", "fixture"}:
        return 0.0
    return None


def token_usage_from_response(response: object) -> TokenUsage | None:
    for usage_key in ("usage", "usage_metadata", "usageMetadata"):
        usage = _field(response, usage_key)
        if usage is None:
            continue
        input_tokens = _int_field(
            usage,
            ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count", "promptTokenCount", "total_input_tokens"),
        )
        output_tokens = _int_field(
            usage,
            (
                "output_tokens",
                "completion_tokens",
                "output_token_count",
                "completion_token_count",
                "candidates_token_count",
                "candidatesTokenCount",
                "response_token_count",
                "total_output_tokens",
            ),
        )
        if input_tokens is not None and output_tokens is not None:
            thought_tokens = _int_field(
                usage,
                (
                    "thoughts_token_count",
                    "thoughtsTokenCount",
                    "thought_token_count",
                    "thinking_tokens",
                    "thought_tokens",
                    "total_thought_tokens",
                ),
            )
            if thought_tokens is not None:
                output_tokens += thought_tokens
            cache_aliases = ("cached_content_token_count", "cachedContentTokenCount")
            reported_cached_input_tokens = _int_field(usage, cache_aliases)
            cache_metadata_present = any(_field(usage, name) is not None for name in cache_aliases)
            cache_metadata_inconsistent = cache_metadata_present and (
                reported_cached_input_tokens is None or reported_cached_input_tokens > input_tokens
            )
            # Invalid cache metadata must not make a billable prompt cheaper.
            cached_input_tokens = 0 if cache_metadata_inconsistent else (reported_cached_input_tokens or 0)
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                reported_cached_input_tokens=reported_cached_input_tokens,
                cache_metadata_inconsistent=cache_metadata_inconsistent,
            )
    return None


def llm_token_prices(provider: str, model: str, config: dict[str, object]) -> tuple[float, float] | None:
    configured = _configured_token_prices(config)
    if configured is not None:
        return configured

    normalized_provider = provider.lower()
    normalized_model = model.lower().removeprefix("models/")
    if normalized_provider == "openai" or normalized_model.startswith("gpt-"):
        if normalized_model == "gpt-5.6-luna":
            return (1.0, 6.0)
    if normalized_provider == "gemini" or normalized_model.startswith("gemini-"):
        if normalized_model in _GEMINI_INTRODUCTORY_FLASH_MODELS:
            # https://ai.google.dev/gemini-api/docs/pricing
            # Official standard paid-tier pricing changes on 2027-01-01.
            return (1.5, 7.5) if _utc_today() >= _GEMINI_FLASH_STANDARD_PRICE_CHANGE else (0.75, 3.75)
        if normalized_model == "gemini-3.5-flash":
            return (1.5, 9.0)
        if normalized_model == "gemini-3.5-flash-lite":
            return (0.3, 2.5)
        if normalized_model == f"gemini-{_LEGACY_FLASH_LITE_VERSION}-flash-lite":
            return (0.25, 1.5)
    return None


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def record_llm_usage(
    meter: CostMeter,
    provider: str,
    model: str,
    config: dict[str, object],
    response: object,
) -> str | None:
    usage = token_usage_from_response(response)
    if usage is None:
        return "usage metadata unavailable"
    try:
        prices = llm_token_prices(provider, model, config)
        cached_price = _gemini_cached_input_price(model, config) if provider.lower() == "gemini" else None
    except (TypeError, ValueError):
        return "token pricing invalid"
    if prices is None:
        return "token pricing unavailable"

    input_per_million, output_per_million = prices
    meter.add_tokens(
        model or provider,
        usage.input_tokens,
        usage.output_tokens,
        input_per_million,
        output_per_million,
        cached_input_tokens=usage.cached_input_tokens,
        cached_input_per_million=cached_price,
    )
    if provider.lower() == "gemini" and usage.cache_metadata_inconsistent:
        # The documented prompt count includes cached content, but real audio
        # responses can report more cached tokens than total prompt tokens.
        # Preserve the full-input estimate without inventing a discount, and
        # distinguish its uncertainty from a request omitted from accounting.
        item = meter.items[-1]
        units = dict(item.units)
        if usage.reported_cached_input_tokens is not None:
            units["reported_cached_input_tokens"] = float(usage.reported_cached_input_tokens)
        meter.items[-1] = item.model_copy(update={
            "kind": "tokens_cache_metadata_estimate", "units": units,
        })
    return None


def record_gemini_context_cost(
    meter: CostMeter,
    model: str,
    config: dict[str, object],
    report: dict[str, object],
) -> str | None:
    """Record conservative context estimates once, after context cleanup.

    Cache creation uses a normal-input reserve; this is not a reported charge.
    Storage uses the greater of observed token-seconds and the total reserved
    lifetime when cleanup could not confirm deletion. Successful generation usage
    is metered separately. Failed requests without usage reserve audio input or
    cached-prefix reads as uncertain estimates; actual charges are unknown.
    """
    creation_tokens = _nonnegative_number(report.get("cache_create_input_tokens_reserved", 0))
    storage_token_seconds = _nonnegative_number(report.get("cache_storage_token_seconds", 0))
    reserved_token_seconds = _nonnegative_number(report.get("cache_storage_token_seconds_reserved", 0))
    unreported_audio_tokens = (
        _int_field(report, ("unreported_uncached_audio_tokens_reserved",))
        if "unreported_uncached_audio_tokens_reserved" in report else 0
    )
    unreported_cached_tokens = (
        _int_field(report, ("unreported_cached_audio_tokens_reserved",))
        if "unreported_cached_audio_tokens_reserved" in report else 0
    )
    if any(value is None for value in (
        creation_tokens, storage_token_seconds, reserved_token_seconds,
        unreported_audio_tokens, unreported_cached_tokens,
    )):
        return "cache context usage metadata invalid"
    storage_token_seconds = max(storage_token_seconds, reserved_token_seconds)
    if creation_tokens == 0 and storage_token_seconds == 0 and unreported_audio_tokens == 0 and unreported_cached_tokens == 0:
        return None

    try:
        token_prices = llm_token_prices("gemini", model, config)
        storage_price = _gemini_cache_storage_price(model, config)
        cached_price = _gemini_cached_input_price(model, config) if unreported_cached_tokens else None
    except (TypeError, ValueError):
        return "cache context pricing invalid"
    if cached_price is None and token_prices is not None:
        # An unknown discount cannot make the uncertain reserve cheaper.
        cached_price = token_prices[0]
    if (
        ((creation_tokens or unreported_audio_tokens) and token_prices is None)
        or (storage_token_seconds and storage_price is None)
        or (unreported_cached_tokens and cached_price is None)
    ):
        return "cache context pricing unavailable"

    if creation_tokens and token_prices is not None:
        meter.items.append(CostItem(
            provider=model,
            kind="cache_create_estimate",
            units={"input_tokens": creation_tokens},
            usd=round(creation_tokens / 1_000_000 * token_prices[0], 6),
        ))
    if storage_token_seconds and storage_price is not None:
        meter.items.append(CostItem(
            provider=model,
            kind="cache_storage_estimate",
            units={"token_seconds": storage_token_seconds},
            usd=round(storage_token_seconds / 3_600 / 1_000_000 * storage_price, 6),
        ))
    if unreported_audio_tokens and token_prices is not None:
        meter.items.append(CostItem(
            provider=model,
            kind="uncertain_audio_input_estimate",
            units={"input_tokens": float(unreported_audio_tokens)},
            usd=round(unreported_audio_tokens / 1_000_000 * token_prices[0], 6),
        ))
    if unreported_cached_tokens and cached_price is not None:
        meter.items.append(CostItem(
            provider=model,
            kind="uncertain_cached_input_estimate",
            units={"cached_input_tokens": float(unreported_cached_tokens)},
            usd=round(unreported_cached_tokens / 1_000_000 * cached_price, 6),
        ))
    return None


def _gemini_cached_input_price(model: str, config: dict[str, object]) -> float | None:
    configured = _configured_optional_price(config, (
        "cached_input_per_million", "cached_input_usd_per_million", "cached_input_per_million_usd",
    ))
    if configured is not None:
        return configured
    # A custom input tariff does not imply the catalog's native cache discount.
    if _configured_token_prices(config) is not None:
        return None
    if model.lower().removeprefix("models/") in _GEMINI_INTRODUCTORY_FLASH_MODELS:
        return 0.15 if _utc_today() >= _GEMINI_FLASH_STANDARD_PRICE_CHANGE else 0.075
    return None


def _gemini_cache_storage_price(model: str, config: dict[str, object]) -> float | None:
    configured = _configured_optional_price(config, ("cache_storage_per_million_token_hour",))
    if configured is not None:
        return configured
    if model.lower().removeprefix("models/") in _GEMINI_INTRODUCTORY_FLASH_MODELS:
        # https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash
        return 1.0 if _utc_today() >= _GEMINI_FLASH_STANDARD_PRICE_CHANGE else 0.5
    return None


def _configured_optional_price(config: dict[str, object], aliases: tuple[str, ...]) -> float | None:
    configured = _price_value(config, aliases)
    if configured is not None:
        return configured
    for nested_key in ("pricing", "cost"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            configured = _configured_optional_price(nested, aliases)
            if configured is not None:
                return configured
    return None


def audio_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0
            return round(frames / float(rate), 3)
    except (wave.Error, OSError, EOFError):
        return 0.0


def _configured_token_prices(config: dict[str, object]) -> tuple[float, float] | None:
    input_price = _price_value(config, ("input_per_million", "input_usd_per_million", "input_per_million_usd"))
    output_price = _price_value(config, ("output_per_million", "output_usd_per_million", "output_per_million_usd"))
    if input_price is not None and output_price is not None:
        return (input_price, output_price)

    for nested_key in ("pricing", "cost"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            nested_prices = _configured_token_prices(nested)
            if nested_prices is not None:
                return nested_prices
    return None


def _has_keyterm_prompting(config: dict[str, object]) -> bool:
    for key in ("keyterms", "character_names"):
        value = config.get(key)
        if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
            return True
    return False


def _assemblyai_dollars_per_hour(config: dict[str, object]) -> float:
    model = str(config.get("model", "universal-3-pro")).lower()
    base = 0.15 if model in {"universal-2", "universal_2", "u2"} else 0.21
    speaker_surcharge = 0.02 if bool(config.get("speaker_labels", True)) else 0.0
    return round(base + speaker_surcharge, 6)


def _price_value(source: dict[str, object], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = source.get(alias)
        if value is None:
            continue
        price = _nonnegative_number(value)
        if price is None:
            raise ValueError(f"{alias} must be finite and non-negative")
        return price
    return None


def _int_field(source: object, aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        value = _field(source, alias)
        if value is None:
            continue
        number = _nonnegative_number(value)
        return int(number) if number is not None and number.is_integer() else None
    return None


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _field(source: object, name: str) -> Any | None:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)
