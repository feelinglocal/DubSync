from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CostItem(BaseModel):
    provider: str
    kind: str
    units: dict[str, float]
    usd: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class CostMeter(BaseModel):
    items: list[CostItem] = Field(default_factory=list)

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
    ) -> None:
        usd = (input_tokens / 1_000_000 * input_per_million) + (output_tokens / 1_000_000 * output_per_million)
        self.items.append(
            CostItem(
                provider=provider,
                kind="tokens",
                units={"input_tokens": float(input_tokens), "output_tokens": float(output_tokens)},
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
    if normalized in {"elevenlabs", "scribe_v2"}:
        surcharge = 0.05 if _has_keyterm_prompting(config) else 0.0
        return round(0.22 + surcharge, 6)
    if normalized in {"openai", "whisper-1"}:
        return 0.36
    if normalized == "assemblyai":
        return _assemblyai_dollars_per_hour(config)
    if normalized in {"whisperx", "fixture"}:
        return 0.0
    return None


def token_usage_from_response(response: object) -> TokenUsage | None:
    for usage_key in ("usage", "usage_metadata"):
        usage = _field(response, usage_key)
        if usage is None:
            continue
        input_tokens = _int_field(
            usage,
            ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count"),
        )
        output_tokens = _int_field(
            usage,
            ("output_tokens", "completion_tokens", "output_token_count", "completion_token_count", "candidates_token_count"),
        )
        if input_tokens is not None and output_tokens is not None:
            return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return None


def llm_token_prices(provider: str, model: str, config: dict[str, object]) -> tuple[float, float] | None:
    configured = _configured_token_prices(config)
    if configured is not None:
        return configured

    normalized_provider = provider.lower()
    normalized_model = model.lower()
    if normalized_provider == "gemini" or normalized_model.startswith("gemini-"):
        if normalized_model == "gemini-3.5-flash":
            return (1.5, 9.0)
        if normalized_model == "gemini-3.5-flash-lite":
            return (0.3, 2.5)
    return None


def record_llm_usage(
    meter: CostMeter,
    provider: str,
    model: str,
    config: dict[str, object],
    response: object,
) -> None:
    usage = token_usage_from_response(response)
    prices = llm_token_prices(provider, model, config)
    if usage is None or prices is None:
        return

    input_per_million, output_per_million = prices
    meter.add_tokens(
        model or provider,
        usage.input_tokens,
        usage.output_tokens,
        input_per_million,
        output_per_million,
    )


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
        return float(value)
    return None


def _int_field(source: object, aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        value = _field(source, alias)
        if value is None:
            continue
        return int(value)
    return None


def _field(source: object, name: str) -> Any | None:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)
