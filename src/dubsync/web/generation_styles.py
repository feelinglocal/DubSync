from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import Cue
from ..style_profile import GenerationConstraints, StyleProfile
from ..text_metrics import display_width

StyleSource = Literal["preset", "custom", "sample"]


class GenerationStyleError(ValueError):
    pass


class GenerationStyleValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_lines_per_cue: int = Field(ge=1, le=4)
    max_chars_per_line: int = Field(ge=10, le=80)
    min_cue_duration_seconds: float = Field(ge=0.2, le=5.0)
    max_cue_duration_seconds: float = Field(ge=0.5, le=20.0)
    min_cps: float = Field(ge=0.0, le=10.0)
    max_cps: float = Field(ge=5.0, le=60.0)
    max_gap_seconds: float = Field(ge=0.1, le=5.0)
    lead_in_ms: int = Field(ge=0, le=1000)
    tail_ms: int = Field(ge=0, le=1000)


class GenerationStyleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: StyleSource
    preset: str | None = None
    values: GenerationStyleValues | None = None


class ResolvedGenerationStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: StyleSource
    preset: str | None = None
    profile: StyleProfile
    constraints: GenerationConstraints


class _Preset(BaseModel):
    id: str
    name: str
    values: GenerationStyleValues


_PRESETS = (
    _Preset(
        id="standard",
        name="DubSync default",
        values=GenerationStyleValues(
            max_lines_per_cue=2,
            max_chars_per_line=26,
            min_cue_duration_seconds=0.5,
            max_cue_duration_seconds=5.0,
            min_cps=2.0,
            max_cps=30.0,
            max_gap_seconds=0.8,
            lead_in_ms=0,
            tail_ms=40,
        ),
    ),
    _Preset(
        id="streaming",
        name="Streaming",
        values=GenerationStyleValues(
            max_lines_per_cue=2,
            max_chars_per_line=42,
            min_cue_duration_seconds=1.0,
            max_cue_duration_seconds=7.0,
            min_cps=2.0,
            max_cps=20.0,
            max_gap_seconds=1.0,
            lead_in_ms=0,
            tail_ms=120,
        ),
    ),
    _Preset(
        id="broadcast",
        name="Broadcast",
        values=GenerationStyleValues(
            max_lines_per_cue=2,
            max_chars_per_line=37,
            min_cue_duration_seconds=1.0,
            max_cue_duration_seconds=6.0,
            min_cps=2.0,
            max_cps=18.0,
            max_gap_seconds=0.6,
            lead_in_ms=0,
            tail_ms=80,
        ),
    ),
    _Preset(
        id="short_form",
        name="Short-form",
        values=GenerationStyleValues(
            max_lines_per_cue=2,
            max_chars_per_line=24,
            min_cue_duration_seconds=0.4,
            max_cue_duration_seconds=3.5,
            min_cps=2.0,
            max_cps=24.0,
            max_gap_seconds=0.5,
            lead_in_ms=0,
            tail_ms=60,
        ),
    ),
)
_PRESETS_BY_ID = {preset.id: preset for preset in _PRESETS}
_CUSTOM_LIMITS = {
    "max_lines_per_cue": {"min": 1, "max": 4, "step": 1},
    "max_chars_per_line": {"min": 10, "max": 80, "step": 1},
    "min_cue_duration_seconds": {"min": 0.2, "max": 5, "step": 0.1},
    "max_cue_duration_seconds": {"min": 0.5, "max": 20, "step": 0.1},
    "min_cps": {"min": 0, "max": 10, "step": 0.5},
    "max_cps": {"min": 5, "max": 60, "step": 0.5},
    "max_gap_seconds": {"min": 0.1, "max": 5, "step": 0.1},
    "lead_in_ms": {"min": 0, "max": 1000, "step": 10},
    "tail_ms": {"min": 0, "max": 1000, "step": 10},
}


def public_generation_styles() -> dict[str, object]:
    return {
        "default_preset": "standard",
        "presets": [preset.model_dump() for preset in _PRESETS],
        "custom_limits": {key: dict(value) for key, value in _CUSTOM_LIMITS.items()},
    }


def parse_generation_style_request(raw_style: str) -> GenerationStyleRequest:
    normalized = raw_style.strip()
    if normalized.lower() == "standard":
        return GenerationStyleRequest(source="preset", preset="standard")
    if not normalized or len(normalized) > 4096:
        raise GenerationStyleError("Unsupported generation style.")
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise GenerationStyleError("Generation style must be valid JSON.") from exc
    _validate_raw_relationships(payload)
    try:
        request = GenerationStyleRequest.model_validate(payload)
    except ValidationError as exc:
        raise GenerationStyleError(_validation_message(exc)) from exc

    if request.source == "preset" and not request.preset:
        raise GenerationStyleError("Choose a subtitle style preset.")
    if request.source == "custom" and request.values is None:
        raise GenerationStyleError("Custom subtitle style values are required.")
    return request


def resolve_generation_style(
    request: GenerationStyleRequest,
    *,
    fps: float,
    sample_cues: list[Cue] | None = None,
) -> ResolvedGenerationStyle:
    if request.source == "preset":
        preset = _PRESETS_BY_ID.get(request.preset or "")
        if preset is None:
            raise GenerationStyleError("Unsupported subtitle style preset.")
        return _resolved_from_values("preset", preset.values, fps=fps, preset=preset.id)

    if request.source == "custom":
        if request.values is None:
            raise GenerationStyleError("Custom subtitle style values are required.")
        _validate_relationships(request.values)
        return _resolved_from_values("custom", request.values, fps=fps)

    if not sample_cues:
        raise GenerationStyleError("An SRT style example is required.")
    return _resolved_from_sample(sample_cues, fps=fps)


def _resolved_from_values(
    source: Literal["preset", "custom"],
    values: GenerationStyleValues,
    *,
    fps: float,
    preset: str | None = None,
) -> ResolvedGenerationStyle:
    _validate_relationships(values)
    profile = StyleProfile(
        fps=fps,
        max_lines_per_cue=values.max_lines_per_cue,
        max_chars_per_line=values.max_chars_per_line,
        min_cue_dur=values.min_cue_duration_seconds,
        lead_in_ms=values.lead_in_ms,
        tail_ms=values.tail_ms,
    )
    constraints = GenerationConstraints(
        max_gap_seconds=values.max_gap_seconds,
        max_cue_duration_seconds=values.max_cue_duration_seconds,
        min_cps=values.min_cps,
        max_cps=values.max_cps,
    )
    return ResolvedGenerationStyle(source=source, preset=preset, profile=profile, constraints=constraints)


def _resolved_from_sample(cues: list[Cue], *, fps: float) -> ResolvedGenerationStyle:
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    if any(cue.duration_ms <= 0 or not cue.plain_text for cue in ordered):
        raise GenerationStyleError("The SRT style example contains an empty or invalid cue.")

    durations = [cue.duration_ms / 1000.0 for cue in ordered]
    widths = [display_width(cue.plain_text) for cue in ordered]
    cue_cps = [width / duration for width, duration in zip(widths, durations)]
    max_lines = max(len(cue.lines) for cue in ordered)
    max_line_width = max(display_width(line) for cue in ordered for line in cue.lines)
    allow_zero_gap = any(left.end_ms == right.start_ms for left, right in zip(ordered, ordered[1:]))

    profile = StyleProfile(
        fps=fps,
        max_lines_per_cue=int(_clamp(max_lines, 1, 4)),
        max_chars_per_line=int(_clamp(max_line_width, 10, 80)),
        min_cue_dur=round(_clamp(min(durations), 0.2, 5.0), 3),
        allow_zero_gap=allow_zero_gap,
        cue_count=len(ordered),
        observed_min_duration=round(min(durations), 3),
        observed_max_duration=round(max(durations), 3),
        notes=["Generation style derived from an uploaded SRT example."],
    )
    constraints = GenerationConstraints(
        max_gap_seconds=0.8,
        max_cue_duration_seconds=round(_clamp(max(durations), 0.5, 20.0), 3),
        min_cps=round(_clamp(min(cue_cps), 0.0, 10.0), 2),
        max_cps=round(_clamp(max(cue_cps), 5.0, 60.0), 2),
    )
    return ResolvedGenerationStyle(source="sample", profile=profile, constraints=constraints)


def _validate_relationships(values: GenerationStyleValues) -> None:
    if values.min_cue_duration_seconds > values.max_cue_duration_seconds:
        raise GenerationStyleError("Custom minimum cue duration cannot exceed maximum cue duration.")
    if values.min_cps > values.max_cps:
        raise GenerationStyleError("Custom minimum CPS cannot exceed maximum CPS.")


def _validate_raw_relationships(payload: object) -> None:
    if not isinstance(payload, dict) or payload.get("source") != "custom":
        return
    values = payload.get("values")
    if not isinstance(values, dict):
        return
    try:
        minimum_duration = float(values["min_cue_duration_seconds"])
        maximum_duration = float(values["max_cue_duration_seconds"])
        minimum_cps = float(values["min_cps"])
        maximum_cps = float(values["max_cps"])
    except (KeyError, TypeError, ValueError):
        return
    if minimum_duration > maximum_duration:
        raise GenerationStyleError("Custom minimum cue duration cannot exceed maximum cue duration.")
    if minimum_cps > maximum_cps:
        raise GenerationStyleError("Custom minimum CPS cannot exceed maximum CPS.")


def _validation_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "Invalid value."))
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ")
    return f"Invalid generation style value for {field}: {message}" if field else message


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
