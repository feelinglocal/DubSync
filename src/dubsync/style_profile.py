from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import Cue
from .subtitle_annotations import cue_has_spoken_text, text_without_bracketed_screen_text
from .text_metrics import display_width

FPS_CANDIDATES = (23.976, 24.0, 25.0, 29.97, 30.0)
_ROBUST_PROFILE_MIN_SAMPLE = 20
_UPPER_STYLE_PERCENTILE = 0.95
_LOWER_STYLE_TRIM_FRACTION = 0.05
_FPS_CONFIDENT_MAX_ERROR_MS = 2.0
_FPS_CONFIDENT_MIN_ADVANTAGE_MS = 0.25


@dataclass(frozen=True)
class FPSDetection:
    fps: float
    confident: bool
    best_error_ms: float
    runner_up_error_ms: float | None = None
    detected_fps: float | None = None
    fallback_fps: float | None = None

    @property
    def used_fallback(self) -> bool:
        return not self.confident


class StyleProfile(BaseModel):
    fps: float = Field(default=30.0, gt=0)
    max_lines_per_cue: int = Field(default=2, ge=1)
    max_chars_per_line: int = Field(default=26, ge=1)
    min_cue_dur: float = Field(default=0.5, ge=0)
    allow_zero_gap: bool = True
    lead_in_ms: int = Field(default=0, ge=0)
    tail_ms: int = Field(default=40, ge=0)
    overlap_policy: Literal["stack", "dash", "flag_only"] = "stack"
    drop_policy: Literal["keep_flagged", "remove"] = "keep_flagged"
    cue_count: int | None = None
    observed_min_duration: float | None = None
    observed_max_duration: float | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps

    def snap_floor(self, ms: int | float) -> int:
        frame = floor(float(ms) / self.frame_ms + 1e-9)
        return int(frame * self.frame_ms)

    def snap_ceil(self, ms: int | float) -> int:
        value = float(ms)
        frame_ms = self.frame_ms
        frame = floor(value / frame_ms + 1e-9)
        snapped = int(frame * frame_ms)
        if snapped < value - 1e-9:
            frame += 1
            snapped = int(frame * frame_ms)
        return snapped

    def is_frame_aligned(self, ms: int, tolerance_ms: int = 1) -> bool:
        floor = self.snap_floor(ms)
        ceil = self.snap_ceil(ms)
        return min(abs(ms - floor), abs(ms - ceil)) <= tolerance_ms


class GenerationConstraints(BaseModel):
    max_gap_seconds: float = Field(default=0.8, gt=0)
    max_cue_duration_seconds: float = Field(default=5.0, gt=0)
    min_cps: float = Field(default=2.0, ge=0)
    max_cps: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _ordered_cps_limits(self) -> "GenerationConstraints":
        if self.min_cps > self.max_cps:
            raise ValueError("min_cps cannot exceed max_cps")
        return self


def _candidate_error(timestamps: list[int], fps: float) -> float:
    frame_ms = 1000.0 / fps
    errors = []
    for timestamp in timestamps:
        frame = round(timestamp / frame_ms)
        snapped = round(frame * frame_ms)
        errors.append(abs(timestamp - snapped))
    return sum(errors) / max(len(errors), 1)


def detect_fps(cues: list[Cue]) -> float:
    return detect_fps_with_confidence(cues).fps


def detect_fps_with_confidence(cues: list[Cue], default: float = 30.0) -> FPSDetection:
    if default <= 0:
        raise ValueError("default FPS must be positive")
    timestamps = [time for cue in cues for time in (cue.start_ms, cue.end_ms)]
    if not timestamps:
        return FPSDetection(
            fps=float(default),
            confident=False,
            best_error_ms=0.0,
            runner_up_error_ms=None,
            detected_fps=None,
            fallback_fps=float(default),
        )
    scored = sorted(((_candidate_error(timestamps, fps), fps) for fps in FPS_CANDIDATES), key=lambda item: item[0])
    best_error, best = scored[0]
    runner_up_error = scored[1][0] if len(scored) > 1 else None
    confident = (
        best_error <= _FPS_CONFIDENT_MAX_ERROR_MS
        and runner_up_error is not None
        and runner_up_error - best_error >= _FPS_CONFIDENT_MIN_ADVANTAGE_MS
    )
    if not confident:
        return FPSDetection(
            fps=float(default),
            confident=False,
            best_error_ms=round(best_error, 3),
            runner_up_error_ms=round(runner_up_error, 3) if runner_up_error is not None else None,
            detected_fps=float(best),
            fallback_fps=float(default),
        )
    return FPSDetection(
        fps=float(best),
        confident=True,
        best_error_ms=round(best_error, 3),
        runner_up_error_ms=round(runner_up_error, 3) if runner_up_error is not None else None,
        detected_fps=float(best),
        fallback_fps=None,
    )


def derive_style_profile(cues: list[Cue]) -> StyleProfile:
    if not cues:
        return StyleProfile()

    dialogue_cues = [cue for cue in cues if cue_has_spoken_text(cue)] or list(cues)
    dialogue_lines = [
        line
        for cue in dialogue_cues
        for line in text_without_bracketed_screen_text(cue.text).splitlines()
        if line.strip()
    ]
    durations = [cue.duration_ms / 1000.0 for cue in dialogue_cues]
    line_counts = [
        max(
            1,
            len(
                [
                    line
                    for line in text_without_bracketed_screen_text(cue.text).splitlines()
                    if line.strip()
                ]
            ),
        )
        for cue in dialogue_cues
    ]
    line_widths = [display_width(line) for line in dialogue_lines] or [
        display_width(line) for cue in dialogue_cues for line in cue.lines
    ]
    max_lines = _robust_upper_limit(line_counts)
    max_chars = _robust_upper_limit(line_widths)
    allow_zero_gap = any(
        left.end_ms == right.start_ms
        for left, right in zip(dialogue_cues, dialogue_cues[1:])
    )
    observed_min_duration = min(durations)
    min_duration = _robust_lower_limit(durations)

    notes: list[str] = []
    if any(line != line.rstrip() for cue in dialogue_cues for line in cue.lines):
        notes.append("trailing text whitespace normalized on write")

    return StyleProfile(
        fps=detect_fps(dialogue_cues),
        max_lines_per_cue=max(2, max_lines),
        max_chars_per_line=max(26, max_chars),
        min_cue_dur=min(round(min_duration, 3), 0.5),
        allow_zero_gap=allow_zero_gap,
        cue_count=len(cues),
        observed_min_duration=round(observed_min_duration, 3),
        observed_max_duration=round(max(durations), 3),
        notes=notes,
    )


def _robust_upper_limit(values: list[int]) -> int:
    ordered = sorted(values)
    if len(ordered) < _ROBUST_PROFILE_MIN_SAMPLE:
        return ordered[-1]
    rank = max(1, ceil(_UPPER_STYLE_PERCENTILE * len(ordered)))
    return ordered[rank - 1]


def _robust_lower_limit(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < _ROBUST_PROFILE_MIN_SAMPLE:
        return ordered[0]
    trimmed_count = floor(_LOWER_STYLE_TRIM_FRACTION * len(ordered))
    return ordered[min(trimmed_count, len(ordered) - 1)]
