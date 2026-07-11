from __future__ import annotations

from math import floor, isclose
from typing import Literal

from pydantic import BaseModel, Field

from .models import Cue
from .text_metrics import display_width

FPS_CANDIDATES = (23.976, 24.0, 25.0, 29.97, 30.0)


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


def _candidate_error(timestamps: list[int], fps: float) -> float:
    frame_ms = 1000.0 / fps
    errors = []
    for timestamp in timestamps:
        frame = round(timestamp / frame_ms)
        floored = int(frame * frame_ms)
        errors.append(abs(timestamp - floored))
    return sum(errors) / max(len(errors), 1)


def detect_fps(cues: list[Cue]) -> float:
    timestamps = [time for cue in cues for time in (cue.start_ms, cue.end_ms)]
    if not timestamps:
        return 30.0
    best = min(FPS_CANDIDATES, key=lambda fps: _candidate_error(timestamps, fps))
    return 30.0 if isclose(best, 29.97) and _candidate_error(timestamps, 30.0) <= 2 else float(best)


def derive_style_profile(cues: list[Cue]) -> StyleProfile:
    if not cues:
        return StyleProfile()

    durations = [cue.duration_ms / 1000.0 for cue in cues]
    max_lines = max(len(cue.lines) for cue in cues)
    observed_chars = max(display_width(line) for cue in cues for line in cue.lines)
    allow_zero_gap = any(left.end_ms == right.start_ms for left, right in zip(cues, cues[1:]))
    min_duration = min(durations)

    notes: list[str] = []
    if any(line != line.rstrip() for cue in cues for line in cue.lines):
        notes.append("trailing text whitespace normalized on write")

    return StyleProfile(
        fps=detect_fps(cues),
        max_lines_per_cue=max(2, max_lines),
        max_chars_per_line=max(26, observed_chars),
        min_cue_dur=min(round(min_duration, 3), 0.5),
        allow_zero_gap=allow_zero_gap,
        cue_count=len(cues),
        observed_min_duration=round(min_duration, 3),
        observed_max_duration=round(max(durations), 3),
        notes=notes,
    )
