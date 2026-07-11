from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Verdict = Literal["keep_srt", "use_audio", "hybrid"]


class Cue(BaseModel):
    index: int
    start_ms: int
    end_ms: int
    lines: list[str]
    speaker_id: str | None = None
    character: str | None = None

    @field_validator("lines")
    @classmethod
    def _has_text_lines(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("cue must contain at least one text line")
        return value

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def plain_text(self) -> str:
        return " ".join(line.strip() for line in self.lines if line.strip())

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def with_timing(self, start_ms: int, end_ms: int) -> "Cue":
        return self.model_copy(update={"start_ms": start_ms, "end_ms": end_ms})

    def with_lines(self, lines: list[str]) -> "Cue":
        return self.model_copy(update={"lines": list(lines)})


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    speaker_id: str | None = None


class TokenMatch(BaseModel):
    cue_id: int
    srt_token_index: int
    asr_word_index: int
    score: float


class AnchorRegion(BaseModel):
    anchor_id: str
    cue_ids: list[int]
    srt_token_indices: list[int]
    asr_word_indices: list[int]
    srt_text: str
    asr_text: str
    start: float
    end: float
    score: float = Field(ge=0.0, le=1.0)


class CueContext(BaseModel):
    cue_id: int
    text: str
    start: float
    end: float


class DivergenceSpan(BaseModel):
    case_id: str
    cue_ids: list[int]
    srt_text: str
    asr_text: str
    start: float | None = None
    end: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    speaker_ids: list[str] = Field(default_factory=list)
    srt_token_indices: list[int] = Field(default_factory=list)
    asr_word_indices: list[int] = Field(default_factory=list)
    context_before: list[CueContext] = Field(default_factory=list)
    context_after: list[CueContext] = Field(default_factory=list)


class AudioSnippet(BaseModel):
    case_id: str
    path: str
    mime_type: str = "audio/wav"
    start: float
    end: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end - self.start)


class AlignmentResult(BaseModel):
    token_matches: list[TokenMatch] = Field(default_factory=list)
    anchor_regions: list[AnchorRegion] = Field(default_factory=list)
    cue_word_indices: dict[int, list[int]] = Field(default_factory=dict)
    anchor_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    divergence_spans: list[DivergenceSpan] = Field(default_factory=list)
    unmatched_cue_ids: list[int] = Field(default_factory=list)


class AdjudicationDecision(BaseModel):
    case_id: str
    verdict: Verdict
    final_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    speaker: str | None = None
    character: str | None = None
    reason: str


class QCFlag(BaseModel):
    kind: str
    cue_ids: list[int] = Field(default_factory=list)
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    confidence: float | None = None
    old_text: str | None = None
    new_text: str | None = None
    start: float | None = None
    end: float | None = None


class StyleIssue(BaseModel):
    kind: str
    cue_id: int
    message: str
    severity: Literal["warning", "error"] = "warning"


class CueScore(BaseModel):
    cue_id: int
    start: float
    end: float
    cps: float = Field(default=0.0, ge=0.0)
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["asr_confidence", "forced_alignment", "unscored"]


class ForcedAlignmentCue(BaseModel):
    cue_id: int
    start: float
    end: float
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class OverlapRegion(BaseModel):
    start: float
    end: float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SpeechRegion(BaseModel):
    start: float
    end: float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
