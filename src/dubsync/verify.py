from __future__ import annotations

from .models import AlignmentResult, Cue, CueScore, ForcedAlignmentCue, QCFlag, StyleIssue, Word
from .style_profile import StyleProfile
from .text_metrics import display_width


def lint_cues(cues: list[Cue], profile: StyleProfile) -> list[StyleIssue]:
    issues: list[StyleIssue] = []
    previous: Cue | None = None
    min_duration_ms = int(profile.min_cue_dur * 1000)

    for cue in cues:
        if cue.start_ms > cue.end_ms:
            issues.append(StyleIssue(kind="negative_duration", cue_id=cue.index, message="Cue start is after end.", severity="error"))
        if cue.duration_ms < min_duration_ms:
            issues.append(StyleIssue(kind="min_duration", cue_id=cue.index, message="Cue is shorter than minimum duration."))
        if not profile.is_frame_aligned(cue.start_ms) or not profile.is_frame_aligned(cue.end_ms):
            issues.append(StyleIssue(kind="frame_grid", cue_id=cue.index, message="Cue timestamp is off the frame grid."))
        if len(cue.lines) > profile.max_lines_per_cue:
            issues.append(StyleIssue(kind="line_count", cue_id=cue.index, message="Cue has too many text lines."))
        for line in cue.lines:
            if display_width(line) > profile.max_chars_per_line:
                issues.append(StyleIssue(kind="line_length", cue_id=cue.index, message="Cue line exceeds profile length."))
        if previous is not None and cue.start_ms < previous.end_ms and _same_or_unknown_speaker(previous, cue):
            issues.append(StyleIssue(kind="overlap", cue_id=cue.index, message="Cue overlaps the previous cue."))
        previous = cue

    return issues


def score_cues(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    forced_alignments: list[ForcedAlignmentCue] | None = None,
) -> list[CueScore]:
    forced_by_cue = {item.cue_id: item for item in forced_alignments or []}
    scores: list[CueScore] = []

    for cue in cues:
        forced = forced_by_cue.get(cue.index)
        if forced is not None:
            scores.append(
                CueScore(
                    cue_id=cue.index,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                    cps=_cue_cps(cue),
                    score=round(forced.score, 4),
                    source="forced_alignment",
                )
            )
            continue

        cue_words = [
            words[index]
            for index in alignment.cue_word_indices.get(cue.index, [])
            if 0 <= index < len(words)
        ]
        if cue_words:
            score = sum(word.confidence for word in cue_words) / len(cue_words)
            source = "asr_confidence"
        else:
            score = 0.0
            source = "unscored"

        scores.append(
            CueScore(
                cue_id=cue.index,
                start=cue.start_ms / 1000.0,
                end=cue.end_ms / 1000.0,
                cps=_cue_cps(cue),
                score=round(score, 4),
                source=source,
            )
        )

    return scores


def cps_sanity_flags(cues: list[Cue], *, max_cps: float = 30.0, min_cps: float = 2.0) -> list[QCFlag]:
    flags: list[QCFlag] = []
    for cue in cues:
        cps = _cue_cps(cue)
        if cps > max_cps:
            flags.append(
                QCFlag(
                    kind="impossible_cps_fast",
                    cue_ids=[cue.index],
                    message=f"Cue reads too fast at {cps:.1f} characters per second.",
                    confidence=round(cps, 3),
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
        elif cue.duration_ms > 2000 and cps < min_cps:
            flags.append(
                QCFlag(
                    kind="impossible_cps_slow",
                    cue_ids=[cue.index],
                    message=f"Cue reads too slowly at {cps:.1f} characters per second.",
                    confidence=round(cps, 3),
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
    return flags


def _cue_cps(cue: Cue) -> float:
    duration_seconds = cue.duration_ms / 1000.0
    if duration_seconds <= 0:
        return 0.0
    return round(display_width(cue.plain_text) / duration_seconds, 3)


def _same_or_unknown_speaker(previous: Cue, cue: Cue) -> bool:
    return previous.speaker_id is None or cue.speaker_id is None or previous.speaker_id == cue.speaker_id
