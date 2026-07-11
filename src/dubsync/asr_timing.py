from __future__ import annotations

from .models import QCFlag, SpeechRegion, Word


def clamp_asr_word_durations(
    words: list[Word],
    regions: list[SpeechRegion],
    *,
    max_word_duration: float = 2.0,
) -> tuple[list[Word], list[QCFlag]]:
    if max_word_duration <= 0:
        raise ValueError("timing.max_word_duration must be positive")

    clamped: list[Word] = []
    flags: list[QCFlag] = []
    sorted_regions = sorted(regions, key=lambda region: (region.start, region.end))
    for index, word in enumerate(words):
        duration = word.end - word.start
        if duration <= max_word_duration:
            clamped.append(word)
            continue

        region = _region_containing_timestamp(word.start, sorted_regions)
        fallback_end = word.start + max_word_duration
        new_end = min(
            word.end,
            fallback_end,
            region.end if region is not None and region.end > word.start else fallback_end,
        )
        new_end = max(word.start, new_end)
        next_word = word.model_copy(update={"end": round(new_end, 3)})
        clamped.append(next_word)
        flags.append(
            QCFlag(
                kind="asr_word_clamped",
                cue_ids=[],
                message="ASR word duration exceeded timing.max_word_duration and was clamped to the containing speech region.",
                old_text=f"{word.text} {word.start:.3f} --> {word.end:.3f}",
                new_text=f"{next_word.text} {next_word.start:.3f} --> {next_word.end:.3f}",
                start=next_word.start,
                end=next_word.end,
                confidence=round(duration, 3),
            )
        )
    return clamped, flags


def _region_containing_timestamp(timestamp: float, regions: list[SpeechRegion]) -> SpeechRegion | None:
    for region in regions:
        if region.start <= timestamp <= region.end:
            return region
    return None
