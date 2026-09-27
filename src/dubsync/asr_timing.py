from __future__ import annotations

from .models import QCFlag, SpeechRegion, Word
from .region_index import SpeechRegionIndex


def clamp_asr_word_durations(
    words: list[Word],
    regions: list[SpeechRegion],
    *,
    max_word_duration: float = 2.0,
    max_region_overrun: float = 0.3,
    max_region_gap: float = 0.2,
) -> tuple[list[Word], list[QCFlag]]:
    if max_word_duration <= 0:
        raise ValueError("timing.max_word_duration must be positive")
    if max_region_overrun < 0 or max_region_gap < 0:
        raise ValueError("speech-region timing limits must be non-negative")

    clamped: list[Word] = []
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for word in words:
        duration = word.end - word.start
        region_end = _speech_envelope_end(word, region_index, max_region_gap)
        duration_is_outlier = duration > max_word_duration
        region_overrun_is_outlier = (
            region_end is not None
            and word.end - region_end > max_region_overrun
        )
        if not duration_is_outlier and not region_overrun_is_outlier:
            clamped.append(word)
            continue

        fallback_end = word.start + max_word_duration
        new_end = min(
            word.end,
            fallback_end if duration_is_outlier else word.end,
            region_end if region_overrun_is_outlier and region_end is not None else word.end,
        )
        new_end = max(word.start, new_end)
        next_word = word.model_copy(update={"end": round(new_end, 3)})
        clamped.append(next_word)
        flags.append(
            QCFlag(
                kind="asr_word_clamped",
                cue_ids=[],
                message="ASR word endpoint exceeded the configured duration or speech region bounds and was clamped.",
                old_text=f"{word.text} {word.start:.3f} --> {word.end:.3f}",
                new_text=f"{next_word.text} {next_word.start:.3f} --> {next_word.end:.3f}",
                start=next_word.start,
                end=next_word.end,
                confidence=round(duration, 3),
            )
        )
    return clamped, flags


def _speech_envelope_end(
    word: Word,
    region_index: SpeechRegionIndex,
    max_region_gap: float,
) -> float | None:
    containing = region_index.first_containing(word.start)
    if containing is None:
        return None

    containing_index, containing_region = containing
    envelope_end = containing_region.end
    # Index the bounded neighborhood without copying the rest of the episode
    # for every word. Most probes stop at the immediately following region.
    for index in range(containing_index + 1, len(region_index.regions)):
        region = region_index.regions[index]
        if region.start > word.end or region.start - envelope_end > max_region_gap:
            break
        envelope_end = max(envelope_end, region.end)
    return envelope_end
