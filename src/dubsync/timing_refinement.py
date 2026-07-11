from __future__ import annotations

from dataclasses import dataclass

from .models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from .style_profile import StyleProfile


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    enabled: bool = True
    start_pad_ms: int = 40
    end_pad_ms: int = 40
    max_end_extension_ms: int = 300
    max_leading_silence_ms: int = 150
    max_trailing_silence_ms: int = 300
    max_word_duration_ms: int = 2000


def refine_cues_to_speech_activity(
    cues: list[Cue],
    regions: list[SpeechRegion],
    profile: StyleProfile,
    config: BoundaryRefinementConfig | None = None,
    *,
    words: list[Word] | None = None,
    alignment: AlignmentResult | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    options = config or BoundaryRefinementConfig()
    if not options.enabled or not regions:
        return cues, []

    refined: list[Cue] = []
    flags: list[QCFlag] = []
    sorted_regions = sorted(regions, key=lambda region: (region.start, region.end))

    for index, cue in enumerate(cues):
        word_window = _word_window_for_cue(cue, words, alignment)
        cue_regions = (
            _regions_from_word_window(word_window, sorted_regions, options)
            if word_window is not None
            else _regions_overlapping_cue(cue, sorted_regions)
        )
        if cue_regions is None:
            refined.append(cue)
            continue

        start_region, end_region = cue_regions
        start_ms = _refined_start_ms(cue, start_region, profile, options)
        end_ms = (
            _word_refined_end_ms(cue, end_region, word_window, profile, options)
            if word_window is not None
            else _refined_end_ms(cue, end_region, profile, options)
        )
        end_cap_ms = None
        if index + 1 < len(cues):
            end_cap_ms = cues[index + 1].start_ms
            end_ms = min(end_ms, end_cap_ms)
        end_ms = max(end_ms, profile.snap_ceil(start_ms + profile.min_cue_dur * 1000))
        if end_cap_ms is not None and end_ms > end_cap_ms:
            end_ms = max(start_ms, end_cap_ms)

        if start_ms == cue.start_ms and end_ms == cue.end_ms:
            refined.append(cue)
            continue

        next_cue = cue.with_timing(start_ms, end_ms)
        refined.append(next_cue)
        flags.append(
            QCFlag(
                kind="timing_refined",
                cue_ids=[cue.index],
                message="Cue boundary adjusted to the detected speech activity envelope.",
                old_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
                new_text=f"{next_cue.start_ms / 1000.0:.3f} --> {next_cue.end_ms / 1000.0:.3f}",
                start=next_cue.start_ms / 1000.0,
                end=next_cue.end_ms / 1000.0,
            )
        )

    return refined, flags


def _regions_overlapping_cue(cue: Cue, regions: list[SpeechRegion]) -> tuple[SpeechRegion, SpeechRegion] | None:
    cue_start = cue.start_ms / 1000.0
    cue_end = cue.end_ms / 1000.0
    overlapping = [
        region
        for region in regions
        if region.end > cue_start and region.start < cue_end
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def _word_window_for_cue(
    cue: Cue,
    words: list[Word] | None,
    alignment: AlignmentResult | None,
) -> list[Word] | None:
    if words is None or alignment is None:
        return None
    matched = [
        words[index]
        for index in alignment.cue_word_indices.get(cue.index, [])
        if 0 <= index < len(words)
    ]
    if not matched:
        return None
    return sorted(matched, key=lambda word: (word.start, word.end))


def _regions_from_word_window(
    word_window: list[Word],
    regions: list[SpeechRegion],
    config: BoundaryRefinementConfig,
) -> tuple[SpeechRegion, SpeechRegion] | None:
    first_word = word_window[0]
    last_word = word_window[-1]
    start_region = _region_containing_timestamp(first_word.start, regions)
    if start_region is None:
        start_region = _region_overlapping_word(first_word, regions)
    end_probe = last_word.start if _is_word_duration_outlier(last_word, config) else last_word.end
    end_region = _region_containing_timestamp(end_probe, regions)
    if end_region is None:
        end_region = _region_containing_timestamp(last_word.start, regions) or _region_overlapping_word(last_word, regions)
    if start_region is None or end_region is None:
        return None
    return start_region, end_region


def _region_containing_timestamp(timestamp: float, regions: list[SpeechRegion]) -> SpeechRegion | None:
    for region in regions:
        if region.start <= timestamp <= region.end:
            return region
    return None


def _region_overlapping_word(word: Word, regions: list[SpeechRegion]) -> SpeechRegion | None:
    for region in regions:
        if region.end > word.start and region.start < word.end:
            return region
    return None


def _is_word_duration_outlier(word: Word, config: BoundaryRefinementConfig) -> bool:
    return (word.end - word.start) * 1000 > config.max_word_duration_ms


def _refined_start_ms(
    cue: Cue,
    first_region: SpeechRegion,
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    first_speech_ms = int(first_region.start * 1000)
    leading_silence_ms = first_speech_ms - cue.start_ms
    if leading_silence_ms <= config.max_leading_silence_ms:
        return cue.start_ms
    return max(0, profile.snap_floor(first_speech_ms - config.start_pad_ms))


def _refined_end_ms(
    cue: Cue,
    last_region: SpeechRegion,
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    speech_end_ms = int(last_region.end * 1000)
    padded_end_ms = profile.snap_ceil(speech_end_ms + config.end_pad_ms)
    end_overrun_ms = speech_end_ms - cue.end_ms
    if config.end_pad_ms < end_overrun_ms <= config.max_end_extension_ms:
        return padded_end_ms
    if cue.end_ms - padded_end_ms > config.max_trailing_silence_ms:
        return padded_end_ms
    return cue.end_ms


def _word_refined_end_ms(
    cue: Cue,
    last_region: SpeechRegion,
    word_window: list[Word],
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    last_word = word_window[-1]
    speech_end_ms = int(last_region.end * 1000)
    padded_region_end_ms = profile.snap_ceil(speech_end_ms + config.end_pad_ms)

    if _is_word_duration_outlier(last_word, config):
        return padded_region_end_ms

    word_end_ms = profile.snap_ceil(last_word.end * 1000 + config.end_pad_ms)
    if cue.end_ms < word_end_ms:
        return word_end_ms

    region_tail_ms = padded_region_end_ms - word_end_ms
    if cue.end_ms < padded_region_end_ms and region_tail_ms <= config.max_trailing_silence_ms:
        return padded_region_end_ms

    if cue.end_ms - word_end_ms > config.max_trailing_silence_ms:
        return max(word_end_ms, min(cue.end_ms, padded_region_end_ms))

    return cue.end_ms
