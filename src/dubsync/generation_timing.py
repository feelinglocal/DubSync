from __future__ import annotations

from math import ceil, floor, isfinite

from .models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from .region_index import SpeechRegionIndex
from .style_profile import GenerationConstraints, StyleProfile
from .text_metrics import display_width
from .timing_refinement import BoundaryRefinementConfig, _acoustic_end_seconds, _quiet_until_seconds


def generated_spoken_spans(words: list[Word], alignment: AlignmentResult) -> dict[int, tuple[int, int]]:
    return {
        cue_id: (
            floor(min(words[index].start for index in indices) * 1000),
            ceil(max(words[index].end for index in indices) * 1000),
        )
        for cue_id, indices in alignment.cue_word_indices.items() if indices
    }


def bound_held_generation_padding(
    cues: list[Cue], words: list[Word], alignment: AlignmentResult,
    held_cue_ids: set[int], profile: StyleProfile,
) -> list[Cue]:
    """Limit generated display margins without selecting an uncertain burst.

    Generated segmentation has no authored interval to preserve. Before its
    uncertain cues become fixed, remove only padding in a frame-aligned gap
    between the complete owned word envelopes. The original Word objects,
    including ambiguous raw endpoints, remain untouched. A true overlap or
    a gap with no safe frame boundary stays visible for review.
    """
    result = list(cues)
    for position in range(len(result) - 1):
        left, right = result[position:position + 2]
        if not held_cue_ids.intersection((left.index, right.index)) or left.end_ms <= right.start_ms:
            continue
        left_indices = alignment.cue_word_indices.get(left.index, [])
        right_indices = alignment.cue_word_indices.get(right.index, [])
        if not left_indices or not right_indices:
            continue
        lower = profile.snap_ceil(max(words[index].end for index in left_indices) * 1000)
        upper = profile.snap_floor(min(words[index].start for index in right_indices) * 1000)
        if lower > upper:
            continue
        boundary = max(lower, min(right.start_ms, upper))
        if left.start_ms < boundary < right.end_ms:
            result[position] = left.with_timing(left.start_ms, min(left.end_ms, boundary))
            result[position + 1] = right.with_timing(max(right.start_ms, boundary), right.end_ms)
    return result


def extend_generated_cues_into_silence(
    cues: list[Cue], words: list[Word], alignment: AlignmentResult,
    regions: list[SpeechRegion], profile: StyleProfile, constraints: GenerationConstraints,
    boundary: BoundaryRefinementConfig, *, media_duration_ms: int | None,
) -> tuple[list[Cue], list[QCFlag], dict[int, int]]:
    """Apply reading-time targets only where the speech detector proves room.

    Starts and word ownership never change. A continuous speech/music region,
    missing VAD, the next word/cue, and the media end all prevent extension.
    The returned original endpoints let acoustic QC ignore only these verified
    display tails, while the exported duration is still checked by style QC.
    """
    if not regions or boundary.min_duration_policy == "acoustic":
        return cues, [], {}
    region_index = SpeechRegionIndex(regions)
    region_starts = [region.start for region in region_index.regions]
    word_starts = sorted(word.start for word in words)
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    result: list[Cue] = []
    flags: list[QCFlag] = []
    original_ends: dict[int, int] = {}
    for position, cue in enumerate(ordered):
        target_ms = profile.snap_ceil(cue.start_ms + min(
            constraints.max_cue_duration_seconds,
            max(profile.min_cue_dur, display_width(cue.plain_text) / constraints.max_cps),
        ) * 1000)
        indices = alignment.cue_word_indices.get(cue.index, [])
        if target_ms <= cue.end_ms or not indices:
            result.append(cue)
            continue
        last_word = max((words[index] for index in indices), key=lambda word: word.end)
        overlapping = region_index.overlapping(last_word.start, last_word.end)
        if not overlapping:
            result.append(cue)
            continue
        end_region = overlapping[-1]
        acoustic_end = _acoustic_end_seconds(last_word, end_region, boundary, word_starts)
        # A burst still sounding beyond this word is not verified silence.
        if end_region.end > acoustic_end + 0.01:
            result.append(cue)
            continue
        quiet_until = _quiet_until_seconds(
            acoustic_end, end_region, region_index.regions, region_starts, word_starts,
        )
        cap_ms = target_ms
        if isfinite(quiet_until):
            cap_ms = min(cap_ms, profile.snap_floor(quiet_until * 1000))
        if position + 1 < len(ordered):
            cap_ms = min(cap_ms, ordered[position + 1].start_ms)
        if media_duration_ms is not None:
            cap_ms = min(cap_ms, profile.snap_floor(media_duration_ms))
        end_ms = max(cue.end_ms, cap_ms)
        updated = cue.with_timing(cue.start_ms, end_ms)
        result.append(updated)
        if end_ms == cue.end_ms:
            continue
        original_ends[cue.index] = cue.end_ms
        flags.append(QCFlag(
            kind="cps_duration_extended", cue_ids=[cue.index], severity="info",
            message="Cue end extended into verified silence to meet the selected duration and reading-speed targets.",
            old_text=f"{cue.start_ms / 1000:.3f} --> {cue.end_ms / 1000:.3f}",
            new_text=f"{updated.start_ms / 1000:.3f} --> {updated.end_ms / 1000:.3f}",
            start=updated.start_ms / 1000, end=updated.end_ms / 1000,
        ))
    by_id = {cue.index: cue for cue in result}
    return [by_id[cue.index] for cue in cues], flags, original_ends
