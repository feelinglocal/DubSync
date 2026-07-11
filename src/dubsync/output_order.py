from __future__ import annotations

from rapidfuzz import fuzz

from .models import Cue, QCFlag
from .style_profile import StyleProfile
from .text_metrics import display_width
from .tokenize import alphanumeric_signature


def finalize_cues_for_output(
    cues: list[Cue],
    profile: StyleProfile,
    *,
    no_overlaps: bool = True,
    max_cps: float | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    merged, flags = _merge_duplicate_overlaps(ordered)
    if max_cps is not None:
        merged, readability_flags = _extend_fast_cues_into_following_gap(merged, profile, max_cps)
        flags.extend(readability_flags)
        merged, merge_flags = _merge_fast_cues_with_following(merged, profile, max_cps)
        flags.extend(merge_flags)
    finalized = merged
    if no_overlaps:
        finalized, overlap_flags = _resolve_residual_overlaps(merged, profile)
        flags.extend(overlap_flags)
    _assert_monotonic_starts(finalized)
    return finalized, flags


def _extend_fast_cues_into_following_gap(
    cues: list[Cue],
    profile: StyleProfile,
    max_cps: float,
) -> tuple[list[Cue], list[QCFlag]]:
    if max_cps <= 0 or not cues:
        return cues, []
    adjusted = list(cues)
    flags: list[QCFlag] = []
    for index in range(len(adjusted)):
        cue = adjusted[index]
        needed_end_ms = _end_for_cps(cue, profile, max_cps)
        if needed_end_ms <= cue.end_ms:
            continue
        if index + 1 >= len(adjusted):
            next_cue = cue.with_timing(cue.start_ms, needed_end_ms)
            adjusted[index] = next_cue
            flags.append(_cps_extension_flag(cue, next_cue))
            continue
        following = adjusted[index + 1]
        if needed_end_ms <= following.start_ms:
            next_cue = cue.with_timing(cue.start_ms, needed_end_ms)
            adjusted[index] = next_cue
            flags.append(_cps_extension_flag(cue, next_cue))
            continue
    return adjusted, flags


def _merge_fast_cues_with_following(
    cues: list[Cue],
    profile: StyleProfile,
    max_cps: float,
) -> tuple[list[Cue], list[QCFlag]]:
    merged: list[Cue] = []
    flags: list[QCFlag] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        if _cue_cps(current) <= max_cps or index + 1 >= len(cues):
            merged.append(current)
            index += 1
            continue
        following = cues[index + 1]
        if _known_different_speakers(current, following):
            merged.append(current)
            index += 1
            continue
        candidate = Cue(
            index=current.index,
            start_ms=current.start_ms,
            end_ms=max(current.end_ms, following.end_ms),
            lines=[current.plain_text, following.plain_text],
            speaker_id=current.speaker_id if current.speaker_id == following.speaker_id else current.speaker_id or following.speaker_id,
            character=current.character if current.character == following.character else current.character or following.character,
        )
        if (
            len(candidate.lines) <= profile.max_lines_per_cue
            and all(display_width(line) <= profile.max_chars_per_line for line in candidate.lines)
            and _cue_cps(candidate) <= max_cps
        ):
            merged.append(candidate)
            flags.append(
                QCFlag(
                    kind="cps_cue_merged",
                    cue_ids=[current.index, following.index],
                    message="Adjacent cues were merged to satisfy timing.max_cps without creating overlaps.",
                    old_text=f"{current.text}\n{following.text}",
                    new_text=candidate.text,
                    start=candidate.start_ms / 1000.0,
                    end=candidate.end_ms / 1000.0,
                )
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged, flags


def _end_for_cps(cue: Cue, profile: StyleProfile, max_cps: float) -> int:
    width = display_width(cue.plain_text)
    if width <= 0:
        return cue.end_ms
    needed_duration_ms = width / max_cps * 1000
    end_ms = profile.snap_ceil(cue.start_ms + needed_duration_ms)
    while _cps_for_width(width, cue.start_ms, end_ms) > max_cps:
        next_end_ms = profile.snap_ceil(end_ms + 1)
        end_ms = next_end_ms if next_end_ms > end_ms else end_ms + 1
    return end_ms


def _cps_for_width(width: int, start_ms: int, end_ms: int) -> float:
    duration_seconds = (end_ms - start_ms) / 1000.0
    if duration_seconds <= 0:
        return float("inf")
    return width / duration_seconds


def _cps_extension_flag(old_cue: Cue, new_cue: Cue) -> QCFlag:
    return QCFlag(
        kind="cps_duration_extended",
        cue_ids=[old_cue.index],
        message="Cue display duration was extended into the following display gap to stay within timing.max_cps.",
        old_text=f"{old_cue.start_ms / 1000.0:.3f} --> {old_cue.end_ms / 1000.0:.3f}",
        new_text=f"{new_cue.start_ms / 1000.0:.3f} --> {new_cue.end_ms / 1000.0:.3f}",
        start=new_cue.start_ms / 1000.0,
        end=new_cue.end_ms / 1000.0,
    )


def _cue_cps(cue: Cue) -> float:
    if cue.duration_ms <= 0:
        return 0.0
    return display_width(cue.plain_text) / (cue.duration_ms / 1000.0)


def _merge_duplicate_overlaps(cues: list[Cue]) -> tuple[list[Cue], list[QCFlag]]:
    merged: list[Cue] = []
    flags: list[QCFlag] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        duplicate_ids = [current.index]
        old_texts = [current.text]
        cursor = index + 1
        while cursor < len(cues) and _is_duplicate_overlap(current, cues[cursor]):
            duplicate = cues[cursor]
            duplicate_ids.append(duplicate.index)
            old_texts.append(duplicate.text)
            current = _merged_duplicate_cue(current, duplicate)
            cursor += 1

        merged.append(current)
        if len(duplicate_ids) > 1:
            flags.append(
                QCFlag(
                    kind="duplicate_cue_merged",
                    cue_ids=duplicate_ids,
                    message="Duplicate overlapping cue text was merged into one output cue.",
                    old_text="\n\n".join(old_texts),
                    new_text=current.text,
                    start=current.start_ms / 1000.0,
                    end=current.end_ms / 1000.0,
                )
            )
        index = cursor
    return merged, flags


def _is_duplicate_overlap(left: Cue, right: Cue) -> bool:
    if right.start_ms >= left.end_ms:
        return False
    left_signature = " ".join(alphanumeric_signature(left.plain_text))
    right_signature = " ".join(alphanumeric_signature(right.plain_text))
    if not left_signature or not right_signature:
        return False
    if left_signature == right_signature:
        return True
    return fuzz.ratio(left_signature, right_signature) >= 90


def _merged_duplicate_cue(left: Cue, right: Cue) -> Cue:
    preferred = _preferred_duplicate_text(left, right)
    return preferred.model_copy(
        update={
            "index": min(left.index, right.index),
            "start_ms": min(left.start_ms, right.start_ms),
            "end_ms": max(left.end_ms, right.end_ms),
            "speaker_id": left.speaker_id if left.speaker_id == right.speaker_id else left.speaker_id or right.speaker_id,
            "character": left.character if left.character == right.character else left.character or right.character,
        }
    )


def _preferred_duplicate_text(left: Cue, right: Cue) -> Cue:
    if right.duration_ms > left.duration_ms:
        return right
    if len(right.plain_text) > len(left.plain_text):
        return right
    return left


def _resolve_residual_overlaps(cues: list[Cue], profile: StyleProfile) -> tuple[list[Cue], list[QCFlag]]:
    adjusted: list[Cue] = []
    flags: list[QCFlag] = []
    min_duration_ms = int(profile.min_cue_dur * 1000)
    for cue in cues:
        if adjusted and _needs_final_timing_separation(adjusted[-1], cue):
            previous = adjusted[-1]
            was_overlap = cue.start_ms < previous.end_ms
            start_ms = _separated_start_ms(previous, cue, profile)
            end_ms = max(cue.end_ms, profile.snap_ceil(start_ms + min_duration_ms))
            next_cue = cue.with_timing(start_ms, end_ms)
            flags.append(
                QCFlag(
                    kind="output_overlap_resolved" if was_overlap else "speaker_transition_gap_inserted",
                    cue_ids=[previous.index, cue.index],
                    message=(
                        "Residual output overlap was resolved during final ordering."
                        if was_overlap
                        else "A visible frame gap was inserted between different detected speakers."
                    ),
                    old_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
                    new_text=f"{next_cue.start_ms / 1000.0:.3f} --> {next_cue.end_ms / 1000.0:.3f}",
                    start=next_cue.start_ms / 1000.0,
                    end=next_cue.end_ms / 1000.0,
                )
            )
            adjusted.append(next_cue)
            continue
        adjusted.append(cue)
    return adjusted, flags


def _needs_final_timing_separation(previous: Cue, cue: Cue) -> bool:
    if cue.start_ms < previous.end_ms:
        return True
    return cue.start_ms == previous.end_ms and _known_different_speakers(previous, cue)


def _separated_start_ms(previous: Cue, cue: Cue, profile: StyleProfile) -> int:
    if profile.allow_zero_gap and not _known_different_speakers(previous, cue):
        return previous.end_ms
    return profile.snap_ceil(previous.end_ms + 1)


def _known_different_speakers(left: Cue, right: Cue) -> bool:
    if left.speaker_id and right.speaker_id and left.speaker_id != right.speaker_id:
        return True
    return bool(left.character and right.character and left.character != right.character)


def _assert_monotonic_starts(cues: list[Cue]) -> None:
    for left, right in zip(cues, cues[1:]):
        if right.start_ms < left.start_ms:
            raise ValueError("final cue ordering is not monotonic by start time")
