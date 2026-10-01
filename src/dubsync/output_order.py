from __future__ import annotations

from math import ceil

from .models import Cue, QCFlag
from .speaker_evidence import speakers_known_different
from .srt_io import validate_cue_timings_for_export
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue
from .text_metrics import display_width
from .tokenize import alphanumeric_signature


def finalize_cues_for_output(
    cues: list[Cue],
    profile: StyleProfile,
    *,
    no_overlaps: bool = True,
    max_cps: float | None = None,
    max_cue_duration_seconds: float | None = None,
    protected_cue_ids: set[int] | None = None,
    fixed_cue_ids: set[int] | None = None,
    preserve_timing: bool = False,
    media_duration_ms: int | None = None,
    merge_duplicates: bool = True,
    spoken_spans: dict[int, tuple[int, int]] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    """Finalize display order without replacing acoustic evidence with reading-time guesses.

    With ``preserve_timing``, physical speech overlaps remain visible review errors;
    reading speed and speaker gaps never shift the spoken boundaries. Source holds
    and screen annotations are kept verbatim, including outside the media.
    Audio generation disables ``merge_duplicates`` because each word occurrence
    has known ownership even when repeated utterances snap to the same onset.

    ``spoken_spans`` (first-word onset, last-word offset in ms per cue) lets
    ``no_overlaps`` separate cues that only collide through display padding,
    frame snapping or an unsynchronized source hold; a hold may then be clipped
    at its acoustic neighbour (``_resolve_overlaps_with_speech_evidence``).
    Without it nothing is moved.

    ``fixed_cue_ids`` retains unresolved word-ambiguity intervals even when
    ordinary source holds could be clipped at a neighbor. Any overlap remains
    an uncertainty finding, not a claim of simultaneous speech.
    """
    if media_duration_ms is not None and media_duration_ms < 0:
        raise ValueError("media_duration_ms must be non-negative")
    # Validate before readability extension or merging can conceal an invalid
    # upstream boundary. Only the creating stage has evidence to repair it.
    validate_cue_timings_for_export(cues)
    fixed = fixed_cue_ids or set()
    protected = (protected_cue_ids or set()) | fixed
    untouched_cues = [
        cue
        for cue in cues
        if cue.index in protected or is_bracketed_screen_text_cue(cue)
    ]
    dialogue_cues = [
        cue
        for cue in cues
        if cue.index not in protected and not is_bracketed_screen_text_cue(cue)
    ]
    flags = _source_order_inversion_flags(dialogue_cues)
    ordered = sorted(dialogue_cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    merged, merge_flags = _merge_duplicate_overlaps(ordered) if merge_duplicates else (ordered, [])
    flags.extend(merge_flags)
    if max_cps is not None and not preserve_timing:
        merged, readability_flags = _extend_fast_cues_into_following_gap(
            merged,
            profile,
            max_cps,
            max_cue_duration_seconds,
        )
        flags.extend(readability_flags)
        merged, merge_flags = _merge_fast_cues_with_following(
            merged,
            profile,
            max_cps,
            max_cue_duration_seconds,
        )
        flags.extend(merge_flags)
    finalized = merged
    if no_overlaps and not preserve_timing:
        finalized, overlap_flags = _resolve_residual_overlaps(merged, profile)
        flags.extend(overlap_flags)
    if media_duration_ms is not None:
        finalized, boundary_flags = _cap_cues_at_media_end(finalized, media_duration_ms)
        flags.extend(boundary_flags)
        flags.extend(
            _outside_media_flag(cue, media_duration_ms)
            for cue in untouched_cues
            if cue.end_ms > media_duration_ms
        )
    combined = sorted(
        [*finalized, *untouched_cues],
        key=lambda cue: (cue.start_ms, cue.end_ms, cue.index),
    )
    if no_overlaps and preserve_timing and spoken_spans is not None:
        combined = _resolve_overlaps_with_speech_evidence(combined, profile, protected, spoken_spans, fixed)
    remaining_overlaps = (
        _unresolved_acoustic_overlap_flags([cue for cue in combined if not is_bracketed_screen_text_cue(cue)])
        if no_overlaps
        else []
    )
    for flag in remaining_overlaps:
        if preserve_timing:
            # One finding per remaining pair. A pair that involves a source
            # hold says so instead of claiming both cues are acoustically timed.
            flags.append(
                flag.model_copy(update={"message": _AMBIGUOUS_OVERLAP_MESSAGE})
                if fixed.intersection(flag.cue_ids)
                else
                flag.model_copy(update={"message": _HELD_OVERLAP_MESSAGE})
                if protected.intersection(flag.cue_ids)
                else flag
            )
        elif protected.intersection(flag.cue_ids):
            # Source holds are excluded from retiming, but overlaps with them
            # still need a visible advisory even when other dialogue is adjusted.
            flags.append(flag.model_copy(update={
                "kind": "output_overlap_preserved",
                "severity": "warning",
                "message": "Uncertain source timing was preserved; this overlap needs review.",
            }))
    _assert_monotonic_starts(combined)
    validate_cue_timings_for_export(combined)
    return combined, flags


_HELD_OVERLAP_MESSAGE = (
    "A cue kept at its source timing overlaps its neighbour and could not be separated "
    "without hiding speech; review the held cue's timing."
)
_AMBIGUOUS_OVERLAP_MESSAGE = (
    "An ASR word overlaps multiple possible speech bursts. Its retained cue interval overlaps "
    "a neighbor; review the uncertain timing before selecting a speech boundary."
)
# A cue without word timing is only clipped while at least this many frames,
# and half of its duration, remain; anything less is left for review.
_MIN_CLIPPED_HOLD_FRAMES = 3
_MAX_OVERLAP_PASSES = 8


def _resolve_overlaps_with_speech_evidence(
    cues: list[Cue],
    profile: StyleProfile,
    protected: set[int],
    spoken_spans: dict[int, tuple[int, int]],
    fixed_cue_ids: set[int] | None = None,
) -> list[Cue]:
    """Separate overlapping cues without delaying or hiding anyone's speech.

    ``spoken_spans`` holds the first-word onset and last-word offset of every
    cue that owns timed words. An overlapping pair gets one shared boundary:

    * a cue timed from its words never starts later, so the boundary is its
      start and the earlier cue's end is trimmed to it, provided that removes
      only display padding or a frame-snap margin and not the earlier cue's
      last word;
    * a later cue kept at source timing (a hold or an unmatched cue) instead
      starts where the earlier cue ends, as long as its own first word, when
      known, is not cut;
    * a cue without any word timing is clipped only while most of it remains;
    * otherwise the words really overlap: simultaneous speech stays as it is.

    Screen-text annotations are not dialogue and are left alone.
    """
    snap_slack_ms = ceil(profile.frame_ms)
    min_hold_ms = ceil(profile.frame_ms * _MIN_CLIPPED_HOLD_FRAMES)

    def acoustic(cue: Cue) -> bool:
        return cue.index not in protected and cue.index in spoken_spans

    def kept_ms(cue: Cue) -> float:
        return max(min_hold_ms, cue.duration_ms / 2)

    def boundary_ms(earlier: Cue, later: Cue) -> int | None:
        earlier_span = spoken_spans.get(earlier.index)
        later_span = spoken_spans.get(later.index)
        if earlier_span is None and later_span is None:
            return None
        earliest = earlier_span[1] - snap_slack_ms if earlier_span is not None else earlier.start_ms + kept_ms(earlier)
        if not acoustic(earlier):
            # A hold can own words spoken outside the interval it is shown in;
            # with or without known words it is never clipped to a sliver.
            earliest = max(earliest, earlier.start_ms + kept_ms(earlier))
        if acoustic(later):
            boundary = later.start_ms
        else:
            latest = later.end_ms - kept_ms(later)
            if later_span is not None:
                latest = min(latest, later_span[0] + snap_slack_ms)
            boundary = earlier.end_ms if earlier.end_ms <= latest else profile.snap_floor(latest)
            boundary = max(boundary, later.start_ms)
        if boundary < earliest or not earlier.start_ms < boundary < later.end_ms:
            return None
        if (earlier.index in (fixed_cue_ids or set()) and boundary < earlier.end_ms) or (
            later.index in (fixed_cue_ids or set()) and boundary > later.start_ms
        ):
            return None
        return boundary

    ordered = list(cues)
    # Delaying a held cue can make it meet the following cue, so the pass is
    # repeated; every change shortens a cue, and a few passes settle real data.
    for _ in range(_MAX_OVERLAP_PASSES):
        changed = False
        resolved: list[Cue] = []
        on_screen: list[int] = []
        for cue in ordered:
            if is_bracketed_screen_text_cue(cue):
                resolved.append(cue)
                continue
            listed_start_ms = cue.start_ms
            still_on_screen: list[int] = []
            for position in on_screen:
                earlier = resolved[position]
                if earlier.end_ms > cue.start_ms:
                    boundary = boundary_ms(earlier, cue)
                    if boundary is not None:
                        resolved[position] = earlier.with_timing(earlier.start_ms, min(earlier.end_ms, boundary))
                        cue = cue.with_timing(max(cue.start_ms, boundary), cue.end_ms)
                        changed = True
                # Later cues are listed by start, so an earlier cue matters to
                # them only while it outlasts this cue's listed start.
                if resolved[position].end_ms > listed_start_ms:
                    still_on_screen.append(position)
            on_screen = [*still_on_screen, len(resolved)]
            resolved.append(cue)
        ordered = sorted(resolved, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
        if not changed:
            break
    return ordered


def _unresolved_acoustic_overlap_flags(cues: list[Cue]) -> list[QCFlag]:
    flags: list[QCFlag] = []
    latest_ending: Cue | None = None
    for cue in cues:
        if latest_ending is not None and cue.start_ms < latest_ending.end_ms:
            flags.append(
                QCFlag(
                    kind="output_overlap_unresolved",
                    cue_ids=[latest_ending.index, cue.index],
                    message=(
                        "Acoustically timed cues overlap. Their speech boundaries were preserved; "
                        "review the simultaneous dialogue instead of delaying an utterance."
                    ),
                    severity="error",
                    start=cue.start_ms / 1000.0,
                    end=min(latest_ending.end_ms, cue.end_ms) / 1000.0,
                )
            )
        if latest_ending is None or cue.end_ms > latest_ending.end_ms:
            latest_ending = cue
    return flags


def _cap_cues_at_media_end(cues: list[Cue], media_duration_ms: int) -> tuple[list[Cue], list[QCFlag]]:
    bounded: list[Cue] = []
    flags: list[QCFlag] = []
    for cue in cues:
        if cue.end_ms <= media_duration_ms:
            bounded.append(cue)
            continue
        if cue.start_ms >= media_duration_ms:
            # No supported timestamp exists: retain reviewable text instead of
            # silently dropping it or inventing speech inside the recording.
            bounded.append(cue)
            flags.append(_outside_media_flag(cue, media_duration_ms))
            continue
        # EOF may lie between video frames. Flooring it could cut the last
        # phoneme; the real recording boundary takes precedence over the grid.
        updated = cue.with_timing(cue.start_ms, media_duration_ms)
        bounded.append(updated)
        flags.append(
            QCFlag(
                kind="media_boundary_clamped",
                cue_ids=[cue.index],
                message="Cue display tail exceeded the recording and was capped at the media boundary.",
                old_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
                new_text=f"{updated.start_ms / 1000.0:.3f} --> {updated.end_ms / 1000.0:.3f}",
                start=updated.start_ms / 1000.0,
                end=updated.end_ms / 1000.0,
            )
        )
    return bounded, flags


def _outside_media_flag(cue: Cue, media_duration_ms: int) -> QCFlag:
    return QCFlag(
        kind="cue_outside_media",
        cue_ids=[cue.index],
        message=(
            f"Cue extends outside the {media_duration_ms / 1000.0:.3f}s recording. "
            "Its text and timing were retained for review because no safe acoustic correction is available."
        ),
        severity="error",
        old_text=cue.text,
        start=cue.start_ms / 1000.0,
        end=cue.end_ms / 1000.0,
    )


def source_order_inversion_flags(
    cues: list[Cue],
    *,
    source_cue_ids: set[int],
    protected_cue_ids: set[int] | None = None,
) -> list[QCFlag]:
    """Capture source narrative conflicts before acoustic sorting erases order.

    Generated insertions and newly split speaker children have acoustic order,
    not independent source positions. Source holds and screen annotations do
    not claim an acoustic boundary and are excluded as in final ordering.
    """
    protected = protected_cue_ids or set()
    return _source_order_inversion_flags([
        cue for cue in cues
        if cue.index in source_cue_ids and cue.index not in protected
        and not is_bracketed_screen_text_cue(cue)
    ])


def _source_order_inversion_flags(cues: list[Cue]) -> list[QCFlag]:
    flags: list[QCFlag] = []
    for left, right in zip(cues, cues[1:]):
        if right.start_ms >= left.start_ms:
            continue
        flags.append(
            QCFlag(
                kind="output_order_inversion",
                cue_ids=[left.index, right.index],
                message=(
                    "Source cue order conflicts with chronological timing; final time sorting would "
                    "reverse this narrative sequence. Acoustic timing review is required."
                ),
                severity="error",
                old_text=f"{left.index}: {left.text}\n{right.index}: {right.text}",
                start=min(left.start_ms, right.start_ms) / 1000.0,
                end=max(left.end_ms, right.end_ms) / 1000.0,
            )
        )
    return flags


def _extend_fast_cues_into_following_gap(
    cues: list[Cue],
    profile: StyleProfile,
    max_cps: float,
    max_cue_duration_seconds: float | None,
) -> tuple[list[Cue], list[QCFlag]]:
    if max_cps <= 0 or not cues:
        return cues, []
    adjusted = list(cues)
    flags: list[QCFlag] = []
    for index in range(len(adjusted)):
        cue = adjusted[index]
        needed_end_ms = _end_for_cps(cue, profile, max_cps, max_cue_duration_seconds)
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
    max_cue_duration_seconds: float | None,
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
            and (
                max_cue_duration_seconds is None
                or candidate.duration_ms <= max_cue_duration_seconds * 1000
            )
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


def _end_for_cps(
    cue: Cue,
    profile: StyleProfile,
    max_cps: float,
    max_cue_duration_seconds: float | None,
) -> int:
    width = display_width(cue.plain_text)
    if width <= 0:
        return cue.end_ms
    needed_duration_ms = width / max_cps * 1000
    end_ms = profile.snap_ceil(cue.start_ms + needed_duration_ms)
    while _cps_for_width(width, cue.start_ms, end_ms) > max_cps:
        next_end_ms = profile.snap_ceil(end_ms + 1)
        end_ms = next_end_ms if next_end_ms > end_ms else end_ms + 1
    if max_cue_duration_seconds is not None:
        duration_cap_end_ms = max(
            cue.start_ms + 1,
            profile.snap_floor(cue.start_ms + max_cue_duration_seconds * 1000),
        )
        end_ms = min(end_ms, duration_cap_end_ms)
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
    if (
        right.start_ms >= left.end_ms
        or left.start_ms != right.start_ms
        or _known_different_speakers(left, right)
    ):
        return False
    left_signature = " ".join(alphanumeric_signature(left.plain_text))
    right_signature = " ".join(alphanumeric_signature(right.plain_text))
    if not left_signature or not right_signature:
        return False
    # Similar words may reverse meaning ("can" / "can't"); separate onsets may
    # be an intentional repetition. Only exact words at the same onset dedupe.
    return left_signature == right_signature


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
    if speakers_known_different(left.speaker_id, right.speaker_id):
        return True
    return bool(left.character and right.character and left.character != right.character)


def _assert_monotonic_starts(cues: list[Cue]) -> None:
    for left, right in zip(cues, cues[1:]):
        if right.start_ms < left.start_ms:
            raise ValueError("final cue ordering is not monotonic by start time")
