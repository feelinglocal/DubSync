from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import AlignmentResult, Cue, QCFlag, Word
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue


@dataclass(frozen=True)
class _CueTiming:
    start_ms: int
    end_ms: int
    min_end_ms: int
    speaker_id: str | None


def rebuild_cues(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    max_word_duration: float = 2.0,
    max_intra_cue_gap: float = 1.5,
) -> tuple[list[Cue], list[QCFlag]]:
    rebuilt: list[Cue] = []
    flags: list[QCFlag] = []
    timings, timing_flags = _cue_timings(
        cues,
        words,
        alignment,
        profile,
        max_word_duration=max_word_duration,
        max_intra_cue_gap=max_intra_cue_gap,
    )
    flags.extend(timing_flags)
    next_start_by_cue = _next_start_by_same_speaker(cues, timings)

    for cue in cues:
        timing = timings.get(cue.index)
        if timing is None:
            should_remove = profile.drop_policy == "remove"
            if is_bracketed_screen_text_cue(cue):
                rebuilt.append(cue)
                continue
            if not should_remove:
                rebuilt.append(cue)
            flags.append(
                QCFlag(
                    kind="dropped_unmatched_cue" if should_remove else "unmatched_cue",
                    cue_ids=[cue.index],
                    message=(
                        "No ASR word timestamps matched this cue; removed by drop_policy."
                        if should_remove
                        else "No ASR word timestamps matched this cue."
                    ),
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        end_ms = _extend_into_available_gap(timing, next_start_by_cue.get(cue.index), profile)
        rebuilt.append(cue.with_timing(timing.start_ms, end_ms).model_copy(update={"speaker_id": timing.speaker_id}))

    return _enforce_monotonic(
        rebuilt,
        profile,
        preserve_source_timing_ids=set(alignment.unmatched_cue_ids),
    ), flags


def _cue_timings(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    max_word_duration: float,
    max_intra_cue_gap: float,
) -> tuple[dict[int, _CueTiming], list[QCFlag]]:
    timings: dict[int, _CueTiming] = {}
    flags: list[QCFlag] = []
    for cue in cues:
        word_indices = alignment.cue_word_indices.get(cue.index, [])
        if not word_indices:
            continue
        matched_words = [words[index] for index in word_indices]
        selected_words, trimmed = _largest_dense_cluster(
            matched_words,
            max_word_duration=max_word_duration,
            max_intra_cue_gap=max_intra_cue_gap,
        )
        if trimmed:
            flags.append(
                QCFlag(
                    kind="timing_outlier_trimmed",
                    cue_ids=[cue.index],
                    message="Cue timing ignored an impossible ASR word duration or intra-cue gap.",
                    old_text=_word_span_label(matched_words),
                    new_text=_word_span_label(selected_words),
                    start=selected_words[0].start if selected_words else None,
                    end=selected_words[-1].end if selected_words else None,
                )
            )
        matched_words = selected_words
        start_ms = max(0, profile.snap_floor(min(word.start for word in matched_words) * 1000 - profile.lead_in_ms))
        end_ms = profile.snap_ceil(max(word.end for word in matched_words) * 1000 + profile.tail_ms)
        min_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
        timings[cue.index] = _CueTiming(
            start_ms=start_ms,
            end_ms=end_ms,
            min_end_ms=min_end_ms,
            speaker_id=_dominant_speaker(matched_words),
        )
    return timings, flags


def _largest_dense_cluster(
    words: list[Word],
    *,
    max_word_duration: float,
    max_intra_cue_gap: float,
) -> tuple[list[Word], bool]:
    if len(words) <= 1:
        return words, False
    sorted_words = sorted(words, key=lambda word: (word.start, word.end))
    clusters: list[list[Word]] = []
    current: list[Word] = []
    previous: Word | None = None
    for word in sorted_words:
        word_is_outlier = _word_duration(word) > max_word_duration
        starts_new_cluster = False
        if previous is not None:
            starts_new_cluster = word.start - previous.end > max_intra_cue_gap or _word_duration(previous) > max_word_duration
        if word_is_outlier and current:
            clusters.append(current)
            current = []
        if starts_new_cluster and current:
            clusters.append(current)
            current = []
        current.append(word)
        if word_is_outlier:
            clusters.append(current)
            current = []
        previous = word
    if current:
        clusters.append(current)
    if len(clusters) <= 1:
        return sorted_words, False
    selected = max(clusters, key=lambda cluster: _cluster_score(cluster, max_word_duration))
    return selected, len(selected) != len(sorted_words)


def _cluster_score(words: list[Word], max_word_duration: float) -> tuple[int, int, float]:
    normal_count = sum(1 for word in words if _word_duration(word) <= max_word_duration)
    span = max(word.end for word in words) - min(word.start for word in words)
    return normal_count, len(words), -span


def _word_duration(word: Word) -> float:
    return word.end - word.start


def _word_span_label(words: list[Word]) -> str:
    if not words:
        return ""
    return " ".join(f"{word.text}({word.start:.3f}-{word.end:.3f})" for word in words)


def _next_start_by_same_speaker(cues: list[Cue], timings: dict[int, _CueTiming]) -> dict[int, int]:
    next_by_cue: dict[int, int] = {}
    next_start_by_speaker: dict[str, int] = {}
    for cue in reversed(cues):
        timing = timings.get(cue.index)
        if timing is None:
            continue
        speaker_key = _speaker_key(timing.speaker_id)
        if speaker_key in next_start_by_speaker:
            next_by_cue[cue.index] = next_start_by_speaker[speaker_key]
        next_start_by_speaker[speaker_key] = timing.start_ms
    return next_by_cue


def _extend_into_available_gap(timing: _CueTiming, next_start_ms: int | None, profile: StyleProfile) -> int:
    if timing.end_ms >= timing.min_end_ms:
        return timing.end_ms
    if next_start_ms is None or timing.end_ms > next_start_ms:
        return timing.min_end_ms
    cap_ms = next_start_ms if profile.allow_zero_gap else profile.snap_floor(max(0, next_start_ms - 1))
    return min(timing.min_end_ms, max(timing.end_ms, cap_ms))


def _dominant_speaker(words: list[Word]) -> str | None:
    speakers = [word.speaker_id for word in words if word.speaker_id]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


def _enforce_monotonic(
    cues: list[Cue],
    profile: StyleProfile,
    *,
    preserve_source_timing_ids: set[int] | None = None,
) -> list[Cue]:
    if not cues:
        return []
    preserved = preserve_source_timing_ids or set()
    adjusted: list[Cue] = []
    previous_by_speaker: dict[str, Cue] = {}
    for cue in cues:
        if cue.index in preserved or is_bracketed_screen_text_cue(cue):
            adjusted.append(cue)
            continue
        speaker_key = _speaker_key(cue.speaker_id)
        previous = previous_by_speaker.get(speaker_key)
        if previous is not None and cue.start_ms < previous.end_ms:
            start_ms = previous.end_ms if profile.allow_zero_gap else profile.snap_ceil(previous.end_ms + 1)
            end_ms = max(cue.end_ms, profile.snap_ceil(start_ms + profile.min_cue_dur * 1000))
            next_cue = cue.with_timing(start_ms, end_ms)
        else:
            next_cue = cue
        adjusted.append(next_cue)
        previous_by_speaker[speaker_key] = next_cue
    return adjusted


def _speaker_key(speaker_id: str | None) -> str:
    return speaker_id or "__unknown__"
