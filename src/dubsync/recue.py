from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from rapidfuzz import fuzz

from .models import AlignmentResult, Cue, QCFlag, Word
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue, speech_text_for_alignment
from .tokenize import alphanumeric_signature


@dataclass(frozen=True)
class _CueTiming:
    start_ms: int
    spoken_end_ms: int
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
    protected_cue_ids: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    rebuilt: list[Cue] = []
    flags: list[QCFlag] = []
    protected = set(protected_cue_ids or ())
    timings, timing_flags = _cue_timings(
        cues,
        words,
        alignment,
        profile,
        max_word_duration=max_word_duration,
        max_intra_cue_gap=max_intra_cue_gap,
        protected_cue_ids=protected,
    )
    flags.extend(timing_flags)
    held_cue_ids = protected | {
        cue_id for flag in timing_flags if flag.kind == "timing_evidence_held"
        for cue_id in flag.cue_ids
    }
    preserved_cue_ids = held_cue_ids | {
        cue.index for cue in cues
        if cue.index not in timings and profile.drop_policy != "remove"
        and not is_bracketed_screen_text_cue(cue)
    }
    next_start_by_cue = _next_acoustic_start_by_cue(cues, timings, preserved_cue_ids)

    for cue in cues:
        if cue.index in held_cue_ids:
            rebuilt.append(cue)
            continue
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
        preserve_source_timing_ids=set(alignment.unmatched_cue_ids) | preserved_cue_ids,
        next_start_by_cue=next_start_by_cue,
    ), flags


def _cue_timings(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    max_word_duration: float,
    max_intra_cue_gap: float,
    protected_cue_ids: set[int],
) -> tuple[dict[int, _CueTiming], list[QCFlag]]:
    timings: dict[int, _CueTiming] = {}
    flags: list[QCFlag] = []
    for cue in cues:
        if cue.index in protected_cue_ids or is_bracketed_screen_text_cue(cue):
            continue
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
        issue = timing_evidence_issue(cue, matched_words)
        if issue is not None:
            flags.append(QCFlag(
                kind="timing_evidence_held", cue_ids=[cue.index], severity="error",
                message=(
                    f"{issue} Approved dialogue and its input timing were preserved for review; "
                    "no neighboring speech boundaries were borrowed."
                ),
                old_text=cue.text, start=cue.start_ms / 1000, end=cue.end_ms / 1000,
            ))
            continue
        start_ms = max(0, profile.snap_floor(min(word.start for word in matched_words) * 1000 - profile.lead_in_ms))
        spoken_end = max(word.end for word in matched_words) * 1000
        end_ms = profile.snap_ceil(spoken_end + profile.tail_ms)
        min_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
        timings[cue.index] = _CueTiming(
            start_ms=start_ms,
            spoken_end_ms=profile.snap_ceil(spoken_end),
            end_ms=end_ms,
            min_end_ms=min_end_ms,
            speaker_id=_dominant_speaker(matched_words),
        )
    return timings, flags


def timing_evidence_issue(cue: Cue, words: list[Word]) -> str | None:
    """Identify clearly unusable anchors without estimating missing speech.

    These conservative bounds detect placeholder timestamps and sparse lexical
    matches, not ordinary reading-speed problems. Display padding cannot turn
    those anchors into reliable timing for a complete source phrase.
    """
    source_tokens = alphanumeric_signature(speech_text_for_alignment(cue))
    if not source_tokens or not words:
        return None
    if any(
        not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start
        for word in words
    ):
        return "Matched ASR words contain invalid timestamps."
    evidence_tokens = alphanumeric_signature(" ".join(word.text for word in words))
    span = max(word.end for word in words) - min(word.start for word in words)
    short_word_count = sum(word.end - word.start <= 0.020 + 1e-9 for word in words)
    if span <= 0.005 + 1e-9 or (
        len(evidence_tokens) >= 2 and (
            span < 0.080 - 1e-9
            or short_word_count * 2 >= len(words) and span < 0.060 * len(evidence_tokens) - 1e-9
        )
    ):
        return (
            f"Matched ASR timing is collapsed: {len(evidence_tokens)} lexical tokens occupy "
            f"{span * 1000:.1f} ms, including {short_word_count} words of at most 20 ms."
        )
    if len(source_tokens) >= 3:
        supported = _ordered_lexical_support(source_tokens, evidence_tokens)
        if supported * 2 < len(source_tokens):
            return (
                f"Sparse lexical timing evidence covers only {supported}/{len(source_tokens)} "
                "spoken source tokens."
            )
    return None


def _ordered_lexical_support(source: list[str], evidence: list[str]) -> int:
    # Use each timestamped token once. Normalization handles numeric/accent
    # aliases; the aligner's 85% spelling tolerance retains minor name variants.
    previous = [0] * (len(evidence) + 1)
    for token in source:
        current = [0]
        for index, candidate in enumerate(evidence, start=1):
            matches = token == candidate or fuzz.ratio(token, candidate, score_cutoff=85) >= 85
            current.append(max(previous[index], current[-1], previous[index - 1] + int(matches)))
        previous = current
    return previous[-1]


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


def _next_acoustic_start_by_cue(
    cues: list[Cue], timings: dict[int, _CueTiming], preserved_cue_ids: set[int],
) -> dict[int, int]:
    # Insertions are initially placed using source SRT times. Once the source
    # cues are retimed, that list can differ from their actual spoken order.
    # A source timing hold is still a display boundary. Omitting it lets the
    # preceding cue's optional padding extend over the preserved dialogue.
    starts = {
        cue.index: timings[cue.index].start_ms if cue.index in timings else cue.start_ms
        for cue in cues
        if cue.index in timings or (
            cue.index in preserved_cue_ids and not is_bracketed_screen_text_cue(cue)
        )
    }
    timed_cues = sorted(
        (cue for cue in cues if cue.index in starts),
        key=lambda cue: (starts[cue.index], cue.index in preserved_cue_ids),
    )
    return {
        cue.index: starts[following.index]
        for cue, following in zip(timed_cues, timed_cues[1:])
    }


def _extend_into_available_gap(timing: _CueTiming, next_start_ms: int | None, profile: StyleProfile) -> int:
    desired_end_ms = max(timing.end_ms, timing.min_end_ms)
    if next_start_ms is None:
        return desired_end_ms
    cap_ms = next_start_ms if profile.allow_zero_gap else profile.snap_floor(max(0, next_start_ms - 1))
    # A following actor limits optional display padding too. A true word
    # overlap remains intact; readability must never manufacture one.
    return max(timing.spoken_end_ms, min(desired_end_ms, cap_ms))


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
    next_start_by_cue: dict[int, int] | None = None,
) -> list[Cue]:
    if not cues:
        return []
    preserved = preserve_source_timing_ids or set()
    adjusted = list(cues)
    previous_by_speaker: dict[str, Cue] = {}
    # Resolve same-speaker overlaps in acoustic order without changing the
    # source-list order used by later reconciliation and source timing holds.
    for position, cue in sorted(enumerate(cues), key=lambda item: item[1].start_ms):
        if cue.index in preserved or is_bracketed_screen_text_cue(cue):
            continue
        speaker_key = _speaker_key(cue.speaker_id)
        previous = previous_by_speaker.get(speaker_key)
        if previous is not None and cue.start_ms < previous.end_ms:
            start_ms = previous.end_ms if profile.allow_zero_gap else profile.snap_ceil(previous.end_ms + 1)
            end_ms = max(cue.end_ms, profile.snap_ceil(start_ms + profile.min_cue_dur * 1000))
            following_start_ms = (next_start_by_cue or {}).get(cue.index)
            if following_start_ms is not None:
                cap_ms = (
                    following_start_ms
                    if profile.allow_zero_gap
                    else profile.snap_floor(max(0, following_start_ms - 1))
                )
                end_ms = max(cue.end_ms, min(end_ms, cap_ms))
            # Conflicting same-speaker anchors may leave no interval before
            # another actor. Keep that real overlap reviewable instead of
            # moving the cue beyond its evidence or inventing more padding.
            next_cue = cue.with_timing(start_ms, end_ms) if end_ms > start_ms else cue
        else:
            next_cue = cue
        adjusted[position] = next_cue
        if previous is None or next_cue.end_ms >= previous.end_ms:
            previous_by_speaker[speaker_key] = next_cue
    return adjusted


def _speaker_key(speaker_id: str | None) -> str:
    return speaker_id or "__unknown__"
