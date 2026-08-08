from __future__ import annotations

from collections import Counter

from .models import AlignmentResult, Cue, QCFlag, Word
from .style_profile import StyleProfile
from .text_metrics import contains_character_level_script, wrap_visual_width


def group_words_for_cues(
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> list[list[Word]]:
    groups = group_word_indices_for_cues(
        words,
        list(range(len(words))),
        profile,
        max_gap_seconds=max_gap_seconds,
        max_cue_duration_seconds=max_cue_duration_seconds,
    )
    return [[words[index] for index in group] for group in groups]


def group_word_indices_for_cues(
    words: list[Word],
    word_indices: list[int],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> list[list[int]]:
    ordered = sorted(
        {
            index
            for index in word_indices
            if 0 <= index < len(words)
            and words[index].text.strip()
            and words[index].end >= words[index].start
            and words[index].start >= 0
        },
        key=lambda index: (words[index].start, words[index].end, index),
    )
    groups: list[list[int]] = []
    current: list[int] = []
    for word_index in ordered:
        if current and _starts_new_cue(
            words,
            current,
            word_index,
            profile,
            max_gap_seconds=max_gap_seconds,
            max_cue_duration_seconds=max_cue_duration_seconds,
        ):
            groups.append(current)
            current = []
        current.append(word_index)
        word = words[word_index]
        if _ends_sentence(word.text) and word.end - words[current[0]].start >= profile.min_cue_dur:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def segment_generated_adlib_cues(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    generated_cue_ids: set[int],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> tuple[list[Cue], AlignmentResult, list[QCFlag], dict[int, list[int]]]:
    if not generated_cue_ids:
        return list(cues), alignment, [], {}

    next_cue_id = max((cue.index for cue in cues), default=0) + 1
    cue_word_indices = {
        cue_id: list(indices)
        for cue_id, indices in alignment.cue_word_indices.items()
    }
    segmented: list[Cue] = []
    flags: list[QCFlag] = []
    expansions: dict[int, list[int]] = {}

    for cue in cues:
        if cue.index not in generated_cue_ids:
            segmented.append(cue)
            continue

        word_groups = group_word_indices_for_cues(
            words,
            cue_word_indices.get(cue.index, []),
            profile,
            max_gap_seconds=max_gap_seconds,
            max_cue_duration_seconds=max_cue_duration_seconds,
        )
        word_groups, text_chunks = _text_chunks_for_word_groups(
            cue.plain_text,
            word_groups,
            profile,
        )
        if len(word_groups) <= 1 or len(text_chunks) <= 1:
            segmented.append(cue)
            continue

        replacement_cues: list[Cue] = []
        replacement_ids: list[int] = []
        for position, (word_group, text_chunk) in enumerate(
            zip(word_groups, text_chunks, strict=True)
        ):
            cue_id = cue.index if position == 0 else next_cue_id
            if position > 0:
                next_cue_id += 1
            replacement_ids.append(cue_id)
            cue_word_indices[cue_id] = list(word_group)
            replacement_cues.append(
                _cue_from_generated_group(
                    cue_id,
                    cue,
                    text_chunk,
                    word_group,
                    words,
                    profile,
                )
            )

        segmented.extend(replacement_cues)
        expansions[cue.index] = replacement_ids
        flags.append(
            QCFlag(
                kind="generated_adlib_segmented",
                cue_ids=replacement_ids,
                message=(
                    "A generated ASR-only dialogue span was split at acoustic, speaker, "
                    "duration, sentence, and house-style boundaries."
                ),
                severity="info",
                old_text=cue.text,
                new_text="\n\n".join(item.text for item in replacement_cues),
                start=replacement_cues[0].start_ms / 1000.0,
                end=replacement_cues[-1].end_ms / 1000.0,
            )
        )

    return (
        segmented,
        alignment.model_copy(update={"cue_word_indices": cue_word_indices}),
        flags,
        expansions,
    )


def _starts_new_cue(
    words: list[Word],
    current: list[int],
    word_index: int,
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> bool:
    previous = words[current[-1]]
    word = words[word_index]
    if word.start - previous.end > max_gap_seconds:
        return True
    if previous.speaker_id and word.speaker_id and previous.speaker_id != word.speaker_id:
        return True
    if _snapped_duration_ms(words[current[0]], word, profile) > max_cue_duration_seconds * 1000:
        return True
    candidate = " ".join(words[index].text.strip() for index in [*current, word_index])
    return len(wrap_visual_width(candidate, profile.max_chars_per_line)) > profile.max_lines_per_cue


def _snapped_duration_ms(first: Word, last: Word, profile: StyleProfile) -> int:
    start_ms = profile.snap_floor(max(0, first.start * 1000 - profile.lead_in_ms))
    end_ms = profile.snap_ceil(last.end * 1000 + profile.tail_ms)
    return end_ms - start_ms


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", "...", "\u2026"))


def _text_chunks_for_word_groups(
    text: str,
    word_groups: list[list[int]],
    profile: StyleProfile,
) -> tuple[list[list[int]], list[str]]:
    units, separator = _split_units(text)
    if not units or not word_groups:
        return word_groups, [text.strip()] if text.strip() else []

    effective_groups = _merge_groups_to_count(word_groups, min(len(word_groups), len(units)))
    text_chunks = _partition_units_by_weights(
        units,
        [len(group) for group in effective_groups],
        separator,
    )
    refined_groups: list[list[int]] = []
    refined_text: list[str] = []
    for word_group, text_chunk in zip(effective_groups, text_chunks, strict=True):
        capacity_chunks = _capacity_chunks(text_chunk, profile)
        if len(capacity_chunks) <= 1 or len(word_group) <= 1:
            refined_groups.append(word_group)
            refined_text.append(text_chunk)
            continue
        capacity_chunks = _merge_text_chunks_to_count(
            capacity_chunks,
            min(len(capacity_chunks), len(word_group)),
            separator,
        )
        word_partitions = _partition_sequence_by_weights(
            word_group,
            [max(1, len(_split_units(chunk)[0])) for chunk in capacity_chunks],
        )
        refined_groups.extend(word_partitions)
        refined_text.extend(capacity_chunks)
    return refined_groups, refined_text


def _split_units(text: str) -> tuple[list[str], str]:
    stripped = text.strip()
    if not stripped:
        return [], " "
    if any(character.isspace() for character in stripped):
        return stripped.split(), " "
    if contains_character_level_script(stripped):
        return list(stripped), ""
    return [stripped], " "


def _partition_units_by_weights(
    units: list[str],
    weights: list[int],
    separator: str,
) -> list[str]:
    boundaries = _weighted_boundaries(len(units), weights)
    chunks: list[str] = []
    start = 0
    for end in boundaries:
        chunks.append(separator.join(units[start:end]).strip())
        start = end
    chunks.append(separator.join(units[start:]).strip())
    return chunks


def _partition_sequence_by_weights(
    values: list[int],
    weights: list[int],
) -> list[list[int]]:
    boundaries = _weighted_boundaries(len(values), weights)
    partitions: list[list[int]] = []
    start = 0
    for end in boundaries:
        partitions.append(values[start:end])
        start = end
    partitions.append(values[start:])
    return partitions


def _weighted_boundaries(item_count: int, weights: list[int]) -> list[int]:
    if len(weights) <= 1:
        return []
    total_weight = max(1, sum(weights))
    boundaries: list[int] = []
    cumulative_weight = 0
    previous = 0
    for position, weight in enumerate(weights[:-1]):
        cumulative_weight += max(0, weight)
        remaining = len(weights) - position - 1
        target = round(item_count * cumulative_weight / total_weight)
        boundary = min(max(previous + 1, target), item_count - remaining)
        boundaries.append(boundary)
        previous = boundary
    return boundaries


def _merge_groups_to_count(groups: list[list[int]], count: int) -> list[list[int]]:
    if count >= len(groups):
        return [list(group) for group in groups]
    boundaries = _weighted_boundaries(len(groups), [1] * count)
    merged: list[list[int]] = []
    start = 0
    for end in [*boundaries, len(groups)]:
        merged.append([value for group in groups[start:end] for value in group])
        start = end
    return merged


def _capacity_chunks(text: str, profile: StyleProfile) -> list[str]:
    units, separator = _split_units(text)
    if not units:
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        candidate = separator.join([*current, unit])
        if current and len(wrap_visual_width(candidate, profile.max_chars_per_line)) > profile.max_lines_per_cue:
            chunks.append(current)
            current = []
        current.append(unit)
    if current:
        chunks.append(current)
    return [separator.join(chunk).strip() for chunk in chunks]


def _merge_text_chunks_to_count(
    chunks: list[str],
    count: int,
    separator: str,
) -> list[str]:
    if count >= len(chunks):
        return list(chunks)
    boundaries = _weighted_boundaries(len(chunks), [1] * count)
    merged: list[str] = []
    start = 0
    for end in [*boundaries, len(chunks)]:
        merged.append(separator.join(chunks[start:end]).strip())
        start = end
    return merged


def _cue_from_generated_group(
    cue_id: int,
    source_cue: Cue,
    text: str,
    word_indices: list[int],
    words: list[Word],
    profile: StyleProfile,
) -> Cue:
    group_words = [words[index] for index in word_indices]
    lines = wrap_visual_width(text, profile.max_chars_per_line) or [text]
    start_ms = profile.snap_floor(max(0, group_words[0].start * 1000 - profile.lead_in_ms))
    spoken_end_ms = profile.snap_ceil(group_words[-1].end * 1000 + profile.tail_ms)
    minimum_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
    speakers = [word.speaker_id for word in group_words if word.speaker_id]
    speaker_id = Counter(speakers).most_common(1)[0][0] if speakers else source_cue.speaker_id
    return Cue(
        index=cue_id,
        start_ms=start_ms,
        end_ms=max(spoken_end_ms, minimum_end_ms, start_ms + 1),
        lines=lines,
        speaker_id=speaker_id,
        character=source_cue.character,
    )
