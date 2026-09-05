from __future__ import annotations

import re
from collections import Counter

from .models import AlignmentResult, Cue, QCFlag, Word
from .style_profile import StyleProfile
from .subtitle_annotations import (
    cue_has_bracketed_screen_text,
    text_without_bracketed_screen_text,
)
from .text_metrics import contains_character_level_script, wrap_visual_width
from .tokenize import alphanumeric_signature


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
    ordered = _ordered_valid_word_indices(words, word_indices)
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

        candidate_word_indices = cue_word_indices.get(cue.index, [])
        retained_word_indices, mapping_status = _retained_word_window(
            cue.plain_text,
            words,
            candidate_word_indices,
        )
        cue_word_indices[cue.index] = retained_word_indices
        mapping_flag = _word_mapping_flag(
            cue,
            words,
            candidate_word_indices,
            retained_word_indices,
            mapping_status,
        )
        word_groups = group_word_indices_for_cues(
            words,
            retained_word_indices,
            profile,
            max_gap_seconds=max_gap_seconds,
            max_cue_duration_seconds=max_cue_duration_seconds,
        )
        word_groups, text_chunks = _text_chunks_for_word_groups(
            cue.plain_text,
            word_groups,
            profile,
            words=words,
        )
        if len(word_groups) <= 1 or len(text_chunks) <= 1:
            segmented.append(cue)
            if mapping_flag is not None:
                flags.append(mapping_flag)
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
        if mapping_flag is not None:
            flags.append(mapping_flag.model_copy(update={"cue_ids": replacement_ids}))
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


def split_overlong_existing_cues(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    source_cue_ids: set[int] | None = None,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> tuple[list[Cue], AlignmentResult, list[QCFlag], dict[int, list[int]]]:
    """Split source-backed cues that cannot satisfy the active line limit.

    This is deliberately timing-gated: when a cue cannot be mapped to valid ASR
    word timing, it is kept and flagged instead of being split by text alone.
    """

    next_cue_id = max((cue.index for cue in cues), default=0) + 1
    cue_word_indices = {
        cue_id: list(indices)
        for cue_id, indices in alignment.cue_word_indices.items()
    }
    split_cues: list[Cue] = []
    flags: list[QCFlag] = []
    expansions: dict[int, list[int]] = {}

    for cue in cues:
        if source_cue_ids is not None and cue.index not in source_cue_ids:
            split_cues.append(cue)
            continue
        if cue_has_bracketed_screen_text(cue):
            split_cues.append(cue)
            continue
        exceeds_explicit_lines = len(cue.lines) > profile.max_lines_per_cue
        exceeds_wrapped_lines = len(wrap_visual_width(cue.plain_text, profile.max_chars_per_line)) > profile.max_lines_per_cue
        if not exceeds_explicit_lines and not exceeds_wrapped_lines:
            split_cues.append(cue)
            continue
        has_markup = _has_inline_subtitle_markup(cue.text)
        markup_can_follow_source_lines = (
            exceeds_explicit_lines and _source_lines_have_self_contained_markup(cue.lines)
        )
        if has_markup and not markup_can_follow_source_lines:
            split_cues.append(cue)
            flags.append(
                QCFlag(
                    kind="sync_cue_line_limit_markup_unsupported",
                    cue_ids=[cue.index],
                    message=(
                        "Cue exceeds the requested maximum line count, but it contains inline "
                        "subtitle styling markup. It was kept for review instead of risking "
                        "unbalanced tags during an automatic split."
                    ),
                    severity="warning",
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        candidate_word_indices = cue_word_indices.get(cue.index, [])
        retained_word_indices, mapping_status = _retained_word_window(
            cue.plain_text,
            words,
            candidate_word_indices,
        )
        cue_word_indices[cue.index] = retained_word_indices
        valid_retained = _ordered_valid_word_indices(words, retained_word_indices)
        if (
            mapping_status == "unavailable"
            or not valid_retained
            or not _has_unique_exact_text_window(cue.plain_text, words, valid_retained)
        ):
            split_cues.append(cue)
            flags.append(
                QCFlag(
                    kind="sync_cue_line_limit_timing_unavailable",
                    cue_ids=[cue.index],
                    message=(
                        "Cue exceeds the requested maximum line count, but no unique valid ASR "
                        "word timing window was available, so it was kept for review instead of "
                        "being text-split without acoustic evidence."
                    ),
                    severity="warning",
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        if exceeds_explicit_lines:
            word_groups, text_chunks = _source_line_chunks_for_limit(
                cue,
                valid_retained,
                words,
                profile,
                max_gap_seconds=max_gap_seconds,
                max_cue_duration_seconds=max_cue_duration_seconds,
            )
        else:
            word_groups = group_word_indices_for_cues(
                words,
                valid_retained,
                profile,
                max_gap_seconds=max_gap_seconds,
                max_cue_duration_seconds=max_cue_duration_seconds,
            )
            word_groups, text_chunks = _text_chunks_for_word_groups(
                cue.plain_text, word_groups, profile, words=words
            )
        if len(word_groups) <= 1 or len(text_chunks) <= 1:
            split_cues.append(cue)
            flags.append(
                QCFlag(
                    kind="sync_cue_line_limit_timing_unavailable",
                    cue_ids=[cue.index],
                    message=(
                        "Cue exceeds the requested maximum line count, but the aligned words "
                        "did not yield a safe acoustic split point."
                    ),
                    severity="warning",
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        replacement_cues: list[Cue] = []
        replacement_ids: list[int] = []
        for position, (word_group, text_chunk) in enumerate(zip(word_groups, text_chunks, strict=True)):
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

        split_cues.extend(replacement_cues)
        expansions[cue.index] = replacement_ids
        flags.append(
            QCFlag(
                kind="sync_cue_line_limit_split",
                cue_ids=replacement_ids,
                message=(
                    "A source SRT cue exceeded the requested maximum line count and was "
                    "split using aligned ASR word timing, speaker, sentence, duration, and "
                    "house-style boundaries."
                ),
                severity="info",
                old_text=cue.text,
                new_text="\n\n".join(item.text for item in replacement_cues),
                start=replacement_cues[0].start_ms / 1000.0,
                end=replacement_cues[-1].end_ms / 1000.0,
            )
        )

    return split_cues, alignment.model_copy(update={"cue_word_indices": cue_word_indices}), flags, expansions


def _source_line_chunks_for_limit(
    cue: Cue,
    valid_word_indices: list[int],
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> tuple[list[list[int]], list[str]]:
    line_word_groups = _word_groups_for_source_lines(cue.lines, valid_word_indices, words)
    if line_word_groups is None:
        return [valid_word_indices], [cue.text]

    cue_word_groups: list[list[int]] = []
    cue_line_groups: list[list[str]] = []
    current_words: list[int] = []
    current_lines: list[str] = []
    for line, line_words in zip(cue.lines, line_word_groups, strict=True):
        if current_words and _starts_new_source_line_cue(
            current_lines,
            current_words,
            line_words,
            words,
            profile,
            max_gap_seconds=max_gap_seconds,
            max_cue_duration_seconds=max_cue_duration_seconds,
        ):
            cue_word_groups.append(current_words)
            cue_line_groups.append(current_lines)
            current_words = []
            current_lines = []
        current_words.extend(line_words)
        current_lines.append(line)
    if current_words:
        cue_word_groups.append(current_words)
        cue_line_groups.append(current_lines)
    return cue_word_groups, ["\n".join(group) for group in cue_line_groups]


def _word_groups_for_source_lines(
    lines: list[str],
    word_indices: list[int],
    words: list[Word],
) -> list[list[int]] | None:
    groups: list[list[int]] = []
    word_position = 0
    for line in lines:
        target_token_count = len(
            alphanumeric_signature(
                _text_without_subtitle_markup(text_without_bracketed_screen_text(line))
            )
        )
        if target_token_count <= 0:
            return None
        group: list[int] = []
        observed_token_count = 0
        while word_position < len(word_indices) and observed_token_count < target_token_count:
            word_index = word_indices[word_position]
            word_token_count = len(alphanumeric_signature(words[word_index].text))
            if word_token_count <= 0:
                return None
            group.append(word_index)
            observed_token_count += word_token_count
            word_position += 1
        if observed_token_count != target_token_count or not group:
            return None
        groups.append(group)
    if word_position != len(word_indices):
        return None
    return groups


def _starts_new_source_line_cue(
    current_lines: list[str],
    current_words: list[int],
    next_words: list[int],
    words: list[Word],
    profile: StyleProfile,
    *,
    max_gap_seconds: float,
    max_cue_duration_seconds: float,
) -> bool:
    if len(current_lines) >= profile.max_lines_per_cue:
        return True
    previous = words[current_words[-1]]
    nxt = words[next_words[0]]
    if nxt.start - previous.end > max_gap_seconds:
        return True
    previous_speaker = next(
        (words[index].speaker_id for index in reversed(current_words) if words[index].speaker_id),
        None,
    )
    next_speaker = next(
        (words[index].speaker_id for index in next_words if words[index].speaker_id),
        None,
    )
    if previous_speaker and next_speaker and previous_speaker != next_speaker:
        return True
    if _ends_sentence(_text_without_subtitle_markup(current_lines[-1])):
        return True
    return (
        _snapped_duration_ms(words[current_words[0]], words[next_words[-1]], profile)
        > max_cue_duration_seconds * 1000
    )


_INLINE_SUBTITLE_MARKUP_RE = re.compile(r"</?[^>\n]+>|{\\[^}\n]+}")
_HTML_SUBTITLE_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<name>[A-Za-z][\w:.-]*)\b[^>]*?(?P<self_closing>/)?\s*>"
)
_VOID_HTML_TAGS = frozenset({"br", "hr"})


def _has_inline_subtitle_markup(text: str) -> bool:
    return bool(_INLINE_SUBTITLE_MARKUP_RE.search(text))


def _source_lines_have_self_contained_markup(lines: list[str]) -> bool:
    """Return whether every source line can move without opening/closing tags elsewhere."""

    for line in lines:
        stack: list[str] = []
        for markup in _INLINE_SUBTITLE_MARKUP_RE.findall(line):
            if markup.startswith("{\\"):
                continue
            tag = _HTML_SUBTITLE_TAG_RE.fullmatch(markup)
            if tag is None:
                return False
            name = tag.group("name").casefold()
            if tag.group("closing"):
                if not stack or stack.pop() != name:
                    return False
            elif not tag.group("self_closing") and name not in _VOID_HTML_TAGS:
                stack.append(name)
        if stack:
            return False
    return True


def _text_without_subtitle_markup(text: str) -> str:
    return _INLINE_SUBTITLE_MARKUP_RE.sub(" ", text)


def _retained_word_window(
    text: str,
    words: list[Word],
    word_indices: list[int],
) -> tuple[list[int], str]:
    ordered = _ordered_valid_word_indices(words, word_indices)
    target_tokens = alphanumeric_signature(
        _text_without_subtitle_markup(text_without_bracketed_screen_text(text))
    )
    if not target_tokens:
        return ordered, "full"
    if not ordered:
        return ordered, "unavailable"

    candidate_tokens: list[str] = []
    token_word_positions: list[int] = []
    for word_position, word_index in enumerate(ordered):
        for token in alphanumeric_signature(words[word_index].text):
            candidate_tokens.append(token)
            token_word_positions.append(word_position)

    if not candidate_tokens:
        return ordered, "unavailable"

    matching_windows = _exact_token_windows(
        candidate_tokens,
        target_tokens,
        token_word_positions,
        limit=2,
    )

    if len(matching_windows) != 1:
        status = "unavailable" if len(candidate_tokens) != len(target_tokens) else "full"
        return ordered, status

    word_start, word_end = next(iter(matching_windows))
    retained = ordered[word_start:word_end]
    return retained, "full" if retained == ordered else "refined"


def _has_unique_exact_text_window(text: str, words: list[Word], word_indices: list[int]) -> bool:
    target_tokens = alphanumeric_signature(
        _text_without_subtitle_markup(text_without_bracketed_screen_text(text))
    )
    if not target_tokens:
        return True
    candidate_tokens: list[str] = []
    token_word_positions: list[int] = []
    for word_position, word_index in enumerate(word_indices):
        for token in alphanumeric_signature(words[word_index].text):
            candidate_tokens.append(token)
            token_word_positions.append(word_position)
    if not candidate_tokens:
        return False
    windows = _exact_token_windows(
        candidate_tokens,
        target_tokens,
        token_word_positions,
        limit=2,
    )
    return windows == {(0, len(word_indices))}


def _exact_token_windows(
    candidate_tokens: list[str],
    target_tokens: list[str],
    token_word_positions: list[int],
    *,
    limit: int,
) -> set[tuple[int, int]]:
    if not target_tokens or len(target_tokens) > len(candidate_tokens):
        return set()
    prefix_lengths = _kmp_prefix_lengths(target_tokens)
    matched = 0
    windows: set[tuple[int, int]] = set()
    for token_position, token in enumerate(candidate_tokens):
        while matched and token != target_tokens[matched]:
            matched = prefix_lengths[matched - 1]
        if token == target_tokens[matched]:
            matched += 1
        if matched != len(target_tokens):
            continue
        token_start = token_position - len(target_tokens) + 1
        windows.add(
            (
                token_word_positions[token_start],
                token_word_positions[token_position] + 1,
            )
        )
        if len(windows) >= limit:
            return windows
        matched = prefix_lengths[matched - 1]
    return windows


def _kmp_prefix_lengths(tokens: list[str]) -> list[int]:
    lengths = [0] * len(tokens)
    matched = 0
    for index in range(1, len(tokens)):
        while matched and tokens[index] != tokens[matched]:
            matched = lengths[matched - 1]
        if tokens[index] == tokens[matched]:
            matched += 1
            lengths[index] = matched
    return lengths


def _ordered_valid_word_indices(words: list[Word], word_indices: list[int]) -> list[int]:
    return sorted(
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


def _word_mapping_flag(
    cue: Cue,
    words: list[Word],
    candidate_word_indices: list[int],
    retained_word_indices: list[int],
    mapping_status: str,
) -> QCFlag | None:
    if mapping_status == "full":
        return None

    valid_candidates = _ordered_valid_word_indices(words, candidate_word_indices)
    start = words[valid_candidates[0]].start if valid_candidates else cue.start_ms / 1000.0
    end = words[valid_candidates[-1]].end if valid_candidates else cue.end_ms / 1000.0
    if mapping_status == "refined":
        return QCFlag(
            kind="generated_adlib_word_window_refined",
            cue_ids=[cue.index],
            message=(
                "Generated dialogue matched one unique contiguous ASR word window; "
                f"{len(valid_candidates) - len(retained_word_indices)} candidate words outside "
                "the retained text were excluded from cue timing and remain unrepresented."
            ),
            severity="warning",
            new_text=cue.text,
            start=start,
            end=end,
        )
    return QCFlag(
        kind="generated_adlib_word_mapping_unavailable",
        cue_ids=[cue.index],
        message=(
            "Generated dialogue could not be mapped to one unique contiguous ASR word window; "
            "the full candidate timing was retained for review."
        ),
        severity="warning",
        new_text=cue.text,
        start=start,
        end=end,
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
    previous_speaker = next(
        (words[index].speaker_id for index in reversed(current) if words[index].speaker_id),
        None,
    )
    if previous_speaker and word.speaker_id and previous_speaker != word.speaker_id:
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
    *,
    words: list[Word],
) -> tuple[list[list[int]], list[str]]:
    units, separator = _split_units(text)
    if not units or not word_groups:
        return word_groups, [text.strip()] if text.strip() else []

    effective_groups = _merge_groups_to_count(word_groups, min(len(word_groups), len(units)))
    text_chunks = _exact_text_chunks_for_word_groups(units, separator, effective_groups, words)
    exact_mapping = text_chunks is not None
    if text_chunks is None:
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
        if exact_mapping:
            word_partitions = _word_groups_for_source_lines(capacity_chunks, word_group, words)
            if word_partitions is None:
                # A provider word can contain several lexical tokens. Keep its
                # acoustic window intact when wrapping would split that word.
                refined_groups.append(word_group)
                refined_text.append(text_chunk)
                continue
        else:
            word_partitions = _partition_sequence_by_weights(
                word_group,
                [max(1, len(_split_units(chunk)[0])) for chunk in capacity_chunks],
            )
        refined_groups.extend(word_partitions)
        refined_text.extend(capacity_chunks)
    return refined_groups, refined_text


def _exact_text_chunks_for_word_groups(
    units: list[str],
    separator: str,
    word_groups: list[list[int]],
    words: list[Word],
) -> list[str] | None:
    """Use lexical boundaries instead of assuming one ASR word per text unit."""
    text_tokens: list[str] = []
    unit_end_by_token_count: dict[int, int] = {}
    for position, unit in enumerate(units, start=1):
        text_tokens.extend(alphanumeric_signature(unit))
        unit_end_by_token_count[len(text_tokens)] = position

    group_tokens = [
        [token for index in group for token in alphanumeric_signature(words[index].text)]
        for group in word_groups
    ]
    if text_tokens != [token for group in group_tokens for token in group]:
        return None
    chunks: list[str] = []
    token_count = 0
    unit_start = 0
    for tokens in group_tokens:
        token_count += len(tokens)
        unit_end = unit_end_by_token_count.get(token_count)
        if unit_end is None or unit_end <= unit_start:
            return None
        chunks.append(separator.join(units[unit_start:unit_end]))
        unit_start = unit_end
    return chunks if unit_start == len(units) else None


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
    lines = [line for line in text.splitlines() if line.strip()] if "\n" in text else wrap_visual_width(text, profile.max_chars_per_line) or [text]
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
