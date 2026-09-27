from __future__ import annotations

import re
import unicodedata

from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag
from .style_profile import StyleProfile
from .text_metrics import contains_character_level_script, display_width, join_word_texts, token_texts, wrap_visual_width
from .tokenize import alphanumeric_signature, tokenize_cues


_TERMINAL_PUNCTUATION_RE = re.compile(r"([,.;:!?…、。！？]+)\s*$")


def apply_adjudication_decisions(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    profile: StyleProfile,
    adlib_cue_ids_by_case: dict[str, int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    by_case = {decision.case_id: decision for decision in decisions}
    cues_by_id = {cue.index: cue for cue in cues}
    adlib_cue_ids_by_case = adlib_cue_ids_by_case or {}
    edits_by_cue: dict[int, list[tuple[int, int, str]]] = {}
    token_ranges_by_cue: dict[int, tuple[int, int]] = {}
    for token in tokenize_cues(cues):
        start, _ = token_ranges_by_cue.get(token.cue_id, (token.token_index, token.token_index))
        token_ranges_by_cue[token.cue_id] = start, token.token_index + 1
    removed_cue_ids: set[int] = set()
    adlib_cues: list[Cue] = []
    updated: list[Cue] = []
    flags: list[QCFlag] = []

    for span in spans:
        decision = by_case.get(span.case_id)
        if decision is None or decision.verdict == "keep_srt":
            continue

        cue_ids = [cue_id for cue_id in span.cue_ids if cue_id in cues_by_id]
        if not decision.final_text.strip():
            if cue_ids:
                should_remove = profile.drop_policy == "remove"
                if should_remove:
                    removed_cue_ids.update(cue_ids)
                flags.append(
                    QCFlag(
                        kind="dropped_adjudicated_cue" if should_remove else "dropped_line_candidate",
                        cue_ids=cue_ids,
                        message=(
                            f"Adjudication verdict {decision.verdict} returned empty text; removed by drop_policy."
                            if should_remove
                            else f"Adjudication verdict {decision.verdict} returned empty spoken text; preserving source cue for review."
                        ),
                        confidence=decision.confidence,
                        old_text="\n".join(cues_by_id[cue_id].text for cue_id in cue_ids),
                        new_text="",
                        start=span.start,
                        end=span.end,
                    )
                )
            continue

        if not cue_ids:
            adlib_cue_id = adlib_cue_ids_by_case.get(span.case_id)
            if adlib_cue_id is None:
                continue
            lines = flow_text_to_lines(decision.final_text, profile.max_chars_per_line, profile.max_lines_per_cue)
            if adlib_cue_id in cues_by_id:
                _record_cue_edit(edits_by_cue, cues_by_id[adlib_cue_id], decision.final_text)
                continue
            adlib_cues.append(
                Cue(
                    index=adlib_cue_id,
                    start_ms=int((span.start or 0.0) * 1000),
                    end_ms=int((span.end or span.start or 0.0) * 1000),
                    lines=lines,
                    speaker_id=decision.speaker,
                    character=decision.character,
                )
            )
            flags.append(
                QCFlag(
                    kind="adlib_inserted",
                    cue_ids=[adlib_cue_id],
                    message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                    confidence=decision.confidence,
                    old_text=None,
                    new_text="\n".join(lines),
                    start=span.start,
                    end=span.end,
                )
            )
            continue

        replacement_texts = (
            [
                _cue_text_with_span_replacement(
                    cues_by_id[cue_ids[0]],
                    span,
                    decision.final_text,
                    _cue_local_span_bounds(span, token_ranges_by_cue.get(cue_ids[0])),
                )
            ]
            if len(cue_ids) == 1
            else _split_text_for_cues(decision.final_text, len(cue_ids))
        )
        for cue_id, text in zip(cue_ids, replacement_texts, strict=False):
            _record_cue_edit(edits_by_cue, cues_by_id[cue_id], text)
        flags.append(
            QCFlag(
                kind="text_changed",
                cue_ids=cue_ids,
                message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                confidence=decision.confidence,
                old_text="\n".join(cues_by_id[cue_id].text for cue_id in cue_ids),
                new_text=None,
                start=span.start,
                end=span.end,
            )
        )

    replacements_by_cue: dict[int, list[str]] = {}
    for cue_id, edits in edits_by_cue.items():
        text = cues_by_id[cue_id].plain_text
        # Every edit refers to the original cue. Applying from the end avoids
        # shifting a later correction when an earlier replacement changes length.
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        replacements_by_cue[cue_id] = flow_text_to_lines(
            text, profile.max_chars_per_line, profile.max_lines_per_cue,
        )
    for flag in flags:
        if flag.kind == "text_changed":
            flag.new_text = "\n".join(
                "\n".join(replacements_by_cue[cue_id])
                if cue_id in replacements_by_cue else cues_by_id[cue_id].text
                for cue_id in flag.cue_ids
            )

    for cue in cues:
        if cue.index in removed_cue_ids:
            continue
        replacement = replacements_by_cue.get(cue.index)
        if replacement is None:
            updated.append(cue)
            continue

        updated.append(cue.with_lines(replacement))

    return _merge_adlibs_positionally(updated, adlib_cues), flags


def _cue_local_span_bounds(
    span: DivergenceSpan, cue_range: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if not span.srt_token_indices or cue_range is None:
        return None
    indices = sorted(set(span.srt_token_indices))
    start, end = cue_range
    if indices[0] < start or indices[-1] >= end or indices != list(range(indices[0], indices[-1] + 1)):
        return None
    return indices[0] - start, indices[-1] - start + 1


def _record_cue_edit(edits_by_cue: dict[int, list[tuple[int, int, str]]], cue: Cue, replacement: str) -> None:
    source = cue.plain_text
    start = 0
    while start < min(len(source), len(replacement)) and source[start] == replacement[start]:
        start += 1
    end, replacement_end = len(source), len(replacement)
    while end > start and replacement_end > start and source[end - 1] == replacement[replacement_end - 1]:
        end -= 1
        replacement_end -= 1
    edits = edits_by_cue.setdefault(cue.index, [])
    if start == end and start == replacement_end:
        return
    # Retain the existing later-decision precedence for conflicting edits, while
    # keeping every disjoint accepted correction on this cue.
    edits[:] = [
        edit for edit in edits
        if not (edit[0] == start or max(edit[0], start) < min(edit[1], end)
                or edit[0] < start < edit[1] or start < edit[0] < end)
    ]
    edits.append((start, end, replacement[start:replacement_end]))


def _merge_adlibs_positionally(cues: list[Cue], adlib_cues: list[Cue]) -> list[Cue]:
    if not adlib_cues:
        return cues
    pending = sorted(adlib_cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    merged: list[Cue] = []
    cursor = 0
    for cue in cues:
        while cursor < len(pending) and pending[cursor].start_ms < cue.start_ms:
            merged.append(pending[cursor])
            cursor += 1
        merged.append(cue)
    merged.extend(pending[cursor:])
    return merged


def flow_text_to_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    wrapped = wrap_visual_width(text, max_chars)
    if not wrapped:
        return [""]
    if len(wrapped) <= max_lines:
        return wrapped
    head = wrapped[: max_lines - 1]
    if contains_character_level_script(text):
        source = " ".join(text.split())
        cursor = 0
        for line in head:
            start = source.find(line, cursor)
            if start < 0:
                break  # A hyphenated Latin word may have been split by wrapping.
            cursor = start + len(line)
        else:
            # Keep original adjacency even when a wrap falls inside an embedded
            # Latin name; script-aware joining alone cannot recover that boundary.
            return [*head, source[cursor:].lstrip()]
    tail = join_word_texts(wrapped[max_lines - 1 :])
    return [*head, tail]


def _split_text_for_cues(text: str, cue_count: int) -> list[str]:
    if cue_count <= 1:
        return [text.strip()]

    units, separator = _split_units(text)
    if not units:
        return [""] * cue_count

    chunks: list[list[str]] = []
    current: list[str] = []
    target_width = max(1, display_width(text) / cue_count)

    for index, unit in enumerate(units):
        remaining_units = len(units) - index
        remaining_chunks = cue_count - len(chunks)
        candidate = separator.join([*current, unit])
        current_width = display_width(separator.join(current))
        if current and (current_width >= target_width or display_width(candidate) > target_width) and remaining_units >= remaining_chunks:
            chunks.append(current)
            current = []
        current.append(unit)

    chunks.append(current)

    while len(chunks) < cue_count:
        chunks.append([])

    if len(chunks) > cue_count:
        head = chunks[: cue_count - 1]
        tail = [unit for chunk in chunks[cue_count - 1 :] for unit in chunk]
        chunks = [*head, tail]

    return [separator.join(chunk).strip() for chunk in chunks]


def _split_units(text: str) -> tuple[list[str], str]:
    stripped = text.strip()
    if not stripped:
        return [], " "
    if " " in stripped:
        return stripped.split(), " "
    if contains_character_level_script(stripped):
        return list(stripped), ""
    return [stripped], " "


def _cue_text_with_span_replacement(
    cue: Cue, span: DivergenceSpan, final_text: str,
    span_token_bounds: tuple[int, int] | None = None,
) -> str:
    replacement = final_text.strip()
    cue_signature = alphanumeric_signature(cue.plain_text)
    span_signature = alphanumeric_signature(span.srt_text)
    if not cue_signature or not span_signature or len(span_signature) >= len(cue_signature):
        return replacement

    bounds = span_token_bounds or _find_subsequence_bounds(cue_signature, span_signature)
    if bounds is None:
        return replacement

    cue_tokens = token_texts(cue.plain_text)
    start, end = bounds
    final_signature = alphanumeric_signature(replacement)
    source_bounds = _source_token_bounds(cue.plain_text, cue_tokens)
    if source_bounds is not None:
        # Splice into the authored text: rebuilding tokens loses quotes and
        # punctuation and invents spaces in Japanese sentences.
        before = cue.plain_text[: source_bounds[start][0]]
        after = cue.plain_text[source_bounds[end - 1][1] :]
        if _starts_with_sequence(final_signature, cue_signature[:start]):
            before = cue.plain_text[: source_bounds[0][0]]
        if _ends_with_sequence(final_signature, cue_signature[end:]):
            after = cue.plain_text[source_bounds[-1][1] :]
        if _TERMINAL_PUNCTUATION_RE.search(replacement):
            after = re.sub(r"^[,.;:!?…、。！？]+", "", after)
        return _join_preserved_boundary(_join_preserved_boundary(before, replacement), after)

    before_tokens = cue_tokens[:start]
    after_tokens = cue_tokens[end:]
    if _starts_with_sequence(final_signature, cue_signature[:start]):
        before_tokens = []
    if _ends_with_sequence(final_signature, cue_signature[end:]):
        after_tokens = []
    pieces = [*before_tokens, replacement, *after_tokens]
    text = join_word_texts(piece.strip() for piece in pieces if piece.strip())
    return _restore_terminal_punctuation(text, cue.plain_text)


def _source_token_bounds(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    """Locate comparison tokens without normalizing the displayed source."""
    normalized_parts: list[str] = []
    original_bounds: dict[int, int] = {0: 0}
    normalized_length = 0
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and (
            unicodedata.category(text[end]).startswith("M") or text[end] in "\uff9e\uff9f"
        ):
            end += 1
        part = unicodedata.normalize("NFKC", text[start:end])
        normalized_parts.append(part)
        normalized_length += len(part)
        original_bounds[normalized_length] = end
        start = end

    normalized = "".join(normalized_parts)
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        comparison_token = unicodedata.normalize("NFKC", token)
        start = normalized.find(comparison_token, cursor)
        end = start + len(comparison_token)
        if start not in original_bounds or end not in original_bounds:
            return None
        bounds.append((original_bounds[start], original_bounds[end]))
        cursor = end
    return bounds


def _join_preserved_boundary(left: str, right: str) -> str:
    # The adjudicator may include an existing surrounding quote or punctuation.
    for size in range(min(len(left), len(right)), 0, -1):
        overlap = right[:size]
        if left.endswith(overlap) and all(not char.isalnum() for char in overlap):
            return left + right[size:]
    return left + right


def _find_subsequence_bounds(haystack: list[str], needle: list[str]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        joined = "".join(needle)
        for start, value in enumerate(haystack):
            if value == joined:
                return start, start + 1
        return None
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
        if "".join(haystack[start : start + len(needle)]) == "".join(needle):
            return start, start + len(needle)
    joined = "".join(needle)
    for start, value in enumerate(haystack):
        if value == joined:
            return start, start + 1
    return None


def _starts_with_sequence(value: list[str], prefix: list[str]) -> bool:
    return bool(prefix) and len(value) >= len(prefix) and value[: len(prefix)] == prefix


def _ends_with_sequence(value: list[str], suffix: list[str]) -> bool:
    return bool(suffix) and len(value) >= len(suffix) and value[-len(suffix) :] == suffix


def _restore_terminal_punctuation(text: str, source_text: str) -> str:
    stripped = text.rstrip()
    if not stripped or _TERMINAL_PUNCTUATION_RE.search(stripped):
        return text
    match = _TERMINAL_PUNCTUATION_RE.search(source_text.rstrip())
    if match is None:
        return text
    return f"{stripped}{match.group(1)}"
