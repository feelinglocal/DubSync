from __future__ import annotations

import re

from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag
from .style_profile import StyleProfile
from .text_metrics import contains_character_level_script, display_width, token_texts, wrap_visual_width
from .tokenize import alphanumeric_signature


_TERMINAL_PUNCTUATION_RE = re.compile(r"([,.;:!?…]+)\s*$")


def apply_adjudication_decisions(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    profile: StyleProfile,
    adlib_cue_ids_by_case: dict[str, int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    by_case = {decision.case_id: decision for decision in decisions}
    cues_by_id = {cue.index: cue for cue in cues}
    cue_token_offsets = _cue_token_offsets(cues)
    adlib_cue_ids_by_case = adlib_cue_ids_by_case or {}
    replacements_by_cue: dict[int, list[str]] = {}
    token_edits_by_cue: dict[int, list[tuple[int, int, str]]] = {}
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
            if len(cue_ids) == 1:
                cue_id = cue_ids[0]
                cue = cues_by_id[cue_id]
                bounds = _span_token_bounds_for_cue(
                    cue,
                    span,
                    cue_token_offsets[cue_id],
                )
                if _is_partial_cue_span(cue, span, bounds):
                    if bounds is not None:
                        token_edits_by_cue.setdefault(cue_id, []).append(
                            (bounds[0], bounds[1], "")
                        )
                        changed_text = _apply_token_edits(
                            cue.plain_text,
                            token_edits_by_cue[cue_id],
                        )
                    else:
                        changed_text = _cue_text_with_span_replacement(cue, span, "")
                        replacements_by_cue[cue_id] = flow_text_to_lines(
                            changed_text,
                            profile.max_chars_per_line,
                            profile.max_lines_per_cue,
                        )
                    flags.append(
                        QCFlag(
                            kind="text_changed",
                            cue_ids=[cue_id],
                            message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                            confidence=decision.confidence,
                            old_text=cue.text,
                            new_text=changed_text,
                            start=span.start,
                            end=span.end,
                        )
                    )
                    continue
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
                cue = cues_by_id[adlib_cue_id]
                insertion_offset = _anchored_insertion_offset(
                    span,
                    adlib_cue_id,
                    cue,
                )
                if insertion_offset is not None:
                    token_edits_by_cue.setdefault(adlib_cue_id, []).append(
                        (insertion_offset, insertion_offset, decision.final_text)
                    )
                    changed_text = _apply_token_edits(
                        cue.plain_text,
                        token_edits_by_cue[adlib_cue_id],
                    )
                    flags.append(
                        QCFlag(
                            kind="text_changed",
                            cue_ids=[adlib_cue_id],
                            message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                            confidence=decision.confidence,
                            old_text=cue.text,
                            new_text=changed_text,
                            start=span.start,
                            end=span.end,
                        )
                    )
                    continue
                replacements_by_cue[adlib_cue_id] = lines
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

        if len(cue_ids) == 1 and span.srt_token_indices:
            cue_id = cue_ids[0]
            cue = cues_by_id[cue_id]
            bounds = _span_token_bounds_for_cue(
                cue,
                span,
                cue_token_offsets[cue_id],
            )
            if bounds is not None:
                localized_replacement = _localized_indexed_replacement(
                    cue,
                    span,
                    bounds,
                    decision.final_text,
                )
                token_edits_by_cue.setdefault(cue_id, []).append(
                    (bounds[0], bounds[1], localized_replacement)
                )
                changed_text = _apply_token_edits(
                    cue.plain_text,
                    token_edits_by_cue[cue_id],
                )
                flags.append(
                    QCFlag(
                        kind="text_changed",
                        cue_ids=[cue_id],
                        message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                        confidence=decision.confidence,
                        old_text=cue.text,
                        new_text=changed_text,
                        start=span.start,
                        end=span.end,
                    )
                )
                continue

        replacement_texts = (
            [_cue_text_with_span_replacement(cues_by_id[cue_ids[0]], span, decision.final_text)]
            if len(cue_ids) == 1
            else _split_text_for_cues(decision.final_text, len(cue_ids))
        )
        replacement_lines = {
            cue_id: flow_text_to_lines(text, profile.max_chars_per_line, profile.max_lines_per_cue)
            for cue_id, text in zip(cue_ids, replacement_texts, strict=False)
            if text.strip()
        }
        removed_cue_ids.update(
            cue_id
            for cue_id, text in zip(cue_ids, replacement_texts, strict=False)
            if not text.strip()
        )
        replacements_by_cue.update(replacement_lines)
        flags.append(
            QCFlag(
                kind="text_changed",
                cue_ids=cue_ids,
                message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                confidence=decision.confidence,
                old_text="\n".join(cues_by_id[cue_id].text for cue_id in cue_ids),
                new_text="\n".join(
                    "\n".join(replacement_lines[cue_id])
                    for cue_id in cue_ids
                    if cue_id in replacement_lines
                ),
                start=span.start,
                end=span.end,
            )
        )

    final_token_edit_text_by_cue: dict[int, str] = {}
    for cue_id, edits in token_edits_by_cue.items():
        changed_text = _apply_token_edits(cues_by_id[cue_id].plain_text, edits)
        final_token_edit_text_by_cue[cue_id] = changed_text
        if not changed_text.strip():
            removed_cue_ids.add(cue_id)
            replacements_by_cue.pop(cue_id, None)
            continue
        replacements_by_cue[cue_id] = flow_text_to_lines(
            changed_text,
            profile.max_chars_per_line,
            profile.max_lines_per_cue,
        )

    flags = [
        flag.model_copy(
            update={"new_text": final_token_edit_text_by_cue[flag.cue_ids[0]]}
        )
        if (
            flag.kind == "text_changed"
            and len(flag.cue_ids) == 1
            and flag.cue_ids[0] in final_token_edit_text_by_cue
        )
        else flag
        for flag in flags
    ]

    for cue in cues:
        if cue.index in removed_cue_ids:
            continue
        replacement = replacements_by_cue.get(cue.index)
        if replacement is None:
            updated.append(cue)
            continue

        updated.append(cue.with_lines(replacement))

    return _merge_adlibs_positionally(updated, adlib_cues), flags


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
    tail = " ".join(wrapped[max_lines - 1 :])
    return [*head, tail]


def _split_text_for_cues(text: str, cue_count: int) -> list[str]:
    if cue_count <= 1:
        return [text.strip()]

    units, separator = _split_units(text)
    if not units:
        return [""] * cue_count

    chunk_count = min(cue_count, len(units))
    chunks: list[list[str]] = []
    current: list[str] = []
    target_width = max(1, display_width(text) / chunk_count)

    for index, unit in enumerate(units):
        remaining_units = len(units) - index
        remaining_chunks_after_current = chunk_count - len(chunks) - 1
        candidate = separator.join([*current, unit])
        current_width = display_width(separator.join(current))
        must_leave_unit_per_chunk = remaining_units <= remaining_chunks_after_current
        width_prefers_split = current_width >= target_width or display_width(candidate) > target_width
        if current and (must_leave_unit_per_chunk or width_prefers_split) and remaining_units >= remaining_chunks_after_current:
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


def _cue_token_offsets(cues: list[Cue]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    offset = 0
    for cue in cues:
        offsets[cue.index] = offset
        offset += len(alphanumeric_signature(cue.plain_text))
    return offsets


def _span_token_bounds_for_cue(
    cue: Cue,
    span: DivergenceSpan,
    cue_token_offset: int,
) -> tuple[int, int] | None:
    cue_signature = alphanumeric_signature(cue.plain_text)
    local_indices = sorted(
        {
            token_index - cue_token_offset
            for token_index in span.srt_token_indices
            if cue_token_offset <= token_index < cue_token_offset + len(cue_signature)
        }
    )
    if local_indices:
        return local_indices[0], local_indices[-1] + 1

    span_signature = alphanumeric_signature(span.srt_text)
    return _find_subsequence_bounds(cue_signature, span_signature)


def _is_partial_cue_span(
    cue: Cue,
    span: DivergenceSpan,
    bounds: tuple[int, int] | None,
) -> bool:
    cue_token_count = len(alphanumeric_signature(cue.plain_text))
    if bounds is not None:
        return bounds != (0, cue_token_count)
    span_signature = alphanumeric_signature(span.srt_text)
    return bool(span_signature) and len(span_signature) < cue_token_count


def _anchored_insertion_offset(
    span: DivergenceSpan,
    cue_id: int,
    cue: Cue,
) -> int | None:
    left_cue_id = span.left_anchor_cue_id
    right_cue_id = span.right_anchor_cue_id
    if left_cue_id is None and right_cue_id is None:
        return None
    if left_cue_id == cue_id and right_cue_id == cue_id:
        return span.insertion_token_offset
    if right_cue_id == cue_id:
        return 0
    if left_cue_id == cue_id:
        return len(alphanumeric_signature(cue.plain_text))
    return None


def _localized_indexed_replacement(
    cue: Cue,
    span: DivergenceSpan,
    bounds: tuple[int, int],
    final_text: str,
) -> str:
    cue_signature = alphanumeric_signature(cue.plain_text)
    final_signature = alphanumeric_signature(final_text)
    if not final_signature:
        return ""

    start, end = bounds
    prefix = cue_signature[:start]
    suffix = cue_signature[end:]
    prefix_count = (
        len(prefix)
        if _starts_with_sequence(final_signature, prefix)
        else 0
    )
    suffix_count = (
        len(suffix)
        if _ends_with_sequence(final_signature, suffix)
        else 0
    )
    if final_signature == [*prefix, *suffix]:
        return ""
    delete_only_span = not span.asr_word_indices and not alphanumeric_signature(span.asr_text)
    if delete_only_span and (
        _starts_with_sequence(suffix, final_signature)
        or _ends_with_sequence(prefix, final_signature)
    ):
        return ""
    if prefix_count + suffix_count >= len(final_signature):
        return final_text.strip()
    if not prefix_count and not suffix_count:
        return final_text.strip()

    final_token_spans = _token_character_spans(final_text)
    if len(final_token_spans) != len(final_signature):
        return final_text.strip()

    start_character = (
        final_token_spans[prefix_count][0]
        if prefix_count
        else 0
    )
    end_character = (
        final_token_spans[len(final_token_spans) - suffix_count][0]
        if suffix_count
        else len(final_text)
    )
    return final_text[start_character:end_character].strip()


def _apply_token_edits(
    source_text: str,
    edits: list[tuple[int, int, str]],
) -> str:
    token_spans = _token_character_spans(source_text)
    if not token_spans:
        inserted = " ".join(replacement.strip() for _, _, replacement in edits if replacement.strip())
        return _restore_terminal_punctuation(inserted, source_text)

    ordered_edits = sorted(
        enumerate(edits),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )
    pieces: list[str] = []
    cursor = 0
    previous_token_end = 0
    for _, (start_token, end_token, replacement) in ordered_edits:
        bounded_start = min(max(0, start_token), len(token_spans))
        bounded_end = min(max(bounded_start, end_token), len(token_spans))
        start_character = (
            token_spans[bounded_start][0]
            if bounded_start < len(token_spans)
            else len(source_text)
        )
        end_character = (
            token_spans[bounded_end - 1][1]
            if bounded_end > bounded_start
            else start_character
        )
        stripped_replacement = replacement.strip()
        replaces_contraction_suffix = (
            bounded_end > bounded_start
            and start_character > cursor
            and source_text[start_character - 1] in {"'", "’"}
        )
        if replaces_contraction_suffix:
            start_character -= 1
            if stripped_replacement:
                stripped_replacement = f" {stripped_replacement}"
        if (
            bounded_end > bounded_start
            and stripped_replacement
            and end_character < len(source_text)
            and source_text[end_character] in ",.;:!?"
            and stripped_replacement.endswith(source_text[end_character])
        ):
            end_character += 1
        if start_character < cursor or bounded_start < previous_token_end:
            continue
        pieces.append(source_text[cursor:start_character])
        if bounded_start == bounded_end and stripped_replacement:
            pieces.append(f" {stripped_replacement} ")
        else:
            pieces.append(stripped_replacement)
        cursor = end_character
        previous_token_end = bounded_end

    pieces.append(source_text[cursor:])
    normalized = re.sub(r"\s+", " ", "".join(pieces)).strip()
    normalized = re.sub(r"\s+([,.;:!?…])", r"\1", normalized)
    return _restore_terminal_punctuation(normalized, source_text)


def _token_character_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in token_texts(text):
        start = text.find(token, cursor)
        if start < 0:
            return []
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def _split_units(text: str) -> tuple[list[str], str]:
    stripped = text.strip()
    if not stripped:
        return [], " "
    if " " in stripped:
        return stripped.split(), " "
    if contains_character_level_script(stripped):
        return list(stripped), ""
    return [stripped], " "


def _cue_text_with_span_replacement(cue: Cue, span: DivergenceSpan, final_text: str) -> str:
    replacement = final_text.strip()
    cue_signature = alphanumeric_signature(cue.plain_text)
    span_signature = alphanumeric_signature(span.srt_text)
    if not cue_signature or not span_signature or len(span_signature) >= len(cue_signature):
        return replacement

    bounds = _find_subsequence_bounds(cue_signature, span_signature)
    if bounds is None:
        return replacement

    cue_tokens = token_texts(cue.plain_text)
    start, end = bounds
    final_signature = alphanumeric_signature(replacement)
    before_tokens = cue_tokens[:start]
    after_tokens = cue_tokens[end:]
    if _starts_with_sequence(final_signature, cue_signature[:start]):
        before_tokens = []
    if _ends_with_sequence(final_signature, cue_signature[end:]):
        after_tokens = []
    pieces = [*before_tokens, replacement, *after_tokens]
    text = " ".join(piece.strip() for piece in pieces if piece.strip())
    return _restore_terminal_punctuation(text, cue.plain_text)


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
