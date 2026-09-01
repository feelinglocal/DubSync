from __future__ import annotations

import re

from .editorial_guard import (
    EditorialGuardError,
    validate_adjudication_editorial_contract,
    validate_editorial_text,
)
from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag
from .style_profile import StyleProfile
from .subtitle_annotations import (
    alignment_token_character_spans,
    bracketed_screen_text_spans,
    cue_has_bracketed_screen_text,
    speech_text_for_alignment,
    text_without_bracketed_screen_text,
)
from .text_metrics import contains_character_level_script, display_width, token_texts, wrap_visual_width
from .tokenize import alphanumeric_signature


_TERMINAL_PUNCTUATION_RE = re.compile(r"([,.;:!?\u2026]+)\s*$")


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
        annotated_cue_ids = [
            cue_id
            for cue_id in cue_ids
            if cue_has_bracketed_screen_text(cues_by_id[cue_id])
        ]
        if not decision.final_text.strip():
            if len(cue_ids) == 1:
                cue_id = cue_ids[0]
                cue = cues_by_id[cue_id]
                bounds = _span_token_bounds_for_cue(
                    cue,
                    span,
                    cue_token_offsets[cue_id],
                )
                if (
                    cue_id in annotated_cue_ids and bounds is not None
                ) or _is_partial_cue_span(cue, span, bounds):
                    if bounds is not None:
                        candidate_edits = [
                            *token_edits_by_cue.get(cue_id, []),
                            (bounds[0], bounds[1], ""),
                        ]
                        changed_text = _apply_cue_token_edits(
                            cue,
                            candidate_edits,
                        )
                        if changed_text is None:
                            flags.append(_screen_text_adjudication_hold(cue_ids, span, decision))
                            continue
                        token_edits_by_cue[cue_id] = candidate_edits
                    else:
                        if cue_id in annotated_cue_ids:
                            flags.append(_screen_text_adjudication_hold(cue_ids, span, decision))
                            continue
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
                if cue_id in annotated_cue_ids:
                    flags.append(_screen_text_adjudication_hold(cue_ids, span, decision))
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

        guard_flag = _editorial_guard_rejection(span, decision, cue_ids)
        if guard_flag is not None:
            flags.append(guard_flag)
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
                    candidate_edits = [
                        *token_edits_by_cue.get(adlib_cue_id, []),
                        (insertion_offset, insertion_offset, decision.final_text),
                    ]
                    changed_text = _apply_cue_token_edits(
                        cue,
                        candidate_edits,
                    )
                    if changed_text is None:
                        flags.append(
                            _screen_text_adjudication_hold([adlib_cue_id], span, decision)
                        )
                        continue
                    token_edits_by_cue[adlib_cue_id] = candidate_edits
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
                if cue_has_bracketed_screen_text(cue):
                    flags.append(_screen_text_adjudication_hold([adlib_cue_id], span, decision))
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
                candidate_edits = [
                    *token_edits_by_cue.get(cue_id, []),
                    (bounds[0], bounds[1], localized_replacement),
                ]
                changed_text = _apply_cue_token_edits(
                    cue,
                    candidate_edits,
                )
                if changed_text is None:
                    flags.append(_screen_text_adjudication_hold(cue_ids, span, decision))
                    continue
                token_edits_by_cue[cue_id] = candidate_edits
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

        if annotated_cue_ids:
            flags.append(_screen_text_adjudication_hold(cue_ids, span, decision))
            continue

        if len(cue_ids) > 1 and span.srt_token_indices and len(alphanumeric_signature(decision.final_text)) == 1:
            applied_multi_cue_edit = False
            for position, cue_id in enumerate(cue_ids):
                cue = cues_by_id[cue_id]
                bounds = _span_token_bounds_for_cue(
                    cue,
                    span,
                    cue_token_offsets[cue_id],
                )
                if bounds is None:
                    continue
                token_edits_by_cue.setdefault(cue_id, []).append(
                    (bounds[0], bounds[1], decision.final_text if position == 0 else "")
                )
                applied_multi_cue_edit = True
            if applied_multi_cue_edit:
                flags.append(
                    QCFlag(
                        kind="text_changed",
                        cue_ids=cue_ids,
                        message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                        confidence=decision.confidence,
                        old_text="\n".join(cues_by_id[cue_id].text for cue_id in cue_ids),
                        new_text=decision.final_text,
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
        changed_text = _apply_cue_token_edits(cues_by_id[cue_id], edits)
        if changed_text is None:
            continue
        final_token_edit_text_by_cue[cue_id] = changed_text
        if not alphanumeric_signature(changed_text):
            removed_cue_ids.add(cue_id)
            replacements_by_cue.pop(cue_id, None)
            continue
        replacements_by_cue[cue_id] = (
            changed_text.splitlines()
            if cue_has_bracketed_screen_text(cues_by_id[cue_id])
            else flow_text_to_lines(
                changed_text,
                profile.max_chars_per_line,
                profile.max_lines_per_cue,
            )
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

    updated, flags = _restore_cues_rejected_by_editorial_guard(
        cues_by_id,
        updated,
        flags,
        changed_cue_ids=set(replacements_by_cue),
    )
    return _merge_adlibs_positionally(updated, adlib_cues), flags


def _editorial_guard_rejection(
    span: DivergenceSpan,
    decision: AdjudicationDecision,
    cue_ids: list[int],
) -> QCFlag | None:
    try:
        validate_adjudication_editorial_contract(
            span,
            decision,
            allow_word_change=decision.verdict in {"use_audio", "hybrid"},
        )
    except EditorialGuardError as exc:
        return QCFlag(
            kind="editorial_guard_rejected",
            cue_ids=cue_ids,
            message=str(exc),
            severity="error",
            confidence=decision.confidence,
            old_text=span.srt_text,
            new_text=decision.final_text,
            start=span.start,
            end=span.end,
        )
    return None


def _screen_text_adjudication_hold(
    cue_ids: list[int],
    span: DivergenceSpan,
    decision: AdjudicationDecision,
) -> QCFlag:
    return QCFlag(
        kind="screen_text_adjudication_held",
        cue_ids=cue_ids,
        message=(
            "Adjudication was held because its alignment-token edit could not be "
            "reconstructed without risking bracketed screen text or its source layout."
        ),
        severity="error",
        confidence=decision.confidence,
        old_text=span.srt_text,
        new_text=decision.final_text,
        start=span.start,
        end=span.end,
    )


def _restore_cues_rejected_by_editorial_guard(
    source_by_id: dict[int, Cue],
    updated: list[Cue],
    flags: list[QCFlag],
    *,
    changed_cue_ids: set[int],
) -> tuple[list[Cue], list[QCFlag]]:
    rejected: dict[int, QCFlag] = {}
    for cue in updated:
        source = source_by_id.get(cue.index)
        if source is None or cue.index not in changed_cue_ids:
            continue
        try:
            validate_editorial_text(source.text, cue.text, allow_word_change=True)
        except EditorialGuardError as exc:
            rejected[cue.index] = QCFlag(
                kind="editorial_guard_rejected",
                cue_ids=[cue.index],
                message=str(exc),
                severity="error",
                old_text=source.text,
                new_text=cue.text,
                start=source.start_ms / 1000.0,
                end=source.end_ms / 1000.0,
            )
    if not rejected:
        return updated, flags

    rejected_ids = set(rejected)
    for flag in flags:
        if flag.kind == "text_changed" and rejected_ids.intersection(flag.cue_ids):
            rejected_ids.update(flag.cue_ids)

    restored = [
        source_by_id[cue.index]
        if cue.index in rejected_ids and cue.index in source_by_id
        else cue
        for cue in updated
    ]
    retained_flags = [
        flag
        for flag in flags
        if not (
            flag.kind == "text_changed"
            and rejected_ids.intersection(flag.cue_ids)
        )
    ]
    guard_flags = [
        flag.model_copy(update={"cue_ids": sorted(rejected_ids)})
        for flag in rejected.values()
    ]
    return restored, [*retained_flags, *guard_flags]


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
        offset += len(alphanumeric_signature(speech_text_for_alignment(cue)))
    return offsets


def _span_token_bounds_for_cue(
    cue: Cue,
    span: DivergenceSpan,
    cue_token_offset: int,
) -> tuple[int, int] | None:
    cue_signature = alphanumeric_signature(speech_text_for_alignment(cue))
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
    cue_token_count = len(alphanumeric_signature(speech_text_for_alignment(cue)))
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
        return len(alphanumeric_signature(speech_text_for_alignment(cue)))
    return None


def _localized_indexed_replacement(
    cue: Cue,
    span: DivergenceSpan,
    bounds: tuple[int, int],
    final_text: str,
) -> str:
    cue_signature = alphanumeric_signature(speech_text_for_alignment(cue))
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
    return _apply_token_edits_with_spans(
        source_text,
        edits,
        _token_character_spans(source_text),
    )


def _apply_cue_token_edits(
    cue: Cue,
    edits: list[tuple[int, int, str]],
) -> str | None:
    if not cue_has_bracketed_screen_text(cue):
        return _apply_token_edits(cue.plain_text, edits)
    token_spans = alignment_token_character_spans(cue)
    annotation_spans = bracketed_screen_text_spans(cue.text)
    if token_spans is None or not _annotated_token_edits_are_safe(
        edits,
        token_spans,
        annotation_spans,
    ):
        return None
    return _apply_token_edits_with_spans(
        cue.text,
        edits,
        list(token_spans),
        append_at_last_token=True,
        preserve_line_breaks=True,
        protected_fragments=_screen_text_protected_fragments(cue),
    )


def _annotated_token_edits_are_safe(
    edits: list[tuple[int, int, str]],
    token_spans: tuple[tuple[int, int], ...],
    annotation_spans: tuple[tuple[int, int], ...],
) -> bool:
    previous_end = 0
    for _, (start_token, end_token, replacement) in sorted(
        enumerate(edits),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    ):
        if (
            start_token < previous_end
            or start_token < 0
            or end_token < start_token
            or end_token > len(token_spans)
            or any(character in replacement for character in "[]\r\n")
        ):
            return False

        if start_token < len(token_spans):
            start_character = token_spans[start_token][0]
        elif token_spans:
            start_character = token_spans[-1][1]
        else:
            return False
        end_character = (
            token_spans[end_token - 1][1]
            if end_token > start_token
            else start_character
        )
        if any(
            start_character < annotation_end and end_character > annotation_start
            for annotation_start, annotation_end in annotation_spans
        ):
            return False
        if start_token == end_token and any(
            annotation_start < start_character < annotation_end
            for annotation_start, annotation_end in annotation_spans
        ):
            return False

        previous_end = end_token
    return True


def _screen_text_protected_fragments(cue: Cue) -> tuple[str, ...]:
    residue_lines = text_without_bracketed_screen_text(cue.text).split("\n")
    standalone_lines = [
        line
        for line, residue in zip(cue.lines, residue_lines, strict=True)
        if line and not residue.strip()
    ]
    annotation_fragments = [
        cue.text[start:end]
        for start, end in bracketed_screen_text_spans(cue.text)
    ]
    return tuple([*standalone_lines, *annotation_fragments])


def _apply_token_edits_with_spans(
    source_text: str,
    edits: list[tuple[int, int, str]],
    token_spans: list[tuple[int, int]],
    *,
    append_at_last_token: bool = False,
    preserve_line_breaks: bool = False,
    protected_fragments: tuple[str, ...] = (),
) -> str:
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
            else token_spans[-1][1]
            if append_at_last_token
            else len(source_text)
        )
        end_character = (
            token_spans[bounded_end - 1][1]
            if bounded_end > bounded_start
            else start_character
        )
        stripped_replacement = replacement.strip()
        if bounded_start == bounded_end == len(token_spans) and stripped_replacement:
            terminal = _TERMINAL_PUNCTUATION_RE.search(source_text.rstrip())
            if terminal is not None and not _TERMINAL_PUNCTUATION_RE.search(stripped_replacement):
                start_character = terminal.start(1)
                end_character = start_character
        replaces_contraction_suffix = (
            bounded_end > bounded_start
            and start_character > cursor
            and source_text[start_character - 1] in {"'", "\u2019"}
        )
        if replaces_contraction_suffix:
            start_character -= 1
            if stripped_replacement:
                stripped_replacement = f" {stripped_replacement}"
        if (
            bounded_end > bounded_start
            and stripped_replacement
            and end_character < len(source_text)
            and source_text[end_character] == "-"
        ):
            end_character += 1
        if (
            bounded_end > bounded_start
            and stripped_replacement
            and end_character < len(source_text)
            and source_text[end_character] in ",.;:!?"
            and _TERMINAL_PUNCTUATION_RE.search(stripped_replacement)
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
    raw_text, protected = _protect_text_fragments("".join(pieces), protected_fragments)
    normalized = re.sub(
        r"[^\S\r\n]+" if preserve_line_breaks else r"\s+",
        " ",
        raw_text,
    ).strip()
    if preserve_line_breaks:
        normalized = re.sub(r" *(\r?\n) *", r"\1", normalized)
    punctuation_spacing = r"[^\S\r\n]+" if preserve_line_breaks else r"\s+"
    normalized = re.sub(rf"{punctuation_spacing}([,.;:!?\u2026])", r"\1", normalized)
    for marker, fragment in protected:
        normalized = normalized.replace(marker, fragment)
    return _restore_terminal_punctuation(normalized, source_text)


def _protect_text_fragments(
    text: str,
    fragments: tuple[str, ...],
) -> tuple[str, list[tuple[str, str]]]:
    protected: list[tuple[str, str]] = []
    for index, fragment in enumerate(sorted(fragments, key=len, reverse=True)):
        if not fragment:
            continue
        marker = f"\ufdd0{index}\ufdd1"
        while marker in text:
            marker += "\ufdd1"
        if fragment not in text:
            continue
        text = text.replace(fragment, marker, 1)
        protected.append((marker, fragment))
    return text, protected


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
    cue_signature = alphanumeric_signature(speech_text_for_alignment(cue))
    span_signature = alphanumeric_signature(span.srt_text)
    if not cue_signature or not span_signature or len(span_signature) >= len(cue_signature):
        return replacement

    bounds = _find_subsequence_bounds(cue_signature, span_signature)
    if bounds is None:
        return replacement

    cue_tokens = [
        cue.plain_text[start:end]
        for start, end in _token_character_spans(cue.plain_text)
    ]
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
