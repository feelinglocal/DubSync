from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite

from .adjudication_regions import is_joint_region
from .editorial_guard import (
    EditorialGuardError,
    validate_adjudication_editorial_contract,
    validate_editorial_text,
)
from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag, TokenMatch, Word
from .style_profile import StyleProfile
from .subtitle_annotations import (
    alignment_token_character_spans,
    bracketed_screen_text_spans,
    cue_has_bracketed_screen_text,
    speech_text_for_alignment,
    text_without_bracketed_screen_text,
)
from .text_metrics import contains_character_level_script, display_width, join_word_texts, token_character_spans, token_texts, wrap_visual_width
from .tokenize import alphanumeric_signature


_TERMINAL_PUNCTUATION_RE = re.compile(r"([,.;:!?\u2026、。！？]+)\s*$")


class ReplacementOwnershipError(ValueError):
    """A source edit crosses speech groups without a safe ownership cut."""


def protected_replacement_targets(span: DivergenceSpan, edits: dict[int, tuple[int, int, str]]) -> set[int]:
    targets = set(edits) - set(span.cue_ids)
    if is_joint_region(span):
        # The joint region newly takes ownership of a formerly retained word
        # in a fully consumed interior cue as well as its external destination.
        targets.update(span.cue_ids[1:])
    return targets


@dataclass(frozen=True)
class WholeCueReplacementPlan:
    edits: dict[int, tuple[int, int, str]]
    word_indices_by_cue: dict[int, list[int]]


def whole_cue_replacement_plan(
    cues: list[Cue], span: DivergenceSpan, final_text: str, words: list[Word] | None,
    *, max_intra_cue_gap: float = 1.5, token_matches: list[TokenMatch] | None = None,
) -> WholeCueReplacementPlan | None:
    """Prevent dense-cluster timing from choosing unrelated replacement speech.

    Only a complete indexed source replacement spanning a gap that re-cue
    would trim needs this guard. A shortened response must identify one exact
    whole-word window. A complete approved sequence may instead retain its
    first group and prefix one word to a proved continuation in the next cue.
    None leaves ordinary single-group and partial-cue edits unchanged.
    """
    if len(span.cue_ids) != 1 or len(span.asr_word_indices) < 2 or words is None:
        return None
    cue = next((cue for cue in cues if cue.index == span.cue_ids[0]), None)
    if cue is None or cue_has_bracketed_screen_text(cue):
        return None
    source_tokens = alphanumeric_signature(speech_text_for_alignment(cue))
    offset = _cue_token_offsets(cues)[cue.index]
    if (
        not source_tokens or span.srt_token_indices != list(range(offset, offset + len(source_tokens)))
        or alphanumeric_signature(span.srt_text) != source_tokens
        or any(index < 0 or index >= len(words) for index in span.asr_word_indices)
    ):
        return None
    selected = [words[index] for index in span.asr_word_indices]
    ordered = sorted(selected, key=lambda word: (word.start, word.end))
    if not any(right.start - left.end > max_intra_cue_gap for left, right in zip(ordered, ordered[1:])):
        return None
    problem = (
        "The complete cue replacement spans separate speech groups, but its approved text has no "
        "unique exact word window within one group. Source text and timing were held for review."
    )
    signatures = [alphanumeric_signature(word.text) for word in selected]
    if (
        span.asr_word_indices != list(range(span.asr_word_indices[0], span.asr_word_indices[-1] + 1))
        or any(not signature for signature in signatures)
        or any(not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start for word in selected)
        or any(right.start < left.end for left, right in zip(selected, selected[1:]))
        or [token for signature in signatures for token in signature] != alphanumeric_signature(span.asr_text)
    ):
        raise ReplacementOwnershipError(problem)
    accepted = alphanumeric_signature(final_text)
    if accepted == [token for signature in signatures for token in signature]:
        continuation = _whole_cue_continuation_plan(
            cues, span, final_text, words, token_matches or [], max_intra_cue_gap,
        )
        if continuation is not None:
            return continuation
    windows: list[tuple[int, int]] = []
    for start in range(len(selected)):
        tokens: list[str] = []
        for end in range(start, len(selected)):
            tokens.extend(signatures[end])
            if tokens == accepted:
                windows.append((start, end + 1))
            if len(tokens) >= len(accepted):
                break
    if len(windows) != 1:
        raise ReplacementOwnershipError(problem)
    start, end = windows[0]
    if any(
        selected[index + 1].start - selected[index].end > max_intra_cue_gap
        for index in range(start, end - 1)
    ):
        raise ReplacementOwnershipError(problem)
    return WholeCueReplacementPlan(
        edits={cue.index: (0, len(source_tokens), final_text)},
        word_indices_by_cue={cue.index: span.asr_word_indices[start:end]},
    )


def _whole_cue_continuation_plan(
    cues: list[Cue], span: DivergenceSpan, final_text: str, words: list[Word],
    token_matches: list[TokenMatch], max_intra_cue_gap: float,
) -> WholeCueReplacementPlan | None:
    """Prove a one-word prefix using the next cue's retained acoustic group."""
    selected = [words[index] for index in span.asr_word_indices]
    cuts = [position for position in range(1, len(selected))
            if selected[position].start - selected[position - 1].end > max_intra_cue_gap]
    if len(cuts) != 1 or cuts[0] != len(selected) - 1 or cuts[0] < 2:
        return None
    if len(alphanumeric_signature(selected[-1].text)) != 1 or _has_sentence_terminal(selected[-1].text):
        return None
    source_position = next(index for index, cue in enumerate(cues) if cue.index == span.cue_ids[0])
    if source_position + 1 >= len(cues):
        return None
    source, target = cues[source_position:source_position + 2]
    if (
        target.index != span.right_anchor_cue_id or cue_has_bracketed_screen_text(target)
        or any(mark in source.text + target.text + final_text for mark in "♪♫")
        or span.start != selected[0].start or span.end != selected[-1].end
        or span.right_anchor_start is None or not isfinite(span.right_anchor_start)
        or not 0 <= span.right_anchor_start - selected[-1].end <= 0.2
        or (span.left_anchor_end is not None and (
            not isfinite(span.left_anchor_end) or span.left_anchor_end > selected[0].start
        ))
    ):
        return None
    target_tokens = alphanumeric_signature(speech_text_for_alignment(target))
    offset = _cue_token_offsets(cues)[target.index]
    retained = sorted((match for match in token_matches if match.cue_id == target.index),
                      key=lambda match: match.srt_token_index)
    if (
        len(retained) < 2 or retained[0].srt_token_index != offset
        or retained[0].asr_word_index != span.asr_word_indices[-1] + 1
        or any(
            match.score != 1.0 or not offset <= match.srt_token_index < offset + len(target_tokens)
            or not 0 <= match.asr_word_index < len(words)
            or alphanumeric_signature(words[match.asr_word_index].text) != [target_tokens[match.srt_token_index - offset]]
            for match in retained
        )
        or any(right.srt_token_index <= left.srt_token_index or right.asr_word_index <= left.asr_word_index
               for left, right in zip(retained, retained[1:]))
        or not any(right.srt_token_index == left.srt_token_index + 1 and right.asr_word_index == left.asr_word_index + 1
                   for left, right in zip(retained, retained[1:]))
    ):
        return None
    following = words[retained[0].asr_word_index:retained[-1].asr_word_index + 1]
    evidence = [*selected, *following]
    if (
        following[0].start != span.right_anchor_start
        or any(not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start
               or not alphanumeric_signature(word.text)
               or (word.confidence is not None and word.confidence < 0.8) for word in evidence)
        or any(not 0 <= right.start - left.end <= 0.2
               for group in (selected[:-1], [selected[-1], *following])
               for left, right in zip(group, group[1:]))
    ):
        return None
    speakers = {speaker for speaker in [
        *span.speaker_ids, source.speaker_id, target.speaker_id,
        span.left_anchor_speaker_id, span.right_anchor_speaker_id,
        *(word.speaker_id for word in evidence),
    ] if speaker is not None}
    if len(speakers) > 1:
        return None
    token_spans = _token_character_spans(final_text)
    token_cut = sum(len(alphanumeric_signature(word.text)) for word in selected[:-1])
    if len(token_spans) != len(alphanumeric_signature(final_text)) or not 0 < token_cut < len(token_spans):
        return None
    character_cut = token_spans[token_cut][0]
    if not any(character.isspace() for character in final_text[token_spans[token_cut - 1][1]:character_cut]):
        return None
    return WholeCueReplacementPlan(
        edits={source.index: (0, len(alphanumeric_signature(speech_text_for_alignment(source))), final_text[:character_cut].strip()),
               target.index: (0, 0, final_text[character_cut:].strip())},
        word_indices_by_cue={source.index: span.asr_word_indices[:-1], target.index: span.asr_word_indices[-1:]},
    )


def apply_adjudication_decisions(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    profile: StyleProfile,
    adlib_cue_ids_by_case: dict[str, int] | None = None,
    *,
    protected_cue_ids: set[int] | None = None,
    words: list[Word] | None = None,
    max_intra_cue_gap: float = 1.5,
    token_matches: list[TokenMatch] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    by_case = {decision.case_id: decision for decision in decisions}
    cues_by_id = {cue.index: cue for cue in cues}
    cue_token_offsets = _cue_token_offsets(cues)
    prefix_replacement_targets = single_token_prefix_replacement_targets(cues, spans, decisions)
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
        indexed_edits = None
        if is_joint_region(span):
            try:
                indexed_edits = indexed_multi_cue_replacements(cues, span, decision.final_text, words=words)
                if indexed_edits is None:
                    raise ReplacementOwnershipError("The joint source region could not be reconstructed; source text was held for review.")
            except ReplacementOwnershipError as exc:
                flags.append(QCFlag(
                    kind="adjudication_replacement_ownership_held",
                    cue_ids=list(cue_ids),
                    severity="warning", message=str(exc), confidence=decision.confidence,
                    old_text=span.srt_text, new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
        if not decision.final_text.strip():
            if len(cue_ids) > 1 and span.srt_token_indices:
                edits = indexed_multi_cue_replacements(cues, span, "")
                if edits is None:
                    flags.append(
                        _screen_text_adjudication_hold(cue_ids, span, decision)
                        if annotated_cue_ids else QCFlag(
                            kind="adjudication_span_edit_held", cue_ids=cue_ids,
                            message="The indexed source span could not be reconstructed; preserving source cues for review.",
                            severity="error", confidence=decision.confidence,
                            old_text=span.srt_text, new_text="", start=span.start, end=span.end,
                        )
                    )
                    continue
                partial_ids = [
                    cue_id for cue_id, (start, end, _) in edits.items()
                    if start > 0 or end < len(alphanumeric_signature(cues_by_id[cue_id].plain_text))
                ]
                if partial_ids:
                    for cue_id in partial_ids:
                        token_edits_by_cue.setdefault(cue_id, []).append(edits[cue_id])
                    flags.append(QCFlag(
                        kind="text_changed", cue_ids=partial_ids,
                        message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                        confidence=decision.confidence,
                        old_text="\n".join(cues_by_id[cue_id].text for cue_id in partial_ids),
                        new_text="", start=span.start, end=span.end,
                    ))
                    # Complete cue deletions still follow the configured drop
                    # policy; a partial deletion must never consume residue.
                    cue_ids = [cue_id for cue_id in cue_ids if cue_id not in partial_ids]
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
            # Truncating both bounds can collapse a short ASR word to zero
            # length (4.003-4.004 s), which fails the export. Round, and keep
            # the generated envelope at least one millisecond long.
            adlib_start_ms = max(0, round((span.start or 0.0) * 1000))
            adlib_cues.append(
                Cue(
                    index=adlib_cue_id,
                    start_ms=adlib_start_ms,
                    end_ms=max(adlib_start_ms + 1, round((span.end or span.start or 0.0) * 1000)),
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

        try:
            whole_plan = whole_cue_replacement_plan(
                cues, span, decision.final_text, words, max_intra_cue_gap=max_intra_cue_gap,
                token_matches=token_matches,
            )
            if whole_plan is not None:
                indexed_edits = whole_plan.edits
        except ReplacementOwnershipError as exc:
            flags.append(QCFlag(
                kind="adjudication_replacement_ownership_held", cue_ids=list(cue_ids),
                severity="warning", message=str(exc), confidence=decision.confidence,
                old_text=span.srt_text, new_text=decision.final_text, start=span.start, end=span.end,
            ))
            continue

        if indexed_edits is None and span.srt_token_indices and (
            len(cue_ids) > 1
            or (span.right_anchor_cue_id is not None and span.right_anchor_cue_id not in cue_ids)
        ):
            try:
                indexed_edits = indexed_multi_cue_replacements(
                    cues, span, decision.final_text,
                    replacement_target=prefix_replacement_targets.get(span.case_id), words=words,
                )
            except ReplacementOwnershipError as exc:
                flags.append(QCFlag(
                    kind="adjudication_replacement_ownership_held",
                    cue_ids=list(cue_ids),
                    severity="warning", message=str(exc), confidence=decision.confidence,
                    old_text=span.srt_text, new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue

        if len(cue_ids) == 1 and span.srt_token_indices and indexed_edits is None:
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

        if (len(cue_ids) > 1 or indexed_edits is not None) and span.srt_token_indices:
            edits = indexed_edits
            if edits is None:
                flags.append(QCFlag(
                    kind="adjudication_span_edit_held",
                    cue_ids=cue_ids,
                    message="The indexed source span could not be reconstructed; preserving source cues for review.",
                    severity="error",
                    confidence=decision.confidence,
                    old_text=span.srt_text,
                    new_text=decision.final_text,
                    start=span.start,
                    end=span.end,
                ))
                continue
            # A separate held span may share an edited cue. Only newly targeted
            # neighbors need this guard; keep independently approved text edits.
            if protected_replacement_targets(span, edits) & (protected_cue_ids or set()):
                flags.append(QCFlag(
                    kind="adjudication_replacement_ownership_held",
                    cue_ids=list(edits), severity="warning",
                    message=(
                        "The replacement would change word ownership inside a protected cue. The complete "
                        "replacement was held so its spoken word cannot be lost during source restoration."
                    ),
                    confidence=decision.confidence, old_text=span.srt_text,
                    new_text=decision.final_text, start=span.start, end=span.end,
                ))
                continue
            for cue_id, edit in edits.items():
                token_edits_by_cue.setdefault(cue_id, []).append(edit)
            flags.append(QCFlag(
                kind="text_changed",
                cue_ids=list(edits),
                message=f"Adjudication verdict {decision.verdict}: {decision.reason}",
                confidence=decision.confidence,
                old_text="\n".join(cues_by_id[cue_id].text for cue_id in edits),
                new_text=decision.final_text,
                start=span.start,
                end=span.end,
            ))
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
        if any(start < end and not text.strip() for start, end, text in edits):
            changed_text = _remove_deleted_dialogue_turn_markers(cues_by_id[cue_id], changed_text)
        if (
            any(start == 0 and end > 0 and not text.strip() for start, end, text in edits)
            and not cue_has_bracketed_screen_text(cues_by_id[cue_id])
            and cues_by_id[cue_id].text.lstrip()[:1].isalnum()
        ):
            # A removed opening word can leave its separator before the next
            # spoken word. Preserve authored leading punctuation and markup.
            changed_text = re.sub(r"^[,;:]+\s*", "", changed_text.lstrip())
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
            update={"new_text": "\n".join(final_token_edit_text_by_cue[cue_id] for cue_id in flag.cue_ids)}
        )
        if (
            flag.kind == "text_changed"
            and flag.cue_ids
            and all(cue_id in final_token_edit_text_by_cue for cue_id in flag.cue_ids)
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


def _remove_deleted_dialogue_turn_markers(source: Cue, text: str) -> str:
    """Remove presentation residue only after a confirmed whole-turn deletion."""
    marker = re.compile(r"(?<!\S)[-–—](?=[\s.!?,;:…])")
    original_markers = list(marker.finditer(source.text))
    current_markers = list(marker.finditer(text))
    if (
        cue_has_bracketed_screen_text(source)
        or len(original_markers) < 2 or len(current_markers) < 2
        or source.text[:original_markers[0].start()].strip()
        or text[:current_markers[0].start()].strip()
    ):
        return text
    turns = [
        (match.group(), text[match.end():current_markers[i + 1].start() if i + 1 < len(current_markers) else len(text)].strip())
        for i, match in enumerate(current_markers)
    ]
    retained = [(dash, content) for dash, content in turns if not re.fullmatch(r"[\s.!?,;:…]*", content)]
    if len(retained) == len(turns):
        return text
    separator = "\n" if "\n" in source.text else " "
    candidate = (retained[0][1] if len(retained) == 1 else
                 separator.join(f"{dash} {content}" for dash, content in retained))
    return candidate if alphanumeric_signature(candidate) == alphanumeric_signature(text) else text


def indexed_multi_cue_replacements(
    cues: list[Cue],
    span: DivergenceSpan,
    final_text: str,
    *,
    replacement_target: int | None = None,
    words: list[Word] | None = None,
) -> dict[int, tuple[int, int, str]] | None:
    """Partition one exact source-token edit without consuming its cue residue.

    Replacement tokens follow the source span's contribution to each cue,
    preferring nearby corroborated sentence boundaries where available.
    An anchored single-cue tail can transfer its replacement to the next cue.
    The pipeline uses these same pieces to assign acoustic evidence, so text
    and timing cannot independently choose different cue boundaries.
    """
    cue_ids = list(dict.fromkeys(span.cue_ids))
    indices = sorted(set(span.srt_token_indices))
    cues_by_id = {cue.index: cue for cue in cues}
    if (
        not cue_ids
        or not indices
        or indices != list(range(indices[0], indices[-1] + 1))
        or any(cue_id not in cues_by_id for cue_id in cue_ids)
        or any(cue_has_bracketed_screen_text(cues_by_id[cue_id]) for cue_id in cue_ids)
    ):
        return None
    offsets = _cue_token_offsets(cues)
    bounds_by_cue: dict[int, tuple[int, int]] = {}
    covered_indices: list[int] = []
    covered_tokens: list[str] = []
    for cue_id in cue_ids:
        signature = alphanumeric_signature(speech_text_for_alignment(cues_by_id[cue_id]))
        local_indices = [index - offsets[cue_id] for index in indices
                         if offsets[cue_id] <= index < offsets[cue_id] + len(signature)]
        if not local_indices:
            return None
        start, end = local_indices[0], local_indices[-1] + 1
        bounds_by_cue[cue_id] = (start, end)
        covered_indices.extend(range(offsets[cue_id] + start, offsets[cue_id] + end))
        covered_tokens.extend(signature[start:end])
    if covered_indices != indices or covered_tokens != alphanumeric_signature(span.srt_text):
        return None

    token_spans = _token_character_spans(final_text)
    if len(token_spans) != len(alphanumeric_signature(final_text)):
        return None
    # Alignment tokens split apostrophes and numeric punctuation. Keep such
    # lexical units together so "aren't" cannot become "aren'" / "t".
    unit_spans: list[tuple[int, int]] = []
    for start, end in token_spans:
        if (
            unit_spans
            and not any(character.isspace() for character in final_text[unit_spans[-1][1]:start])
            and not contains_character_level_script(final_text[unit_spans[-1][0]:end])
        ):
            unit_spans[-1] = (unit_spans[-1][0], end)
        else:
            unit_spans.append((start, end))
    sentence_boundaries = _replacement_sentence_boundaries(final_text, unit_spans)
    if is_joint_region(span):
        return _joint_region_replacement_edits(cues_by_id, bounds_by_cue, span, final_text, unit_spans, words)
    if replacement_target is None:
        acoustic_edits = _acoustic_tail_replacement_edits(
            cues_by_id, bounds_by_cue, span, final_text, unit_spans, words,
        )
        if acoustic_edits is not None:
            return acoustic_edits
    anchored_target = _anchored_prefix_replacement_target(
        cues_by_id, bounds_by_cue, span, final_text,
    )
    if replacement_target is None:
        replacement_target = anchored_target
    if anchored_target is not None and replacement_target == anchored_target and anchored_target not in cue_ids:
        # A source-only tail can precede an ASR-only prefix of the next cue.
        # Retain the next cue's text and insert before its exact right anchor.
        return {
            **{cue_id: (*bounds, "") for cue_id, bounds in bounds_by_cue.items()},
            anchored_target: (0, 0, final_text),
        }
    if len(cue_ids) < 2:
        return None
    if replacement_target is None:
        replacement_target = _contained_sentence_replacement_target(
            cues_by_id, bounds_by_cue, span, final_text, sentence_boundaries,
        )
    if replacement_target is None and len(unit_spans) > 1:
        replacement_target = _continuation_prefix_replacement_target(
            cues_by_id, bounds_by_cue, span, final_text,
        )
    if len(unit_spans) == 1:
        replacement_target = replacement_target or cue_ids[0]
    if replacement_target in cue_ids:
        return {cue_id: (*bounds, final_text if cue_id == replacement_target else "")
                for cue_id, bounds in bounds_by_cue.items()}

    total_weight = len(covered_tokens)
    cumulative_weight = 0
    boundaries: list[int] = []
    for start, end in bounds_by_cue.values():
        cumulative_weight += end - start
        boundaries.append((len(unit_spans) * cumulative_weight * 2 + total_weight) // (2 * total_weight))

    previous_boundary = 0
    previous_character = 0
    result: dict[int, tuple[int, int, str]] = {}
    for position, (cue_id, (start, end)) in enumerate(bounds_by_cue.items()):
        boundary = boundaries[position]
        if position < len(boundaries) - 1 and _has_sentence_terminal(cues_by_id[cue_id].plain_text):
            # Do not move a cut more than two complete lexical units, cross
            # another cut, or resolve equally near sentence breaks arbitrarily.
            candidates = [
                candidate for candidate in sentence_boundaries
                if previous_boundary < candidate < boundaries[position + 1]
                and abs(candidate - boundary) <= 2
                and not _is_preserved_source_internal_boundary(
                    cues_by_id, bounds_by_cue, covered_tokens,
                    final_text, unit_spans[candidate][0],
                )
            ]
            if candidates:
                distance = min(abs(candidate - boundary) for candidate in candidates)
                nearest = [candidate for candidate in candidates if abs(candidate - boundary) == distance]
                if len(nearest) == 1:
                    boundary = nearest[0]
        end_character = unit_spans[boundary][0] if boundary < len(unit_spans) else len(final_text)
        result[cue_id] = (start, end, final_text[previous_character:end_character].strip())
        previous_boundary = boundary
        previous_character = end_character
    return result


def _anchored_prefix_replacement_target(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
    max_gap_seconds: float = 0.2,
) -> int | None:
    """Place one unchanged ASR word with its unambiguous retained continuation.

    Deleting the unspoken tail of one cue and prefix of another can collapse
    into a single spoken word. Source proportions cannot determine ownership
    when both sides retain text; use the matched words surrounding that edit.
    """
    signature = alphanumeric_signature(final_text)
    target_id = _replacement_continuation_target(cues_by_id, bounds_by_cue, span)
    if (
        len(signature) != 1
        or signature != alphanumeric_signature(span.asr_text)
        or _has_sentence_terminal(final_text)
        or not span.asr_word_indices
        or target_id is None
        or len(set(span.speaker_ids)) > 1
        or (
            span.right_anchor_speaker_id is not None
            and span.speaker_ids
            and set(span.speaker_ids) != {span.right_anchor_speaker_id}
        )
        or not all(value is not None and isfinite(value) for value in (
            span.start, span.end, span.left_anchor_end, span.right_anchor_start,
        ))
    ):
        return None
    if (
        not span.left_anchor_end < span.start < span.end <= span.right_anchor_start
        or span.start - span.left_anchor_end <= max_gap_seconds
        or span.right_anchor_start - span.end > max_gap_seconds
    ):
        return None
    return target_id


def _replacement_continuation_target(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
) -> int | None:
    """Require retained source tokens on both sides of one contiguous edit."""
    cue_ids = list(bounds_by_cue)
    first_id, last_id = cue_ids[0], cue_ids[-1]
    target_id = span.right_anchor_cue_id
    first_count = len(alphanumeric_signature(cues_by_id[first_id].plain_text))
    first_start, first_end = bounds_by_cue[first_id]
    if (
        target_id not in cues_by_id
        or cue_has_bracketed_screen_text(cues_by_id[target_id])
        or span.left_anchor_cue_id != first_id
        or not 0 < first_start < first_end == first_count
    ):
        return None
    for cue_id in cue_ids[1:]:
        start, end = bounds_by_cue[cue_id]
        count = len(alphanumeric_signature(cues_by_id[cue_id].plain_text))
        if start != 0 or (cue_id != last_id and end != count):
            return None
    last_count = len(alphanumeric_signature(cues_by_id[last_id].plain_text))
    if target_id == last_id:
        return target_id if bounds_by_cue[last_id][1] < last_count else None
    ordered_ids = list(cues_by_id)
    last_position = ordered_ids.index(last_id)
    if (
        bounds_by_cue[last_id][1] == last_count
        and last_position + 1 < len(ordered_ids)
        and ordered_ids[last_position + 1] == target_id
    ):
        return target_id
    return None


def _acoustic_tail_replacement_edits(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
    unit_spans: list[tuple[int, int]],
    words: list[Word] | None,
    min_phrase_gap_seconds: float = 0.8,
    max_anchor_gap_seconds: float = 0.2,
) -> dict[int, tuple[int, int, str]] | None:
    """Split a rewritten tail only at one measured gap between anchored phrases.

    The gap follows the default speech-grouping threshold. No cue endpoint is
    inferred here: both resulting pieces retain complete, existing ASR words.
    """
    signature = alphanumeric_signature(final_text)
    target_id = _replacement_continuation_target(cues_by_id, bounds_by_cue, span)
    if (
        words is None or len(bounds_by_cue) != 1 or target_id is None
        or len(signature) < 2 or signature != alphanumeric_signature(span.asr_text)
        or not all(value is not None and isfinite(value) for value in (
            span.start, span.end, span.left_anchor_end, span.right_anchor_start,
        ))
    ):
        return None
    # A pause within a local rewrite does not by itself propose a transfer.
    # Without both close retained continuations, keep the ordinary local edit.
    if not (
        0 <= span.start - span.left_anchor_end <= max_anchor_gap_seconds
        and 0 <= span.right_anchor_start - span.end <= max_anchor_gap_seconds
    ):
        return None
    character_cut = _acoustic_replacement_cut(
        span, final_text, unit_spans, words, min_phrase_gap_seconds, max_anchor_gap_seconds,
    )
    if character_cut is None:
        return None
    source_id, bounds = next(iter(bounds_by_cue.items()))
    return {
        source_id: (*bounds, final_text[:character_cut].strip()),
        target_id: (0, 0, final_text[character_cut:].strip()),
    }


def _joint_region_replacement_edits(
    cues_by_id: dict[int, Cue], bounds_by_cue: dict[int, tuple[int, int]], span: DivergenceSpan,
    final_text: str, unit_spans: list[tuple[int, int]], words: list[Word] | None,
) -> dict[int, tuple[int, int, str]]:
    target_id = _replacement_continuation_target(cues_by_id, bounds_by_cue, span)
    if (
        words is None or not 2 <= len(bounds_by_cue) <= 3
        or target_id is None or target_id in bounds_by_cue
        or not alphanumeric_signature(final_text)
        or alphanumeric_signature(final_text) != alphanumeric_signature(span.asr_text)
        or not all(value is not None and isfinite(value) for value in (
            span.start, span.end, span.left_anchor_end, span.right_anchor_start,
        ))
        or any(any(marker in cues_by_id[cue_id].text for marker in ("♪", "♫"))
               for cue_id in [*bounds_by_cue, target_id])
    ):
        raise ReplacementOwnershipError("The joint region needs exact approved words and complete source anchors; source text was held for review.")
    character_cut = _acoustic_replacement_cut(span, final_text, unit_spans, words)
    if character_cut is None:
        raise ReplacementOwnershipError("The joint region has no unique measured phrase boundary; source text was held for review.")
    middle_id = list(bounds_by_cue)[1]
    return {
        **{cue_id: (*bounds, final_text[:character_cut].strip() if cue_id == middle_id else "")
           for cue_id, bounds in bounds_by_cue.items()},
        target_id: (0, 0, final_text[character_cut:].strip()),
    }


def _acoustic_replacement_cut(
    span: DivergenceSpan, final_text: str, unit_spans: list[tuple[int, int]], words: list[Word],
    min_phrase_gap_seconds: float = 0.8, max_anchor_gap_seconds: float = 0.2,
) -> int | None:
    """Find a cut using complete actual words; both text and timing share it."""
    signature = alphanumeric_signature(final_text)
    indices = span.asr_word_indices
    if (
        not indices or indices != list(range(indices[0], indices[-1] + 1))
        or indices[0] < 0 or indices[-1] >= len(words)
    ):
        raise ReplacementOwnershipError("The replacement's complete acoustic word sequence is unavailable; source text was held for review.")
    spoken = [words[index] for index in indices]
    if (
        alphanumeric_signature(" ".join(word.text for word in spoken)) != signature
        or any(not alphanumeric_signature(word.text) for word in spoken)
        or any(not isfinite(word.start) or not isfinite(word.end) or word.start >= word.end for word in spoken)
        or any(left.end > right.start for left, right in zip(spoken, spoken[1:]))
        or abs(spoken[0].start - span.start) > 1e-6 or abs(spoken[-1].end - span.end) > 1e-6
    ):
        raise ReplacementOwnershipError("The replacement's word evidence is inconsistent; source text was held for review.")
    cuts = [position for position in range(1, len(spoken))
            if spoken[position].start - spoken[position - 1].end >= min_phrase_gap_seconds]
    if not cuts:
        return None
    if len(cuts) != 1:
        raise ReplacementOwnershipError("The replacement crosses several speech gaps with no unique cue boundary; source text was held for review.")
    known_speakers = {word.speaker_id for word in spoken if word.speaker_id} | set(span.speaker_ids)
    known_speakers.update(speaker for speaker in (
        span.left_anchor_speaker_id, span.right_anchor_speaker_id,
    ) if speaker)
    if len(known_speakers) > 1:
        raise ReplacementOwnershipError("The replacement crosses conflicting speaker evidence; source text was held for review.")
    if not (
        0 <= span.start - span.left_anchor_end <= max_anchor_gap_seconds
        and 0 <= span.right_anchor_start - span.end <= max_anchor_gap_seconds
    ):
        raise ReplacementOwnershipError("The speech groups do not join both retained word anchors closely enough; source text was held for review.")
    token_cut = sum(len(alphanumeric_signature(word.text)) for word in spoken[:cuts[0]])
    token_spans = _token_character_spans(final_text)
    if not 0 < token_cut < len(token_spans):
        raise ReplacementOwnershipError("The acoustic cut has no complete spoken tokens on both sides; source text was held for review.")
    character_cut = token_spans[token_cut][0]
    if character_cut not in {start for start, _ in unit_spans}:
        raise ReplacementOwnershipError("The proposed acoustic cut would split a lexical unit; source text was held for review.")
    return character_cut


def _has_sentence_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?\u2026\u3002\uff01\uff1f]+[\"'\u201d\u2019\u00bb)\]]*\s*$", text))


def _replacement_sentence_boundaries(
    text: str,
    unit_spans: list[tuple[int, int]],
) -> list[int]:
    return [
        position for position in range(1, len(unit_spans))
        if re.fullmatch(
            r"[.!?\u2026\u3002\uff01\uff1f]+[\"'\u201d\u2019\u00bb)\]]*\s*",
            text[unit_spans[position - 1][1]:unit_spans[position][0]],
        )
    ]


def _is_preserved_source_internal_boundary(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    source_tokens: list[str],
    final_text: str,
    character_boundary: int,
) -> bool:
    """Do not promote unchanged internal punctuation, such as a title's dot."""
    prefix = alphanumeric_signature(final_text[:character_boundary])
    suffix = alphanumeric_signature(final_text[character_boundary:])
    source_offset = len(prefix)
    if (
        not suffix
        or source_tokens[:source_offset + 1] != [*prefix, suffix[0]]
    ):
        return False
    for cue_id, (start, end) in bounds_by_cue.items():
        if source_offset < end - start:
            local_boundary = start + source_offset
            text = cues_by_id[cue_id].plain_text
            token_spans = _token_character_spans(text)
            return (
                start < local_boundary < end
                and _has_sentence_terminal(text[:token_spans[local_boundary][0]])
            )
        source_offset -= end - start
    return False


def _contained_sentence_replacement_target(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
    sentence_boundaries: list[int],
) -> int | None:
    """Keep one confirmed sentence in the complete cue containing its audio.

    A replaced tail in the preceding cue must leave a source prefix. Requiring
    exact time containment and complete sentence endings avoids collapsing
    ambiguous speech or consuming another source cue to infer a new timing.
    """
    if (
        len(bounds_by_cue) != 2
        or sentence_boundaries
        or not _has_sentence_terminal(final_text)
        or span.start is None
        or span.end is None
        or span.start >= span.end
    ):
        return None
    prefix_id, target_id = bounds_by_cue
    prefix, target = cues_by_id[prefix_id], cues_by_id[target_id]
    prefix_count = len(alphanumeric_signature(prefix.plain_text))
    target_count = len(alphanumeric_signature(target.plain_text))
    prefix_start, prefix_end = bounds_by_cue[prefix_id]
    if (
        not (0 < prefix_start < prefix_end == prefix_count)
        or bounds_by_cue[target_id] != (0, target_count)
        or not _has_sentence_terminal(prefix.plain_text)
        or not _has_sentence_terminal(target.plain_text)
        or prefix.end_ms / 1000.0 > span.start
        or not (target.start_ms / 1000.0 <= span.start < span.end <= target.end_ms / 1000.0)
    ):
        return None
    return target_id


def _continuation_prefix_replacement_target(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
) -> int | None:
    """Keep an unfinished single-speaker phrase with its retained continuation."""
    if (
        len(set(span.speaker_ids)) != 1
        or not span.speaker_ids[0]
        or re.search(r"[.!?\u2026\u3002\uff01\uff1f]", final_text)
    ):
        return None
    cue_ids = list(bounds_by_cue)
    for cue_id in cue_ids[:-1]:
        signature = alphanumeric_signature(cues_by_id[cue_id].plain_text)
        if bounds_by_cue[cue_id] != (0, len(signature)):
            return None
    target_id = cue_ids[-1]
    text = cues_by_id[target_id].plain_text
    signature = alphanumeric_signature(text)
    start, end = bounds_by_cue[target_id]
    final_signature = alphanumeric_signature(final_text)
    if (
        start != 0
        or not (0 < end < len(signature))
        or final_signature[-end:] != signature[:end]
    ):
        return None
    # A retained suffix after a sentence end is another sentence, not the
    # continuation that provides ownership for this unfinished phrase.
    token_spans = _token_character_spans(text)
    if re.search(r"[.!?\u2026\u3002\uff01\uff1f]", text[token_spans[end - 1][1]:token_spans[end][0]]):
        return None
    return target_id


def single_token_prefix_replacement_targets(
    cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
) -> dict[str, int]:
    """Keep a collapsed source span with its surviving sentence continuation.

    This only applies when explicit contiguous token indices consume complete
    preceding cues and the prefix of the final cue. The same target must own
    the replacement's ASR words and text; proportional cue assignment cannot
    represent this case without producing a one-word orphan.
    """
    cues_by_id = {cue.index: cue for cue in cues}
    offsets = _cue_token_offsets(cues)
    by_case = {decision.case_id: decision for decision in decisions}
    targets: dict[str, int] = {}
    for span in spans:
        decision = by_case.get(span.case_id)
        cue_ids = list(dict.fromkeys(span.cue_ids))
        indices = sorted(set(span.srt_token_indices))
        if (
            decision is None
            or decision.verdict not in {"use_audio", "hybrid"}
            or len(alphanumeric_signature(decision.final_text)) != 1
            or len(cue_ids) < 2
            or not indices
            or indices != list(range(indices[0], indices[-1] + 1))
            or any(cue_id not in cues_by_id for cue_id in cue_ids)
            or any(cue_has_bracketed_screen_text(cues_by_id[cue_id]) for cue_id in cue_ids)
        ):
            continue
        covered_tokens: list[str] = []
        for position, cue_id in enumerate(cue_ids):
            cue = cues_by_id[cue_id]
            signature = alphanumeric_signature(speech_text_for_alignment(cue))
            bounds = _span_token_bounds_for_cue(cue, span, offsets[cue_id])
            if bounds is None or bounds[0] != 0:
                break
            if position < len(cue_ids) - 1:
                if bounds[1] != len(signature):
                    break
            elif bounds[1] >= len(signature):
                break
            covered_tokens.extend(signature[:bounds[1]])
        else:
            if covered_tokens == alphanumeric_signature(span.srt_text):
                targets[span.case_id] = cue_ids[-1]
    return targets


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
            and source_text[end_character] in ",.;:!?、。！？"
            and _TERMINAL_PUNCTUATION_RE.search(stripped_replacement)
        ):
            end_character += 1
        if start_character < cursor or bounded_start < previous_token_end:
            continue
        pieces.append(source_text[cursor:start_character])
        if bounded_start == bounded_end and stripped_replacement:
            left = source_text[start_character - 1:start_character] if start_character else ""
            right = source_text[end_character:end_character + 1]
            prefix = "" if join_word_texts((left, stripped_replacement)) == left + stripped_replacement else " "
            suffix = "" if join_word_texts((stripped_replacement, right)) == stripped_replacement + right else " "
            pieces.append(f"{prefix}{stripped_replacement}{suffix}")
        else:
            pieces.append(stripped_replacement)
        cursor = end_character
        previous_token_end = bounded_end

    pieces.append(source_text[cursor:])
    assembled = ""
    for piece in pieces:
        assembled = _join_preserved_boundary(assembled, piece)
    raw_text, protected = _protect_text_fragments(assembled, protected_fragments)
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
    return token_character_spans(text, token_texts(text)) or []


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
    source_bounds = token_character_spans(cue.plain_text, cue_tokens)
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
