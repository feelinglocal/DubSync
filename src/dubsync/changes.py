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
from .speaker_evidence import has_known_different_speakers
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

        guard_flag = _editorial_guard_rejection(span, decision, cue_ids, cues_by_id, cue_token_offsets)
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
    unchanged_cue_ids: set[int] = set()
    capitalized_source_words = source_capitalized_words(cues)
    for cue_id, edits in token_edits_by_cue.items():
        edits = _recase_prefixed_token_edits(
            cues_by_id[cue_id], edits, words, token_matches or [],
            cue_token_offsets[cue_id], capitalized_source_words,
        )
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
        if changed_text.split() == cues_by_id[cue_id].text.split():
            # An approved wording equal to the source (the far-away part of
            # its case was placed elsewhere) keeps the authored lines.
            unchanged_cue_ids.add(cue_id)
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
        # Nothing changed for the viewer: there is no change to report.
        if not (flag.kind == "text_changed" and flag.cue_ids and set(flag.cue_ids) <= unchanged_cue_ids)
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


def source_capitalized_words(cues: list[Cue]) -> frozenset[str]:
    """Source capitals inside a sentence protect names and German nouns."""
    protected: set[str] = set()
    for cue in cues:
        text = speech_text_for_alignment(cue)
        for position, (start, end) in enumerate(_token_character_spans(text)):
            token = text[start:end]
            # A dialogue marker can follow a completed sentence ("foto. -").
            # Its next initial is sentence casing, not source-name evidence.
            preceding = re.sub(r"(?<!\S)[-–—]\s*$", "", text[:start])
            if position and token[:1].isupper() and not _has_sentence_terminal(preceding):
                protected.add(token.casefold())
    return frozenset(protected)


def recase_prefix_join(
    prefix: str, retained: str, *, matched_initial: str | None,
    capitalized_source_words: frozenset[str],
) -> tuple[str, str]:
    """Transfer an authored initial capital without guessing the next word's case.

    Only the two boundary initials can change. Quoted or styled openings do
    not establish a new sentence here; source names, interior capitals and
    capitalized/missing ASR evidence keep the retained word's authored case.
    """
    prefix_spans, retained_spans = _token_character_spans(prefix), _token_character_spans(retained)
    if not prefix_spans or not retained_spans:
        return prefix, retained
    prefix_start, _ = prefix_spans[0]
    start, end = retained_spans[0]
    initial = retained[start:end]
    if (
        prefix[:prefix_start].strip() or retained[:start].strip()
        or not initial[:1].isupper()
        or any(mark in prefix for mark in _DOUBLE_QUOTATION_MARKS)
    ):
        return prefix, retained
    capital = prefix[prefix_start].upper()
    candidate = prefix[:prefix_start] + capital + prefix[prefix_start + 1:]
    if len(capital) == 1 and alphanumeric_signature(candidate) == alphanumeric_signature(prefix):
        prefix = candidate

    if (
        matched_initial is None or _has_sentence_terminal(prefix)
        or initial == "I" or any(character.isupper() for character in initial[1:])
        or initial.casefold() in capitalized_source_words
    ):
        return prefix, retained
    spoken_spans = _token_character_spans(matched_initial)
    if len(spoken_spans) != 1:
        return prefix, retained
    spoken = matched_initial[slice(*spoken_spans[0])]
    lower = initial[0].lower()
    candidate = retained[:start] + lower + retained[start + 1:]
    if (
        spoken.islower() and spoken.casefold() == initial.casefold() and len(lower) == 1
        and alphanumeric_signature(candidate) == alphanumeric_signature(retained)
    ):
        retained = candidate
    return prefix, retained


def _recase_prefixed_token_edits(
    cue: Cue, edits: list[tuple[int, int, str]], words: list[Word] | None,
    token_matches: list[TokenMatch], cue_token_offset: int,
    capitalized_source_words: frozenset[str],
) -> list[tuple[int, int, str]]:
    prefix_positions = [position for position, (start, end, text) in enumerate(edits)
                        if start == end == 0 and alphanumeric_signature(text)]
    if not prefix_positions or cue_has_bracketed_screen_text(cue):
        return edits
    source = cue.plain_text
    source_spans = _token_character_spans(source)
    if not source_spans:
        return edits
    matches = [match for match in token_matches
               if match.cue_id == cue.index and match.srt_token_index == cue_token_offset]
    matched_initial = None
    if (
        words is not None and len(matches) == 1 and matches[0].score == 1.0
        and 0 <= matches[0].asr_word_index < len(words)
        and not any(start <= 0 < end for start, end, _ in edits)
    ):
        matched_initial = words[matches[0].asr_word_index].text
    prefix = join_word_texts(edits[position][2] for position in prefix_positions)
    recased_prefix, retained = recase_prefix_join(
        prefix, source, matched_initial=matched_initial,
        capitalized_source_words=capitalized_source_words,
    )
    recased = list(edits)
    if recased_prefix != prefix:
        position = prefix_positions[0]
        start, end, text = recased[position]
        character = _token_character_spans(text)[0][0]
        recased[position] = (start, end, text[:character] + text[character].upper() + text[character + 1:])
    if retained != source:
        recased.append((0, 1, retained[slice(*source_spans[0])]))
    return recased


def indexed_multi_cue_replacements(
    cues: list[Cue],
    span: DivergenceSpan,
    final_text: str,
    *,
    replacement_target: int | None = None,
    words: list[Word] | None = None,
    ownership: dict[int, list[int]] | None = None,
) -> dict[int, tuple[int, int, str]] | None:
    """Partition one exact source-token edit without consuming its cue residue.

    With word timing, every spoken phrase goes to the cue at whose time it
    was spoken when the measured gaps decide that (``ownership`` then receives
    the ASR word indices of each piece). Otherwise replacement tokens follow
    the source span's contribution to each cue, preferring nearby corroborated
    sentence boundaries where available.
    An anchored single-cue tail can transfer its replacement to the next cue.
    The pipeline uses these same pieces to assign acoustic evidence, so text
    and timing cannot independently choose different cue boundaries.
    """
    cue_ids = list(dict.fromkeys(span.cue_ids))
    cues_by_id = {cue.index: cue for cue in cues}
    bounds_by_cue = indexed_span_bounds(cues, span)
    if bounds_by_cue is None:
        return None
    covered_tokens = alphanumeric_signature(span.srt_text)

    unit_spans = lexical_unit_spans(final_text)
    if unit_spans is None:
        return None
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
    leading = _leading_phrase_of_previous_cue(cues_by_id, bounds_by_cue, span, final_text, words)
    if leading is not None:
        # The phrase ends the previous cue's sentence; the rest of the wording
        # is distributed over the case's own cues like any other replacement.
        previous_id, previous_edit, word_count, rest_text, rest_span = leading
        rest_ownership: dict[int, list[int]] = {}
        rest_edits = indexed_multi_cue_replacements(
            cues, rest_span, rest_text, replacement_target=replacement_target, words=words, ownership=rest_ownership,
        )
        if rest_edits is not None and previous_id not in rest_edits:
            if ownership is not None and rest_ownership:
                ownership.clear()
                ownership.update({previous_id: span.asr_word_indices[:word_count], **rest_ownership})
            return {previous_id: previous_edit, **rest_edits}
    spoken_placement = _spoken_phrase_placement(cues_by_id, bounds_by_cue, span, final_text, words)
    if spoken_placement is not None:
        edits, word_indices_by_cue = spoken_placement
        if ownership is not None:
            ownership.clear()
            ownership.update(word_indices_by_cue)
        return edits
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


# Spoken words closer together than this are one phrase and stay in one cue
# unless the wording itself is divided by the source-based rules.
_PHRASE_GAP_SECONDS = 0.3
# A phrase is placed by timing only when the next-best cue is this much farther.
_PLACEMENT_MARGIN_SECONDS = 0.25
# A completely replaced cue is expected at its source time, moved by the
# offset of the matched words around the case, within this tolerance.
_EXPECTED_CUE_PAD_SECONDS = 0.5


def _spoken_phrase_placement(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
    words: list[Word] | None,
) -> tuple[dict[int, tuple[int, int, str]], dict[int, list[int]]] | None:
    """Give every spoken phrase to the cue at whose time it was spoken.

    Source token share cannot know that "Aqui," follows a 1.7 s pause and
    belongs to the next cue, or that "Ah," is spoken 21 s after the cue whose
    tail it replaces. A cue that keeps matched words is spoken where those
    words are; a completely replaced cue near its source time. Phrases keep
    their spoken order. None unless every phrase is clearly nearer to one cue
    than to its neighbours and the approved text can be cut at the same words.
    """
    cue_ids = list(bounds_by_cue)
    indices = span.asr_word_indices
    if (
        words is None or len(cue_ids) < 2 or not indices
        or indices != list(range(indices[0], indices[-1] + 1))
        or indices[0] < 0 or indices[-1] >= len(words)
    ):
        return None
    spoken = [words[index] for index in indices]
    if any(
        not isfinite(word.start) or not isfinite(word.end) or word.end < word.start
        or not alphanumeric_signature(word.text)
        for word in spoken
    ) or any(right.start < left.start for left, right in zip(spoken, spoken[1:])):
        return None

    # Phrase = run of words without a measurable pause; (first, end) positions.
    phrases: list[tuple[int, int]] = []
    first = 0
    for position in range(1, len(spoken) + 1):
        if position == len(spoken) or spoken[position].start - spoken[position - 1].end >= _PHRASE_GAP_SECONDS:
            phrases.append((first, position))
            first = position
    times = [(spoken[first].start, max(word.end for word in spoken[first:end])) for first, end in phrases]

    offsets = [
        anchor_time - cue_time
        for anchor_id, anchor_time, cue_time in (
            (span.left_anchor_cue_id, span.left_anchor_end,
             cues_by_id[span.left_anchor_cue_id].end_ms / 1000 if span.left_anchor_cue_id in cues_by_id else None),
            (span.right_anchor_cue_id, span.right_anchor_start,
             cues_by_id[span.right_anchor_cue_id].start_ms / 1000 if span.right_anchor_cue_id in cues_by_id else None),
        )
        if anchor_id is not None and anchor_id not in bounds_by_cue
        and cue_time is not None and anchor_time is not None and isfinite(anchor_time)
    ]
    low, high = (min(offsets), max(offsets)) if offsets else (0.0, 0.0)

    def distances(cue_position: int) -> list[float] | None:
        cue_id = cue_ids[cue_position]
        cue = cues_by_id[cue_id]
        start, end = bounds_by_cue[cue_id]
        token_count = len(alphanumeric_signature(speech_text_for_alignment(cue)))
        if cue_position == 0 and start > 0:
            anchor = span.left_anchor_end
            if span.left_anchor_cue_id != cue_id or anchor is None or not isfinite(anchor):
                return None
            return [max(0.0, phrase_start - anchor) for phrase_start, _ in times]
        if cue_position == len(cue_ids) - 1 and end < token_count:
            anchor = span.right_anchor_start
            if span.right_anchor_cue_id != cue_id or anchor is None or not isfinite(anchor):
                return None
            return [max(0.0, anchor - phrase_end) for _, phrase_end in times]
        if (start, end) != (0, token_count):
            return None
        window_start = cue.start_ms / 1000 + low - _EXPECTED_CUE_PAD_SECONDS
        window_end = cue.end_ms / 1000 + high + _EXPECTED_CUE_PAD_SECONDS
        return [max(0.0, window_start - phrase_end, phrase_start - window_end) for phrase_start, phrase_end in times]

    distance_by_cue = [distances(position) for position in range(len(cue_ids))]
    if any(row is None for row in distance_by_cue):
        return None

    # Cheapest assignment of the phrases to the cues in spoken order.
    costs = [[distance_by_cue[cue_position][0] for cue_position in range(len(cue_ids))]]
    previous_cue: list[list[int]] = [[0] * len(cue_ids)]
    for phrase in range(1, len(phrases)):
        row: list[float] = []
        origins: list[int] = []
        best = 0
        for cue_position in range(len(cue_ids)):
            if costs[-1][cue_position] < costs[-1][best]:
                best = cue_position
            row.append(costs[-1][best] + distance_by_cue[cue_position][phrase])
            origins.append(best)
        costs.append(row)
        previous_cue.append(origins)
    assigned = [0] * len(phrases)
    assigned[-1] = min(range(len(cue_ids)), key=costs[-1].__getitem__)
    for phrase in range(len(phrases) - 1, 0, -1):
        assigned[phrase - 1] = previous_cue[phrase][assigned[phrase]]
    for phrase, cue_position in enumerate(assigned):
        lowest = assigned[phrase - 1] if phrase else 0
        highest = assigned[phrase + 1] if phrase + 1 < len(phrases) else len(cue_ids) - 1
        own = distance_by_cue[cue_position][phrase]
        if any(
            distance_by_cue[other][phrase] - own < _PLACEMENT_MARGIN_SECONDS
            for other in range(lowest, highest + 1) if other != cue_position
        ):
            return None

    # Text and words are cut at the same place only for a wording that follows
    # the spoken words one by one (spelling may differ). A changed word count
    # would lend a phrase the timing of words its text does not contain.
    unit_spans = lexical_unit_spans(final_text)
    token_spans = _token_character_spans(final_text)
    token_starts: list[int] = []
    token_count = 0
    for word in spoken:
        token_starts.append(token_count)
        token_count += len(alphanumeric_signature(word.text))
    if unit_spans is None or len(token_spans) != token_count:
        return None
    cuts: list[int] = []
    for phrase in range(1, len(phrases)):
        if assigned[phrase] == assigned[phrase - 1]:
            continue
        offset = _token_cut_offset(final_text, token_spans, unit_spans, token_starts[phrases[phrase][0]])
        if offset is None:
            return None
        cuts.append(offset)
    edges = [0, *cuts, len(final_text)]
    pieces: dict[int, str] = {}
    owned: dict[int, list[int]] = {cue_id: [] for cue_id in cue_ids}
    piece = 0
    for phrase, cue_position in enumerate(assigned):
        if phrase and cue_position != assigned[phrase - 1]:
            piece += 1
        pieces[cue_ids[cue_position]] = final_text[edges[piece]:edges[piece + 1]].strip()
        owned[cue_ids[cue_position]].extend(indices[phrases[phrase][0]:phrases[phrase][1]])
    if any(not alphanumeric_signature(text) for text in pieces.values()):
        # The adjudicator left out the words of a phrase: nothing places its cue.
        return None
    return {cue_id: (*bounds_by_cue[cue_id], pieces.get(cue_id, "")) for cue_id in cue_ids}, owned


_LEADING_PHRASE_MAX_TOKENS = 3


def _leading_phrase_of_previous_cue(
    cues_by_id: dict[int, Cue],
    bounds_by_cue: dict[int, tuple[int, int]],
    span: DivergenceSpan,
    final_text: str,
    words: list[Word] | None,
    min_phrase_gap_seconds: float = 0.8,
    max_anchor_gap_seconds: float = 0.2,
) -> tuple[int, tuple[int, int, str], int, str, DivergenceSpan] | None:
    """Find a short phrase that still belongs to the cue before the case.

    The mirror of the acoustic tail transfer: "em casa." directly follows the
    previous cue's last word and precedes a measured pause, while the case
    starts with the next cue. It ends the previous cue's sentence. Returns the
    previous cue, its insertion edit, the number of spoken words moved, the
    remaining wording and the case reduced to the remaining words.
    """
    previous_id = span.left_anchor_cue_id
    first_id = next(iter(bounds_by_cue))
    indices = span.asr_word_indices
    anchor_end = span.left_anchor_end
    if (
        words is None or previous_id is None or previous_id in bounds_by_cue or previous_id not in cues_by_id
        or bounds_by_cue[first_id][0] != 0 or anchor_end is None or not isfinite(anchor_end)
        or len(indices) < 2 or indices != list(range(indices[0], indices[-1] + 1))
        or indices[0] < 0 or indices[-1] >= len(words)
    ):
        return None
    previous = cues_by_id[previous_id]
    spoken = [words[index] for index in indices]
    if (
        cue_has_bracketed_screen_text(previous) or any(mark in previous.text for mark in "♪♫")
        or any(
            not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start
            or not alphanumeric_signature(word.text)
            for word in spoken
        )
        or any(right.start < left.end for left, right in zip(spoken, spoken[1:]))
        or not 0 <= spoken[0].start - anchor_end <= max_anchor_gap_seconds
    ):
        return None
    word_count = next(
        (position for position in range(1, len(spoken))
         if spoken[position].start - spoken[position - 1].end >= min_phrase_gap_seconds),
        None,
    )
    if word_count is None:
        return None
    phrase = spoken[:word_count]
    token_count = sum(len(alphanumeric_signature(word.text)) for word in phrase)
    if (
        token_count > _LEADING_PHRASE_MAX_TOKENS
        or any(right.start - left.end > max_anchor_gap_seconds for left, right in zip(phrase, phrase[1:]))
        or has_known_different_speakers([span.left_anchor_speaker_id, *(word.speaker_id for word in phrase)])
    ):
        return None
    unit_spans = lexical_unit_spans(final_text)
    token_spans = _token_character_spans(final_text)
    if unit_spans is None or len(token_spans) != sum(len(alphanumeric_signature(word.text)) for word in spoken):
        return None
    cut = _token_cut_offset(final_text, token_spans, unit_spans, token_count)
    if cut is None or not alphanumeric_signature(final_text[cut:]):
        return None
    rest = spoken[word_count:]
    previous_count = len(alphanumeric_signature(speech_text_for_alignment(previous)))
    rest_span = span.model_copy(update={
        "asr_word_indices": indices[word_count:],
        "asr_text": join_word_texts(word.text for word in rest),
        "start": rest[0].start,
        "left_anchor_end": phrase[-1].end,
        "left_anchor_speaker_id": phrase[-1].speaker_id or span.left_anchor_speaker_id,
    })
    return (
        previous_id, (previous_count, previous_count, final_text[:cut].strip()),
        word_count, final_text[cut:].strip(), rest_span,
    )


def indexed_span_bounds(cues: list[Cue], span: DivergenceSpan) -> dict[int, tuple[int, int]] | None:
    """Alignment-token bounds of an exact source span inside each of its cues.

    None unless the token indices are contiguous, lie in plain dialogue cues
    and reproduce the span's source text.
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
    return bounds_by_cue


def lexical_unit_spans(text: str) -> list[tuple[int, int]] | None:
    """Character spans of the units an approved text may be cut between.

    Alignment tokens split apostrophes and numeric punctuation. Such tokens
    stay one unit, so "aren't" cannot become "aren'" / "t".
    """
    token_spans = _token_character_spans(text)
    if len(token_spans) != len(alphanumeric_signature(text)):
        return None
    unit_spans: list[tuple[int, int]] = []
    for start, end in token_spans:
        if (
            unit_spans
            and not any(character.isspace() for character in text[unit_spans[-1][1]:start])
            and not contains_character_level_script(text[unit_spans[-1][0]:end])
        ):
            unit_spans[-1] = (unit_spans[-1][0], end)
        else:
            unit_spans.append((start, end))
    return unit_spans


def lexical_edit_costs(left: list[str], right: list[str]) -> list[list[int]]:
    rows = [list(range(len(right) + 1))]
    for left_position, left_token in enumerate(left, start=1):
        previous = rows[-1]
        row = [left_position]
        for right_position, right_token in enumerate(right, start=1):
            row.append(min(
                previous[right_position - 1] + (left_token != right_token),
                previous[right_position] + 1,
                row[-1] + 1,
            ))
        rows.append(row)
    return rows


def replacement_text_cuts(final_text: str, spoken: list[Word], boundaries: list[int]) -> list[int] | None:
    """Character offsets of an approved text at boundaries between spoken words.

    Each boundary is a position in ``spoken``: the cut lies before that word.
    Identical wording is cut at the same token. Changed wording is cut where
    every optimal lexical alignment agrees, or where exactly one of them keeps
    an identical word next to the cut. None when a cut is ambiguous or would
    split a lexical unit. A cut at 0 or at the text end means the approved
    text has no words on that side.
    """
    unit_spans = lexical_unit_spans(final_text)
    if unit_spans is None:
        return None
    token_spans = _token_character_spans(final_text)
    final_tokens = alphanumeric_signature(final_text)
    spoken_tokens: list[str] = []
    confident: list[bool] = []
    word_token_starts: list[int] = []
    for word in spoken:
        signature = alphanumeric_signature(word.text)
        if not signature:
            return None
        word_token_starts.append(len(spoken_tokens))
        spoken_tokens.extend(signature)
        confident.extend([anchor_confidence_is_acceptable(word.confidence)] * len(signature))
    exact = final_tokens == spoken_tokens
    forward = [] if exact else lexical_edit_costs(final_tokens, spoken_tokens)
    backward = [] if exact else lexical_edit_costs(final_tokens[::-1], spoken_tokens[::-1])
    cuts: list[int] = []
    for boundary in boundaries:
        if not 0 < boundary < len(spoken):
            return None
        spoken_cut = word_token_starts[boundary]
        if exact:
            candidates = [spoken_cut]
        else:
            candidates = [
                final_cut for final_cut in range(len(final_tokens) + 1)
                if forward[final_cut][spoken_cut]
                + backward[len(final_tokens) - final_cut][len(spoken_tokens) - spoken_cut] == forward[-1][-1]
            ]
            if len(candidates) > 1:
                candidates = [
                    final_cut for final_cut in candidates
                    if (final_cut > 0 and confident[spoken_cut - 1]
                        and final_tokens[final_cut - 1] == spoken_tokens[spoken_cut - 1])
                    or (final_cut < len(final_tokens) and confident[spoken_cut]
                        and final_tokens[final_cut] == spoken_tokens[spoken_cut])
                ]
        if len(candidates) != 1:
            return None
        offset = _token_cut_offset(final_text, token_spans, unit_spans, candidates[0])
        if offset is None or (cuts and offset < cuts[-1]):
            return None
        cuts.append(offset)
    return cuts


def anchor_confidence_is_acceptable(confidence: float | None) -> bool:
    # Providers without word confidences (MAI) report None: unknown, not zero.
    # Only a known low confidence disqualifies an exact lexical anchor.
    return confidence is None or confidence >= 0.8


def _token_cut_offset(
    text: str, token_spans: list[tuple[int, int]], unit_spans: list[tuple[int, int]], token_cut: int,
) -> int | None:
    """Character offset of a cut before a token; None inside a lexical unit."""
    if token_cut <= 0:
        return 0
    if token_cut >= len(token_spans):
        return len(text)
    offset = token_spans[token_cut][0]
    if offset not in {start for start, _ in unit_spans}:
        return None
    # An opening quote or dash belongs to the word it precedes.
    previous_end = token_spans[token_cut - 1][1]
    spaces = [position for position, character in enumerate(text[previous_end:offset]) if character.isspace()]
    return previous_end + spaces[-1] + 1 if spaces else offset


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
        not span.left_anchor_end < span.start < span.end
        or span.start >= span.right_anchor_start
        # Independently repaired edges can share one 10 ms detector hop.
        # This changes word ownership only, never either acoustic timestamp.
        or span.end - span.right_anchor_start > 0.010 + 1e-9
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
    if re.search(r"[.!?\u2026\u3002\uff01\uff1f]", final_text):
        return None
    speakers = [speaker for speaker in dict.fromkeys(span.speaker_ids) if speaker]
    if has_known_different_speakers(speakers):
        return None
    if len(speakers) != 1:
        # Without one diarized actor (MAI without diarization, or labels of
        # two unrelated chunk scopes) the phrase must be acoustically joined
        # to the retained continuation: complete adjacent words that run
        # into the right anchor. Otherwise this rule proves nothing, and the
        # proportional fallback would leave one-word orphan cues.
        indices = span.asr_word_indices
        if (
            not indices or indices != list(range(indices[0], indices[-1] + 1))
            or span.right_anchor_cue_id != list(bounds_by_cue)[-1]
            or span.end is None or span.right_anchor_start is None
            or not isfinite(span.end) or not isfinite(span.right_anchor_start)
            or not -0.05 <= span.right_anchor_start - span.end <= 0.2
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


def _span_source_context(
    span: DivergenceSpan, cue_ids: list[int], cues_by_id: dict[int, Cue], cue_token_offsets: dict[int, int],
) -> str:
    """Authored text of the span's tokens with the punctuation directly around them."""
    context: list[str] = []
    for cue_id in cue_ids:
        cue = cues_by_id[cue_id]
        bounds = _span_token_bounds_for_cue(cue, span, cue_token_offsets[cue_id]) if span.srt_token_indices else None
        token_spans = (
            list(alignment_token_character_spans(cue) or []) if cue_has_bracketed_screen_text(cue)
            else _token_character_spans(cue.plain_text)
        )
        text = cue.text if cue_has_bracketed_screen_text(cue) else cue.plain_text
        if bounds is None or not token_spans or bounds[1] > len(token_spans):
            context.append(text)
            continue
        start, end = bounds
        context.append(text[
            token_spans[start - 1][1] if start else 0:
            token_spans[end][0] if end < len(token_spans) else len(text)
        ])
    return "\n".join(context)


def _editorial_guard_rejection(
    span: DivergenceSpan,
    decision: AdjudicationDecision,
    cue_ids: list[int],
    cues_by_id: dict[int, Cue] | None = None,
    cue_token_offsets: dict[int, int] | None = None,
) -> QCFlag | None:
    source_context = applied_text = None
    if cues_by_id is not None and cue_token_offsets is not None:
        source_context = _span_source_context(span, cue_ids, cues_by_id, cue_token_offsets)
        if len(cue_ids) == 1 and span.srt_token_indices:
            # A wording that repeats the cue's unchanged words is applied
            # without them; only the applied part can add a mark.
            cue = cues_by_id[cue_ids[0]]
            bounds = _span_token_bounds_for_cue(cue, span, cue_token_offsets[cue.index])
            if bounds is not None:
                applied_text = _localized_indexed_replacement(cue, span, bounds, decision.final_text)
    try:
        validate_adjudication_editorial_contract(
            span,
            decision,
            allow_word_change=decision.verdict in {"use_audio", "hybrid"},
            source_context=source_context,
            applied_text=applied_text,
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
            validate_editorial_text(source.text, cue.text, allow_word_change=True, allow_removed_quotations=True)
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
            elif terminal is not None and terminal.group(1) == "." and stripped_replacement[0].islower():
                # A lower-case continuation that brings its own ending carries
                # the sentence on: the full stop moves to the new end.
                start_character, end_character = terminal.span(1)
        ends_with_title = bounded_end > bounded_start and _is_title_abbreviation(source_text, *token_spans[bounded_end - 1])
        if ends_with_title and bounded_end == bounded_start + 1 and _is_spoken_title(
            source_text[start_character:end_character], stripped_replacement,
        ):
            # "Sr." is how the script writes the spoken "senhor": the same word.
            continue
        # Only a contraction suffix ("gibt's" -> "gibt es") gives up its
        # apostrophe. An elision ("l'homme") or an opening quote ('oi') keeps it.
        replaces_contraction_suffix = (
            bounded_end > bounded_start
            and start_character > max(cursor, 1)
            and source_text[start_character - 1] in {"'", "\u2019"}
            and source_text[start_character - 2].isalnum()
            and source_text[slice(*token_spans[bounded_start])].casefold() in _CONTRACTION_SUFFIXES
        )
        if replaces_contraction_suffix:
            start_character -= 1
            if stripped_replacement:
                stripped_replacement = f" {stripped_replacement}"
        restored_before = restored_after = ""
        if bounded_end > bounded_start and not any(mark in stripped_replacement for mark in _DOUBLE_QUOTATION_MARKS):
            start_character, end_character, restored_before, restored_after = _balanced_quote_removal(
                source_text, start_character, end_character,
            )
        if (
            ends_with_title
            and not stripped_replacement.endswith(".")
            and end_character < len(source_text)
            and source_text[end_character] == "."
            and alphanumeric_signature(source_text[end_character + 1:])
        ):
            # The abbreviation's own period goes with it; the sentence continues.
            end_character += 1
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
        retained = source_text[cursor:start_character]
        # A quotation that keeps words on one side of the edit keeps its mark there.
        pieces.append(f"{retained.rstrip()}{restored_before} " if restored_before else retained)
        if bounded_start == bounded_end and stripped_replacement:
            left = source_text[start_character - 1:start_character] if start_character else ""
            right = source_text[end_character:end_character + 1]
            prefix = "" if join_word_texts((left, stripped_replacement)) == left + stripped_replacement else " "
            suffix = "" if join_word_texts((stripped_replacement, right)) == stripped_replacement + right else " "
            pieces.append(f"{prefix}{stripped_replacement}{suffix}")
        else:
            pieces.append(stripped_replacement)
        if restored_after:
            pieces.append(f" {restored_after}")
            while end_character < len(source_text) and source_text[end_character] in " \t":
                end_character += 1
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
    if not _DANGLING_SEPARATOR_RE.search(source_text):
        # A removed clause leaves its separator before the next mark
        # ("a pol\u00edcia,." / "a oferecer,,"); authored clusters stay untouched.
        normalized = _DANGLING_SEPARATOR_RE.sub("", normalized)
    for marker, fragment in protected:
        normalized = normalized.replace(marker, fragment)
    return _restore_terminal_punctuation(normalized, source_text)


_DANGLING_SEPARATOR_RE = re.compile(r"[,;:]+(?=[.!?\u2026,;:])")
_DOUBLE_QUOTATION_MARKS = '"\u201c\u201d\u201e\u201f\u00ab\u00bb'
# English and German contraction suffixes; the word before them is complete.
_CONTRACTION_SUFFIXES = frozenset({"s", "t", "m", "d", "re", "ve", "ll"})
# Written title abbreviations and the words actors say for them (accent-folded).
_TITLE_ABBREVIATIONS: dict[str, frozenset[str]] = {
    "sr": frozenset({"senhor", "senor"}), "sra": frozenset({"senhora", "senora"}),
    "srta": frozenset({"senhorita", "senorita"}), "srs": frozenset({"senhores", "senores"}),
    "sras": frozenset({"senhoras", "senoras"}),
    "dr": frozenset({"doutor", "doctor", "doktor", "docteur"}), "dra": frozenset({"doutora", "doctora"}),
    "prof": frozenset({"professor", "profesor", "professeur"}), "profa": frozenset({"professora", "profesora"}),
    "mr": frozenset({"mister"}), "mrs": frozenset({"missus", "missis"}), "ms": frozenset(),
    "hr": frozenset({"herr"}), "fr": frozenset({"frau"}),
    "mme": frozenset({"madame"}), "mlle": frozenset({"mademoiselle"}), "m": frozenset({"monsieur"}),
    "d": frozenset({"dom", "dona", "don"}), "st": frozenset(), "jr": frozenset({"junior"}),
}


def _is_title_abbreviation(text: str, start: int, end: int) -> bool:
    return text[start:end].casefold() in _TITLE_ABBREVIATIONS and text[end:end + 1] == "."


def _is_spoken_title(abbreviation: str, replacement: str) -> bool:
    spoken = alphanumeric_signature(replacement)
    return len(spoken) == 1 and spoken[0] in _TITLE_ABBREVIATIONS.get(abbreviation.casefold(), frozenset())


def _balanced_quote_removal(text: str, start: int, end: int) -> tuple[int, int, str, str]:
    """Never leave half of a quotation behind when an edit removes one of its marks.

    The partner directly beside the removed range goes with it (the whole
    quotation is gone). A partner farther away still encloses retained words:
    the removed mark is named so the caller puts it back beside them. Returns
    the adjusted range, a mark to restore before it and one to restore after.
    """
    positions = [index for index, character in enumerate(text) if character in _DOUBLE_QUOTATION_MARKS]
    restored_before = restored_after = ""
    for opening, closing in zip(positions[0::2], positions[1::2]):
        if start <= opening < end <= closing:
            if closing == end:
                end += 1
            else:
                restored_after = text[opening]
        elif opening < start <= closing < end:
            if opening == start - 1:
                start -= 1
            else:
                restored_before = text[closing]
    return start, end, restored_before, restored_after


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
