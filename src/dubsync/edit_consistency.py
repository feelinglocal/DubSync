"""Keep an approved wording and its timing evidence together.

An adjudicated replacement changes a cue's text and, separately, the ASR
words that time it. When only one of the two could be applied, the viewer saw
new words at the old source time, or one-letter leftovers in the source
slots. Every function here turns such a half-applied edit into one of two
consistent outcomes: the wording is shown at the acoustic time of the words
the adjudicator heard, or source text and source timing are kept together.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from .adjudication_regions import heard_accent_collision
from .cue_segmentation import join_one_letter_residues
from .models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, TokenMatch, Word
from .style_profile import StyleProfile
from .subtitle_annotations import speech_text_for_alignment
from .text_metrics import join_word_texts
from .tokenize import alphanumeric_signature, tokenize_cues

ApplyText = Callable[[list[AdjudicationDecision]], tuple[list[Cue], list[QCFlag]]]
Rebuild = Callable[[list[Cue]], tuple[list[Cue], list[QCFlag]]]

_HELD_KIND = "adjudication_replacement_ownership_held"


def held_decisions(
    decisions: list[AdjudicationDecision], spans: list[DivergenceSpan], case_ids: set[str], reason: str,
) -> list[AdjudicationDecision]:
    """Return the decisions with the given cases turned into source keeps."""
    if not case_ids:
        return decisions
    source_text = {span.case_id: span.srt_text for span in spans}
    return [
        decision.model_copy(update={
            "verdict": "keep_srt", "final_text": source_text.get(decision.case_id, decision.final_text),
            "reason": f"{reason} Proposed {decision.verdict}: {decision.final_text!r}.",
        })
        if decision.case_id in case_ids and decision.verdict != "keep_srt" else decision
        for decision in decisions
    ]


def hold_fragmenting_replacements(
    cues: list[Cue], spans: list[DivergenceSpan], decisions: list[AdjudicationDecision], apply_text: ApplyText,
) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    """Hold every replacement whose pieces would leave a one-letter cue.

    Distributing "E o Yuanzhu" over three source cues produced the cues "E."
    and "o.": pieces of a sentence that the adjudicator approved as a whole.
    A complete approved one-letter answer ("É.") is not a leftover and stays;
    the residue of an approved deletion is resolved after the rebuild, when
    its word timing is known (``settle_one_letter_residues``).
    """
    by_case = {decision.case_id: decision for decision in decisions}
    edited, flags = apply_text(decisions)
    fragments = _one_letter_leftovers(cues, edited)
    if not fragments:
        return decisions, []
    edited_by_id = {cue.index: cue for cue in edited}
    held_case_ids: set[str] = set()
    hold_flags: list[QCFlag] = []
    for flag in flags:
        if flag.kind != "text_changed" or not fragments.intersection(flag.cue_ids):
            continue
        for span, decision in _decided_spans_of_flag(flag, spans, by_case):
            approved = alphanumeric_signature(decision.final_text)
            if not approved or span.case_id in held_case_ids or not any(
                _signature(edited_by_id[cue_id]) != approved
                for cue_id in flag.cue_ids if cue_id in fragments
            ):
                continue
            held_case_ids.add(span.case_id)
            hold_flags.append(QCFlag(
                kind=_HELD_KIND, cue_ids=list(flag.cue_ids), severity="warning",
                message=(
                    "The approved wording could only be placed by leaving a one-letter cue; "
                    "the complete replacement was held and the source text kept for review."
                ),
                confidence=decision.confidence, old_text=span.srt_text, new_text=decision.final_text,
                start=span.start, end=span.end,
            ))
    return held_decisions(
        decisions, spans, held_case_ids,
        "The replacement would leave a one-letter cue; source text was kept for review.",
    ), hold_flags


def hold_edits_beside_accent_anchors(
    cues: list[Cue], spans: list[DivergenceSpan], decisions: list[AdjudicationDecision],
    token_matches: list[TokenMatch], words: list[Word],
) -> tuple[list[AdjudicationDecision], list[QCFlag]]:
    """Hold every edit of a cue that would keep an unheard accent anchor beside new wording.

    A retained one-letter match whose accent differs from the ASR word
    ("é" / "E") is no evidence for its spelling; the audio question of the
    adjacent case normally includes it (``extend_boundary_anchor_regions``).
    When it could not, an approved edit beside it delivered the script's word
    inside the new wording ("é eu ia lá saber"). The cue keeps its script
    wording and timing, and the held proposals are listed for review.
    """
    tokens = tokenize_cues(cues)
    first_token: dict[int, int] = {}
    for token in tokens:
        first_token.setdefault(token.cue_id, token.token_index)
    claimed = {index for span in spans for index in span.srt_token_indices}
    matched: dict[int, list[TokenMatch]] = defaultdict(list)
    for match in token_matches:
        matched[match.srt_token_index].append(match)
    by_case = {decision.case_id: decision for decision in decisions}

    def edited(span: DivergenceSpan) -> tuple[list[int], list[tuple[int, int]]]:
        """The cues a case edits and the retained (token, cue) positions on both sides of it."""
        source = span.srt_token_indices
        if source:
            if not 0 <= source[0] <= source[-1] < len(tokens):
                return list(span.cue_ids), []
            return list(span.cue_ids), [
                (source[0] - 1, tokens[source[0]].cue_id), (source[-1] + 1, tokens[source[-1]].cue_id),
            ]
        cue_id = span.right_anchor_cue_id
        if span.insertion_token_offset is None or cue_id != span.left_anchor_cue_id or cue_id not in first_token:
            return [], []
        position = first_token[cue_id] + span.insertion_token_offset
        return [cue_id], [(position - 1, cue_id), (position, cue_id)]

    edits = [
        (span, decision, *edited(span)) for span in spans
        if (decision := by_case.get(span.case_id)) is not None and decision.verdict != "keep_srt"
        and alphanumeric_signature(decision.final_text) != alphanumeric_signature(span.srt_text)
    ]
    anchors: dict[int, tuple[str, str]] = {}
    for _, _, _, neighbours in edits:
        for neighbour, cue_id in neighbours:
            if (
                not 0 <= neighbour < len(tokens) or neighbour in claimed or tokens[neighbour].cue_id != cue_id
                or len(matched[neighbour]) != 1 or matched[neighbour][0].score != 1.0
                or not 0 <= matched[neighbour][0].asr_word_index < len(words)
            ):
                continue
            word = words[matched[neighbour][0].asr_word_index]
            if heard_accent_collision(tokens[neighbour], word):
                anchors.setdefault(cue_id, (tokens[neighbour].text, word.text))
    if not anchors:
        return decisions, []
    held_case_ids: set[str] = set()
    hold_flags: list[QCFlag] = []
    for span, decision, cue_ids, _ in edits:
        cue_id = next((cue_id for cue_id in cue_ids if cue_id in anchors), None)
        if cue_id is None:
            continue
        written, heard = anchors[cue_id]
        held_case_ids.add(span.case_id)
        hold_flags.append(QCFlag(
            kind="adjudication_span_edit_held", cue_ids=cue_ids, severity="error",
            message=(
                f"The script word '{written}' beside this approved wording was heard as '{heard}' but was not "
                "part of the AI question, so the cue would show it inside the new wording. The script wording "
                "was kept for review."
            ),
            confidence=decision.confidence, old_text=span.srt_text, new_text=decision.final_text,
            start=span.start, end=span.end,
        ))
    return held_decisions(
        decisions, spans, held_case_ids,
        "A retained word beside the edit has an accent the audio does not confirm; source text was kept for review.",
    ), hold_flags


def settle_edits_with_held_timing(
    edited_cues: list[Cue],
    rebuilt: list[Cue],
    recue_flags: list[QCFlag],
    flags: list[QCFlag],
    *,
    source_cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    rebuild: Rebuild,
    max_intra_cue_gap: float,
    max_word_duration: float,
) -> tuple[list[Cue], list[QCFlag], list[QCFlag]]:
    """Resolve cues whose wording was changed while their timing was held.

    The rebuild holds a cue at source timing when its words do not lexically
    support its text. For a cue with approved new wording that is expected:
    the adjudicator kept a source spelling ("Sun Yu" for the ASR's "Sonho"),
    wrote digits as words, or added a word to a line that is only partly
    spoken. When the words the cue owns form one dense utterance, they are
    where its wording is spoken and the cue is timed from them. Otherwise the
    new wording has no place of its own (a word 14 s away, collapsed
    timestamps): source text and source timing are kept together. A cue that
    only lost words shows source words at source timing and is left alone.
    """
    source_by_id = {cue.index: cue for cue in source_cues}
    edited_by_id = {cue.index: cue for cue in edited_cues}
    changed_ids = {cue_id for flag in flags if flag.kind == "text_changed" for cue_id in flag.cue_ids}
    held_edited = {
        cue_id for flag in recue_flags if flag.kind == "timing_evidence_held"
        for cue_id in flag.cue_ids
        if cue_id in source_by_id and cue_id in changed_ids and cue_id in edited_by_id
        and not _is_subsequence(_signature(edited_by_id[cue_id]), _signature(source_by_id[cue_id]))
    }
    if not held_edited:
        return rebuilt, recue_flags, flags

    evidence_text: dict[int, str] = {}
    for cue_id in held_edited:
        owned = [index for index in alignment.cue_word_indices.get(cue_id, []) if 0 <= index < len(words)]
        ordered = sorted((words[index] for index in owned), key=lambda word: (word.start, word.end))
        if (
            ordered
            and all(word.end - word.start <= max_word_duration for word in ordered)
            and all(right.start - left.end <= max_intra_cue_gap for left, right in zip(ordered, ordered[1:]))
        ):
            evidence_text[cue_id] = join_word_texts(word.text for word in ordered)

    rescued: set[int] = set()
    if evidence_text:
        # Time the candidates exactly like every other cue: the rebuild only
        # needs to see the words they own as their lexical evidence.
        probe = [cue.with_lines([evidence_text[cue.index]]) if cue.index in evidence_text else cue for cue in edited_cues]
        probe_rebuilt, probe_flags = rebuild(probe)
        still_held = {
            cue_id for flag in probe_flags if flag.kind == "timing_evidence_held" for cue_id in flag.cue_ids
        }
        rescued = set(evidence_text) - still_held
        if rescued:
            lines_by_id = {cue.index: cue.lines for cue in edited_cues}
            rebuilt = [
                cue.with_lines(lines_by_id[cue.index]) if cue.index in evidence_text else cue
                for cue in probe_rebuilt
            ]
            recue_flags = probe_flags

    return _keep_source_together(
        rebuilt, recue_flags, flags, held_edited - rescued, source_cues,
        "The approved wording had no usable word timing of its own; source text and "
        "source timing were kept together for review.",
    )


def settle_one_letter_residues(
    rebuilt: list[Cue],
    recue_flags: list[QCFlag],
    flags: list[QCFlag],
    *,
    source_cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    profile: StyleProfile,
    fixed_cue_ids: set[int],
    split_cue_ids: set[int] | None = None,
) -> tuple[list[Cue], AlignmentResult, list[QCFlag], list[QCFlag]]:
    """Never export the one-letter residue of an approved deletion as a cue.

    'e não deixei que ele conseguisse' was delivered as the cue "e" for 100 ms.
    The spoken letter is joined to the cue it is spoken with; without such a
    neighbour the deletion is not applied and the source cue is kept.

    ``split_cue_ids`` names source cues that were divided into several cues
    (a speaker turn): a short first piece such as "É." is a complete turn
    there, not what a deletion left behind.
    """
    leftovers = _one_letter_leftovers(source_cues, rebuilt) - (split_cue_ids or set())
    if not leftovers:
        return rebuilt, alignment, recue_flags, flags
    by_case = {decision.case_id: decision for decision in decisions}
    rebuilt_by_id = {cue.index: cue for cue in rebuilt}
    residues: set[int] = set()
    for flag in flags:
        if flag.kind != "text_changed" or not leftovers.intersection(flag.cue_ids):
            continue
        # A complete approved one-letter answer ("É.") is not a residue.
        approved = [alphanumeric_signature(decision.final_text) for _, decision in _decided_spans_of_flag(flag, spans, by_case)]
        residues.update(
            cue_id for cue_id in leftovers.intersection(flag.cue_ids)
            if _signature(rebuilt_by_id[cue_id]) not in approved
        )
    if not residues:
        return rebuilt, alignment, recue_flags, flags
    rebuilt, alignment, join_flags, joined = join_one_letter_residues(
        rebuilt, words, alignment, profile, residue_cue_ids=residues, fixed_cue_ids=fixed_cue_ids,
    )
    if joined:
        # The deletion and the join are one reported change of the receiving cue.
        flags = [
            flag for flag in flags
            if not (flag.kind == "text_changed" and flag.cue_ids and set(flag.cue_ids) <= joined)
        ]
        recue_flags = [flag for flag in recue_flags if not (flag.cue_ids and set(flag.cue_ids) <= joined)]
        flags = [*flags, *join_flags]
    rebuilt, recue_flags, flags = _keep_source_together(
        rebuilt, recue_flags, flags, residues - joined, source_cues,
        "The approved deletion would leave a one-letter cue without an adjacent cue to join; "
        "source text and source timing were kept together for review.",
    )
    return rebuilt, alignment, recue_flags, flags


def _keep_source_together(
    rebuilt: list[Cue], recue_flags: list[QCFlag], flags: list[QCFlag],
    unresolved: set[int], source_cues: list[Cue], message: str,
) -> tuple[list[Cue], list[QCFlag], list[QCFlag]]:
    """Put back source text and timing of every cue of the unresolved edits."""
    if not unresolved:
        return rebuilt, recue_flags, flags
    source_by_id = {cue.index: cue for cue in source_cues}
    restore: set[int] = set(unresolved)
    hold_flags: list[QCFlag] = []
    retained: list[QCFlag] = []
    for flag in flags:
        if flag.kind == "text_changed" and unresolved.intersection(flag.cue_ids):
            group = [cue_id for cue_id in flag.cue_ids if cue_id in source_by_id]
            restore.update(group)
            hold_flags.append(QCFlag(
                kind=_HELD_KIND, cue_ids=group, severity="warning", message=message,
                confidence=flag.confidence,
                old_text="\n".join(source_by_id[cue_id].text for cue_id in group),
                new_text=flag.new_text, start=flag.start, end=flag.end,
            ))
            continue
        retained.append(flag)
    # Another edit of a restored cue is undone with it.
    retained = [
        flag for flag in retained
        if not (flag.kind == "text_changed" and restore.intersection(flag.cue_ids))
    ]
    present = {cue.index for cue in rebuilt}
    position = {cue.index: index for index, cue in enumerate(source_cues)}
    missing = sorted((cue_id for cue_id in restore if cue_id not in present), key=position.__getitem__)
    restored: list[Cue] = []
    for cue in rebuilt:
        while missing and cue.index in position and position[missing[0]] < position[cue.index]:
            restored.append(source_by_id[missing.pop(0)])
        restored.append(
            source_by_id[cue.index].model_copy(update={"speaker_id": cue.speaker_id, "character": cue.character})
            if cue.index in restore else cue
        )
    restored.extend(source_by_id[cue_id] for cue_id in missing)
    # A restored cue is at source timing: findings about its word evidence are moot.
    recue_flags = [
        flag for flag in recue_flags
        if not (
            flag.kind in {"timing_evidence_held", "timing_outlier_trimmed"}
            and flag.cue_ids and set(flag.cue_ids) <= restore
        )
    ]
    return restored, recue_flags, [*retained, *hold_flags]


def _decided_spans_of_flag(
    flag: QCFlag, spans: list[DivergenceSpan], by_case: dict[str, AdjudicationDecision],
) -> list[tuple[DivergenceSpan, AdjudicationDecision]]:
    """The applied decisions a change flag describes (it carries their window)."""
    return [
        (span, decision) for span in spans
        if (decision := by_case.get(span.case_id)) is not None and decision.verdict != "keep_srt"
        and (span.start, span.end) == (flag.start, flag.end)
        and set(flag.cue_ids) & {*span.cue_ids, span.right_anchor_cue_id}
    ]


def _one_letter_leftovers(source_cues: list[Cue], edited: list[Cue]) -> set[int]:
    source_by_id = {cue.index: cue for cue in source_cues}
    return {
        cue.index for cue in edited
        if cue.index in source_by_id and _letter_count(cue) <= 1 < _letter_count(source_by_id[cue.index])
    }


def _is_subsequence(part: list[str], whole: list[str]) -> bool:
    remaining = iter(whole)
    return all(token in remaining for token in part)


def _signature(cue: Cue) -> list[str]:
    return alphanumeric_signature(speech_text_for_alignment(cue))


def _letter_count(cue: Cue) -> int:
    return sum(character.isalnum() for character in speech_text_for_alignment(cue))
