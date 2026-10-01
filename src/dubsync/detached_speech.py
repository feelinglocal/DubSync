"""Keep words spoken seconds away from a cue out of that cue.

The aligner puts every ASR word between two matched words into one divergence
case, regardless of time. An approved wording could therefore carry another
actor's line or a late interjection into a cue shown at a different time
("no ano novo. Alô?" with "Alô?" spoken 14 s later). After adjudication each
such case is divided at its large acoustic gaps: the group spoken at the cue's
own time edits the cue, every other approved group becomes a pure insertion
with its own timing. A wording that cannot be divided is held as a whole.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .adjudication_regions import (
    JOINT_REGION_PREFIX, PROTECTED_SOURCE_PREFIX, SONG_CAPTION_PREFIX, SPEECH_REPEAT_PREFIX,
)
from .changes import indexed_span_bounds, replacement_text_cuts
from .edit_consistency import held_decisions
from .models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SpeechRegion, Word
from .subtitle_annotations import speech_text_for_alignment
from .text_metrics import join_word_texts
from .tokenize import alphanumeric_signature

DETACHED_SPEECH_PREFIX = "detached-"
_DERIVED_PREFIXES = (
    JOINT_REGION_PREFIX, PROTECTED_SOURCE_PREFIX, SPEECH_REPEAT_PREFIX, SONG_CAPTION_PREFIX,
    DETACHED_SPEECH_PREFIX,
)
_HELD_KIND = "adjudication_replacement_ownership_held"
# A cue without retained words is expected near its source time, moved by the
# offset its matched neighbours show.
_EXPECTED_WINDOW_PAD_SECONDS = 0.5
_NEIGHBOUR_SEARCH_CUES = 4
# A far interjection of at most this many words may be a mistimed ASR word
# when speech without any word is heard this close to the cue's own words.
_MISTIMED_WORD_MAX_TOKENS = 2
_ISOLATED_WORD_GAP_SECONDS = 0.3
_UNLABELLED_SPEECH_REACH_SECONDS = 0.6
_REGION_EDGE_TOLERANCE_SECONDS = 0.05


@dataclass(frozen=True)
class _Group:
    indices: list[int]
    start: float
    end: float


@dataclass(frozen=True)
class _Separation:
    spans: list[DivergenceSpan]
    decisions: list[AdjudicationDecision]
    flags: list[QCFlag]


def separate_detached_speech(
    cues: list[Cue],
    alignment: AlignmentResult,
    decisions: list[AdjudicationDecision],
    words: list[Word],
    *,
    max_intra_cue_gap: float = 1.5,
    protected_cue_ids: set[int] | None = None,
    speech_regions: list[SpeechRegion] | None = None,
) -> tuple[AlignmentResult, list[AdjudicationDecision], list[QCFlag]]:
    """Divide approved cases at acoustic gaps above ``max_intra_cue_gap``.

    Returns the alignment with the divided cases, their decisions and the
    flags of wordings that were held or left out. Cases without a large gap,
    kept source text and derived regions are returned unchanged.
    """
    by_case = {decision.case_id: decision for decision in decisions}
    protected = protected_cue_ids or set()
    spans: list[DivergenceSpan] = []
    replaced: dict[str, list[AdjudicationDecision]] = {}
    held_case_ids: set[str] = set()
    flags: list[QCFlag] = []
    for span in alignment.divergence_spans:
        decision = by_case.get(span.case_id)
        result = _separate(span, decision, cues, alignment, words, max_intra_cue_gap, protected, speech_regions)
        if result is None:
            spans.append(span)
            continue
        flags.extend(result.flags)
        if not result.spans:
            held_case_ids.add(span.case_id)
            spans.append(span)
            continue
        spans.extend(result.spans)
        replaced[span.case_id] = result.decisions
    if not replaced and not held_case_ids:
        return alignment, decisions, flags
    decisions = held_decisions(
        decisions, alignment.divergence_spans, held_case_ids,
        "The approved wording spans speech groups that are seconds apart and cannot be divided between them.",
    )
    divided: list[AdjudicationDecision] = []
    for decision in decisions:
        divided.extend(replaced.get(decision.case_id, [decision]))
    return alignment.model_copy(update={"divergence_spans": spans}), divided, flags


def _separate(
    span: DivergenceSpan, decision: AdjudicationDecision | None, cues: list[Cue],
    alignment: AlignmentResult, words: list[Word], max_gap: float, protected: set[int],
    speech_regions: list[SpeechRegion] | None,
) -> _Separation | None:
    if decision is None or span.case_id.startswith(_DERIVED_PREFIXES):
        return None
    kept = decision.verdict == "keep_srt"
    insertion = not span.cue_ids and not span.srt_token_indices
    # Kept source text of one cue is still timed from the case's words: only
    # the group at the cue's own time may time it.
    if (kept and (insertion or len(set(span.cue_ids)) != 1)) or (not kept and not decision.final_text.strip()):
        return None
    groups = _acoustic_groups(span, words, max_gap)
    # Even a single word can be seconds away from the cue of its case.
    if groups is None or (len(groups) < 2 and insertion):
        return None
    spoken = [words[index] for index in span.asr_word_indices]
    boundaries = [group.indices[0] - span.asr_word_indices[0] for group in groups[1:]]
    cuts = None if kept else replacement_text_cuts(decision.final_text, spoken, boundaries)
    if insertion:
        return _separated_insertion(span, decision, groups, cuts, words)
    return _separated_replacement(
        span, decision, groups, cuts, cues, alignment, words, max_gap, protected, speech_regions,
    )


def _acoustic_groups(span: DivergenceSpan, words: list[Word], max_gap: float) -> list[_Group] | None:
    indices = span.asr_word_indices
    if (
        not indices or indices != list(range(indices[0], indices[-1] + 1))
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
    runs: list[list[int]] = [[indices[0]]]
    for index, previous, word in zip(indices[1:], spoken, spoken[1:]):
        if word.start - previous.end > max_gap:
            runs.append([])
        runs[-1].append(index)
    return [_Group(run, words[run[0]].start, max(words[index].end for index in run)) for run in runs]


def _pieces(final_text: str, cuts: list[int]) -> list[str]:
    edges = [0, *cuts, len(final_text)]
    return [final_text[start:end].strip() for start, end in zip(edges, edges[1:])]


def _separated_insertion(
    span: DivergenceSpan, decision: AdjudicationDecision, groups: list[_Group],
    cuts: list[int] | None, words: list[Word],
) -> _Separation | None:
    # Speech inserted between two words of one cue belongs to that cue's own
    # pause; text that cannot be divided stays one generated candidate, which
    # the ad-lib segmentation times from its words.
    if cuts is None or (
        span.left_anchor_cue_id is not None and span.left_anchor_cue_id == span.right_anchor_cue_id
    ):
        return None
    spans: list[DivergenceSpan] = []
    decisions: list[AdjudicationDecision] = []
    for number, (group, text) in enumerate(zip(groups, _pieces(decision.final_text, cuts))):
        if not alphanumeric_signature(text):
            continue
        case_id = span.case_id if not spans else f"{DETACHED_SPEECH_PREFIX}{number}-{span.case_id}"
        spans.append(_group_span(span, group, words, case_id))
        decisions.append(decision.model_copy(update={"case_id": case_id, "final_text": text}))
    return _Separation(spans, decisions, []) if len(spans) > 1 else None


def _separated_replacement(
    span: DivergenceSpan, decision: AdjudicationDecision, groups: list[_Group],
    cuts: list[int] | None, cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    max_gap: float, protected: set[int], speech_regions: list[SpeechRegion] | None,
) -> _Separation | None:
    bounds = indexed_span_bounds(cues, span)
    if bounds is None:
        return None
    cues_by_id = {cue.index: cue for cue in cues}
    cue_ids = list(bounds)
    token_counts = {
        cue_id: len(alphanumeric_signature(speech_text_for_alignment(cues_by_id[cue_id]))) for cue_id in cue_ids
    }
    consumed = [cue_id for cue_id in cue_ids if bounds[cue_id] == (0, token_counts[cue_id])]
    # A cue that keeps matched words before or after the case is spoken where
    # those words are; a completely replaced cue near its source time.
    keeps_prefix = bounds[cue_ids[0]][0] > 0
    keeps_suffix = bounds[cue_ids[-1]][1] < token_counts[cue_ids[-1]]
    left_home = keeps_prefix and span.left_anchor_cue_id == cue_ids[0] and _finite(span.left_anchor_end)
    right_home = keeps_suffix and span.right_anchor_cue_id == cue_ids[-1] and _finite(span.right_anchor_start)
    if (keeps_prefix and not left_home) or (keeps_suffix and not right_home):
        # The retained words carry no timing: nothing says where the cue is spoken.
        return None
    if len(cue_ids) == 1 and left_home and right_home:
        # The cue's own retained words surround the gap: the pause is its own.
        return None

    home = [False] * len(groups)
    if left_home and groups[0].start - span.left_anchor_end <= max_gap:
        home[0] = True
    if right_home and span.right_anchor_start - groups[-1].end <= max_gap:
        home[-1] = True
    for cue_id in consumed:
        window = _expected_window(cues_by_id[cue_id], cues, alignment, words)
        for position, group in enumerate(groups):
            if group.end > window[0] and group.start < window[1]:
                home[position] = True
    whole_cues_only = len(consumed) == len(cue_ids)
    kept = decision.verdict == "keep_srt"
    if not any(home):
        # Only complete cues: the whole-cue planner selects one exact word
        # window or holds. A partly retained cue has no approved word nearby.
        if whole_cues_only:
            return None
        wordless = span.model_copy(update={
            "asr_word_indices": [], "asr_text": "", "speaker_ids": [], "confidence": 0.0,
        })
        if kept:
            # None of the words is spoken with the kept cue: they time nothing.
            return _Separation([wordless], [decision], [])
        return _separated_without_home(
            span, wordless, decision, groups, cuts, words, left_home, protected, speech_regions,
        )
    first, last = home.index(True), len(home) - 1 - home[::-1].index(True)
    if first == 0 and last == len(groups) - 1:
        return None
    home_group = _Group(
        [index for group in groups[first:last + 1] for index in group.indices],
        groups[first].start, max(group.end for group in groups[first:last + 1]),
    )
    home_span = _group_span(span, home_group, words, span.case_id, keep_source=True)
    if kept:
        return _Separation([home_span], [decision], [])
    if cuts is None:
        return None if whole_cues_only else _held(span, decision)

    pieces = _pieces(decision.final_text, cuts)
    edges = [0, *cuts, len(decision.final_text)]
    # A short word the ASR timed far away while unlabelled speech is heard
    # directly beside the cue's own words was most likely spoken there: its
    # text stays with the cue, its misplaced timestamp times nothing.
    text_first, text_last = first, last
    if first > 0 and _is_mistimed_beside(groups[first - 1], groups[first], words, speech_regions):
        text_first = first - 1
    if last + 1 < len(groups) and _is_mistimed_beside(groups[last + 1], groups[last], words, speech_regions):
        text_last = last + 1
    home_text = decision.final_text[edges[text_first]:edges[text_last + 1]].strip()
    home_decision = (
        decision.model_copy(update={"final_text": home_text}) if alphanumeric_signature(home_text)
        else decision.model_copy(update={
            "verdict": "keep_srt", "final_text": span.srt_text,
            "reason": f"{decision.reason} None of the approved words is spoken at the cue's own time; source text was kept.",
        })
    )
    spans: list[DivergenceSpan] = []
    decisions: list[AdjudicationDecision] = []
    flags: list[QCFlag] = []
    first_home_word, last_home_word = words[home_group.indices[0]], words[home_group.indices[-1]]
    for number, (group, text) in enumerate(zip(groups, pieces)):
        if first <= number <= last:
            if number == first:
                spans.append(home_span)
                decisions.append(home_decision)
            continue
        if not alphanumeric_signature(text) or text_first <= number <= text_last:
            continue
        leading = number < first
        anchor_update = {
            "right_anchor_cue_id": cue_ids[0], "right_anchor_start": home_group.start,
            "right_anchor_speaker_id": first_home_word.speaker_id,
        } if leading else {
            "left_anchor_cue_id": cue_ids[-1], "left_anchor_end": home_group.end,
            "left_anchor_speaker_id": last_home_word.speaker_id,
        }
        outer_anchor = span.left_anchor_cue_id if leading else span.right_anchor_cue_id
        if outer_anchor in protected:
            flags.append(_left_out_flag(span, decision, text, group))
            continue
        case_id = f"{DETACHED_SPEECH_PREFIX}{number}-{span.case_id}"
        spans.append(_group_span(span, group, words, case_id).model_copy(update=anchor_update))
        decisions.append(decision.model_copy(update={"case_id": case_id, "final_text": text}))
    return _Separation(spans, decisions, flags)


def _separated_without_home(
    span: DivergenceSpan, wordless: DivergenceSpan, decision: AdjudicationDecision, groups: list[_Group],
    cuts: list[int] | None, words: list[Word], after_retained_words: bool, protected: set[int],
    speech_regions: list[SpeechRegion] | None,
) -> _Separation | None:
    """Every approved word is spoken seconds away from the partly retained cue.

    The audio has other words elsewhere instead of the cue's replaced tokens:
    the tokens are removed like any approved deletion and each group becomes
    an insertion with its own timing (next to a neighbouring cue it joins that
    cue, as a far word directly before the next line always did). Only a
    wording that follows the spoken words one by one can be placed that way.
    """
    spoken_tokens = sum(len(alphanumeric_signature(words[index].text)) for index in span.asr_word_indices)
    if len(groups) > 1 and cuts is None or len(alphanumeric_signature(decision.final_text)) != spoken_tokens:
        return _held(span, decision) if len(groups) > 1 else None
    anchor_time = span.left_anchor_end if after_retained_words else span.right_anchor_start
    if len(groups) == 1 and _is_mistimed_beside(
        groups[0], _Group([], anchor_time, anchor_time), words, speech_regions,
    ):
        return None
    pieces = _pieces(decision.final_text, cuts or [])
    if any(not alphanumeric_signature(piece) for piece in pieces):
        return _held(span, decision) if len(groups) > 1 else None
    spans = [wordless]
    decisions = [decision.model_copy(update={
        "final_text": "",
        "reason": f"{decision.reason} The approved words are spoken seconds away and were placed at their own time.",
    })]
    flags: list[QCFlag] = []
    outer_anchors = {span.left_anchor_cue_id, span.right_anchor_cue_id} - set(span.cue_ids)
    for number, (group, text) in enumerate(zip(groups, pieces)):
        if outer_anchors & protected:
            flags.append(_left_out_flag(span, decision, text, group))
            continue
        case_id = f"{DETACHED_SPEECH_PREFIX}{number}-{span.case_id}"
        spans.append(_group_span(span, group, words, case_id))
        decisions.append(decision.model_copy(update={"case_id": case_id, "final_text": text}))
    if not after_retained_words:
        # The words precede the cue's retained words: keep time order.
        spans = [*spans[1:], spans[0]]
        decisions = [*decisions[1:], decisions[0]]
    return _Separation(spans, decisions, flags)


def _left_out_flag(span: DivergenceSpan, decision: AdjudicationDecision, text: str, group: _Group) -> QCFlag:
    # Like every insertion next to a cue without audio evidence, the words may
    # be that cue's own line: they are reported, not generated.
    return QCFlag(
        kind=_HELD_KIND, cue_ids=list(span.cue_ids), severity="warning",
        message=(
            "Approved words spoken seconds away from this cue stand next to a source cue without "
            "audio evidence; they were left out instead of being shown at the wrong time."
        ),
        confidence=decision.confidence, old_text=span.srt_text, new_text=text,
        start=group.start, end=group.end,
    )


def _is_mistimed_beside(
    group: _Group, home: _Group, words: list[Word], speech_regions: list[SpeechRegion] | None,
) -> bool:
    """A short far word while speech without any ASR word adjoins the cue's own words.

    MAI timed "Ah." 19 s before 'Chama a polícia.' although an unlabelled
    burst is heard 0.25 s before "Chama"; the reviewer kept "Ah." in the cue.
    """
    if not speech_regions or sum(
        len(alphanumeric_signature(words[index].text)) for index in group.indices
    ) > _MISTIMED_WORD_MAX_TOKENS:
        return False
    leading = group.end <= home.start
    # A word that runs into other speech at its own time is spoken there.
    outer = group.indices[0] - 1 if leading else group.indices[-1] + 1
    if 0 <= outer < len(words) and (
        group.start - words[outer].end if leading else words[outer].start - group.end
    ) < _ISOLATED_WORD_GAP_SECONDS:
        return False
    if leading:
        return any(
            region.start >= group.end - _REGION_EDGE_TOLERANCE_SECONDS
            and region.end <= home.start + _REGION_EDGE_TOLERANCE_SECONDS
            and home.start - region.end <= _UNLABELLED_SPEECH_REACH_SECONDS
            for region in speech_regions
        )
    return any(
        region.start >= home.end - _REGION_EDGE_TOLERANCE_SECONDS
        and region.end <= group.start + _REGION_EDGE_TOLERANCE_SECONDS
        and region.start - home.end <= _UNLABELLED_SPEECH_REACH_SECONDS
        for region in speech_regions
    )


def _finite(value: float | None) -> bool:
    return value is not None and isfinite(value)


def _held(span: DivergenceSpan, decision: AdjudicationDecision) -> _Separation:
    return _Separation([], [], [QCFlag(
        kind=_HELD_KIND, cue_ids=list(span.cue_ids), severity="warning",
        message=(
            "The approved wording spans speech groups that are seconds apart, and it cannot be told which "
            "of its words belong to this cue. Source text was kept for review."
        ),
        confidence=decision.confidence, old_text=span.srt_text, new_text=decision.final_text,
        start=span.start, end=span.end,
    )])


def _group_span(
    span: DivergenceSpan, group: _Group, words: list[Word], case_id: str, *, keep_source: bool = False,
) -> DivergenceSpan:
    spoken = [words[index] for index in group.indices]
    confidences = [word.confidence for word in spoken if word.confidence is not None]
    update: dict[str, object] = {
        "case_id": case_id,
        "asr_word_indices": list(group.indices),
        "asr_text": join_word_texts(word.text for word in spoken),
        "start": group.start, "end": group.end,
        "confidence": min(confidences) if confidences else 0.0,
        "speaker_ids": sorted({word.speaker_id for word in spoken if word.speaker_id}),
    }
    if not keep_source:
        update.update({"cue_ids": [], "srt_text": "", "srt_token_indices": [], "insertion_token_offset": None})
    return span.model_copy(update=update)


def _expected_window(
    cue: Cue, cues: list[Cue], alignment: AlignmentResult, words: list[Word],
) -> tuple[float, float]:
    """Where a cue without retained words is expected to be spoken."""
    position = next(index for index, candidate in enumerate(cues) if candidate.index == cue.index)
    offsets: list[float] = []
    for step in (-1, 1):
        neighbour_position = position + step
        for _ in range(_NEIGHBOUR_SEARCH_CUES):
            if not 0 <= neighbour_position < len(cues):
                break
            neighbour = cues[neighbour_position]
            owned = [
                words[index] for index in alignment.cue_word_indices.get(neighbour.index, [])
                if 0 <= index < len(words)
            ]
            if owned:
                offsets.append(min(word.start for word in owned) - neighbour.start_ms / 1000.0)
                break
            neighbour_position += step
    low, high = (min(offsets), max(offsets)) if offsets else (0.0, 0.0)
    return (
        cue.start_ms / 1000.0 + low - _EXPECTED_WINDOW_PAD_SECONDS,
        cue.end_ms / 1000.0 + high + _EXPECTED_WINDOW_PAD_SECONDS,
    )
