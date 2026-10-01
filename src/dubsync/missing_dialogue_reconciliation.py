"""Ask about an entire missing dialogue cue without making its neighbours editable.

ASR absence alone never removes text. A native clear answer may confirm an
omission in a fully covered, acoustically anchored gap. Audible wording also
needs one independent speech chain before it can supply cue timing.
Provider words and their ownership are read-only throughout this path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import ceil, floor, isfinite

from .adjudication_regions import is_song_caption_cue
from .asr_timing import MIN_OWNED_OVERLAP_SECONDS
from .models import AdjudicationDecision, AlignmentResult, Cue, CueContext, DivergenceSpan, QCFlag, SpeechRegion, Word
from .style_profile import StyleProfile
from .subtitle_annotations import cue_has_bracketed_screen_text, speech_text_for_alignment
from .text_metrics import join_word_texts
from .tokenize import alphanumeric_signature, tokenize_cues


MISSING_DIALOGUE_POLICY_VERSION = 1
MISSING_DIALOGUE_RESIDUAL_PREFIX = "missing-dialogue-residual-v1-"
_MAX_ANCHORED_WINDOW_SECONDS = 16.0
_MAX_TARGET_TOKENS = 16
# Match asr_timing's speech-chain convention without changing global VAD.
_MAX_SPEECH_CHAIN_GAP_SECONDS = 0.2
_EPSILON = 1e-7
_RELEASED_HOLD_KINDS = frozenset({
    "missing_audio_source_cue_held", "missing_audio_timing_held", "missing_audio_source_cue_restored",
    "unmatched_source_cue", "dropped_line_candidate",
})


@dataclass(frozen=True)
class MissingDialogueQuestion:
    span: DivergenceSpan
    parent_case_id: str
    left_word_indices: tuple[int, ...]
    right_word_indices: tuple[int, ...]
    read_only_source_tokens: tuple[str, ...]

    @property
    def cue_id(self) -> int:
        return self.span.cue_ids[0]

    def record(self) -> dict[str, object]:
        return {
            "span": self.span.model_dump(mode="json"), "parent_case_id": self.parent_case_id,
            "left_word_indices": list(self.left_word_indices), "right_word_indices": list(self.right_word_indices),
            "read_only_source_tokens": list(self.read_only_source_tokens),
        }


@dataclass(frozen=True)
class MissingDialogueResolution:
    cues: list[Cue]
    alignment: AlignmentResult
    flags: list[QCFlag]
    resolved_cue_ids: set[int]
    spoken_spans: dict[int, tuple[int, int]]
    outcomes: list[dict[str, object]]


@dataclass(frozen=True)
class MissingDialogueEvidence:
    questions: list[MissingDialogueQuestion]
    context: dict[str, object]
    decisions: list[AdjudicationDecision]
    flags: list[QCFlag]

    def artifact(self) -> dict[str, object]:
        content = {"context": self.context, "decisions": [d.model_dump(mode="json") for d in self.decisions],
                   "flags": [flag.model_dump(mode="json") for flag in self.flags]}
        return {**content, "receipt_sha256": _digest(content),
                "questions": [question.record() for question in self.questions]}


def _owners(alignment: AlignmentResult) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for cue_id, indices in alignment.cue_word_indices.items():
        for index in indices:
            result.setdefault(index, set()).add(cue_id)
    return result


def _anchor_indices(cue_id, alignment, words, tokens, owners, uncertain):
    indices = tuple(alignment.cue_word_indices.get(cue_id, ()))
    if not indices or indices != tuple(sorted(set(indices))):
        return ()
    if any(index < 0 or index >= len(words) or index in uncertain or owners.get(index) != {cue_id}
           or not isfinite(words[index].start) or not isfinite(words[index].end)
           or not 0 <= words[index].start < words[index].end
           or (words[index].confidence is not None and words[index].confidence < .7)
           for index in indices):
        return ()
    lexical = tuple(index for index in indices if alphanumeric_signature(words[index].text))
    matched = {
        match.asr_word_index for match in alignment.token_matches
        if match.cue_id == cue_id and match.asr_word_index in lexical and match.score >= .8
        and 0 <= match.srt_token_index < len(tokens) and tokens[match.srt_token_index].cue_id == cue_id
        and tokens[match.srt_token_index].normalized in alphanumeric_signature(words[match.asr_word_index].text)
    }
    # Both boundaries have a lexical source match, not merely an inferred
    # assignment of an unmatched word to an adjacent cue.
    if not lexical or lexical[0] not in matched or lexical[-1] not in matched:
        return ()
    if any(words[a].start > words[b].start or words[a].end > words[b].end
           for a, b in zip(lexical, lexical[1:])):
        return ()
    return lexical


def _lexical_words_in_gap(words: list[Word], start: float, end: float) -> bool:
    return any(alphanumeric_signature(word.text) and
               (not isfinite(word.start) or not isfinite(word.end)
                or (word.start < end - _EPSILON and word.end > start + _EPSILON))
               for word in words)


def build_missing_dialogue_questions(
    cues: list[Cue], alignment: AlignmentResult, words: list[Word], regions: list[SpeechRegion] | None, *,
    uncertain_word_indices: set[int] | None = None, audio_duration_seconds: float,
) -> list[MissingDialogueQuestion]:
    """Return narrowly editable questions; original mixed spans remain held.

    The question window contains both complete neighbouring acoustic anchors.
    Its editable source indices belong exclusively to one complete cue. Its
    gap boundaries come from independently matched words, never source times.
    """
    if (regions is None or alignment.diagnostics.unresolved or not isfinite(audio_duration_seconds)
            or audio_duration_seconds <= 0
            or any(not isfinite(r.start) or not isfinite(r.end) or not 0 <= r.start < r.end for r in regions)):
        return []
    tokens = tokenize_cues(cues)
    missing = set(alignment.diagnostics.missing_audio_cue_ids)
    owners = _owners(alignment)
    uncertain = uncertain_word_indices or set()
    result: list[MissingDialogueQuestion] = []
    for position, cue in enumerate(cues):
        if (cue.index not in missing or position == 0 or position + 1 >= len(cues)
                or is_song_caption_cue(cue) or cue_has_bracketed_screen_text(cue)
                or alignment.cue_word_indices.get(cue.index)):
            continue
        own = [token.token_index for token in tokens if token.cue_id == cue.index]
        parents = [span for span in alignment.divergence_spans
                   if cue.index in span.cue_ids or set(own).intersection(span.srt_token_indices)]
        if len(parents) != 1 or not own or len(own) > _MAX_TARGET_TOKENS:
            continue
        parent = parents[0]
        source_indices = parent.srt_token_indices
        if (not set(own) <= set(source_indices) or source_indices != sorted(set(source_indices))
                or source_indices[0] < 0 or source_indices[-1] >= len(tokens)
                or [tokens[index].normalized for index in source_indices] != alphanumeric_signature(parent.srt_text)
                or any(tokens[index].cue_id not in parent.cue_ids for index in source_indices)
                or set(parent.cue_ids) & missing != {cue.index}
                or parent.asr_word_indices or alphanumeric_signature(parent.asr_text)):
            continue
        left, right = cues[position - 1], cues[position + 1]
        if (parent.left_anchor_cue_id != left.index or parent.right_anchor_cue_id != right.index
                or {left.index, right.index} & missing
                or any(is_song_caption_cue(item) or cue_has_bracketed_screen_text(item) for item in (left, right))):
            continue
        left_indices = _anchor_indices(left.index, alignment, words, tokens, owners, uncertain)
        right_indices = _anchor_indices(right.index, alignment, words, tokens, owners, uncertain)
        if not left_indices or not right_indices:
            continue
        start, end = words[left_indices[0]].start, words[right_indices[-1]].end
        gap_start, gap_end = words[left_indices[-1]].end, words[right_indices[0]].start
        if (not start < gap_start < gap_end < end or end > audio_duration_seconds
                or end - start > _MAX_ANCHORED_WINDOW_SECONDS
                or _lexical_words_in_gap(words, gap_start, gap_end)):
            continue
        # Independent timestamps must themselves be covered by detected audio.
        if any(not any(r.start < words[index].end and r.end > words[index].start for r in regions)
               for index in (*left_indices, *right_indices)):
            continue
        def context(item: Cue) -> CueContext:
            return CueContext(cue_id=item.index, text=item.plain_text,
                              start=item.start_ms / 1000, end=item.end_ms / 1000)
        question = DivergenceSpan(
            case_id=f"missing-dialogue-v{MISSING_DIALOGUE_POLICY_VERSION}-{parent.case_id}-cue-{cue.index}",
            cue_ids=[cue.index], srt_text=speech_text_for_alignment(cue), asr_text="",
            srt_token_indices=own, asr_word_indices=[], start=start, end=end,
            context_before=[context(item) for item in cues[max(0, position - 2):position]],
            context_after=[context(item) for item in cues[position + 1:position + 3]],
            left_anchor_cue_id=left.index, right_anchor_cue_id=right.index,
            left_anchor_end=gap_start, right_anchor_start=gap_end,
            left_anchor_speaker_id=words[left_indices[-1]].speaker_id,
            right_anchor_speaker_id=words[right_indices[0]].speaker_id,
        )
        result.append(MissingDialogueQuestion(question, parent.case_id, left_indices, right_indices,
                      tuple(tokens[index].normalized for index in source_indices if index not in own)))
    return result


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def with_missing_dialogue_residual_questions(
    alignment: AlignmentResult, questions: list[MissingDialogueQuestion], cues: list[Cue],
) -> AlignmentResult:
    """Expose the nonmissing fragment to ordinary native-audio adjudication.

    The original mixed case stays held. Each additional case owns only its
    exact source-token residue in one acoustically anchored neighbouring cue;
    no neighbouring word is reassigned and no missing-cue hold is released.
    """
    tokens = tokenize_cues(cues)
    parents = {span.case_id: span for span in alignment.divergence_spans}
    added: list[DivergenceSpan] = []
    for question in questions:
        parent = parents[question.parent_case_id]
        for cue_id in dict.fromkeys(parent.cue_ids):
            if cue_id == question.cue_id:
                continue
            case_id = f"{MISSING_DIALOGUE_RESIDUAL_PREFIX}{parent.case_id}-cue-{cue_id}"
            if case_id in parents:
                continue
            indices = [index for index in parent.srt_token_indices if tokens[index].cue_id == cue_id]
            if not indices or any(set(indices).intersection(span.srt_token_indices)
                                  for span in alignment.divergence_spans if span.case_id != parent.case_id):
                continue
            position = next(index for index, cue in enumerate(cues) if cue.index == cue_id)
            def context(items):
                return [CueContext(cue_id=cue.index, text=cue.plain_text,
                                   start=cue.start_ms / 1000, end=cue.end_ms / 1000) for cue in items]
            added.append(question.span.model_copy(update={
                "case_id": case_id, "cue_ids": [cue_id], "srt_token_indices": indices,
                "srt_text": join_word_texts(tokens[index].text for index in indices),
                "context_before": context(cues[max(0, position - 2):position]),
                "context_after": context(cues[position + 1:position + 3]),
            }))
    return alignment.model_copy(update={"divergence_spans": [*alignment.divergence_spans, *added]}) if added else alignment


def reconciliation_context(questions, cues, alignment, words, regions, *, audio_sha256):
    return {
        "policy_version": MISSING_DIALOGUE_POLICY_VERSION,
        "audio_required": True, "audio_sha256": audio_sha256,
        "questions_sha256": _digest([question.record() for question in questions]),
        "source_sha256": _digest([cue.model_dump(mode="json") for cue in cues]),
        "alignment_sha256": _digest(alignment.model_dump(mode="json")),
        "words_sha256": _digest([word.model_dump(mode="json") for word in words]),
        "regions_sha256": _digest([region.model_dump(mode="json") for region in regions]),
    }


def validate_reconciliation_artifact(payload, context, questions):
    error = "Missing-dialogue evidence is stale or invalid; resume from adjudicate to ask the bounded audio question again."
    if not isinstance(payload, dict) or payload.get("context") != context:
        raise ValueError(error)
    content = {key: payload.get(key) for key in ("context", "decisions", "flags")}
    if (payload.get("receipt_sha256") != _digest(content)
            or payload.get("questions") != [question.record() for question in questions]):
        raise ValueError(error)
    try:
        decisions = [AdjudicationDecision.model_validate(item) for item in payload["decisions"]]
        flags = [QCFlag.model_validate(item) for item in payload["flags"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    case_ids = [decision.case_id for decision in decisions]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) - {question.span.case_id for question in questions}:
        raise ValueError(error)
    return decisions, flags


def _contains(sequence, part):
    return bool(part) and any(sequence[index:index + len(part)] == part for index in range(len(sequence) - len(part) + 1))


def _independent_gap_activity(question, words, regions, bursts):
    """Exclude whole anchor-owned chains when their edges cross a word anchor.

    An ASR edge can sit inside its own speech burst. That neighbour activity
    must not prevent a separate, clearly heard utterance from using its own
    chain. Join activity before testing ownership, so no piece of a connected
    utterance can be discarded or timed independently.
    """
    start, end = question.span.left_anchor_end, question.span.right_anchor_start
    if not any(region.start < start - _EPSILON or region.end > end + _EPSILON for region in bursts):
        return bursts
    if any(not isfinite(region.start) or not isfinite(region.end) or region.start >= region.end for region in regions):
        return bursts
    chains = []
    members = []
    for region in sorted(regions, key=lambda item: (item.start, item.end)):
        if chains and region.start - chains[-1].end < _MAX_SPEECH_CHAIN_GAP_SECONDS - _EPSILON:
            chains[-1] = SpeechRegion(start=chains[-1].start, end=max(chains[-1].end, region.end))
            members[-1].append(region)
        else:
            chains.append(region)
            members.append([region])
    left = words[question.left_word_indices[-1]]
    right = words[question.right_word_indices[0]]

    def owned(word):
        needed = min(MIN_OWNED_OVERLAP_SECONDS, (word.end - word.start) / 2)
        result = []
        for index, chain_members in enumerate(members):
            overlap, covered_end = 0.0, word.start
            for member in chain_members:
                start, end = max(covered_end, word.start, member.start), min(word.end, member.end)
                overlap += max(0.0, end - start)
                covered_end = max(covered_end, end)
            if overlap >= needed - _EPSILON:
                result.append(index)
        return result

    left_owned, right_owned = owned(left), owned(right)
    excluded = set()
    if len(left_owned) == 1 and left_owned[0] not in right_owned:
        index = left_owned[0]
        if chains[index].start < start < chains[index].end:
            excluded.add(index)
    if len(right_owned) == 1 and right_owned[0] not in left_owned:
        index = right_owned[0]
        if chains[index].start < end < chains[index].end:
            excluded.add(index)
    return [chain for index, chain in enumerate(chains) if index not in excluded
            and chain.start < end - _EPSILON and chain.end > start + _EPSILON]


def _resolution_reason(question, decision, sources, alignment, words, regions, flags):
    if decision is None:
        return "pending_audio_question", None
    if decision.evidence != "heard_clearly" or decision.confidence != 1:
        return "unconfirmed_audio", None
    if any(flag.cue_ids and question.cue_id in flag.cue_ids and flag.kind in {
        "adjudication_audio_unavailable", "audio_snippet_unavailable", "llm_provider_unavailable",
        "invalid_llm_response", "low_confidence_adjudication", "adjudication_review_unavailable",
    } for flag in flags):
        return "unconfirmed_audio", None
    span = question.span
    source = sources[question.cue_id]
    if (alignment.diagnostics.unresolved or alignment.cue_word_indices.get(question.cue_id)
            or is_song_caption_cue(source) or cue_has_bracketed_screen_text(source)):
        return "source_or_ownership_changed", None
    owners = _owners(alignment)
    if any(index >= len(words) or owners.get(index) != {cue_id}
           for cue_id, indices in ((span.left_anchor_cue_id, question.left_word_indices),
                                  (span.right_anchor_cue_id, question.right_word_indices)) for index in indices):
        return "anchor_ownership_changed", None
    gap_start, gap_end = span.left_anchor_end, span.right_anchor_start
    if (words[question.left_word_indices[-1]].end != gap_start
            or words[question.right_word_indices[0]].start != gap_end
            or _lexical_words_in_gap(words, gap_start, gap_end)):
        return "acoustic_ownership_changed", None
    heard, final = alphanumeric_signature(decision.heard_text or ""), alphanumeric_signature(decision.final_text)
    if heard != final or (decision.verdict == "keep_srt" and decision.final_text != span.srt_text):
        return "wording_does_not_match_hearing", None
    bursts = [region for region in regions if region.start < gap_end - _EPSILON and region.end > gap_start + _EPSILON]
    if not heard:
        if decision.final_text.strip() or (decision.heard_text or "").strip() or decision.verdict == "keep_srt":
            return "wording_does_not_match_hearing", None
        return ("audio_confirmed_omission", None) if not bursts else ("untranscribed_activity_remains", None)
    bursts = _independent_gap_activity(question, words, regions, bursts)
    read_only = list(question.read_only_source_tokens)
    target = alphanumeric_signature(span.srt_text)
    if read_only and not _contains(target, read_only) and _contains(heard, read_only):
        return "neighbor_wording_echo", None
    for cue_id, indices in ((span.left_anchor_cue_id, question.left_word_indices),
                            (span.right_anchor_cue_id, question.right_word_indices)):
        neighbor = [token for index in indices for token in alphanumeric_signature(words[index].text)]
        source_neighbor = alphanumeric_signature(sources[cue_id].plain_text)
        if (_contains(neighbor, heard) or _contains(source_neighbor, heard)
                or (_contains(heard, neighbor) and not _contains(target, neighbor))
                or (_contains(heard, source_neighbor) and not _contains(target, source_neighbor))):
            return "neighbor_wording_echo", None
    if not bursts:
        return "no_unique_speech_burst", None
    if any(not isfinite(burst.start) or not isfinite(burst.end)
           or not gap_start + _EPSILON < burst.start < burst.end < gap_end - _EPSILON
           for burst in bursts):
        return "speech_burst_crosses_anchor", None
    ordered = sorted(bursts, key=lambda burst: (burst.start, burst.end))
    chain_start, chain_end = ordered[0].start, ordered[0].end
    for burst in ordered[1:]:
        # All activity must belong to the same chain. Never select only one
        # plausible cluster, and keep an exact 200 ms pause separate despite
        # floating-point subtraction at the threshold.
        if burst.start - chain_end >= _MAX_SPEECH_CHAIN_GAP_SECONDS - _EPSILON:
            return "no_unique_speech_burst", None
        chain_end = max(chain_end, burst.end)
    return "audio_confirmed_utterance", SpeechRegion(start=chain_start, end=chain_end)


def reconcile_missing_dialogue(
    rebuilt: list[Cue], source_cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    regions: list[SpeechRegion], questions: list[MissingDialogueQuestion], decisions: list[AdjudicationDecision],
    profile: StyleProfile, *, flags: list[QCFlag],
) -> MissingDialogueResolution:
    sources = {cue.index: cue for cue in source_cues}
    by_case = {decision.case_id: decision for decision in decisions}
    replacement: dict[int, Cue | None] = {}
    spoken: dict[int, tuple[int, int]] = {}
    outcomes: list[dict[str, object]] = []
    change_flags: list[QCFlag] = []
    for question in questions:
        decision = by_case.get(question.span.case_id)
        outcome, burst = _resolution_reason(question, decision, sources, alignment, words, regions, flags)
        outcomes.append({"case_id": question.span.case_id, "cue_id": question.cue_id, "outcome": outcome,
                         "native_evidence": decision.evidence if decision else None,
                         "heard_text": decision.heard_text if decision else None})
        if outcome not in {"audio_confirmed_omission", "audio_confirmed_utterance"}:
            continue
        source = sources[question.cue_id]
        if burst is None:
            replacement[source.index] = None
        else:
            start_ms = profile.snap_floor(burst.start * 1000)
            end_ms = profile.snap_ceil(burst.end * 1000 + profile.tail_ms)
            # Padding may use only the established gap. It never trims the
            # burst or extends into the independently owned next utterance.
            end_ms = min(end_ms, profile.snap_floor(question.span.right_anchor_start * 1000))
            if end_ms < burst.end * 1000 or start_ms < question.span.left_anchor_end * 1000 or end_ms <= start_ms:
                outcomes[-1]["outcome"] = "no_safe_frame_boundary"
                continue
            lines = source.lines if alphanumeric_signature(source.plain_text) == alphanumeric_signature(decision.final_text) else [decision.final_text.strip()]
            replacement[source.index] = source.with_lines(lines).with_timing(start_ms, end_ms)
            spoken[source.index] = (floor(burst.start * 1000), ceil(burst.end * 1000))
        change_flags.append(QCFlag(
            kind="missing_dialogue_audio_reconciled", severity="info", cue_ids=[source.index],
            old_text=source.plain_text, new_text=decision.final_text, confidence=1,
            message=("Complete anchored audio confirmed this source cue was omitted; only this cue was removed."
                     if burst is None else "Native audio confirmed this whole cue; its timing uses one independent speech chain with no borrowed ASR words."),
            start=burst.start if burst else question.span.left_anchor_end,
            end=burst.end if burst else question.span.right_anchor_start,
        ))
    resolved = set(replacement)
    # Resolving Tao does not approve the neighbouring "chega" from the
    # original mixed case. Retain one actionable wording finding for that
    # residual fragment without making it editable or changing its timing.
    residual_flags: list[QCFlag] = []
    tokens = tokenize_cues(source_cues) if resolved else []
    parents = {span.case_id: span for span in alignment.divergence_spans}
    for question in questions:
        parent = parents.get(question.parent_case_id)
        if question.cue_id not in resolved or parent is None:
            continue
        for cue_id in set(parent.cue_ids) - resolved:
            if f"{MISSING_DIALOGUE_RESIDUAL_PREFIX}{parent.case_id}-cue-{cue_id}" in parents:
                # Its independent ordinary audio case already carries the
                # accepted edit or the existing uncertainty finding.
                continue
            residue = [tokens[index].text for index in parent.srt_token_indices
                       if 0 <= index < len(tokens) and tokens[index].cue_id == cue_id]
            if residue:
                residual_flags.append(QCFlag(
                    kind="missing_audio_source_cue_held", cue_ids=[cue_id], severity="error",
                    old_text=" ".join(residue), start=parent.start, end=parent.end,
                    message="This neighbouring source fragment remains unconfirmed; the separate audio question assessed only the missing cue. Its wording was retained for review.",
                ))
    attempted = {question.cue_id: outcome["outcome"] for question, outcome in zip(questions, outcomes)
                 if question.span.case_id in by_case}
    def clean(items):
        cleaned = []
        for flag in items:
            if flag.kind in _RELEASED_HOLD_KINDS and set(flag.cue_ids) & resolved:
                remaining = [cue_id for cue_id in flag.cue_ids if cue_id not in resolved]
                if remaining:
                    cleaned.append(flag.model_copy(update={"cue_ids": remaining}))
            else:
                # The original mixed span was held, but its target now had a
                # separate audio question. Do not tell the customer it was
                # never sent, or claim that clear wording establishes timing.
                target_outcomes = {attempted[cue_id] for cue_id in flag.cue_ids if cue_id in attempted}
                if flag.kind == "missing_audio_source_cue_held" and target_outcomes:
                    message = (
                        "Audio confirmed the target wording, but no unique independent speech chain established its timing; retained the source cue for review."
                        if target_outcomes <= {"no_unique_speech_burst", "speech_burst_crosses_anchor", "no_safe_frame_boundary"}
                        else "The bounded audio question did not establish both the whole cue's wording and independent acoustic ownership; retained the source cue for review."
                    )
                    cleaned.append(flag.model_copy(update={"message": message}))
                else:
                    cleaned.append(flag)
        return cleaned
    output = [replacement.get(cue.index, cue) for cue in rebuilt if replacement.get(cue.index, cue) is not None]
    adjusted = alignment if not resolved else alignment.model_copy(update={
        "diagnostics": alignment.diagnostics.model_copy(update={
            "missing_audio_cue_ids": [cue_id for cue_id in alignment.diagnostics.missing_audio_cue_ids if cue_id not in resolved],
        }),
        "unmatched_cue_ids": [cue_id for cue_id in alignment.unmatched_cue_ids if cue_id not in resolved],
        "flags": clean(alignment.flags),
    })
    return MissingDialogueResolution(output, adjusted, [*clean(flags), *change_flags, *residual_flags], resolved, spoken, outcomes)
