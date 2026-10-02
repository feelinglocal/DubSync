"""Recover a collapsed whole cue from a unique, source-bracketed ASR word.

This late pass consumes ordinary adjudication's current text and ownership.
Secondary words only identify an existing VAD chain; neither ASR stream is
rewritten. The caller must bind native hearing to the complete fresh question
and the verified source audio, including all otherwise unassigned activity.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil, floor, isfinite

from .adjudication_case_cache import flag_applies_to_case
from .adjudication_regions import is_song_caption_cue
from .asr_crosscheck import compare_word_streams
from .asr_crosscheck_config import cross_check_context
from .asr_timing import has_sufficient_speech_overlap
from .missing_dialogue_reconciliation import MissingDialogueQuestion, MissingDialogueResolution, _digest, _owners
from .models import AdjudicationDecision, AlignmentResult, Cue, CueContext, DivergenceSpan, QCFlag, SpeechRegion, Word
from .recue import timing_evidence_issue
from .style_profile import StyleProfile
from .subtitle_annotations import cue_has_bracketed_screen_text, speech_text_for_alignment
from .text_metrics import token_texts
from .tokenize import tokenize_cues


COLLAPSED_SINGLETON_TIMING_POLICY_VERSION = 1
COLLAPSED_SINGLETON_TIMING_PREFIX = "collapsed-singleton-timing-v1-"
_EPSILON = 1e-7
_PLACEHOLDER_SECONDS = .020
_MAX_WORD_SECONDS = 2.0
_MAX_QUESTION_SECONDS = 16.0
_MAX_TARGET_TOKENS = 16
_CHAIN_GAP_SECONDS = .2
_NEIGHBOR_OVERRUN_SECONDS = .3
_AUDIO_FAILURES = frozenset({
    "adjudication_audio_unavailable", "audio_snippet_unavailable", "llm_provider_unavailable",
    "invalid_llm_response", "low_confidence_adjudication", "adjudication_review_unavailable",
})


@dataclass(frozen=True)
class CollapsedSingletonTimingQuestion(MissingDialogueQuestion):
    collapsed_singleton_proof: dict[str, object] = field(default_factory=dict)
    collapsed_singleton_proof_sha256: str = ""

    def record(self) -> dict[str, object]:
        return {**super().record(), "collapsed_singleton_proof": deepcopy(self.collapsed_singleton_proof),
                "collapsed_singleton_proof_sha256": self.collapsed_singleton_proof_sha256}


def _keys(text: str) -> tuple[str, ...]:
    # Do not use the aligner's spelling/number aliases: Portuguese É != e.
    return tuple(token.casefold() for token in token_texts(text))


def _cue_keys(cue: Cue) -> tuple[str, ...]:
    return _keys(speech_text_for_alignment(cue))


def _spoken(cue: Cue) -> bool:
    return not is_song_caption_cue(cue) and not cue_has_bracketed_screen_text(cue) and bool(_cue_keys(cue))


def _valid_word(word: Word, *, reliable: bool = False) -> bool:
    return (isfinite(word.start) and isfinite(word.end) and 0 <= word.start < word.end
            and (not reliable or (_PLACEHOLDER_SECONDS + _EPSILON < word.end - word.start <= _MAX_WORD_SECONDS
                                 and (word.confidence is None or word.confidence >= .7))))


def _intersects(word: Word | SpeechRegion, start: float, end: float) -> bool:
    return word.start < end - _EPSILON and word.end > start + _EPSILON


def _region_record(region: SpeechRegion) -> dict[str, float]:
    return {"start": region.start, "end": region.end}


def _occurrences(tokens: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    return [i for i in range(len(tokens) - len(target) + 1) if tokens[i:i + len(target)] == target]


@dataclass
class _Evidence:
    sources: dict[int, Cue]
    current: dict[int, Cue]
    alignment: AlignmentResult
    words: list[Word]
    secondary: list[Word]
    regions: list[SpeechRegion]
    context: dict[str, object]
    decisions: dict[str, list[AdjudicationDecision]]
    uncertain: set[int]
    protected: set[int]
    source_tokens: list
    owners: dict[int, set[int]]
    agreement: object
    input_hashes: dict[str, str]

    def word_records(self, indices, *, secondary=False):
        words = self.secondary if secondary else self.words
        return [{"word_index": i, "word": words[i].model_dump(mode="json")} for i in indices]

    def region_ids(self, word):
        return [i for i, region in enumerate(self.regions)
                if has_sufficient_speech_overlap(word, region.start, region.end)]

    def source_token_ids(self, cue_id):
        return [token.token_index for token in self.source_tokens if token.cue_id == cue_id]

    def current_group(self, cue_id):
        cue = self.current.get(cue_id)
        indices = tuple(self.alignment.cue_word_indices.get(cue_id, ()))
        if (cue is None or not _spoken(cue) or cue_id in self.protected or not indices
                or indices != tuple(sorted(set(indices)))
                or any(i < 0 or i >= len(self.words) or self.owners.get(i) != {cue_id} for i in indices)):
            return None
        lexical = tuple(i for i in indices if _keys(self.words[i].text))
        if not lexical or lexical != tuple(i for i in range(lexical[0], lexical[-1] + 1) if _keys(self.words[i].text)):
            return None
        if tuple(key for i in lexical for key in _keys(self.words[i].text)) != _cue_keys(cue):
            return None
        return lexical

    def reliable_group(self, indices, *, secondary=False):
        words = self.secondary if secondary else self.words
        group = [words[i] for i in indices]
        return (bool(group) and all(_valid_word(word, reliable=True) for word in group)
                and len({word.speaker_id for word in group}) == 1 and bool(group[0].speaker_id)
                and all(a.end <= b.start + _EPSILON for a, b in zip(group, group[1:]))
                and (secondary or not self.uncertain.intersection(indices)))

    def anchor_authorization(self, cue_id, indices):
        source, current = self.sources[cue_id], self.current[cue_id]
        source_ids = self.source_token_ids(cue_id)
        if _cue_keys(source) == _cue_keys(current):
            matches = [m for m in self.alignment.token_matches
                       if m.cue_id == cue_id and m.asr_word_index in indices and m.score >= .8
                       and m.srt_token_index in source_ids
                       and _keys(self.source_tokens[m.srt_token_index].text)
                       and all(key in _keys(self.words[m.asr_word_index].text)
                               for key in _keys(self.source_tokens[m.srt_token_index].text))]
            if ({m.srt_token_index for m in matches} != set(source_ids)
                    or {m.asr_word_index for m in matches} != set(indices)):
                return None
            return {"kind": "exact_source_match", "matches": [m.model_dump(mode="json") for m in matches]}
        # A changed source anchor needs the actual complete ordinary case and
        # its accepted native wording, not a coincidentally matching final SRT.
        accepted = []
        for span in self.alignment.divergence_spans:
            decisions = self.decisions.get(span.case_id, [])
            if (set(span.cue_ids) != {cue_id} or span.srt_token_indices != source_ids
                    or _keys(span.srt_text) != _cue_keys(source) or len(decisions) != 1
                    or any(i < 0 or i >= len(self.words) for i in span.asr_word_indices)
                    or tuple(i for i in span.asr_word_indices if _keys(self.words[i].text)) != indices
                    or _keys(span.asr_text) != _cue_keys(current)
                    or span.start is None or span.end is None
                    or abs(span.start - self.words[indices[0]].start) > _EPSILON
                    or abs(span.end - self.words[indices[-1]].end) > _EPSILON):
                continue
            decision = decisions[0]
            if (decision.verdict == "use_audio" and decision.evidence == "heard_clearly" and decision.confidence == 1
                    and not decision.reason.startswith("Dual ASR cross-check:")
                    and _keys(decision.heard_text or "") == _keys(decision.final_text) == _cue_keys(current)):
                accepted.append({"kind": "accepted_current_source_replacement",
                                 "case": span.model_dump(mode="json"), "decision": decision.model_dump(mode="json")})
        return accepted[0] if len(accepted) == 1 else None

    def anchor(self, cue_id, *, side):
        indices = self.current_group(cue_id)
        if indices is None or not self.reliable_group(indices):
            return None
        authorization = self.anchor_authorization(cue_id, indices)
        if authorization is None:
            return None
        primary_tokens = [i for i, token in enumerate(self.agreement.primary_tokens) if token.word_index in indices]
        matched = [self.agreement.token_matches[i] for i in primary_tokens]
        if (not matched or any(i is None for i in matched)
                or matched != list(range(matched[0], matched[-1] + 1))):
            return None
        secondary = tuple(dict.fromkeys(self.agreement.secondary_tokens[i].word_index for i in matched))
        if (not self.reliable_group(secondary, secondary=True)
                or tuple(key for i in secondary for key in _keys(self.secondary[i].text)) != _cue_keys(self.current[cue_id])):
            return None
        boundary = indices[-1] if side == "left" else indices[0]
        other_boundary = secondary[-1] if side == "left" else secondary[0]
        claims = self.region_ids(self.words[boundary])
        if (len(claims) != 1 or self.region_ids(self.secondary[other_boundary]) != claims
                or any(len(self.region_ids(self.words[i])) != 1 for i in indices)
                or any(len(self.region_ids(self.secondary[i])) != 1 for i in secondary)):
            return None
        region = self.regions[claims[0]]
        excess = region.end - self.words[boundary].end if side == "left" else self.words[boundary].start - region.start
        if excess > _NEIGHBOR_OVERRUN_SECONDS + _EPSILON:
            return None
        return {"cue_id": cue_id, "primary_indices": list(indices), "secondary_indices": list(secondary),
                "source": self.sources[cue_id].model_dump(mode="json"),
                "current": self.current[cue_id].model_dump(mode="json"), "authorization": authorization,
                "primary_words": self.word_records(indices), "secondary_words": self.word_records(secondary, secondary=True),
                "boundary_region_index": claims[0], "boundary_region": _region_record(region)}

    def target_indices(self, left, right, target):
        primary_left, primary_right = left["primary_indices"][-1], right["primary_indices"][0]
        secondary_left, secondary_right = left["secondary_indices"][-1], right["secondary_indices"][0]
        if not primary_left < primary_right or not secondary_left < secondary_right:
            return None
        pairs = [(key, i) for i in range(secondary_left + 1, secondary_right) for key in _keys(self.secondary[i].text)]
        hits = _occurrences(tuple(key for key, _ in pairs), target)
        if len(hits) != 1:
            return None
        hit = hits[0]
        indices = tuple(dict.fromkeys(i for _, i in pairs[hit:hit + len(target)]))
        if (tuple(key for i in indices for key in _keys(self.secondary[i].text)) != target
                or not self.reliable_group(indices, secondary=True)):
            return None
        primary_tokens = tuple(key for i in range(primary_left + 1, primary_right) for key in _keys(self.words[i].text))
        return indices if len(_occurrences(primary_tokens, target)) == 1 else None

    def gap_activity(self, primary_index, secondary_indices, left, right, gap_start, gap_end):
        selected = []
        for index in secondary_indices:
            claims = self.region_ids(self.secondary[index])
            if len(claims) != 1:
                return None
            selected.extend(claims)
        selected = sorted(set(selected))
        if any(not gap_start + _EPSILON < self.regions[i].start < self.regions[i].end < gap_end - _EPSILON for i in selected):
            return None
        if any(i != primary_index and _keys(word.text) and any(_intersects(word, self.regions[r].start, self.regions[r].end) for r in selected)
               for i, word in enumerate(self.words)):
            return None
        if any(i not in secondary_indices and _keys(word.text) and any(_intersects(word, self.regions[r].start, self.regions[r].end) for r in selected)
               for i, word in enumerate(self.secondary)):
            return None
        known_groups, known_regions = {}, set()
        primary_range = range(left["primary_indices"][-1] + 1, right["primary_indices"][0])
        for index in primary_range:
            if index == primary_index or not _keys(self.words[index].text):
                continue
            owners = self.owners.get(index, set())
            if len(owners) != 1:
                return None
            cue_id = next(iter(owners))
            indices = self.current_group(cue_id)
            if (indices is None or not self.reliable_group(indices)
                    or any(i not in primary_range or i == primary_index for i in indices)):
                return None
            claims = [self.region_ids(self.words[i]) for i in indices]
            if any(len(c) != 1 or c[0] in selected for c in claims):
                return None
            known_regions.update(c[0] for c in claims)
            known_groups[cue_id] = {"cue": self.current[cue_id].model_dump(mode="json"),
                                   "primary_indices": list(indices), "primary_words": self.word_records(indices),
                                   "region_indices": sorted({c[0] for c in claims})}
        secondary_other = []
        for index in range(left["secondary_indices"][-1] + 1, right["secondary_indices"][0]):
            word = self.secondary[index]
            if index in secondary_indices or not _keys(word.text):
                continue
            claims = self.region_ids(word)
            if (not _valid_word(word, reliable=True) or len(claims) != 1 or claims[0] not in known_regions
                    or not gap_start < word.start < word.end < gap_end):
                return None
            secondary_other.append({"word_index": index, "word": word.model_dump(mode="json"), "region_index": claims[0]})
        anchors = {left["boundary_region_index"], right["boundary_region_index"]}
        remaining = []
        for i, region in enumerate(self.regions):
            if not _intersects(region, gap_start, gap_end) or i in anchors:
                continue
            if not gap_start + _EPSILON < region.start < region.end < gap_end - _EPSILON:
                return None
            if i not in known_regions:
                remaining.append(i)
        chains = []
        for i in remaining:
            if chains and self.regions[i].start - self.regions[chains[-1][-1]].end < _CHAIN_GAP_SECONDS - _EPSILON:
                chains[-1].append(i)
            else:
                chains.append([i])
        if selected not in chains:
            # An adjacent unexplained raw region may be part of this same
            # utterance. Do not choose just the convenient supported fragment.
            return None
        return {"target_region_indices": selected,
                "target_region": {"start": self.regions[selected[0]].start, "end": self.regions[selected[-1]].end},
                "known_primary_cues": list(known_groups.values()), "known_secondary_words": secondary_other,
                "unassigned_non_target_regions": [_region_record(self.regions[i]) for i in remaining if i not in selected]}


def _prepare(current_cues, source_cues, alignment, words, regions, secondary_words, secondary_context,
             decisions, audio_duration_seconds, uncertain, protected, resolved):
    if (not regions or not secondary_words or not isinstance(secondary_context, dict) or alignment.diagnostics.unresolved
            or not isfinite(audio_duration_seconds) or audio_duration_seconds <= 0
            or len({c.index for c in current_cues}) != len(current_cues)
            or len({c.index for c in source_cues}) != len(source_cues)
            or any(not _valid_word(w) for w in (*words, *secondary_words))
            or any(not isfinite(r.start) or not isfinite(r.end) or not 0 <= r.start < r.end <= audio_duration_seconds for r in regions)):
        return None
    config = secondary_context.get("config")
    if (not isinstance(config, dict) or (secondary_context.get("provider"), secondary_context.get("model")) not in {
            ("openrouter", "microsoft/mai-transcribe-2"), ("elevenlabs", "scribe_v2")}
            or cross_check_context(secondary_words, {"asr": config}) != secondary_context):
        return None
    ordered_regions = sorted(regions, key=lambda r: (r.start, r.end))
    if any(a.end > b.start + _EPSILON for a, b in zip(ordered_regions, ordered_regions[1:])):
        return None
    by_case = {}
    for decision in decisions:
        by_case.setdefault(decision.case_id, []).append(decision)
    hashes = {
        "source": _digest([c.model_dump(mode="json") for c in source_cues]),
        "current": _digest([c.model_dump(mode="json") for c in current_cues]),
        "alignment": _digest(alignment.model_dump(mode="json")),
        "primary_words": _digest([w.model_dump(mode="json") for w in words]),
        "secondary_words": secondary_context["words_sha256"],
        "regions": _digest([r.model_dump(mode="json") for r in ordered_regions]),
        "ordinary_decisions": _digest([d.model_dump(mode="json") for d in decisions]),
        "guards": _digest({"uncertain": sorted(uncertain), "protected": sorted(protected), "resolved": sorted(resolved)}),
    }
    return _Evidence({c.index: c for c in source_cues}, {c.index: c for c in current_cues}, alignment,
                     words, secondary_words, ordered_regions, deepcopy(secondary_context), by_case, uncertain,
                     protected, tokenize_cues(source_cues), _owners(alignment), compare_word_streams(words, secondary_words), hashes)


def build_collapsed_singleton_timing_questions(
    current_cues: list[Cue], source_cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    regions: list[SpeechRegion] | None, *, secondary_words: list[Word] | None,
    secondary_context: dict[str, object] | None, decisions: list[AdjudicationDecision], audio_duration_seconds: float,
    uncertain_word_indices: set[int] | None = None, protected_cue_ids: set[int] | None = None,
    resolved_cue_ids: set[int] | None = None,
) -> list[CollapsedSingletonTimingQuestion]:
    """Issue only complete collapsed cues with independent, bounded ownership.

    The source list supplies order, never timing. Current cues/ownership and
    ordinary decisions must describe the same already-adjudicated snapshot.
    """
    uncertain, protected, resolved = set(uncertain_word_indices or ()), set(protected_cue_ids or ()), set(resolved_cue_ids or ())
    evidence = _prepare(current_cues, source_cues, alignment, words, regions, secondary_words, secondary_context,
                        decisions, audio_duration_seconds, uncertain, protected, resolved)
    if evidence is None:
        return []
    current_tokens = tokenize_cues(current_cues)
    result = []
    for position, source in enumerate(source_cues):
        current = evidence.current.get(source.index)
        if (position == 0 or position + 1 == len(source_cues) or source.index in protected | resolved
                or current is None or not _spoken(source) or not _spoken(current) or _cue_keys(source) != _cue_keys(current)):
            continue
        target = _cue_keys(source)
        owned = tuple(alignment.cue_word_indices.get(source.index, ()))
        lexical = evidence.current_group(source.index)
        if (not target or len(target) > _MAX_TARGET_TOKENS or lexical is None or len(lexical) != 1
                or words[lexical[0]].end - words[lexical[0]].start > _PLACEHOLDER_SECONDS + _EPSILON
                or timing_evidence_issue(source, [words[lexical[0]]]) is None):
            continue
        left = evidence.anchor(source_cues[position - 1].index, side="left")
        right = evidence.anchor(source_cues[position + 1].index, side="right")
        if left is None or right is None or not left["primary_indices"][-1] < lexical[0] < right["primary_indices"][0]:
            continue
        if any(_occurrences(_cue_keys(cue), target) for cue in (evidence.current[left["cue_id"]], evidence.current[right["cue_id"]])):
            continue
        secondary_indices = evidence.target_indices(left, right, target)
        if secondary_indices is None:
            continue
        p_left, p_right = words[left["primary_indices"][-1]], words[right["primary_indices"][0]]
        s_left, s_right = secondary_words[left["secondary_indices"][-1]], secondary_words[right["secondary_indices"][0]]
        gap_start, gap_end = max(p_left.end, s_left.end), min(p_right.start, s_right.start)
        start = min(words[left["primary_indices"][0]].start, secondary_words[left["secondary_indices"][0]].start)
        end = max(words[right["primary_indices"][-1]].end, secondary_words[right["secondary_indices"][-1]].end)
        if not 0 <= start < gap_start < gap_end < end <= audio_duration_seconds or end - start > _MAX_QUESTION_SECONDS:
            continue
        if any(not gap_start < secondary_words[i].start < secondary_words[i].end < gap_end for i in secondary_indices):
            continue
        activity = evidence.gap_activity(lexical[0], secondary_indices, left, right, gap_start, gap_end)
        if activity is None:
            continue
        known = activity["known_primary_cues"]
        if any(_occurrences(_keys(item["cue"]["lines"][0]) if len(item["cue"]["lines"]) == 1
                            else _keys(" ".join(item["cue"]["lines"])), target) for item in known):
            continue
        current_target_ids = [token.token_index for token in current_tokens if token.cue_id == source.index]
        proof = {"policy_version": COLLAPSED_SINGLETON_TIMING_POLICY_VERSION, "kind": "source_bracketed_secondary_whole_cue",
                 "input_hashes": evidence.input_hashes, "secondary_context": evidence.context,
                 "source_cue_id": source.index, "source_tokens": list(target), "left_anchor": left, "right_anchor": right,
                 "source_token_indices": evidence.source_token_ids(source.index), "current_token_indices": current_target_ids,
                 "primary_target": evidence.word_records(lexical), "secondary_target": evidence.word_records(secondary_indices, secondary=True),
                 **activity}
        proof_hash = _digest(proof)
        def cue_context(cue_id, indices):
            return CueContext(cue_id=cue_id, text=evidence.current[cue_id].plain_text,
                              start=words[indices[0]].start, end=words[indices[-1]].end)
        span = DivergenceSpan(
            case_id=f"{COLLAPSED_SINGLETON_TIMING_PREFIX}cue-{source.index}-{proof_hash[:16]}",
            # Late hearing receives current cues, which include newly inserted
            # dialogue. Its editable token IDs must use that same cue order.
            cue_ids=[source.index], srt_text=speech_text_for_alignment(source), srt_token_indices=current_target_ids,
            asr_text=words[lexical[0]].text, asr_word_indices=list(lexical), start=start, end=end,
            context_before=[cue_context(left["cue_id"], left["primary_indices"]),
                            *(cue_context(item["cue"]["index"], item["primary_indices"]) for item in known)],
            context_after=[cue_context(right["cue_id"], right["primary_indices"])],
            left_anchor_cue_id=left["cue_id"], right_anchor_cue_id=right["cue_id"], left_anchor_end=gap_start, right_anchor_start=gap_end,
            left_anchor_speaker_id=p_left.speaker_id, right_anchor_speaker_id=p_right.speaker_id,
        )
        receipt_indices = sorted({*left["primary_indices"], *right["primary_indices"], *owned,
                                  *(i for item in known for i in item["primary_indices"])})
        result.append(CollapsedSingletonTimingQuestion(
            span=span, parent_case_id="", left_word_indices=tuple(left["primary_indices"]), right_word_indices=tuple(right["primary_indices"]),
            read_only_source_tokens=tuple(key for item in known for key in _keys(" ".join(item["cue"]["lines"]))),
            purpose="collapsed_singleton_timing", target_word_indices=owned, evidence_word_indices=lexical,
            word_evidence_sha256=_digest(evidence.word_records(receipt_indices)),
            collapsed_singleton_proof=deepcopy(proof), collapsed_singleton_proof_sha256=proof_hash,
        ))
    return result


def reconcile_collapsed_singleton_timing(
    current_cues: list[Cue], source_cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    regions: list[SpeechRegion], questions: list[CollapsedSingletonTimingQuestion], hearing_decisions: list[AdjudicationDecision],
    profile: StyleProfile, *, secondary_words: list[Word] | None, secondary_context: dict[str, object] | None,
    decisions: list[AdjudicationDecision], audio_duration_seconds: float, uncertain_word_indices: set[int] | None = None,
    protected_cue_ids: set[int] | None = None, resolved_cue_ids: set[int] | None = None, flags: list[QCFlag] | None = None,
) -> MissingDialogueResolution:
    """Revalidate the issued proof, then apply only a confirmed cue envelope.

    ``decisions`` are ordinary anchor decisions; ``hearing_decisions`` must be
    fresh answers bound by the caller to these questions and the current audio.
    Existing flags and all alignment/word ownership are retained for the caller.
    """
    fresh = build_collapsed_singleton_timing_questions(
        current_cues, source_cues, alignment, words, regions, secondary_words=secondary_words, secondary_context=secondary_context,
        decisions=decisions, audio_duration_seconds=audio_duration_seconds, uncertain_word_indices=uncertain_word_indices,
        protected_cue_ids=protected_cue_ids, resolved_cue_ids=resolved_cue_ids,
    )
    by_case = {question.span.case_id: question for question in fresh}
    question_counts = Counter(question.span.case_id for question in questions)
    cue_counts = Counter(question.cue_id for question in questions)
    answers = {}
    for decision in hearing_decisions:
        answers.setdefault(decision.case_id, []).append(decision)
    replacements, spoken, outcomes, added_flags = {}, {}, [], []
    current = {cue.index: cue for cue in current_cues}
    existing_flags = list(flags or ())
    for question in questions:
        decisions_for_case = answers.get(question.span.case_id, [])
        answer = decisions_for_case[0] if len(decisions_for_case) == 1 else None
        expected = by_case.get(question.span.case_id)
        reason = "audio_confirmed_utterance"
        if (not isinstance(question, CollapsedSingletonTimingQuestion) or expected is None
                or question.record() != expected.record() or question_counts[question.span.case_id] != 1 or cue_counts[question.cue_id] != 1):
            reason = "collapsed_singleton_proof_changed"
        elif len(decisions_for_case) > 1:
            reason = "ambiguous_audio_decisions"
        elif answer is None:
            reason = "pending_audio_question"
        elif (answer.evidence != "heard_clearly" or answer.confidence != 1
              or any(flag.kind in _AUDIO_FAILURES and flag_applies_to_case(flag, question.span) for flag in existing_flags)):
            reason = "unconfirmed_audio"
        elif (_keys(answer.heard_text or "") != _keys(question.span.srt_text)
              or _keys(answer.final_text) != _keys(question.span.srt_text)
              or answer.verdict == "keep_srt" and answer.final_text != question.span.srt_text):
            reason = "wording_does_not_match_hearing"
        outcome = {"case_id": question.span.case_id, "cue_id": question.cue_id, "outcome": reason,
                   "native_evidence": answer.evidence if answer else None, "heard_text": answer.heard_text if answer else None}
        outcomes.append(outcome)
        if reason != "audio_confirmed_utterance":
            continue
        region = expected.collapsed_singleton_proof["target_region"]
        start_ms = profile.snap_floor(region["start"] * 1000)
        end_ms = min(profile.snap_ceil(region["end"] * 1000 + profile.tail_ms),
                     profile.snap_floor(question.span.right_anchor_start * 1000))
        if (start_ms < question.span.left_anchor_end * 1000 or end_ms < region["end"] * 1000 - _EPSILON
                or end_ms <= start_ms):
            outcome["outcome"] = "no_safe_frame_boundary"
            continue
        replacements[question.cue_id] = current[question.cue_id].with_timing(start_ms, end_ms)
        spoken[question.cue_id] = (floor(region["start"] * 1000 + _EPSILON), ceil(region["end"] * 1000 - _EPSILON))
        added_flags.append(QCFlag(
            kind="collapsed_singleton_audio_reconciled", severity="info", cue_ids=[question.cue_id], confidence=1,
            old_text=current[question.cue_id].plain_text, new_text=current[question.cue_id].plain_text,
            start=region["start"], end=region["end"],
            message="Native audio confirmed this complete cue; unique secondary words identify its existing speech region. Primary ASR words and ownership were retained.",
        ))
    return MissingDialogueResolution([replacements.get(cue.index, cue) for cue in current_cues], alignment,
                                    [*existing_flags, *added_flags], set(replacements), spoken, outcomes)
