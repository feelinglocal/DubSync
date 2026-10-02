"""Reuse a verified whole-cue absence between accepted, independently matched anchors.

This narrowly handles a quiet raw-VAD gap hidden by the speech-chain join rule.
It does not claim that either neighbour owns an entire joined chain, infer an
absence from ASR agreement, or issue a new hearing question. The caller first
validates the saved native receipt and the source-audio identity.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
import re

from .adjudication_case_cache import flag_applies_to_case
from .adjudication_regions import is_song_caption_cue
from .asr_crosscheck import compare_word_streams
from .asr_crosscheck_config import cross_check_context
from .asr_timing import has_sufficient_speech_overlap
from .missing_dialogue_reconciliation import (
    MISSING_DIALOGUE_RECEIPT_POLICY_VERSION, MissingDialogueEvidence, MissingDialogueResolution, _digest, _owners,
)
from .models import AdjudicationDecision, AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from .subtitle_annotations import cue_has_bracketed_screen_text, speech_text_for_alignment
from .text_metrics import token_texts
from .tokenize import tokenize_cues


ACCEPTED_ANCHOR_OMISSION_POLICY_VERSION = 1
_EPSILON = 1e-7
_MAX_GAP_SECONDS = .2
_MAX_QUESTION_SECONDS = 16.0
_MAX_TARGET_TOKENS = 16
_RELEASED_HOLDS = frozenset({
    "missing_audio_source_cue_held", "missing_audio_timing_held", "missing_audio_source_cue_restored",
    "unmatched_source_cue", "dropped_line_candidate",
})
_AUDIO_FAILURES = frozenset({
    "adjudication_audio_unavailable", "audio_snippet_unavailable", "llm_provider_unavailable",
    "invalid_llm_response", "low_confidence_adjudication", "adjudication_review_unavailable",
})


def _keys(text):
    return tuple(token.casefold() for token in token_texts(text))


def _cue_keys(cue):
    return _keys(speech_text_for_alignment(cue))


def _spoken(cue):
    return not is_song_caption_cue(cue) and not cue_has_bracketed_screen_text(cue) and bool(_cue_keys(cue))


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _word_valid(word, *, reliable=False):
    return (isfinite(word.start) and isfinite(word.end) and 0 <= word.start < word.end
            and (not reliable or (.020 + _EPSILON < word.end - word.start <= 2.0
                                 and (word.confidence is None or word.confidence >= .7))))


def _intersects(item, start, end):
    return item.start < end - _EPSILON and item.end > start + _EPSILON


def _native_clear(decision):
    return (decision.verdict == "use_audio" and decision.evidence == "heard_clearly" and decision.confidence == 1
            and not decision.reason.startswith("Dual ASR cross-check:")
            and _keys(decision.heard_text or "") == _keys(decision.final_text))


def _contains(tokens, target):
    return any(tokens[i:i + len(target)] == target for i in range(len(tokens) - len(target) + 1))


@dataclass
class _Evidence:
    current: dict
    sources: dict
    current_order: list
    source_order: list
    tokens: list
    alignment: AlignmentResult
    words: list[Word]
    secondary: list[Word]
    regions: list[SpeechRegion]
    owners: dict
    agreement: object
    decisions: dict
    clips: dict
    flags: list[QCFlag]
    duration: float
    uncertain: set
    protected: set
    resolved: set

    def word_records(self, indices, *, secondary=False):
        words = self.secondary if secondary else self.words
        return [{"word_index": i, "word": words[i].model_dump(mode="json")} for i in indices]

    def failure(self, span):
        return any(f.kind in _AUDIO_FAILURES and flag_applies_to_case(f, span) for f in self.flags)

    def clip(self, case_id, start, end):
        clips = self.clips.get(case_id, [])
        if len(clips) != 1:
            return None
        clip = clips[0]
        a, b, size = clip.get("start"), clip.get("end"), clip.get("size_bytes")
        if (not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                or not isfinite(a) or not isfinite(b) or not 0 <= a < b <= self.duration
                or a > start + _EPSILON or b < end - _EPSILON
                or not _sha256(clip.get("sha256")) or clip.get("mime_type") != "audio/wav"
                or not isinstance(size, int) or isinstance(size, bool) or size < 44):
            return None
        return deepcopy(clip)

    def group(self, cue_id):
        cue = self.current.get(cue_id)
        owned = tuple(self.alignment.cue_word_indices.get(cue_id, ()))
        if (cue is None or not _spoken(cue) or not owned or owned != tuple(sorted(set(owned)))
                or any(i < 0 or i >= len(self.words) or self.owners.get(i) != {cue_id} for i in owned)):
            return None
        lexical = tuple(i for i in owned if _keys(self.words[i].text))
        if (not lexical or lexical != tuple(i for i in range(lexical[0], lexical[-1] + 1) if _keys(self.words[i].text))
                or tuple(key for i in lexical for key in _keys(self.words[i].text)) != _cue_keys(cue)
                or not self.reliable(lexical)):
            return None
        return lexical

    def reliable(self, indices, *, secondary=False):
        words = self.secondary if secondary else self.words
        group = [words[i] for i in indices]
        return (bool(group) and all(_word_valid(w, reliable=True) for w in group)
                and bool(group[0].speaker_id) and len({w.speaker_id for w in group}) == 1
                and all(a.end <= b.start + _EPSILON for a, b in zip(group, group[1:]))
                and (secondary or not self.uncertain.intersection(indices)))

    def authorization(self, cue_id, owned):
        """Reconstruct current lexical content from unchanged tokens and native edits."""
        source_ids = [t.token_index for t in self.tokens if t.cue_id == cue_id]
        if not source_ids:
            return None
        edits, edited_tokens = {}, set()
        for span in self.alignment.divergence_spans:
            answers = self.decisions.get(span.case_id, [])
            if set(span.cue_ids) != {cue_id} or not answers:
                continue
            if len(answers) != 1:
                return None
            answer = answers[0]
            if not _native_clear(answer):
                continue
            token_ids = span.srt_token_indices
            if (not token_ids or token_ids != list(range(token_ids[0], token_ids[-1] + 1))
                    or not set(token_ids) <= set(source_ids) or edited_tokens.intersection(token_ids)
                    or _keys(span.srt_text) != tuple(key for i in token_ids for key in _keys(self.tokens[i].text))
                    or span.start is None or span.end is None or not isfinite(span.start) or not isfinite(span.end)
                    or not 0 <= span.start < span.end <= self.duration or self.failure(span)):
                return None
            indices = tuple(span.asr_word_indices)
            if (indices != tuple(sorted(set(indices))) or any(i not in owned for i in indices)
                    or _keys(span.asr_text) != tuple(key for i in indices for key in _keys(self.words[i].text))
                    or _keys(answer.final_text) != _keys(span.asr_text)
                    or indices and (span.start > self.words[indices[0]].start + _EPSILON
                                    or span.end < self.words[indices[-1]].end - _EPSILON)):
                return None
            clip = self.clip(span.case_id, span.start, span.end)
            if clip is None:
                return None
            edited_tokens.update(token_ids)
            edits[token_ids[0]] = {"case": span.model_dump(mode="json"), "decision": answer.model_dump(mode="json"),
                                   "clip": clip, "word_indices": list(indices), "source_token_indices": list(token_ids)}
        output, ordered_words, unchanged, position = [], [], [], 0
        while position < len(source_ids):
            token_id = source_ids[position]
            edit = edits.get(token_id)
            if edit is not None:
                output.extend(key for i in edit["word_indices"] for key in _keys(self.words[i].text))
                ordered_words.extend(edit["word_indices"])
                position += len(edit["source_token_indices"])
                continue
            token = self.tokens[token_id]
            matches = [m for m in self.alignment.token_matches if m.cue_id == cue_id and m.srt_token_index == token_id
                       and m.asr_word_index in owned and m.score >= .8
                       and all(key in _keys(self.words[m.asr_word_index].text) for key in _keys(token.text))]
            indices = {m.asr_word_index for m in matches}
            if len(indices) != 1:
                return None
            output.extend(_keys(token.text))
            ordered_words.append(next(iter(indices)))
            unchanged.extend(m.model_dump(mode="json") for m in matches)
            position += 1
        if (tuple(output) != _cue_keys(self.current[cue_id]) or tuple(dict.fromkeys(ordered_words)) != owned
                or any(a > b for a, b in zip(ordered_words, ordered_words[1:]))):
            return None
        return {"source": self.sources[cue_id].model_dump(mode="json"), "current": self.current[cue_id].model_dump(mode="json"),
                "source_token_indices": source_ids, "unchanged_token_matches": unchanged, "native_partial_edits": list(edits.values())}

    def boundary(self, cue_id, indices, *, side):
        owned = self.group(cue_id)
        if owned is None or not indices or indices != tuple(sorted(set(indices))):
            return None
        if indices != (owned[-len(indices):] if side == "left" else owned[:len(indices)]):
            return None
        authorization = self.authorization(cue_id, owned)
        if authorization is None:
            return None
        primary_tokens = [i for i, token in enumerate(self.agreement.primary_tokens) if token.word_index in indices]
        matched = [self.agreement.token_matches[i] for i in primary_tokens]
        if (not matched or any(i is None for i in matched)
                or matched != list(range(matched[0], matched[-1] + 1))):
            return None
        secondary = tuple(dict.fromkeys(self.agreement.secondary_tokens[i].word_index for i in matched))
        if (not self.reliable(secondary, secondary=True)
                or tuple(key for i in secondary for key in _keys(self.secondary[i].text)) !=
                   tuple(key for i in indices for key in _keys(self.words[i].text))):
            return None
        primary_regions = [{j for j, r in enumerate(self.regions) if has_sufficient_speech_overlap(self.words[i], r.start, r.end)} for i in indices]
        secondary_regions = [{j for j, r in enumerate(self.regions) if has_sufficient_speech_overlap(self.secondary[i], r.start, r.end)} for i in secondary]
        if (any(len(ids) != 1 for ids in [*primary_regions, *secondary_regions])
                or set().union(*primary_regions) != set().union(*secondary_regions)):
            return None
        return {"cue_id": cue_id, "authorization": authorization, "owned_primary_word_indices": list(owned),
                "primary_word_indices": list(indices), "secondary_word_indices": list(secondary),
                "primary_words": self.word_records(indices), "secondary_words": self.word_records(secondary, secondary=True),
                "boundary_raw_region_indices": sorted(set().union(*primary_regions))}


def _prepare(current, sources, alignment, words, regions, missing, secondary, context, decisions,
             audio_hash, manifest, duration, uncertain, protected, resolved, flags):
    if (missing is None or not secondary or not isinstance(context, dict) or not isinstance(manifest, dict)
            or not isinstance(manifest.get("snippets"), list) or not _sha256(audio_hash)
            or not isfinite(duration) or duration <= 0 or alignment.diagnostics.unresolved
            or len({c.index for c in sources}) != len(sources) or len({c.index for c in current}) != len(current)
            or any(not _word_valid(w) for w in (*words, *secondary))
            or any(not isfinite(r.start) or not isfinite(r.end) or not 0 <= r.start < r.end <= duration for r in regions)):
        return None
    expected = {
        "policy_version": MISSING_DIALOGUE_RECEIPT_POLICY_VERSION, "audio_required": True, "audio_sha256": audio_hash,
        "questions_sha256": _digest([q.record() for q in missing.questions]),
        "source_sha256": _digest([c.model_dump(mode="json") for c in sources]),
        "words_sha256": _digest([w.model_dump(mode="json") for w in words]),
        "regions_sha256": _digest([r.model_dump(mode="json") for r in regions]),
    }
    if (any(missing.context.get(key) != value for key, value in expected.items())
            or not _sha256(missing.context.get("alignment_sha256"))):
        return None
    config = context.get("config")
    if (not isinstance(config, dict) or (context.get("provider"), context.get("model")) not in {
            ("openrouter", "microsoft/mai-transcribe-2"), ("elevenlabs", "scribe_v2")}):
        return None
    try:
        if cross_check_context(secondary, {"asr": config}) != context:
            return None
    except (TypeError, ValueError):
        return None
    ordered_regions = sorted(regions, key=lambda r: (r.start, r.end))
    if any(a.end > b.start + _EPSILON for a, b in zip(ordered_regions, ordered_regions[1:])):
        return None
    by_case, clips = {}, {}
    for decision in decisions:
        by_case.setdefault(decision.case_id, []).append(decision)
    for clip in manifest["snippets"]:
        if isinstance(clip, dict) and isinstance(clip.get("case_id"), str):
            clips.setdefault(clip["case_id"], []).append(clip)
    return _Evidence({c.index: c for c in current}, {c.index: c for c in sources}, [c.index for c in current],
                     [c.index for c in sources], tokenize_cues(sources), alignment, words, secondary, ordered_regions,
                     _owners(alignment), compare_word_streams(words, secondary), by_case, clips,
                     [*flags, *missing.flags], duration, uncertain, protected, resolved)


def _proof(question, answer, evidence, missing, secondary_context, audio_hash):
    span, target = question.span, question.cue_id
    if (question.purpose != "missing_dialogue" or not span.case_id.startswith("missing-dialogue-v1-")
            or span.cue_ids != [target] or target not in evidence.sources or target not in evidence.current
            or target in evidence.protected | evidence.resolved or evidence.alignment.cue_word_indices.get(target)
            or not _spoken(evidence.sources[target]) or not _spoken(evidence.current[target])
            or _cue_keys(evidence.sources[target]) != _cue_keys(evidence.current[target])
            or _keys(span.srt_text) != _cue_keys(evidence.sources[target])
            or len(_keys(span.srt_text)) > _MAX_TARGET_TOKENS or span.asr_word_indices or _keys(span.asr_text)
            or span.srt_token_indices != [t.token_index for t in evidence.tokens if t.cue_id == target]):
        return None, "source_or_ownership_changed"
    if (answer is None or not _native_clear(answer) or answer.final_text.strip() or (answer.heard_text or "").strip()
            or evidence.failure(span)):
        return None, "unconfirmed_audio"
    position = evidence.source_order.index(target)
    if position == 0 or position + 1 == len(evidence.source_order):
        return None, "unproven_current_anchors"
    left_id, right_id = evidence.source_order[position - 1], evidence.source_order[position + 1]
    if (span.left_anchor_cue_id != left_id or span.right_anchor_cue_id != right_id
            or {left_id, right_id} & (evidence.protected | evidence.resolved)
            or left_id not in evidence.current or right_id not in evidence.current):
        return None, "unproven_current_anchors"
    current_position = evidence.current_order.index(target)
    if (current_position == 0 or current_position + 1 == len(evidence.current_order)
            or evidence.current_order[current_position - 1:current_position + 2] != [left_id, target, right_id]
            or any(_contains(_cue_keys(evidence.current[i]), _keys(span.srt_text)) for i in (left_id, right_id))):
        return None, "competing_current_cue"
    left = evidence.boundary(left_id, question.left_word_indices, side="left")
    right = evidence.boundary(right_id, question.right_word_indices, side="right")
    if left is None or right is None:
        return None, "unproven_current_anchors"
    lp, rp = left["primary_word_indices"][-1], right["primary_word_indices"][0]
    ls, rs = left["secondary_word_indices"][-1], right["secondary_word_indices"][0]
    if (lp >= rp or ls >= rs or any(_keys(evidence.words[i].text) for i in range(lp + 1, rp))
            or any(_keys(evidence.secondary[i].text) for i in range(ls + 1, rs))):
        return None, "competing_word_activity"
    a, b = evidence.words[lp].end, evidence.words[rp].start
    sa, sb = evidence.secondary[ls].end, evidence.secondary[rs].start
    ca, cb = max(a, sa), min(b, sb)
    if (span.left_anchor_end is None or span.right_anchor_start is None
            or not isfinite(span.left_anchor_end) or not isfinite(span.right_anchor_start)
            or abs(span.left_anchor_end - a) > _EPSILON or abs(span.right_anchor_start - b) > _EPSILON
            or not _EPSILON < b - a < _MAX_GAP_SECONDS - _EPSILON
            or not _EPSILON < sb - sa < _MAX_GAP_SECONDS - _EPSILON or not ca + _EPSILON < cb):
        return None, "unproven_raw_gap"
    # Check the entire original primary gap too: the intersection must not
    # conceal a small unexplained pulse just outside the secondary boundary.
    if (any(_intersects(r, a, b) for r in evidence.regions)
            or any(_keys(w.text) and _intersects(w, a, b) for w in evidence.words)
            or any(i not in {*left["secondary_word_indices"], *right["secondary_word_indices"]}
                   and _keys(w.text) and _intersects(w, a, b) for i, w in enumerate(evidence.secondary))):
        return None, "raw_gap_activity_remains"
    if (span.start is None or span.end is None or not isfinite(span.start) or not isfinite(span.end)
            or not 0 <= span.start <= a < b <= span.end <= evidence.duration
            or span.end - span.start > _MAX_QUESTION_SECONDS):
        return None, "unverified_audio_receipt"
    clip = evidence.clip(span.case_id, min(span.start, evidence.secondary[left["secondary_word_indices"][0]].start),
                         max(span.end, evidence.secondary[right["secondary_word_indices"][-1]].end))
    if clip is None:
        return None, "unverified_audio_receipt"
    return {"policy_version": ACCEPTED_ANCHOR_OMISSION_POLICY_VERSION, "kind": "native_absence_in_verified_raw_gap",
            "source_cue_id": target, "source": evidence.sources[target].model_dump(mode="json"),
            "current": evidence.current[target].model_dump(mode="json"), "question": question.record(),
            "native_absence": answer.model_dump(mode="json"), "native_clip": clip,
            "audio_sha256": audio_hash, "existing_native_receipt_sha256": missing.artifact()["receipt_sha256"],
            "existing_native_context": deepcopy(missing.context), "secondary_context": deepcopy(secondary_context),
            "left_anchor": left, "right_anchor": right,
            "primary_gap": {"start": a, "end": b}, "secondary_gap": {"start": sa, "end": sb},
            "conservative_gap": {"start": ca, "end": cb}, "raw_regions_in_primary_gap": [],
            "ownership_sha256": _digest(evidence.alignment.cue_word_indices),
            "guards_sha256": _digest({"protected": sorted(evidence.protected), "resolved": sorted(evidence.resolved),
                                       "uncertain": sorted(evidence.uncertain)}),
            "does_not_assign_joined_chain_to_neighbors": True}, "audio_confirmed_omission"


def reconcile_accepted_anchor_omissions(
    current_cues: list[Cue], source_cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    regions: list[SpeechRegion], missing_dialogue: MissingDialogueEvidence | None, *,
    secondary_words: list[Word] | None, secondary_context: dict[str, object] | None,
    decisions: list[AdjudicationDecision], verified_audio_sha256: str | None,
    audio_snippet_manifest: dict[str, object] | None, audio_duration_seconds: float,
    uncertain_word_indices: set[int] | None = None, protected_cue_ids: set[int] | None = None,
    resolved_cue_ids: set[int] | None = None, flags: list[QCFlag] | None = None,
) -> MissingDialogueResolution:
    """Remove only a complete native-confirmed absence with a fresh raw-gap proof.

    ``missing_dialogue`` must already have passed the pipeline's original
    source/audio receipt validation. ``verified_audio_sha256`` and the snippet
    manifest describe that same audio. Ordinary ``decisions`` establish only
    accepted current anchor wording; no ASR or model timestamp is rewritten.
    """
    existing = list(flags or ())
    uncertain, protected, resolved = set(uncertain_word_indices or ()), set(protected_cue_ids or ()), set(resolved_cue_ids or ())
    evidence = _prepare(current_cues, source_cues, alignment, words, regions, missing_dialogue, secondary_words,
                        secondary_context, decisions, verified_audio_sha256, audio_snippet_manifest, audio_duration_seconds,
                        uncertain, protected, resolved, existing)
    removed, outcomes, added = set(), [], []
    questions = missing_dialogue.questions if missing_dialogue is not None else []
    case_counts = Counter(q.span.case_id for q in questions)
    cue_counts = Counter(q.cue_id for q in questions)
    answers = {}
    for answer in missing_dialogue.decisions if missing_dialogue is not None else []:
        answers.setdefault(answer.case_id, []).append(answer)
    for question in questions:
        if question.purpose != "missing_dialogue":
            continue
        matching = answers.get(question.span.case_id, [])
        answer = matching[0] if len(matching) == 1 else None
        proof, reason = None, "unverified_audio_receipt"
        if evidence is not None:
            if case_counts[question.span.case_id] == 1 and cue_counts[question.cue_id] == 1:
                proof, reason = _proof(question, answer, evidence, missing_dialogue, secondary_context, verified_audio_sha256)
            else:
                reason = "ambiguous_audio_question"
        outcome = {"case_id": question.span.case_id, "cue_id": question.cue_id, "outcome": reason,
                   "native_evidence": answer.evidence if answer else None, "heard_text": answer.heard_text if answer else None}
        outcomes.append(outcome)
        if proof is None:
            continue
        outcome.update({"accepted_anchor_omission_proof": proof, "accepted_anchor_omission_proof_sha256": _digest(proof)})
        removed.add(question.cue_id)
        added.append(QCFlag(kind="accepted_anchor_omission_reconciled", severity="info", cue_ids=[question.cue_id], confidence=1,
            old_text=evidence.current[question.cue_id].plain_text, new_text="", start=proof["conservative_gap"]["start"],
            end=proof["conservative_gap"]["end"], message="The existing complete native hearing confirmed this cue was absent. Accepted current anchors and independent ASR correspondence bound a gap with no raw speech activity; words and ownership were retained."))
    def clean(items):
        result = []
        for flag in items:
            if flag.kind in _RELEASED_HOLDS and removed.intersection(flag.cue_ids):
                remaining = [i for i in flag.cue_ids if i not in removed]
                if remaining:
                    result.append(flag.model_copy(update={"cue_ids": remaining}))
            else:
                result.append(flag)
        return result
    adjusted = alignment if not removed else alignment.model_copy(update={
        "diagnostics": alignment.diagnostics.model_copy(update={
            "missing_audio_cue_ids": [i for i in alignment.diagnostics.missing_audio_cue_ids if i not in removed]}),
        "unmatched_cue_ids": [i for i in alignment.unmatched_cue_ids if i not in removed], "flags": clean(alignment.flags),
    })
    return MissingDialogueResolution([c for c in current_cues if c.index not in removed], adjusted,
                                     [*clean(existing), *added], removed, {}, outcomes)
