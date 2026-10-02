"""Bound a spoken source cue and its untranscribed laugh for a shared display."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, floor, isfinite
import re
import unicodedata

from .adjudication_case_cache import flag_applies_to_case
from .adjudication_regions import is_song_caption_cue
from .asr_timing import has_sufficient_speech_overlap
from .missing_dialogue_reconciliation import (
    MissingDialogueQuestion, MissingDialogueResolution, _anchor_indices, _digest, _owners, _sentence_punctuation_indices,
)
from .models import AlignmentResult, Cue, CueContext, DivergenceSpan, QCFlag, SourcePairEvidence, SpeechRegion, Word
from .recue import timing_evidence_issue
from .speaker_evidence import speakers_known_different
from .subtitle_annotations import cue_has_bracketed_screen_text
from .text_metrics import display_width, join_word_texts
from .tokenize import alphanumeric_signature, tokenize_cues


SOURCE_PAIR_TIMING_POLICY_VERSION = 2
SOURCE_PAIR_TIMING_PREFIX = "source-pair-timing-v2-"
_EPSILON = 1e-7
_MAX_CHAIN_GAP_SECONDS = .2
_MAX_LAUGH_TAIL_SECONDS = 1.5
_MAX_QUESTION_SECONDS = 16
_MARKUP = re.compile(r"</?[^>\n]+>|{\\[^}\n]+}")
_LAUGH = re.compile(r"(?:(?:ha){2,6}|(?:he){2,6}|(?:hi){2,6}|(?:ho){2,6}|(?:ja){2,6}|[はハ]{2,6}|[ふフ]{2,6}|[へヘ]{2,6}|[ほホ]{2,6}|哈{2,6}|하{2,6})")
_RELEASED_HOLD_KINDS = frozenset({
    "missing_audio_source_cue_held", "missing_audio_timing_held", "missing_audio_source_cue_restored",
    "unmatched_source_cue", "unmatched_cue", "dropped_line_candidate", "timing_evidence_held",
})
_AUDIO_FAILURE_KINDS = frozenset({
    "adjudication_audio_unavailable", "audio_snippet_unavailable", "llm_provider_unavailable",
    "invalid_llm_response", "low_confidence_adjudication", "adjudication_review_unavailable",
})


@dataclass(frozen=True)
class SourcePairTimingQuestion(MissingDialogueQuestion):
    original_pair_texts: tuple[str, str] = ("", "")
    accepted_pair_texts: tuple[str, str] = ("", "")
    anchor_speaker_id: str = ""
    utterance_start_seconds: float = 0.0
    utterance_end_seconds: float = 0.0
    audio_duration_seconds: float = 0.0
    pair_evidence_sha256: str = ""

    def record(self):
        return {**super().record(), "source_pair": {
            "policy_version": SOURCE_PAIR_TIMING_POLICY_VERSION,
            "original_pair_texts": list(self.original_pair_texts),
            "accepted_pair_texts": list(self.accepted_pair_texts),
            "anchor_speaker_id": self.anchor_speaker_id,
            "utterance_start_seconds": self.utterance_start_seconds,
            "utterance_end_seconds": self.utterance_end_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
            "pair_evidence_sha256": self.pair_evidence_sha256,
        }}


def _plain_dialogue(cue):
    return (len(cue.lines) == 1 and bool(cue.plain_text) and not is_song_caption_cue(cue)
            and not cue_has_bracketed_screen_text(cue) and not _MARKUP.search(cue.text)
            and not re.match(r"^\s*[-–—]\s", cue.text))


def _short_laugh(cue):
    value = unicodedata.normalize("NFKC", "".join(alphanumeric_signature(cue.plain_text))).casefold()
    return _plain_dialogue(cue) and _LAUGH.fullmatch(value) is not None


def _reliable_words(cue, alignment, words, owners, uncertain):
    indices = tuple(alignment.cue_word_indices.get(cue.index, ()))
    if not indices or indices != tuple(sorted(set(indices))) or any(
        i < 0 or i >= len(words) or i in uncertain or owners.get(i) != {cue.index}
        or not isfinite(words[i].start) or not isfinite(words[i].end)
        or not 0 <= words[i].start < words[i].end or words[i].end - words[i].start > 2
        or words[i].confidence is not None and words[i].confidence < .7
        for i in indices
    ):
        return ()
    lexical = tuple(i for i in indices if alphanumeric_signature(words[i].text))
    if (not lexical or len({words[i].speaker_id for i in lexical}) != 1 or not words[lexical[0]].speaker_id
            or any(words[a].end > words[b].start + _EPSILON or words[b].start - words[a].end > .3
                   for a, b in zip(lexical, lexical[1:]))
            or timing_evidence_issue(cue, [words[i] for i in lexical]) is not None):
        return ()
    return lexical


def _pair_envelope(first, right, words, regions):
    """Keep all raw activity after the first word, including its untimed laugh.

    The containing region can begin in earlier source cues. Its beginning is
    never the pair's onset. Activity crossing the next actor's word anchor
    remains ambiguous; a possible laugh in its preroll cannot be discarded.
    """
    start, right_start = words[first[0]].start, words[right[0]].start
    members = [region for region in regions if region.end > start + _EPSILON and region.start < right_start - _EPSILON]
    if not members or not members[0].start <= start < members[0].end:
        return None
    end = start
    for position, region in enumerate(members):
        if region.end >= right_start - _EPSILON:
            return None
        if position and region.start - end >= _MAX_CHAIN_GAP_SECONDS - _EPSILON:
            return None
        end = max(end, region.end)
    if (not 0 < end - words[first[-1]].end <= _MAX_LAUGH_TAIL_SECONDS
            or end >= right_start - _EPSILON
            or any(not any(has_sufficient_speech_overlap(words[i], region.start, region.end) for region in members)
                   for i in first)
            or any(i not in first and alphanumeric_signature(word.text)
                   and word.start < end - _EPSILON and word.end > start + _EPSILON
                   for i, word in enumerate(words))):
        return None
    return start, end


def _pair_evidence(alignment, words, regions, parent):
    # Whole-stream receipts also catch a newly inserted foreign word or owner.
    return _digest({"words": [word.model_dump(mode="json") for word in words],
                    "regions": [region.model_dump(mode="json") for region in regions],
                    "ownership": alignment.cue_word_indices, "parent": parent.model_dump(mode="json")})


def _confirmed_voice_relation(question, decision):
    evidence = decision.source_pair_evidence
    if (not isinstance(evidence, SourcePairEvidence)
            or evidence.sequence != "first_then_second" or evidence.voice_relation not in {"same", "different"}
            or alphanumeric_signature(evidence.first_text) != alphanumeric_signature(question.accepted_pair_texts[0])
            or alphanumeric_signature(evidence.second_text) != alphanumeric_signature(question.accepted_pair_texts[1])
            or evidence.intervening_speech is not False or evidence.candidate_complete is not True
            or evidence.candidate_start_clipped is not False or evidence.candidate_end_clipped is not False
            or evidence.laugh_outside_candidate is not False
            or evidence.candidate_audio_id != question.span.case_id + "-candidate"):
        return None
    return evidence.voice_relation


def build_source_pair_timing_questions(
    current: list[Cue], source_cues: list[Cue], alignment: AlignmentResult, words: list[Word],
    regions: list[SpeechRegion] | None, *, audio_duration_seconds: float,
    uncertain_word_indices: set[int] | None = None, protected_cue_ids: set[int] | None = None,
    resolved_cue_ids: set[int] | None = None,
) -> list[SourcePairTimingQuestion]:
    """Ask after ordinary adjudication, retaining its accepted first-cue text.

    Existing missing holds may protect the unowned laugh; they do not veto
    this fresh complete question. Protected first/outer anchors and a laugh
    already independently resolved by the ordinary path are ineligible.
    """
    if (not regions or alignment.diagnostics.unresolved or not isfinite(audio_duration_seconds)
            or audio_duration_seconds <= 0
            or any(not isfinite(r.start) or not isfinite(r.end) or not 0 <= r.start < r.end <= audio_duration_seconds for r in regions)
            or any(a.end > b.start + _EPSILON for a, b in zip(regions, regions[1:]))
            or any(alphanumeric_signature(w.text) and (not isfinite(w.start) or not isfinite(w.end)) for w in words)):
        return []
    by_id, owners = {cue.index: cue for cue in current}, _owners(alignment)
    uncertain, protected, resolved = uncertain_word_indices or set(), protected_cue_ids or set(), resolved_cue_ids or set()
    tokens, accepted_tokens = tokenize_cues(source_cues), tokenize_cues(current)
    result = []
    for position in range(1, len(source_cues) - 2):
        previous, original_first, original_laugh, following = source_cues[position - 1:position + 3]
        first, laugh = by_id.get(original_first.index), by_id.get(original_laugh.index)
        if (first is None or laugh is None or original_laugh.index in resolved
                or {previous.index, original_first.index, following.index} & protected
                or not all(_plain_dialogue(cue) for cue in (previous, original_first, original_laugh, following, first, laugh))
                or not _short_laugh(original_laugh)
                or alphanumeric_signature(laugh.plain_text) != alphanumeric_signature(original_laugh.plain_text)):
            continue
        first_words = _reliable_words(first, alignment, words, owners, uncertain)
        left = _anchor_indices(previous.index, alignment, words, tokens, owners, uncertain)
        right = _anchor_indices(following.index, alignment, words, tokens, owners, uncertain)
        if (not first_words or not left or not right
                or left != _reliable_words(previous, alignment, words, owners, uncertain)
                or right != _reliable_words(following, alignment, words, owners, uncertain)
                or alphanumeric_signature(first.plain_text) != [token for i in first_words for token in alphanumeric_signature(words[i].text)]
                or words[left[-1]].end > words[first_words[0]].start + _EPSILON
                or not speakers_known_different(words[first_words[0]].speaker_id, words[right[0]].speaker_id)):
            continue
        laugh_owned = tuple(alignment.cue_word_indices.get(laugh.index, ()))
        target_tokens = {token.token_index for token in tokens if token.cue_id == laugh.index}
        parents = [span for span in alignment.divergence_spans if laugh.index in span.cue_ids
                   and target_tokens <= set(span.srt_token_indices)]
        if len(parents) != 1:
            continue
        parent = parents[0]
        punctuation = tuple(sorted(set((*laugh_owned, *parent.asr_word_indices))))
        if (not target_tokens or set(parent.cue_ids) - {first.index, laugh.index} or alphanumeric_signature(parent.asr_text)
                or punctuation and (not _sentence_punctuation_indices(punctuation, words)
                                    or any(owners.get(i, set()) - {laugh.index} for i in punctuation))):
            continue
        envelope = _pair_envelope(first_words, right, words, regions)
        if envelope is None:
            continue
        start, end = envelope
        window_start, window_end = words[left[0]].start, words[right[-1]].end
        if not 0 <= window_start < start < end < window_end <= audio_duration_seconds or window_end - window_start > _MAX_QUESTION_SECONDS:
            continue
        source_text = first.text + "\n" + laugh.text
        span = DivergenceSpan(
            case_id=f"{SOURCE_PAIR_TIMING_PREFIX}{first.index}-{laugh.index}", cue_ids=[first.index, laugh.index],
            srt_text=source_text, asr_text=join_word_texts(words[i].text for i in first_words),
            srt_token_indices=[t.token_index for t in accepted_tokens if t.cue_id in {first.index, laugh.index}],
            asr_word_indices=list(first_words), start=window_start, end=window_end,
            speaker_ids=[words[first_words[0]].speaker_id],
            left_anchor_cue_id=previous.index, right_anchor_cue_id=following.index,
            left_anchor_end=words[left[-1]].end, right_anchor_start=words[right[0]].start,
            left_anchor_speaker_id=words[left[-1]].speaker_id, right_anchor_speaker_id=words[right[0]].speaker_id,
            context_before=[CueContext(cue_id=previous.index, text=previous.text, start=words[left[0]].start, end=words[left[-1]].end)],
            context_after=[CueContext(cue_id=following.index, text=following.text, start=words[right[0]].start, end=words[right[-1]].end)],
        )
        result.append(SourcePairTimingQuestion(
            span=span, parent_case_id=parent.case_id, left_word_indices=left, right_word_indices=right,
            read_only_source_tokens=tuple(alphanumeric_signature(previous.plain_text + " " + following.plain_text)),
            purpose="source_pair_timing", target_word_indices=tuple(alignment.cue_word_indices[first.index]),
            evidence_word_indices=punctuation, original_pair_texts=(original_first.text, original_laugh.text),
            accepted_pair_texts=(first.text, laugh.text), anchor_speaker_id=words[first_words[0]].speaker_id,
            utterance_start_seconds=start, utterance_end_seconds=end, audio_duration_seconds=audio_duration_seconds,
            pair_evidence_sha256=_pair_evidence(alignment, words, regions, parent),
        ))
    return result


def reconcile_source_pair_timing(
    current, source_cues, alignment, words, regions, questions, decisions, profile, *, flags,
    uncertain_word_indices=None, protected_cue_ids=None, resolved_cue_ids=None,
) -> MissingDialogueResolution:
    """Merge only a freshly confirmed whole pair; primary evidence is immutable."""
    by_case = {decision.case_id: decision for decision in decisions}
    decision_counts = Counter(decision.case_id for decision in decisions)
    question_counts = Counter(question.span.case_id for question in questions)
    by_id = {cue.index: cue for cue in current}
    replacements, removed, spoken, outcomes, change_flags = {}, set(), {}, [], []
    for question in questions:
        decision = by_case.get(question.span.case_id)
        ids = question.span.cue_ids
        outcome = "pending_audio_question" if decision is None else "unconfirmed_audio"
        confirmed_voice = None
        target = alphanumeric_signature(question.span.srt_text)
        fresh = build_source_pair_timing_questions(
            current, source_cues, alignment, words, regions, audio_duration_seconds=question.audio_duration_seconds,
            uncertain_word_indices=uncertain_word_indices, protected_cue_ids=protected_cue_ids,
            resolved_cue_ids=resolved_cue_ids,
        )
        current_proof = next((item for item in fresh if item.span.case_id == question.span.case_id), None)
        if question_counts[question.span.case_id] != 1 or decision_counts[question.span.case_id] > 1:
            outcome = "duplicate_audio_evidence"
        elif current_proof is None or current_proof.record() != question.record():
            outcome = "source_or_acoustic_evidence_changed"
        elif decision is not None and decision.evidence == "heard_clearly" and decision.confidence == 1 and not any(
            flag.kind in _AUDIO_FAILURE_KINDS and flag_applies_to_case(flag, question.span) for flag in flags
        ):
            voice = _confirmed_voice_relation(question, decision)
            if voice is None:
                outcome = "pair_candidate_evidence_unconfirmed"
            elif ((voice == "same" and decision.speaker != question.anchor_speaker_id)
                    or (voice == "different" and decision.speaker is not None)
                    or alphanumeric_signature(decision.heard_text or "") != target
                    or alphanumeric_signature(decision.final_text) != target
                    or decision.verdict == "keep_srt" and decision.final_text != question.span.srt_text):
                outcome = "wording_or_voice_unconfirmed"
            else:
                start, end = question.utterance_start_seconds, question.utterance_end_seconds
                start_ms, end_ms = profile.snap_floor(start * 1000), profile.snap_ceil(end * 1000 + profile.tail_ms)
                end_ms = min(end_ms, profile.snap_floor(question.span.right_anchor_start * 1000))
                lines = [("- " if voice == "different" else "") + text for text in question.accepted_pair_texts]
                if (profile.max_lines_per_cue < 2 or any(display_width(line) > profile.max_chars_per_line for line in lines)
                        or end_ms < end * 1000 or start_ms > start * 1000 or end_ms <= start_ms):
                    outcome = "no_safe_display_union"
                else:
                    replacements[ids[0]] = by_id[ids[0]].with_lines(lines).with_timing(start_ms, end_ms).model_copy(
                        update={"speaker_id": question.anchor_speaker_id if voice == "same" else None,
                                "character": by_id[ids[0]].character if voice == "same" else None})
                    removed.add(ids[1])
                    spoken[ids[0]] = (floor(start * 1000), ceil(end * 1000 - _EPSILON))
                    confirmed_voice = voice
                    outcome = "audio_confirmed_source_exchange" if voice == "different" else "audio_confirmed_source_pair"
                    change_flags.append(QCFlag(
                        kind="source_exchange_audio_reconciled" if voice == "different" else "source_pair_audio_reconciled",
                        cue_ids=list(ids), severity="info", confidence=1,
                        old_text=question.span.srt_text, new_text=replacements[ids[0]].text, start=start, end=end,
                        message=("Native audio confirmed a complete sequential exchange in two voices; separate dialogue lines share its acoustic envelope without estimating an internal boundary."
                                 if voice == "different" else
                                 "Native audio confirmed both source parts in one voice; the spoken cue and its laugh share one complete acoustic envelope without estimating an internal boundary."),
                    ))
        outcomes.append({"case_id": question.span.case_id, "cue_id": ids[0], "source_cue_ids": list(ids),
                         "merged_cue_id": ids[0] if confirmed_voice else None,
                         "outcome": outcome, "native_evidence": decision.evidence if decision else None,
                         "heard_text": decision.heard_text if decision else None, "speaker": decision.speaker if decision else None,
                         "voice_relation": confirmed_voice,
                         "line_provenance": ([{"line": 1, "source_cue_id": ids[0], "voice": "first"},
                                              {"line": 2, "source_cue_id": ids[1],
                                               "voice": "second" if confirmed_voice == "different" else "first"}]
                                             if confirmed_voice else [])})
    resolved = set(replacements) | removed

    def clean(items):
        result = []
        for flag in items:
            if flag.kind in _RELEASED_HOLD_KINDS and resolved.intersection(flag.cue_ids):
                remaining = [cue_id for cue_id in flag.cue_ids if cue_id not in resolved]
                if remaining:
                    result.append(flag.model_copy(update={"cue_ids": remaining}))
            else:
                result.append(flag)
        return result

    adjusted = alignment if not resolved else alignment.model_copy(update={
        "diagnostics": alignment.diagnostics.model_copy(update={
            "missing_audio_cue_ids": [cue_id for cue_id in alignment.diagnostics.missing_audio_cue_ids if cue_id not in resolved]}),
        "unmatched_cue_ids": [cue_id for cue_id in alignment.unmatched_cue_ids if cue_id not in resolved],
        "flags": clean(alignment.flags),
    })
    output = [replacements.get(cue.index, cue) for cue in current if cue.index not in removed]
    return MissingDialogueResolution(output, adjusted, [*clean(flags), *change_flags], resolved, spoken, outcomes)
