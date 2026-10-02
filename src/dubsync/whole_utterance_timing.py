"""Positive whole-cue hearing can recover a cue envelope, never word times.

Missing-dialogue omission rules remain separate. This path requires a fresh
complete source hearing and one independent group of all remaining activity
between trustworthy neighbouring anchors. ASR evidence stays unchanged.
"""
from __future__ import annotations

from math import isfinite

from .adjudication_regions import is_song_caption_cue
from .asr_timing import has_sufficient_speech_overlap
from .missing_dialogue_reconciliation import MissingDialogueQuestion, _anchor_indices, _contains, _digest, _owners
from .models import AlignmentResult, Cue, CueContext, DivergenceSpan, SpeechRegion, Word
from .recue import _trimmed_lexical_timing_issue, select_cue_word_window, timing_evidence_issue
from .secondary_acoustic_ownership import (SecondaryAcousticEvidence, secondary_target, supported_neighbor_boundary,
                                          valid_secondary_proof)
from .subtitle_annotations import cue_has_bracketed_screen_text, speech_text_for_alignment
from .text_metrics import join_word_texts
from .tokenize import alphanumeric_signature, tokenize_cues


WHOLE_UTTERANCE_TIMING_POLICY_VERSION = 2
_EPSILON = 1e-7
_CHAIN_GAP_SECONDS = .2
_PLACEHOLDER_SECONDS = .020
_MAX_QUESTION_SECONDS = 16.0
_MAX_TARGET_TOKENS = 16


def _valid_indices(indices, words, *, max_word_duration=2.0):
    return (tuple(indices) == tuple(sorted(set(indices))) and all(
        0 <= index < len(words) and isfinite(words[index].start) and isfinite(words[index].end)
        and 0 <= words[index].start < words[index].end
        and words[index].end - words[index].start <= max_word_duration
        and (words[index].confidence is None or words[index].confidence >= .7)
        for index in indices
    ))


def _word_receipt(question, words):
    indices = sorted(set((*question.left_word_indices, *question.right_word_indices,
                          *question.target_word_indices, *question.evidence_word_indices)))
    return _digest([{"word_index": index, "word": words[index].model_dump(mode="json")} for index in indices])


def _intersects(word, start, end):
    return word.start < end - _EPSILON and word.end > start + _EPSILON


def _unanchored_neighbor_echo(question, sources, words, burst):
    target = alphanumeric_signature(question.span.srt_text)
    echo = any(
        _contains([token for index in indices for token in alphanumeric_signature(words[index].text)], target)
        or _contains(alphanumeric_signature(sources[cue_id].plain_text), target)
        for cue_id, indices in ((question.span.left_anchor_cue_id, question.left_word_indices),
                                (question.span.right_anchor_cue_id, question.right_word_indices))
    )
    if not echo:
        return False
    # A separate primary lexical anchor can distinguish a repeated utterance
    # ("three minutes" after "three minutes, then..."). Untranscribed VAD
    # alone cannot distinguish that repetition from a model quoting context.
    return not any(
        index not in question.discarded_word_indices
        and burst.start <= words[index].start < words[index].end <= burst.end
        and any(token in target for token in alphanumeric_signature(words[index].text))
        for index in question.evidence_word_indices
    )


def _positive_gap_burst(question, words, regions):
    """Remove only boundary-owned raw regions, then account for all activity.

    Joining every region first would fuse a separate short reaction with its
    neighbour merely because their pause is under 200 ms. A raw region itself
    is indivisible: target evidence inside an anchor's region prevents release.
    """
    start, end = question.span.left_anchor_end, question.span.right_anchor_start
    left, right = words[question.left_word_indices[-1]], words[question.right_word_indices[0]]
    claims = [[i for i, region in enumerate(regions) if has_sufficient_speech_overlap(word, region.start, region.end)]
              for word in (left, right)]
    if any(len(indices) != 1 for indices in claims) or claims[0] == claims[1]:
        return None
    evidence, discarded = set(question.evidence_word_indices), set(question.discarded_word_indices)
    target_proof = secondary_target(question)
    mapped = {item["primary_word_index"] for item in target_proof["primary_mappings"]} if target_proof else set()
    remaining, excluded = [], []
    lexical = [(i, word) for i, word in enumerate(words) if alphanumeric_signature(word.text)]
    if any(not isfinite(word.start) or not isfinite(word.end) or not 0 <= word.start < word.end for _, word in lexical):
        return None
    for index, region in enumerate(regions):
        if region.start >= end - _EPSILON or region.end <= start + _EPSILON:
            continue
        side = 0 if index == claims[0][0] else 1 if index == claims[1][0] else None
        if side is not None:
            anchor_indices = set(question.left_word_indices if side == 0 else question.right_word_indices)
            excess = region.end - left.end if side == 0 else right.start - region.start
            if (excess > question.neighbor_boundary_allowance_seconds + _EPSILON
                    and not supported_neighbor_boundary(question, "left" if side == 0 else "right", region)):
                return None
            # Discarded placeholders are allowed only when the original
            # whole-cue timing guard already rejected that separated group.
            if any(i in evidence - discarded - mapped and _intersects(word, region.start, region.end) for i, word in lexical):
                return None
            if any(i not in anchor_indices | discarded | mapped and _intersects(word, max(start, region.start), min(end, region.end))
                   for i, word in lexical):
                return None
            excluded.append(region)
            continue
        if not start + _EPSILON < region.start < region.end < end - _EPSILON:
            return None
        if any(i not in evidence and _intersects(word, region.start, region.end) for i, word in lexical):
            return None
        remaining.append(region)
    if not remaining:
        return None
    remaining.sort(key=lambda region: (region.start, region.end))
    burst_start, burst_end = remaining[0].start, remaining[0].end
    for region in remaining[1:]:
        if region.start - burst_end >= _CHAIN_GAP_SECONDS - _EPSILON:
            return None
        burst_end = max(burst_end, region.end)
    if target_proof and target_proof["region"] != {"start": burst_start, "end": burst_end}:
        return None
    for index in evidence:
        word = words[index]
        if index in mapped:
            continue  # The unique secondary token proves this existing region, never a replacement word time.
        if index in discarded:
            if not any(region.start <= word.start and word.end <= region.end and
                       has_sufficient_speech_overlap(word, region.start, region.end) for region in excluded):
                return None
        elif not (burst_start <= word.start < word.end <= burst_end and
                  any(has_sufficient_speech_overlap(word, region.start, region.end) for region in remaining)):
            return None
    return SpeechRegion(start=burst_start, end=burst_end)


def build_whole_utterance_timing_questions(
    cues: list[Cue], alignment: AlignmentResult, words: list[Word], regions: list[SpeechRegion] | None, *,
    audio_duration_seconds: float, uncertain_word_indices: set[int] | None = None,
    max_word_duration: float = 2.0, max_intra_cue_gap: float = 1.5,
    max_neighbor_boundary_overrun: float = .3,
    secondary_words: list[Word] | None = None, secondary_context: dict[str, object] | None = None,
) -> list[MissingDialogueQuestion]:
    if (not regions or alignment.diagnostics.unresolved or not isfinite(audio_duration_seconds)
            or audio_duration_seconds <= 0 or not isfinite(max_neighbor_boundary_overrun) or max_neighbor_boundary_overrun < 0
            or any(not isfinite(r.start) or not isfinite(r.end)
            or not 0 <= r.start < r.end <= audio_duration_seconds for r in regions)):
        return []
    tokens, owners = tokenize_cues(cues), _owners(alignment)
    uncertain = uncertain_word_indices or set()
    secondary = SecondaryAcousticEvidence.prepare(cues, words, secondary_words, secondary_context)
    missing = set(alignment.diagnostics.missing_audio_cue_ids)
    result = []
    for position, cue in enumerate(cues):
        if position == 0 or position + 1 == len(cues) or is_song_caption_cue(cue) or cue_has_bracketed_screen_text(cue):
            continue
        own_tokens = [token.token_index for token in tokens if token.cue_id == cue.index]
        if not own_tokens or len(own_tokens) > _MAX_TARGET_TOKENS:
            continue
        owned = tuple(alignment.cue_word_indices.get(cue.index, ()))
        if not _valid_indices(owned, words, max_word_duration=max_word_duration) or any(owners.get(i) != {cue.index} for i in owned):
            continue
        lexical = tuple(i for i in owned if alphanumeric_signature(words[i].text))
        target_proof = secondary.whole_target(cue, lexical, uncertain, regions) if secondary else None
        parent, discarded, read_only = None, (), ()
        if target_proof:
            # A spelling divergence may leave phonetic interior words unowned.
            # They remain read-only evidence of this complete primary phrase.
            source_parents = [span for span in alignment.divergence_spans if cue.index in span.cue_ids]
            local = {i for span in source_parents if set(span.cue_ids) == {cue.index} for i in span.asr_word_indices}
            evidence = tuple(i for i in range(lexical[0], lexical[-1] + 1) if alphanumeric_signature(words[i].text))
            if (not _valid_indices(evidence, words, max_word_duration=max_word_duration)
                    or any(i not in lexical and i not in local or owners.get(i, set()) - {cue.index} for i in evidence)
                    or len({words[i].speaker_id for i in evidence}) != 1 or not words[evidence[0]].speaker_id):
                continue
        elif lexical:
            if alphanumeric_signature(speech_text_for_alignment(cue)) != alphanumeric_signature(join_word_texts(words[i].text for i in lexical)):
                continue
            selected, trimmed = select_cue_word_window(cue, [words[i] for i in lexical], max_word_duration=max_word_duration,
                                                       max_intra_cue_gap=max_intra_cue_gap)
            issue = timing_evidence_issue(cue, selected)
            if issue is None and trimmed:
                issue = _trimmed_lexical_timing_issue(cue, [words[i] for i in lexical], selected, max_word_duration)
            if issue is None:
                continue
            selected_ids = {id(word) for word in selected}
            discarded = tuple(i for i in lexical if id(words[i]) not in selected_ids)
            if any(words[i].end - words[i].start > _PLACEHOLDER_SECONDS + _EPSILON for i in discarded):
                continue
            evidence = lexical
        else:
            parents = [span for span in alignment.divergence_spans
                       if cue.index in span.cue_ids or set(own_tokens).intersection(span.srt_token_indices)]
            if len(parents) != 1:
                continue
            parent = parents[0]
            source = parent.srt_token_indices
            if (not set(own_tokens) <= set(source) or source != sorted(set(source)) or not source
                    or source[0] < 0 or source[-1] >= len(tokens)
                    or [tokens[i].normalized for i in source] != alphanumeric_signature(parent.srt_text)
                    or any(tokens[i].cue_id not in parent.cue_ids for i in source)
                    or not set(parent.cue_ids) <= {cues[position - 1].index, cue.index, cues[position + 1].index}
                    or not _valid_indices(parent.asr_word_indices, words, max_word_duration=max_word_duration)):
                continue
            evidence = tuple(i for i in parent.asr_word_indices if alphanumeric_signature(words[i].text))
            if evidence and (set(parent.cue_ids) != {cue.index} or any(owners.get(i, set()) - {cue.index} for i in evidence)):
                continue
            read_only = tuple(tokens[i].normalized for i in source if i not in own_tokens)
        mapped = {item["primary_word_index"] for item in target_proof["primary_mappings"]} if target_proof else set()
        if any(i in uncertain for i in evidence if i not in discarded and i not in mapped):
            continue
        left, right = cues[position - 1], cues[position + 1]
        if ({left.index, right.index} & missing or
                any(is_song_caption_cue(item) or cue_has_bracketed_screen_text(item) for item in (left, right))):
            continue
        left_indices = _anchor_indices(left.index, alignment, words, tokens, owners, uncertain)
        right_indices = _anchor_indices(right.index, alignment, words, tokens, owners, uncertain)
        if not left_indices or not right_indices:
            continue
        if any(not _valid_indices(indices, words, max_word_duration=max_word_duration) for indices in (left_indices, right_indices)):
            continue
        boundary_words = [words[left_indices[-1]], words[right_indices[0]]]
        if any(word.end - word.start <= _PLACEHOLDER_SECONDS + _EPSILON for word in boundary_words):
            continue
        start, end = words[left_indices[0]].start, words[right_indices[-1]].end
        gap_start, gap_end = boundary_words[0].end, boundary_words[1].start
        if not start < gap_start < gap_end < end <= audio_duration_seconds or end - start > _MAX_QUESTION_SECONDS:
            continue
        if any(not any(has_sufficient_speech_overlap(words[i], r.start, r.end) for r in regions)
               for i in (*left_indices, *right_indices)):
            continue
        def context(items):
            return [CueContext(cue_id=item.index, text=item.plain_text, start=item.start_ms / 1000, end=item.end_ms / 1000)
                    for item in items]
        span = DivergenceSpan(
            case_id=f"whole-utterance-timing-v{WHOLE_UTTERANCE_TIMING_POLICY_VERSION}-cue-{cue.index}",
            cue_ids=[cue.index], srt_text=speech_text_for_alignment(cue), srt_token_indices=own_tokens,
            asr_text=join_word_texts(words[i].text for i in evidence), asr_word_indices=list(evidence), start=start, end=end,
            context_before=context(cues[max(0, position - 2):position]), context_after=context(cues[position + 1:position + 3]),
            left_anchor_cue_id=left.index, right_anchor_cue_id=right.index, left_anchor_end=gap_start, right_anchor_start=gap_end,
            left_anchor_speaker_id=boundary_words[0].speaker_id, right_anchor_speaker_id=boundary_words[1].speaker_id,
        )
        question = MissingDialogueQuestion(span, parent.case_id if parent else "", left_indices, right_indices, read_only,
                                           "whole_utterance_timing", owned, evidence, discarded,
                                           neighbor_boundary_allowance_seconds=max_neighbor_boundary_overrun)
        if secondary:
            neighbor_proofs = []
            for neighbor, boundary, side in ((left, boundary_words[0], "left"), (right, boundary_words[1], "right")):
                claimed = [r for r in regions if has_sufficient_speech_overlap(boundary, r.start, r.end)]
                if len(claimed) != 1:
                    continue
                region = claimed[0]
                excess = region.end - boundary.end if side == "left" else boundary.start - region.start
                if excess > max_neighbor_boundary_overrun + _EPSILON:
                    proof = secondary.neighbor_boundary(neighbor, region, side=side, gap_start=gap_start, gap_end=gap_end,
                                                        allowance=max_neighbor_boundary_overrun)
                    if proof:
                        neighbor_proofs.append(proof)
            proof = secondary.proof(neighbor_proofs, target_proof)
            if proof:
                question = MissingDialogueQuestion(**{**question.__dict__, "secondary_acoustic_proof": proof,
                                                       "secondary_acoustic_proof_sha256": _digest(proof)})
        burst = _positive_gap_burst(question, words, regions)
        if burst is None or _unanchored_neighbor_echo(question, {item.index: item for item in cues}, words, burst):
            continue
        result.append(MissingDialogueQuestion(**{**question.__dict__, "word_evidence_sha256": _word_receipt(question, words)}))
    return result


def whole_utterance_resolution_reason(question, decision, sources, alignment, words, regions):
    """A positive question cannot authorize omission or infer internal words."""
    source = sources[question.cue_id]
    target = alphanumeric_signature(speech_text_for_alignment(source))
    if (not target or alphanumeric_signature(decision.heard_text or "") != target
            or alphanumeric_signature(decision.final_text) != target
            or decision.verdict == "keep_srt" and decision.final_text != question.span.srt_text):
        return "wording_does_not_match_hearing", None
    if (alignment.diagnostics.unresolved or is_song_caption_cue(source) or cue_has_bracketed_screen_text(source)
            or tuple(alignment.cue_word_indices.get(source.index, ())) != question.target_word_indices):
        return "source_or_ownership_changed", None
    owners = _owners(alignment)
    for cue_id, indices in ((source.index, question.target_word_indices),
                            (question.span.left_anchor_cue_id, question.left_word_indices),
                            (question.span.right_anchor_cue_id, question.right_word_indices)):
        if not _valid_indices(indices, words) or any(owners.get(i) != {cue_id} for i in indices):
            return "acoustic_ownership_changed", None
    if (not _valid_indices(question.evidence_word_indices, words)
            or any(owners.get(i, set()) - {source.index} for i in question.evidence_word_indices)
            or _word_receipt(question, words) != question.word_evidence_sha256):
        return "acoustic_ownership_changed", None
    if any(not isfinite(r.start) or not isfinite(r.end) or not 0 <= r.start < r.end for r in regions):
        return "acoustic_ownership_changed", None
    if not valid_secondary_proof(question, sources, regions):
        return "secondary_acoustic_evidence_changed", None
    burst = _positive_gap_burst(question, words, regions)
    if burst is not None and _unanchored_neighbor_echo(question, sources, words, burst):
        return "neighbor_wording_echo", None
    return ("audio_confirmed_utterance", burst) if burst is not None else ("no_unique_speech_burst", None)
