"""Read-only secondary ownership proofs for whole-utterance timing questions.

The cross-check runtime verifies audio provenance before this module receives
its context. These proofs only identify an existing VAD region or corroborate
one neighbouring boundary. They never replace primary words or timestamps.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from .aligner import align_cues_to_words
from .asr_crosscheck import CROSS_CHECK_POLICY_VERSION, compare_word_streams
from .asr_timing import has_sufficient_speech_overlap
from .missing_dialogue_reconciliation import _digest, _owners
from .models import Cue, SpeechRegion, Word
from .subtitle_annotations import speech_text_for_alignment
from .tokenize import alphanumeric_signature, tokenize_cues


SECONDARY_ACOUSTIC_POLICY_VERSION = 1
_EPSILON = 1e-7


def _intersects(word, start, end):
    return word.start < end - _EPSILON and word.end > start + _EPSILON


def _region_record(region):
    return {"start": region.start, "end": region.end}


def _same_region(record, region):
    return record == _region_record(region)


@dataclass
class SecondaryAcousticEvidence:
    words: list[Word]
    context: dict[str, object]
    cues: list[Cue]
    alignment: object
    tokens: list
    owners: dict[int, set[int]]
    agreement: object

    @classmethod
    def prepare(cls, cues, primary_words, secondary_words, context):
        if not secondary_words or not isinstance(context, dict):
            return None
        if (context.get("policy_version") != CROSS_CHECK_POLICY_VERSION
                or (context.get("provider"), context.get("model")) not in {
                    ("openrouter", "microsoft/mai-transcribe-2"), ("elevenlabs", "scribe_v2")}
                or not isinstance(context.get("config"), dict)):
            return None
        if any(not isfinite(w.start) or not isfinite(w.end) or not 0 <= w.start < w.end for w in secondary_words):
            return None
        if context.get("words_sha256") != _digest([w.model_dump(mode="json") for w in secondary_words]):
            return None
        language = context["config"].get("language_code", context["config"].get("language"))
        alignment = align_cues_to_words(cues, secondary_words, language=language)
        if alignment.diagnostics.unresolved:
            return None
        return cls(secondary_words, deepcopy(context), cues, alignment, tokenize_cues(cues),
                   _owners(alignment), compare_word_streams(primary_words, secondary_words))

    def _source_group(self, cue, *, exact=False):
        source = [t for t in self.tokens if t.cue_id == cue.index]
        lexical = [i for i in self.alignment.cue_word_indices.get(cue.index, [])
                   if alphanumeric_signature(self.words[i].text)]
        if not source or not lexical or lexical != sorted(set(lexical)):
            return None
        source_ids = {t.token_index for t in source}
        matched = {(m.srt_token_index, m.asr_word_index) for m in self.alignment.token_matches
                   if m.cue_id == cue.index and m.asr_word_index in lexical and m.score >= .8
                   and m.srt_token_index in source_ids
                   and self.tokens[m.srt_token_index].normalized in alphanumeric_signature(self.words[m.asr_word_index].text)}
        covered = {token for token, _ in matched}
        if (len(covered) * 3 < len(source) * 2
                or (source[0].token_index, lexical[0]) not in matched
                or (source[-1].token_index, lexical[-1]) not in matched):
            return None
        indices = [i for i in range(lexical[0], lexical[-1] + 1) if alphanumeric_signature(self.words[i].text)]
        for index in indices:
            owners = self.owners.get(index, set())
            if owners == {cue.index}:
                continue
            parents = [s for s in self.alignment.divergence_spans if index in s.asr_word_indices]
            if owners or len(parents) != 1 or set(parents[0].cue_ids) != {cue.index}:
                return None
        words = [self.words[i] for i in indices]
        if (any(not .020 + _EPSILON < w.end - w.start <= 2
                or w.confidence is not None and w.confidence < .7 for w in words)
                or any(a.end > b.start + _EPSILON for a, b in zip(words, words[1:]))):
            return None
        speakers = {w.speaker_id for w in words}
        if len(speakers) != 1 or not next(iter(speakers), None):
            return None
        target = alphanumeric_signature(speech_text_for_alignment(cue))
        spoken = [token for word in words for token in alphanumeric_signature(word.text)]
        if exact and (spoken != target or covered != source_ids):
            return None
        return {
            "cue_id": cue.index, "source_tokens": target, "matched_source_tokens": sorted(covered),
            "word_indices": indices, "word_owners": {str(i): sorted(self.owners.get(i, set())) for i in indices},
            "words": [{"word_index": i, "word": self.words[i].model_dump(mode="json")} for i in indices],
        }

    def neighbor_boundary(self, cue, region, *, side, gap_start, gap_end, allowance):
        group = self._source_group(cue)
        if group is None:
            return None
        indices = group["word_indices"]
        first, last = self.words[indices[0]], self.words[indices[-1]]
        boundary = first if side == "right" else last
        excess = first.start - region.start if side == "right" else region.end - last.end
        if excess > allowance + _EPSILON or not has_sufficient_speech_overlap(boundary, region.start, region.end):
            return None
        start, end = max(gap_start, region.start), min(gap_end, region.end)
        if (not start < end or any(i not in indices and alphanumeric_signature(w.text) and _intersects(w, start, end)
                                  for i, w in enumerate(self.words))):
            return None
        return {"side": side, "region": _region_record(region), "scope": {"start": start, "end": end}, "group": group}

    def whole_target(self, cue, primary_owned, uncertain, regions):
        uncertain = set(primary_owned) & set(uncertain)
        if not uncertain:
            return None
        group = self._source_group(cue, exact=True)
        if group is None:
            return None
        indices = group["word_indices"]
        first, last = self.words[indices[0]], self.words[indices[-1]]
        candidates = [r for r in regions if r.start <= first.start < last.end <= r.end]
        if len(candidates) != 1:
            return None
        region = candidates[0]
        if (any(i not in indices and alphanumeric_signature(w.text) and _intersects(w, region.start, region.end)
                for i, w in enumerate(self.words))
                or any(sum(has_sufficient_speech_overlap(self.words[i], r.start, r.end) for r in regions) != 1 for i in indices)):
            return None
        mappings = []
        for index in sorted(uncertain):
            primary_tokens = [i for i, token in enumerate(self.agreement.primary_tokens) if token.word_index == index]
            if not primary_tokens or any(self.agreement.token_matches[i] is None for i in primary_tokens):
                return None
            secondary_ids = {self.agreement.secondary_tokens[self.agreement.token_matches[i]].word_index for i in primary_tokens}
            if len(secondary_ids) != 1 or not secondary_ids <= set(indices):
                return None
            secondary_id = next(iter(secondary_ids))
            if (alphanumeric_signature(self.agreement.primary_words[index].text)
                    != alphanumeric_signature(self.words[secondary_id].text)):
                return None
            mappings.append({"primary_word_index": index, "secondary_word_index": secondary_id})
        return {"region": _region_record(region), "group": group, "primary_mappings": mappings}

    def proof(self, neighbor_regions, target):
        if not neighbor_regions and target is None:
            return None
        return {"policy_version": SECONDARY_ACOUSTIC_POLICY_VERSION, "context": deepcopy(self.context),
                "neighbor_regions": neighbor_regions, "target": target}


def valid_secondary_proof(question, sources, regions):
    proof = question.secondary_acoustic_proof
    if proof is None:
        return question.secondary_acoustic_proof_sha256 == ""
    if (not isinstance(proof, dict) or _digest(proof) != question.secondary_acoustic_proof_sha256
            or proof.get("policy_version") != SECONDARY_ACOUSTIC_POLICY_VERSION):
        return False
    parts = list(proof["neighbor_regions"])
    if proof["target"] is not None:
        parts.append(proof["target"])
    for part in parts:
        group = part["group"]
        if (group["cue_id"] not in sources
                or group["source_tokens"] != alphanumeric_signature(speech_text_for_alignment(sources[group["cue_id"]]))
                or not any(_same_region(part["region"], region) for region in regions)):
            return False
    return True


def supported_neighbor_boundary(question, side, region):
    proof = question.secondary_acoustic_proof
    return bool(proof and any(part["side"] == side and _same_region(part["region"], region)
                              for part in proof["neighbor_regions"]))


def secondary_target(question):
    proof = question.secondary_acoustic_proof
    return proof["target"] if proof else None
