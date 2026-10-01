"""Reuse independent wording decisions without coupling their keys to a batch."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from pydantic import ValidationError

from .cache import CacheKey, JsonDiskCache
from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag, Word


_TRANSIENT = frozenset({
    "audio_snippet_unavailable", "adjudication_audio_unavailable", "adjudication_review_unavailable",
    "invalid_llm_response", "llm_provider_unavailable",
})


def case_cache_key(
    span: DivergenceSpan, *, model: str, params: dict, policy_context: dict,
    source_cues: Sequence[Cue], source_words: Sequence[Word],
    word_evidence_sha256: str | None = None,
) -> CacheKey:
    positions = {cue.index: position for position, cue in enumerate(source_cues)}
    ids = {*span.cue_ids, span.left_anchor_cue_id, span.right_anchor_cue_id,
           *(cue.cue_id for cue in (*span.context_before, *span.context_after))}
    selected = set()
    for cue_id in ids:
        if cue_id in positions:
            position = positions[cue_id]
            selected.update(range(max(0, position - 2), min(len(source_cues), position + 3)))
    payload = span.model_dump(mode="json", exclude={"case_id"})
    # The full word-stream digest preserves the reviewer's ownership evidence.
    # Editing another divergent source span still leaves this case reusable.
    words_sha = word_evidence_sha256 or word_stream_digest(source_words)
    return CacheKey.from_payload({
        "kind": "adjudication-case-v1", "span": payload, "policy": policy_context,
        "local_source": [source_cues[i].model_dump(mode="json") for i in sorted(selected)],
        "words_sha256": words_sha,
    }, model=model, params=params)


def word_stream_digest(words: Sequence[Word]) -> str:
    digest = hashlib.sha256()
    for word in words:
        digest.update(json.dumps(word.model_dump(mode="json"), sort_keys=True,
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_case(cache: JsonDiskCache, key: CacheKey, span: DivergenceSpan):
    try:
        payload = cache.read(key)
        if not isinstance(payload, dict) or not isinstance(payload.get("flags"), list):
            return None
        decision = AdjudicationDecision.model_validate(payload["decision"])
        flags = [QCFlag.model_validate(item) for item in payload["flags"]]
    except (KeyError, TypeError, ValueError, ValidationError):
        return None
    if any(flag.kind in _TRANSIENT for flag in flags):
        return None
    return decision.model_copy(update={"case_id": span.case_id}), flags


def write_case(
    cache: JsonDiskCache, key: CacheKey, span: DivergenceSpan,
    decision: AdjudicationDecision, flags: Sequence[QCFlag],
) -> None:
    if decision.case_id != span.case_id or any(flag.kind in _TRANSIENT for flag in flags):
        return
    cache.write(key, {"decision": decision.model_dump(mode="json"),
                      "flags": [flag.model_dump(mode="json") for flag in flags]})
