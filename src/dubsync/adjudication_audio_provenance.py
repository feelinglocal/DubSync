"""Bind actual transient native-hearing clips to reusable adjudication cases.

Only an active loader capture can originate provenance. Cached receipts may
outlive the deleted WAV, but must still match the current case key, decision,
span and verified full-audio context. Legacy manifests are never upgraded, and a
snippet row of the right shape is never a receipt.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from threading import RLock
import wave

from .cache import CacheKey
from .models import AdjudicationDecision, AudioSnippet, DivergenceSpan


AUDIO_PROVENANCE_POLICY_VERSION = 1
_EPSILON = 1e-6
_RECORD_FIELDS = {"case_id", "mime_type", "start", "end", "sha256", "size_bytes", "persisted"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _audio_identity(context) -> str | None:
    if not isinstance(context, dict):
        return None
    focused = context.get("focused_snippets", context)
    if (not isinstance(focused, dict) or not isinstance(focused.get("strategy"), str)
            or not focused["strategy"] or not _sha256(focused.get("audio_sha256"))):
        return None
    identity = focused["audio_sha256"]
    if "focused_snippets" in context:
        episode = context.get("episode_audio")
        if (not isinstance(episode, dict) or not _sha256(episode.get("source_sha256"))
                or episode.get("normalized_sha256") != identity
                or not _number(episode.get("duration_seconds")) or episode["duration_seconds"] <= 0
                or not isinstance(episode.get("options"), dict)):
            return None
    try:
        _digest(context)
    except (TypeError, ValueError):
        return None
    return identity


def _valid_record(record, span: DivergenceSpan) -> bool:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        return False
    start, end = record.get("start"), record.get("end")
    return (record.get("case_id") == span.case_id and record.get("mime_type") == "audio/wav"
            and record.get("persisted") is False and _sha256(record.get("sha256"))
            and isinstance(record.get("size_bytes"), int) and not isinstance(record["size_bytes"], bool)
            and record["size_bytes"] >= 44
            and _number(start) and _number(end) and 0 <= start < end
            and _number(span.start) and _number(span.end) and 0 <= span.start < span.end
            and start <= span.start + _EPSILON and end >= span.end - _EPSILON)


def _capture(snippet: AudioSnippet, span: DivergenceSpan) -> dict | None:
    if (not isinstance(snippet, AudioSnippet) or snippet.case_id != span.case_id
            or snippet.mime_type != "audio/wav" or not _number(snippet.start)
            or not _number(snippet.end) or not 0 <= snippet.start < snippet.end):
        return None
    try:
        path = Path(snippet.path)
        with wave.open(str(path), "rb") as stream:
            frames, rate = stream.getnframes(), stream.getframerate()
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or rate != 16000 or frames <= 0:
                return None
            remaining = frames * 2
            while remaining:
                chunk = stream.readframes(min(remaining // 2, 65536))
                if not chunk or len(chunk) > remaining:
                    return None
                remaining -= len(chunk)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        record = {"case_id": span.case_id, "mime_type": snippet.mime_type,
                  "start": snippet.start, "end": min(snippet.end, snippet.start + frames / rate),
                  "sha256": digest, "size_bytes": path.stat().st_size, "persisted": False}
        return record if _valid_record(record, span) else None
    except (OSError, ValueError, EOFError, wave.Error):
        return None


class AudioProvenanceRecorder:
    """Wrap a bounded loader without retaining its WAV files or changing clips.

    Capture is committed only after the consumer returns successfully and
    the actual bytes still match. Distinct concurrent claims for a case or
    conflicting retry bytes cannot become a reusable receipt.
    """
    def __init__(self, load: Callable):
        self._load = load
        self._records: dict[str, dict] = {}
        self._active: set[str] = set()
        self._conflicts: set[str] = set()
        self._lock = RLock()

    @contextmanager
    def load(self, spans: list[DivergenceSpan]):
        counts = Counter(span.case_id for span in spans)
        requested = {span.case_id: span.model_copy(deep=True) for span in spans}
        with self._lock:
            self._conflicts.update(case_id for case_id, count in counts.items() if count != 1)
            self._conflicts.update(self._active.intersection(requested))
            self._active.update(requested)
        try:
            with self._load(spans) as snippets:
                before = {case_id: _capture(snippets.get(case_id), span)
                          for case_id, span in requested.items()}
                yield snippets
                after = {case_id: _capture(snippets.get(case_id), span)
                         for case_id, span in requested.items()}
                with self._lock:
                    for case_id in requested:
                        record = before[case_id]
                        prior = self._records.get(case_id)
                        if (record is None or record != after[case_id]
                                or prior is not None and prior != record):
                            self._records.pop(case_id, None)
                            if record is not None and (record != after[case_id] or prior is not None):
                                self._conflicts.add(case_id)
                        elif case_id not in self._conflicts:
                            self._records[case_id] = record
        except BaseException:
            with self._lock:
                for case_id in requested:
                    self._records.pop(case_id, None)
            raise
        finally:
            with self._lock:
                self._active.difference_update(requested)

    def manifest(self) -> dict[str, object]:
        with self._lock:
            return {"capture_policy_version": AUDIO_PROVENANCE_POLICY_VERSION,
                    "snippets": deepcopy([self._records[case_id] for case_id in sorted(self._records)
                                          if case_id not in self._conflicts])}


def _bindings(key, span, decision, audio_context) -> dict | None:
    audio_sha = _audio_identity(audio_context)
    if (audio_sha is None or not _sha256(key.digest) or decision.case_id != span.case_id
            or decision.evidence is None or decision.reason.startswith("Dual ASR cross-check:")):
        return None
    try:
        return {"case_key_sha256": key.digest, "audio_context_sha256": _digest(audio_context),
                "audio_sha256": audio_sha,
                "span_sha256": _digest(span.model_dump(mode="json", exclude={"case_id"})),
                "decision_sha256": _digest(decision.model_dump(mode="json", exclude={"case_id"}))}
    except (TypeError, ValueError):
        return None


def bind_case_audio_provenance(
    key: CacheKey, span: DivergenceSpan, decision: AdjudicationDecision,
    manifest: dict, audio_context: dict | None,
) -> dict | None:
    """Bind only this run's successful active capture to its native decision.

    The caller stores this alongside that exact successful case-cache value.
    ``audio_context`` must come from the current verified snippet source.
    A plain historical snippet manifest is intentionally insufficient.
    """
    if (not isinstance(manifest, dict) or manifest.get("capture_policy_version") != AUDIO_PROVENANCE_POLICY_VERSION
            or not isinstance(manifest.get("snippets"), list)):
        return None
    records = [record for record in manifest["snippets"]
               if isinstance(record, dict) and record.get("case_id") == span.case_id]
    bindings = _bindings(key, span, decision, audio_context)
    if bindings is None or len(records) != 1 or not _valid_record(records[0], span):
        return None
    content = {"policy_version": AUDIO_PROVENANCE_POLICY_VERSION, "case_id": span.case_id,
               **bindings, "snippet": deepcopy(records[0])}
    return {**content, "receipt_sha256": _digest(content)}


def cached_case_audio_snippet(
    provenance: object, key: CacheKey, span: DivergenceSpan, decision: AdjudicationDecision,
    audio_context: dict | None,
) -> dict | None:
    """Validate a matched case-cache receipt before rebinding its local ID.

    The caller must have accepted the ordinary case-cache hit first. The
    hashes detect stale or corrupted artifacts; they are not signatures that
    can authenticate an adversary who can rewrite the local cache itself.
    """
    bindings = _bindings(key, span, decision, audio_context)
    if bindings is None or not isinstance(provenance, dict):
        return None
    content = {name: value for name, value in provenance.items() if name != "receipt_sha256"}
    try:
        if (provenance.get("policy_version") != AUDIO_PROVENANCE_POLICY_VERSION
                or provenance.get("receipt_sha256") != _digest(content)
                or any(provenance.get(name) != value for name, value in bindings.items())):
            return None
    except (TypeError, ValueError):
        return None
    original_id = provenance.get("case_id")
    if not isinstance(original_id, str) or not original_id:
        return None
    record = provenance.get("snippet")
    if not _valid_record(record, span.model_copy(update={"case_id": original_id})):
        return None
    return {**deepcopy(record), "case_id": span.case_id}


def receipt_audio_snippet(
    receipt: object, span: DivergenceSpan, decision: AdjudicationDecision, audio_sha256: str | None,
) -> dict | None:
    """Return the clip a receipt binds to exactly this question, answer and audio.

    ``receipt`` must be one the caller bound or validated against the current
    case key in this run (``bind_case_audio_provenance`` or a matched
    ``cached_case_audio_snippet``). This rechecks its digest, the clip bytes
    digest and interval, and that it was captured for this span and answer.
    """
    if (not isinstance(receipt, dict) or not _sha256(audio_sha256) or decision.case_id != span.case_id
            or decision.evidence is None or decision.reason.startswith("Dual ASR cross-check:")):
        return None
    content = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    try:
        if (receipt.get("policy_version") != AUDIO_PROVENANCE_POLICY_VERSION
                or receipt.get("receipt_sha256") != _digest(content)
                or not _sha256(receipt.get("case_key_sha256")) or not _sha256(receipt.get("audio_context_sha256"))
                or receipt.get("audio_sha256") != audio_sha256
                or receipt.get("span_sha256") != _digest(span.model_dump(mode="json", exclude={"case_id"}))
                or receipt.get("decision_sha256") != _digest(decision.model_dump(mode="json", exclude={"case_id"}))):
            return None
    except (TypeError, ValueError):
        return None
    original_id = receipt.get("case_id")
    record = receipt.get("snippet")
    if (not isinstance(original_id, str) or not original_id
            or not _valid_record(record, span.model_copy(update={"case_id": original_id}))):
        return None
    return {**deepcopy(record), "case_id": span.case_id}
