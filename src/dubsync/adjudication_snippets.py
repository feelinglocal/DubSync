from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, RLock

from .audio_snippets import (
    SNIPPET_WAV_ALLOWANCE_BYTES,
    AudioSnippetError,
    _max_snippet_bytes,
    _snippet_window,
    extract_audio_snippets,
)
from .cost import audio_seconds
from .models import AudioSnippet, DivergenceSpan, QCFlag


SNIPPET_BATCH_STRATEGY_VERSION = "bounded_batches_v2"
AudioSnippetExtractor = Callable[..., list[AudioSnippet]]


class BoundedAudioSnippetBatchSource:
    """Create optional adjudication audio only for the active LLM batch."""

    def __init__(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        pad_seconds: float,
        max_duration_seconds: float,
        max_snippets_per_batch: int,
        max_audio_duration_seconds: float,
        extractor: AudioSnippetExtractor = extract_audio_snippets,
        max_concurrent_batches: int = 1,
    ) -> None:
        if (
            isinstance(max_concurrent_batches, bool)
            or not isinstance(max_concurrent_batches, int)
            or not 1 <= max_concurrent_batches <= 4
        ):
            raise ValueError("max_concurrent_batches must be an integer between 1 and 4")
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.pad_seconds = pad_seconds
        self.max_duration_seconds = max_duration_seconds
        self.max_snippets_per_batch = max_snippets_per_batch
        self.max_audio_duration_seconds = max_audio_duration_seconds
        self.extractor = extractor
        self.max_concurrent_batches = max_concurrent_batches
        self.max_total_bytes = _max_snippet_bytes(None)
        self.audio_duration_seconds = audio_seconds(audio_path)
        self._audio_sha256 = _sha256_file(audio_path)
        self._batch_index = 0
        self._candidate_case_ids: tuple[str, ...] = ()
        self._selected: dict[str, dict[str, object]] = {}
        self._fallback_case_ids: tuple[str, ...] = ()
        self._fallback_cue_ids: tuple[int, ...] = ()
        self._fallback_batch_count = 0
        self._total_processed_bytes = 0
        self._peak_batch_bytes = 0
        self._lock = RLock()
        self._capacity = Condition(self._lock)
        self._reserved_bytes = 0
        self._active_batch_dirs: dict[int, Path] = {}
        self._peak_concurrent_bytes = 0
        self._storage_failed = False

    @contextmanager
    def load(self, spans: list[DivergenceSpan]) -> Iterator[dict[str, AudioSnippet]]:
        batch = [span.model_copy(deep=True) for span in spans]
        with self._lock:
            self._batch_index += 1
            batch_index = self._batch_index
            self._candidate_case_ids = _ordered_union(
                self._candidate_case_ids,
                (span.case_id for span in batch),
            )
        if (
            self.audio_duration_seconds > 0
            and self.audio_duration_seconds > self.max_audio_duration_seconds
        ):
            self._record_fallback(batch)
            yield {}
            return
        batch_dir = self.output_dir / f"batch-{batch_index:04d}"
        reservation = self._batch_reservation_bytes(batch)
        self._reserve_batch(batch_index, batch_dir, reservation)
        snippets: list[AudioSnippet] = []
        try:
            try:
                snippets = self.extractor(
                    self.audio_path,
                    batch,
                    batch_dir,
                    pad_seconds=self.pad_seconds,
                    max_duration_seconds=self.max_duration_seconds,
                    fail_on_budget_exceeded=False,
                    max_snippets=self.max_snippets_per_batch,
                    max_total_bytes=reservation,
                )
                records, batch_bytes = _snippet_records(snippets, batch_dir)
                if batch_bytes > reservation:
                    raise AudioSnippetError("Audio snippet extractor exceeded its reserved job storage")
            except (AudioSnippetError, OSError):
                self._record_fallback(batch)
                yield {}
                return

            with self._lock:
                self._selected = {**self._selected, **records}
                selected_case_ids = set(records)
                missing_spans = [span for span in batch if span.case_id not in selected_case_ids]
                if missing_spans:
                    self._record_fallback(missing_spans)
                self._total_processed_bytes += batch_bytes
                self._peak_batch_bytes = max(self._peak_batch_bytes, batch_bytes)
                self._update_peak_concurrent_bytes()
            yield {snippet.case_id: snippet for snippet in snippets}
        finally:
            self._finish_batch(batch_index, batch_dir, reservation)

    def _batch_reservation_bytes(self, spans: list[DivergenceSpan]) -> int:
        """Reserve the same PCM/window bounds used by the extractor, before I/O."""
        predicted = 0
        count = 0
        for span in spans:
            if count >= self.max_snippets_per_batch:
                break
            if span.start is None or span.end is None or span.end <= span.start:
                continue
            start, end = _snippet_window(
                span.start, span.end, self.pad_seconds, self.max_duration_seconds,
            )
            predicted += math.ceil((end - start) * 32_000) + SNIPPET_WAV_ALLOWANCE_BYTES
            count += 1
        # The extractor still applies its normal per-file budget checks. Large
        # batches can use the full job cap, but must wait for other batches.
        return max(1, min(self.max_total_bytes, predicted))

    def _reserve_batch(self, index: int, directory: Path, size: int) -> None:
        with self._capacity:
            self._capacity.wait_for(lambda: self._storage_failed or (
                len(self._active_batch_dirs) < self.max_concurrent_batches
                and self._reserved_bytes + size <= self.max_total_bytes
            ))
            if self._storage_failed:
                raise AudioSnippetError("Previous audio snippet cleanup failed; job storage cannot be reused")
            self._reserved_bytes += size
            self._active_batch_dirs[index] = directory

    def _update_peak_concurrent_bytes(self) -> None:
        # Called under the cleanup lock. Include observed files from batches
        # still being extracted, not only batches already awaiting the provider.
        current_bytes = 0
        for directory in self._active_batch_dirs.values():
            for path in directory.rglob("*"):
                try:
                    if path.is_file():
                        current_bytes += path.stat().st_size
                except FileNotFoundError:
                    # FFmpeg can atomically replace its own .partial file.
                    continue
        self._peak_concurrent_bytes = max(self._peak_concurrent_bytes, current_bytes)

    def _finish_batch(self, index: int, directory: Path, size: int) -> None:
        with self._capacity:
            cleaned = False
            try:
                self._update_peak_concurrent_bytes()
                _remove_transient_batch_files(directory)
                if any(path.is_file() for path in directory.rglob("*")):
                    raise AudioSnippetError("Audio snippet cleanup left files occupying job storage")
                cleaned = True
            finally:
                if cleaned:
                    self._reserved_bytes -= size
                    self._active_batch_dirs.pop(index)
                else:
                    # Keep the allocation charged and wake blocked callers so
                    # cleanup failure cannot cause either overspend or deadlock.
                    self._storage_failed = True
                self._capacity.notify_all()

    def cache_context(self) -> dict[str, object]:
        return {
            "strategy": SNIPPET_BATCH_STRATEGY_VERSION,
            "audio_sha256": self._audio_sha256,
            "pad_seconds": self.pad_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "max_snippets_per_batch": self.max_snippets_per_batch,
            "max_audio_duration_seconds": self.max_audio_duration_seconds,
            "max_total_bytes": self.max_total_bytes,
            "max_concurrent_batches": self.max_concurrent_batches,
        }

    def manifest(self) -> dict[str, object]:
        selected_records = [
            self._selected[case_id]
            for case_id in self._candidate_case_ids
            if case_id in self._selected
        ]
        return {
            "strategy_version": SNIPPET_BATCH_STRATEGY_VERSION,
            "storage_mode": "bounded_batches",
            "candidate_count": len(self._candidate_case_ids),
            "max_snippets_per_batch": self.max_snippets_per_batch,
            "max_audio_duration_seconds": self.max_audio_duration_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
            "selected_count": len(selected_records),
            "fallback_count": len(self._fallback_case_ids),
            "fallback_case_ids": list(self._fallback_case_ids),
            "fallback_batch_count": self._fallback_batch_count,
            "total_processed_bytes": self._total_processed_bytes,
            "peak_batch_bytes": self._peak_batch_bytes,
            "peak_concurrent_bytes": self._peak_concurrent_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_concurrent_batches": self.max_concurrent_batches,
            "snippets": selected_records,
        }

    def flags(self) -> list[QCFlag]:
        if not self._fallback_case_ids:
            return []
        return [
            QCFlag(
                kind="audio_snippet_unavailable",
                cue_ids=list(self._fallback_cue_ids),
                message=(
                    "Optional adjudication audio was unavailable for "
                    f"{len(self._fallback_case_ids)} divergence cases; text and "
                    "word-timestamp evidence were used without changing cue timing."
                ),
                severity="warning",
            )
        ]

    def _record_fallback(self, spans: list[DivergenceSpan]) -> None:
        with self._lock:
            self._fallback_case_ids = _ordered_union(
                self._fallback_case_ids,
                (span.case_id for span in spans),
            )
            self._fallback_cue_ids = _ordered_union(
                self._fallback_cue_ids,
                (cue_id for span in spans for cue_id in span.cue_ids),
            )
            self._fallback_batch_count += 1


def _snippet_records(
    snippets: list[AudioSnippet],
    batch_dir: Path,
) -> tuple[dict[str, dict[str, object]], int]:
    resolved_batch_dir = batch_dir.resolve()
    records: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for snippet in snippets:
        path = Path(snippet.path).resolve()
        if not path.is_relative_to(resolved_batch_dir) or not path.is_file():
            raise AudioSnippetError("Audio snippet extractor returned an invalid transient path")
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        records[snippet.case_id] = {
            "case_id": snippet.case_id,
            "mime_type": snippet.mime_type,
            "start": snippet.start,
            "end": snippet.end,
            "sha256": _sha256_file(path),
            "size_bytes": size_bytes,
            "persisted": False,
        }
    return records, total_bytes


def _remove_transient_batch_files(batch_dir: Path) -> None:
    if not batch_dir.exists():
        return
    for path in batch_dir.iterdir():
        if path.is_file() and not path.is_symlink():
            path.unlink(missing_ok=True)
    try:
        batch_dir.rmdir()
    except OSError:
        return
    try:
        batch_dir.parent.rmdir()
    except OSError:
        pass


def _ordered_union(existing: tuple, additions) -> tuple:
    ordered = list(existing)
    seen = set(existing)
    for item in additions:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
