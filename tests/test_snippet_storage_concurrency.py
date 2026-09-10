from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

import dubsync.adjudication_snippets as snippet_batches
from dubsync.adjudication_snippets import BoundedAudioSnippetBatchSource
from dubsync.audio_snippets import AudioSnippetError
from dubsync.models import DivergenceSpan


def _spans(count, *, offset=0):
    return [
        DivergenceSpan(
            case_id=f"case-{offset + index}", cue_ids=[offset + index + 1],
            srt_text="source dialogue", asr_text="spoken dialogue",
            start=(offset + index) * 30.0, end=(offset + index) * 30.0 + 20.0,
        )
        for index in range(count)
    ]


def _source(tmp_path, monkeypatch, *, budget, **options):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"offline source fixture")
    monkeypatch.setenv("DUBSYNC_MAX_AUDIO_SNIPPET_BYTES", str(budget))
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 1000.0)

    def cut(_audio, output, start, end, _ffmpeg, _timeout, _cap):
        output.write_bytes(bytes(round((end - start) * 32_000)))

    monkeypatch.setattr("dubsync.audio_snippets._cut_wav_snippet", cut)
    return BoundedAudioSnippetBatchSource(
        audio, tmp_path / "snippets", pad_seconds=0, max_duration_seconds=20,
        max_snippets_per_batch=8, max_audio_duration_seconds=14400, **options,
    )


def test_concurrent_batches_wait_for_job_storage_before_writing(tmp_path, monkeypatch):
    budget = 1024 * 1024
    source = _source(tmp_path, monkeypatch, budget=budget, max_concurrent_batches=4)
    attempted = Barrier(4, timeout=5)
    first_ready, another_ready, release_first = Event(), Event(), Event()
    lock = Lock()
    entries = []
    live_bytes = []
    active_bytes = {}

    def work(span):
        attempted.wait()
        with source.load([span]) as snippets:
            with lock:
                entries.append(span.case_id)
                is_first = len(entries) == 1
                active_bytes[span.case_id] = sum(Path(snippet.path).stat().st_size for snippet in snippets.values())
                live_bytes.append(sum(active_bytes.values()))
                (first_ready if is_first else another_ready).set()
            assert release_first.wait(5)
            assert span.case_id in snippets
            with lock:
                active_bytes.pop(span.case_id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, span) for span in _spans(4)]
        try:
            assert first_ready.wait(5)
            # One full clip fits this job; a second cannot coexist with it.
            wrote_another_batch_before_release = another_ready.wait(0.25)
        finally:
            release_first.set()
        for future in futures:
            future.result(timeout=5)

    assert not wrote_another_batch_before_release
    assert max(live_bytes) <= budget
    assert len(entries) == 4
    assert not list(source.output_dir.rglob("*.wav"))


def test_default_eight_case_batches_keep_four_way_concurrency_and_report_live_bytes(tmp_path, monkeypatch):
    budget = 32 * 1024 * 1024
    source = _source(tmp_path, monkeypatch, budget=budget, max_concurrent_batches=4)
    ready = Barrier(4, timeout=5)

    def work(index):
        with source.load(_spans(8, offset=index * 8)) as snippets:
            ready.wait()
            assert len(snippets) == 8
            assert all(Path(snippet.path).exists() for snippet in snippets.values())
            ready.wait()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))

    manifest = source.manifest()
    assert manifest["selected_count"] == 32
    assert manifest["fallback_count"] == 0
    assert manifest["peak_batch_bytes"] == 8 * 20 * 32_000
    assert manifest["peak_concurrent_bytes"] == 4 * 8 * 20 * 32_000
    assert manifest["peak_concurrent_bytes"] <= manifest["max_total_bytes"] == budget
    assert manifest["max_concurrent_batches"] == 4
    assert not list(source.output_dir.rglob("*.wav"))


def test_allocation_is_not_reused_until_files_are_deleted(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, budget=1024 * 1024, max_concurrent_batches=4)
    cleanup_entered, allow_cleanup, second_ready = Event(), Event(), Event()
    real_cleanup = snippet_batches._remove_transient_batch_files

    def delayed_cleanup(directory):
        if directory.name == "batch-0001":
            cleanup_entered.set()
            assert allow_cleanup.wait(5)
        real_cleanup(directory)

    monkeypatch.setattr(snippet_batches, "_remove_transient_batch_files", delayed_cleanup)

    def work(span):
        with source.load([span]) as snippets:
            assert span.case_id in snippets
            if span.case_id == "case-1":
                second_ready.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(work, _spans(1)[0])
        try:
            assert cleanup_entered.wait(5)
            second = pool.submit(work, _spans(1, offset=1)[0])
            assert not second_ready.wait(0.25)
            assert len(list(source.output_dir.rglob("*.wav"))) == 1
        finally:
            allow_cleanup.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert second_ready.is_set()
    assert not list(source.output_dir.rglob("*.wav"))


@pytest.mark.parametrize("cleanup_behavior", ["raise", "leave_files"])
def test_cleanup_failure_wakes_waiters_without_reusing_occupied_storage(tmp_path, monkeypatch, cleanup_behavior):
    source = _source(tmp_path, monkeypatch, budget=1024 * 1024, max_concurrent_batches=4)
    attempted = Barrier(4, timeout=5)
    extracted = []

    def broken_cleanup(_directory):
        if cleanup_behavior == "raise":
            raise PermissionError("The WAV is still in use")

    monkeypatch.setattr(snippet_batches, "_remove_transient_batch_files", broken_cleanup)

    def work(span):
        attempted.wait()
        with source.load([span]) as snippets:
            extracted.append(span.case_id)
            assert span.case_id in snippets

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, span) for span in _spans(4)]
        for future in futures:
            with pytest.raises((PermissionError, AudioSnippetError)):
                future.result(timeout=5)

    assert len(extracted) == 1
    assert sum(path.stat().st_size for path in source.output_dir.rglob("*.wav")) == 640_000
    with pytest.raises(AudioSnippetError, match="cleanup failed"):
        with source.load(_spans(1, offset=4)):
            pytest.fail("Occupied storage must not be handed to another batch")


def test_one_oversized_batch_respects_original_job_limit_and_flags_missing_clips(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, budget=1024 * 1024, max_concurrent_batches=4)

    with source.load(_spans(8)) as snippets:
        assert list(snippets) == ["case-0"]
        assert sum(Path(snippet.path).stat().st_size for snippet in snippets.values()) == 640_000

    manifest = source.manifest()
    assert manifest["selected_count"] == 1
    assert manifest["fallback_count"] == 7
    assert source.flags()[0].cue_ids == list(range(2, 9))
    assert not list(source.output_dir.rglob("*.wav"))
