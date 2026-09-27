from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dubsync.adjudication import AdjudicationEngine
from dubsync.adjudication_snippets import BoundedAudioSnippetBatchSource
from dubsync.models import AudioSnippet, DivergenceSpan
from dubsync.providers import ProviderError


def spans(count):
    return [DivergenceSpan(case_id=f"case-{i}", cue_ids=[i+1], srt_text=f"source{i}", asr_text=f"spoken{i}", start=i, end=i+.5) for i in range(count)]


def test_parallel_batches_are_bounded_and_results_keep_case_ownership():
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active, peak, batch_sizes = 0, 0, []
    class Adapter:
        def adjudicate(self, batch):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                batch_sizes.append(len(batch))
            barrier.wait()
            with lock:
                active -= 1
            return [dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text, confidence=.95, reason="Audible words") for span in reversed(batch)]
    source = spans(9)
    engine = AdjudicationEngine(Adapter(), max_batch_spans=3, max_concurrent_batches=3)
    decisions, flags = engine.adjudicate(source)
    assert peak == 3 and max(batch_sizes) <= 3
    assert [(d.case_id, d.final_text) for d in decisions] == [(s.case_id, s.asr_text) for s in source]
    assert flags == []


def test_one_parallel_batch_failure_holds_only_its_source_cases():
    class Adapter:
        def adjudicate(self, batch):
            if any(span.case_id == "case-0" for span in batch):
                raise ProviderError("Uncertain request")
            return [dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text, confidence=.95, reason="Audible words") for span in batch]
    source = spans(4)
    decisions, flags = AdjudicationEngine(Adapter(), max_batch_spans=2, max_concurrent_batches=2).adjudicate(source)
    assert [d.final_text for d in decisions] == ["source0", "source1", "spoken2", "spoken3"]
    assert {tuple(f.cue_ids) for f in flags if f.kind == "llm_provider_unavailable"} == {(1,), (2,)}


def test_fatal_batch_error_stops_unscheduled_calls_and_waits_for_active_request():
    first_started, fatal_seen, release_first = threading.Event(), threading.Event(), threading.Event()
    calls = []
    class Adapter:
        def adjudicate(self, batch):
            case_id = batch[0].case_id
            calls.append(case_id)
            if case_id == "case-0":
                first_started.set()
                assert release_first.wait(5)
                return []
            if case_id == "case-1":
                assert first_started.wait(5)
                fatal_seen.set()
                raise RuntimeError("Fatal adapter invariant")
            return []
    def release():
        assert fatal_seen.wait(5)
        release_first.set()
    helper = threading.Thread(target=release)
    helper.start()
    try:
        with pytest.raises(RuntimeError, match="Fatal adapter invariant"):
            AdjudicationEngine(Adapter(), max_batch_spans=1, max_concurrent_batches=2).adjudicate(spans(8))
    finally:
        release_first.set()
        helper.join(5)
    assert set(calls) == {"case-0", "case-1"}


def test_timed_out_multi_case_batch_is_recovered_once_as_individual_cases():
    calls = []
    class Adapter:
        def adjudicate(self, batch):
            calls.append([span.case_id for span in batch])
            if len(batch) > 1:
                raise ProviderError("Gemini request failed") from TimeoutError("bounded request expired")
            return [dict(case_id=batch[0].case_id, verdict="use_audio", final_text=batch[0].asr_text, confidence=.95, reason="Verified individually")]
    decisions, flags = AdjudicationEngine(Adapter(), max_batch_spans=2, max_concurrent_batches=2,
                                         retry_timed_out_batches=True).adjudicate(spans(2))
    assert [d.final_text for d in decisions] == ["spoken0", "spoken1"]
    assert calls[0] == ["case-0", "case-1"]
    assert sorted(calls[1:]) == [["case-0"], ["case-1"]]
    assert not any(f.kind == "llm_provider_unavailable" for f in flags)


def test_authentication_failure_is_not_retried_as_individual_paid_cases():
    calls = []
    class Adapter:
        def adjudicate(self, batch):
            calls.append(batch)
            raise ProviderError("Authentication failed", code="authentication")
    decisions, flags = AdjudicationEngine(Adapter(), retry_timed_out_batches=True).adjudicate(spans(2))
    assert len(calls) == 1
    assert all(d.verdict == "keep_srt" for d in decisions)
    assert any(f.kind == "llm_provider_unavailable" for f in flags)


@pytest.mark.parametrize("kwargs", [{"max_batch_spans":0}, {"max_batch_spans":26}, {"max_concurrent_batches":0}, {"max_concurrent_batches":5}, {"max_concurrent_batches":True}])
def test_invalid_batch_limits_reject_before_calls(kwargs):
    with pytest.raises(ValueError):
        AdjudicationEngine(object(), **kwargs)


def test_concurrent_snippet_batches_have_unique_owned_paths_and_complete_manifest(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"header")
    barrier = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    seen_paths = []
    def extractor(path, batch, directory, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{batch[0].case_id}.wav"
        target.write_bytes(b"unique audio bytes")
        return [AudioSnippet(case_id=batch[0].case_id, path=str(target), mime_type="audio/wav", start=batch[0].start, end=batch[0].end)]
    source = BoundedAudioSnippetBatchSource(audio, tmp_path/"snippets", pad_seconds=2, max_duration_seconds=20, max_snippets_per_batch=25, max_audio_duration_seconds=14400, extractor=extractor, max_concurrent_batches=4)
    def work(span):
        with source.load([span]) as batch:
            path = Path(batch[span.case_id].path)
            with lock:
                seen_paths.append(path)
            barrier.wait()
            assert all(path.exists() for path in seen_paths)
            barrier.wait()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, spans(4)))
    assert len({p.parent for p in seen_paths}) == 4
    assert all(not p.exists() for p in seen_paths)
    manifest = source.manifest()
    assert manifest["selected_count"] == 4
    assert {row["case_id"] for row in manifest["snippets"]} == {s.case_id for s in spans(4)}
