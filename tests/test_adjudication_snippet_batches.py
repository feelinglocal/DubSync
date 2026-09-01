from __future__ import annotations

from pathlib import Path

import pytest

from dubsync.adjudication_snippets import BoundedAudioSnippetBatchSource
from dubsync.audio_snippets import AudioSnippetError
from dubsync.models import AudioSnippet, DivergenceSpan


def _span(index: int) -> DivergenceSpan:
    return DivergenceSpan(
        case_id=f"case-{index}",
        cue_ids=[index],
        srt_text=f"source {index}",
        asr_text=f"spoken {index}",
        start=float(index),
        end=float(index) + 0.5,
    )


def test_bounded_source_streams_every_batch_and_removes_transient_audio(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"normalized episode audio")
    extraction_calls: list[list[str]] = []
    extraction_options: list[dict[str, object]] = []
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 10.0)

    def fake_extract(_audio_path, spans, output_dir, **kwargs):
        extraction_calls.append([span.case_id for span in spans])
        extraction_options.append(dict(kwargs))
        snippets: list[AudioSnippet] = []
        for span in spans:
            snippet_path = output_dir / f"{span.case_id}.wav"
            snippet_path.parent.mkdir(parents=True, exist_ok=True)
            snippet_path.write_bytes(f"audio for {span.case_id}".encode())
            snippets.append(
                AudioSnippet(
                    case_id=span.case_id,
                    path=str(snippet_path),
                    start=span.start or 0.0,
                    end=span.end or 0.0,
                )
            )
        return snippets

    source = BoundedAudioSnippetBatchSource(
        audio_path,
        tmp_path / "streamed-snippets",
        pad_seconds=2.0,
        max_duration_seconds=20.0,
        max_snippets_per_batch=4,
        max_audio_duration_seconds=60.0,
        extractor=fake_extract,
    )
    batches = [[_span(index) for index in range(1, 26)], [_span(index) for index in range(26, 31)]]

    for batch in batches:
        with source.load(batch) as snippets:
            assert list(snippets) == [span.case_id for span in batch]
            assert all(Path(snippet.path).is_file() for snippet in snippets.values())
        assert all(not Path(snippet.path).exists() for snippet in snippets.values())

    assert extraction_calls == [
        [f"case-{index}" for index in range(1, 26)],
        [f"case-{index}" for index in range(26, 31)],
    ]
    assert {options["max_snippets"] for options in extraction_options} == {4}
    assert {options["fail_on_budget_exceeded"] for options in extraction_options} == {False}
    manifest = source.manifest()
    assert manifest["storage_mode"] == "bounded_batches"
    assert manifest["max_snippets_per_batch"] == 4
    assert manifest["candidate_count"] == 30
    assert manifest["selected_count"] == 30
    assert manifest["fallback_count"] == 0
    assert [item["case_id"] for item in manifest["snippets"]] == [f"case-{index}" for index in range(1, 31)]
    assert source.flags() == []
    assert source.cache_context()["audio_sha256"]
    assert source.cache_context()["max_snippets_per_batch"] == 4
    assert source.cache_context()["max_audio_duration_seconds"] == 60.0


@pytest.mark.parametrize(
    "failure",
    [
        AudioSnippetError("Audio snippets would exceed the job storage budget"),
        OSError(1455, "The paging file is too small for this operation to complete"),
    ],
)
def test_bounded_source_degrades_only_a_failed_optional_batch(tmp_path, monkeypatch, failure):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"normalized episode audio")
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 10.0)

    def failing_extract(*_args, **_kwargs):
        raise failure

    source = BoundedAudioSnippetBatchSource(
        audio_path,
        tmp_path / "streamed-snippets",
        pad_seconds=2.0,
        max_duration_seconds=20.0,
        max_snippets_per_batch=4,
        max_audio_duration_seconds=60.0,
        extractor=failing_extract,
    )
    spans = [_span(1), _span(2)]

    with source.load(spans) as snippets:
        assert snippets == {}

    manifest = source.manifest()
    assert manifest["candidate_count"] == 2
    assert manifest["selected_count"] == 0
    assert manifest["fallback_count"] == 2
    assert manifest["fallback_case_ids"] == ["case-1", "case-2"]
    assert [flag.kind for flag in source.flags()] == ["audio_snippet_unavailable"]
    assert source.flags()[0].cue_ids == [1, 2]


def test_bounded_source_skips_optional_snippets_for_long_audio(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"normalized episode audio")
    extraction_calls: list[object] = []
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 45 * 60.0)

    def unexpected_extract(*args, **kwargs):
        extraction_calls.append((args, kwargs))
        return []

    source = BoundedAudioSnippetBatchSource(
        audio_path,
        tmp_path / "streamed-snippets",
        pad_seconds=2.0,
        max_duration_seconds=20.0,
        max_snippets_per_batch=4,
        max_audio_duration_seconds=30 * 60.0,
        extractor=unexpected_extract,
    )
    spans = [_span(1), _span(2)]

    with source.load(spans) as snippets:
        assert snippets == {}

    assert extraction_calls == []
    manifest = source.manifest()
    assert manifest["candidate_count"] == 2
    assert manifest["selected_count"] == 0
    assert manifest["fallback_count"] == 2
    assert manifest["max_audio_duration_seconds"] == 30 * 60.0


def test_bounded_source_marks_unselected_capped_spans_as_text_only(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"normalized episode audio")
    monkeypatch.setattr("dubsync.adjudication_snippets.audio_seconds", lambda _path: 10.0)

    def capped_extract(_audio_path, spans, output_dir, **_kwargs):
        first = spans[0]
        snippet_path = output_dir / f"{first.case_id}.wav"
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet_path.write_bytes(b"audio for first case only")
        return [
            AudioSnippet(
                case_id=first.case_id,
                path=str(snippet_path),
                start=first.start or 0.0,
                end=first.end or 0.0,
            )
        ]

    source = BoundedAudioSnippetBatchSource(
        audio_path,
        tmp_path / "streamed-snippets",
        pad_seconds=2.0,
        max_duration_seconds=20.0,
        max_snippets_per_batch=1,
        max_audio_duration_seconds=60.0,
        extractor=capped_extract,
    )

    with source.load([_span(1), _span(2), _span(3)]) as snippets:
        assert list(snippets) == ["case-1"]

    manifest = source.manifest()
    assert manifest["selected_count"] == 1
    assert manifest["fallback_count"] == 2
    assert manifest["fallback_case_ids"] == ["case-2", "case-3"]
    assert source.flags()[0].cue_ids == [2, 3]
