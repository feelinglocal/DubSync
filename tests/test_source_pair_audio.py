from pathlib import Path
import wave

import pytest

from dubsync.models import AudioSnippet
from dubsync.providers import ProviderError
from dubsync.source_pair_audio import SourcePairAudioAdapter
from test_source_pair_timing import _case, _questions


def _wave(path, duration):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * round(duration * 16000))


def _setup(tmp_path, fault=None):
    question = _questions(_case())[0]
    seen = []
    class Native:
        def adjudicate_with_audio(self, spans, snippets):
            seen.append((spans, snippets))
            return []
    def extract(audio, spans, directory, **kwargs):
        assert kwargs["pad_seconds"] == 0 and len(spans) == 1
        assert audio == tmp_path / "original.wav"
        directory.mkdir(parents=True, exist_ok=True)
        span = spans[0]
        if fault == "missing":
            return []
        path = directory / "candidate.wav"
        _wave(path, span.end - span.start - (.1 if fault == "truncated" else 0))
        return [AudioSnippet(case_id=span.case_id, path=str(path),
            start=span.start + (.1 if fault == "bounds" else 0), end=span.end)]
    adapter = SourcePairAudioAdapter(Native(), [question], tmp_path / "original.wav", tmp_path / "clips", extractor=extract)
    full = tmp_path / "full.wav"
    _wave(full, question.span.end - question.span.start)
    context = AudioSnippet(case_id=question.span.case_id, path=str(full), start=question.span.start, end=question.span.end)
    return question, adapter, context, seen


def test_pair_audio_binds_an_exact_candidate_excerpt_and_keeps_full_context(tmp_path):
    q, adapter, context, seen = _setup(tmp_path)
    span = q.span.model_copy(update={"prompt_scene_id": 1, "prompt_scene_position": 1})
    adapter.adjudicate_with_audio([span], {q.span.case_id: context})
    sent = seen[0][1]
    assert set(sent) == {q.span.case_id, q.span.case_id + "-candidate"}
    assert sent[q.span.case_id] == context
    candidate = sent[q.span.case_id + "-candidate"]
    assert (candidate.start, candidate.end) == (41.360, 42.465)
    records = adapter.manifest()
    assert records[0]["case_id"] == q.span.case_id
    assert records[0]["candidate_audio_id"] == candidate.case_id
    assert len(records[0]["sha256"]) == 64 and records[0]["frames"] == 17680


@pytest.mark.parametrize("fault", ["missing", "truncated", "bounds"])
def test_pair_audio_never_calls_provider_without_its_complete_candidate(tmp_path, fault):
    q, adapter, context, seen = _setup(tmp_path, fault)
    with pytest.raises(ProviderError, match="candidate"):
        adapter.adjudicate_with_audio([q.span], {q.span.case_id: context})
    assert seen == []


def test_pair_audio_does_not_attach_another_cases_candidate(tmp_path):
    q, adapter, context, seen = _setup(tmp_path)
    other = q.span.model_copy(update={"case_id": "ordinary-other"})
    adapter.adjudicate_with_audio([other], {other.case_id: context})
    assert set(seen[0][1]) == {other.case_id} and adapter.manifest() == []


def test_hybrid_primary_and_selected_reviewer_receive_the_same_two_clips(tmp_path):
    from dubsync.hybrid_adjudication import HybridAdjudicationAdapter
    from test_source_pair_timing import _decision
    q, wrapper, context, primary_seen = _setup(tmp_path)
    reviewed = []
    def review(**kwargs):
        reviewed.append(kwargs)
        return [_decision(q).model_dump()], []
    wrapper.adapter = HybridAdjudicationAdapter(wrapper.adapter, review)
    result = wrapper.adjudicate_with_audio([q.span], {q.span.case_id: context})
    expected = {q.span.case_id, q.span.case_id + "-candidate"}
    assert set(primary_seen[0][1]) == expected
    assert set(reviewed[0]["audio_snippets"]) == expected
    assert result[0]["source_pair_evidence"]["candidate_audio_id"] == q.span.case_id + "-candidate"


def test_ordinary_decision_dump_and_strict_candidate_field_remain_separate():
    from pydantic import ValidationError
    from dubsync.models import AdjudicationDecision
    from test_source_pair_timing import _decision
    q = _questions(_case())[0]
    positive = _decision(q).model_dump()
    assert positive["source_pair_evidence"]["candidate_complete"] is True
    ordinary = {key: value for key, value in positive.items() if key != "source_pair_evidence"}
    assert AdjudicationDecision.model_validate(ordinary).model_dump() == ordinary
    positive["source_pair_evidence"]["candidate_complete"] = "true"
    with pytest.raises(ValidationError):
        AdjudicationDecision.model_validate(positive)
