from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Lock

import pytest

from dubsync.hybrid_adjudication import HybridAdjudicationAdapter, triage_decisions
from dubsync.models import AudioSnippet, Cue, DivergenceSpan, Word
from dubsync.providers import ProviderError


def span(case_id="case-1", source="original words", asr="spoken words", start=1.0, end=2.0):
    return DivergenceSpan(case_id=case_id, cue_ids=[1], srt_text=source, asr_text=asr,
                          start=start, end=end, asr_word_indices=[0, 1])


def decision(item, text=None, verdict="use_audio", confidence=.99):
    return dict(case_id=item.case_id, verdict=verdict,
                final_text=item.asr_text if text is None else text,
                confidence=confidence, reason="Recorded synthetic response")


def clips(tmp_path, spans):
    result = {}
    for item in spans:
        path = tmp_path / f"{item.case_id}.wav"
        path.write_bytes(b"synthetic clip bytes")
        result[item.case_id] = AudioSnippet(case_id=item.case_id, path=str(path),
                                           start=item.start - .1, end=item.end + .1)
    return result


class Primary:
    def __init__(self, replies=None, error=None):
        self.replies, self.error = replies, error
        self.calls, self.events = [], []
        self.lock = Lock()
        self.closed = False

    def adjudicate_with_audio(self, spans, snippets):
        with self.lock:
            self.calls.append(([s.case_id for s in spans], dict(snippets)))
            self.events.append({"usage_metadata": {"prompt_token_count": 1}})
        if self.error:
            raise self.error
        return deepcopy(self.replies) if self.replies is not None else [decision(s) for s in spans]

    def drain_usage_events(self):
        with self.lock:
            events, self.events = self.events, []
            return events

    def set_episode_context(self, cues):
        self.context = cues

    def set_episode_words(self, words):
        self.words = words

    def set_audio_context(self, path, **kwargs):
        self.audio_context_args = (path, kwargs)

    def audio_context_report(self):
        return {"enabled": False, "primary_report": True}

    def close(self):
        self.closed = True


def forbidden_review(**kwargs):
    raise AssertionError("This case must not call the reviewer")


@pytest.mark.parametrize("text,asr", [("SPOKEN, words!", "spoken words"), ("Olá, olá!", "Olá olá")])
def test_owned_asr_agreement_preserves_punctuation_and_real_repetitions(tmp_path, text, asr):
    item = span(asr=asr)
    primary = Primary([decision(item, text)])
    adapter = HybridAdjudicationAdapter(primary, forbidden_review)
    result = adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))
    assert result[0]["final_text"] == text
    assert result[0]["reason"].startswith("[hybrid:primary]")
    assert adapter.route_report()["counts"]["primary"] == 1


@pytest.mark.parametrize("source,asr,reply", [
    ("pode Não pode", "", "Não. De jeito nenhum."),
    ("vai ser destruída Não sei o", "Que", "Que que eu fiz de errado?"),
])
def test_partial_edit_copying_retained_neighbor_is_reviewed_not_trimmed(tmp_path, source, asr, reply):
    item = span(source=source, asr=asr)
    called = []
    def review(**kwargs):
        called.append(kwargs)
        assert kwargs["primary_decisions"][item.case_id]["final_text"] == reply
        assert kwargs["reasons"][item.case_id] == ["wording_differs_from_owned_asr"]
        return [decision(item)], []
    result = HybridAdjudicationAdapter(Primary([decision(item, reply)]), review).adjudicate_with_audio(
        [item], clips(tmp_path, [item]))
    assert len(called) == 1
    assert result[0]["final_text"] == asr
    assert result[0]["reason"].startswith("[hybrid:fallback]")


def test_high_confidence_wrong_source_keep_is_escalated(tmp_path):
    item = span(source="estão com ele não pode denunciar", asr="E")
    def review(**kwargs):
        assert kwargs["reasons"][item.case_id] == ["source_differs_from_owned_asr"]
        return [decision(item)], []
    adapter = HybridAdjudicationAdapter(Primary([decision(item, item.srt_text, "keep_srt", 1)]), review)
    result = adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))
    assert result[0]["final_text"] == "E"


def test_source_keep_agreement_is_not_sent_for_review(tmp_path):
    item = span(source="spoken words", asr="spoken words")
    result = HybridAdjudicationAdapter(Primary([decision(item, item.srt_text, "keep_srt")]), forbidden_review)
    assert result.adjudicate_with_audio([item], clips(tmp_path, [item]))[0]["verdict"] == "keep_srt"


@pytest.mark.parametrize("payload,reason", [
    ([], "primary_missing_decision"),
    ([{"case_id": "case-1"}], "primary_invalid_decision"),
    ([{"case_id": "other"}], "primary_unexpected_case_id"),
])
def test_pure_triage_rejects_unbound_or_invalid_ids(payload, reason):
    assert reason in triage_decisions([span()], payload)["case-1"]


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 1.1, True, "0.99"])
def test_primary_confidence_must_be_finite_numeric_probability(confidence):
    item = span()
    assert "primary_invalid_decision" in triage_decisions([item], [decision(item, confidence=confidence)])[item.case_id]


def test_low_confidence_exact_agreement_still_requires_review():
    item = span()
    assert triage_decisions([item], [decision(item, confidence=.69)])[item.case_id] == ["low_primary_confidence"]


def test_duplicate_primary_id_is_never_last_write_wins():
    item = span()
    assert "primary_duplicate_decision" in triage_decisions([item], [decision(item), decision(item)])[item.case_id]


@pytest.mark.parametrize("gate", [0, -1, 1.1, float("nan"), float("inf"), True, "0.7"])
def test_invalid_gate_cannot_make_source_holds_pass_confidence(gate):
    with pytest.raises(ProviderError):
        HybridAdjudicationAdapter(Primary(), forbidden_review, confidence_gate=gate)


def test_response_position_is_never_a_case_alias(tmp_path):
    first, second = span(), span("case-2", asr="other wording")
    def review(**kwargs):
        return [decision(second), decision(first)], []
    primary = Primary([decision(first, "wrong"), decision(second, "wrong")])
    output = HybridAdjudicationAdapter(primary, review).adjudicate_with_audio(
        [first, second], clips(tmp_path, [first, second]))
    assert [d["case_id"] for d in output] == [first.case_id, second.case_id]
    assert [d["final_text"] for d in output] == [first.asr_text, second.asr_text]


def test_duplicate_input_ids_fail_before_any_provider_call(tmp_path):
    item = span()
    primary = Primary()
    with pytest.raises(ValueError):
        HybridAdjudicationAdapter(primary, forbidden_review).adjudicate_with_audio(
            [item, item], clips(tmp_path, [item]))
    assert primary.calls == []


def test_malformed_primary_batch_is_reviewed_without_unbound_decisions(tmp_path):
    item = span()
    def review(**kwargs):
        assert kwargs["primary_decisions"] == {}
        assert kwargs["reasons"] == {item.case_id: ["primary_invalid_batch"]}
        return [decision(item)], []
    adapter = HybridAdjudicationAdapter(Primary({"unbound": "response"}), review)
    assert adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))[0]["verdict"] == "use_audio"


def test_partial_batch_keeps_missing_clip_held_while_reviewing_available_case(tmp_path):
    first, second = span(), span("case-2", start=3, end=4)
    def review(**kwargs):
        assert [s.case_id for s in kwargs["spans"]] == [first.case_id]
        assert set(kwargs["audio_snippets"]) == {first.case_id}
        assert [s.case_id for s in kwargs["batch_spans"]] == [first.case_id, second.case_id]
        return [decision(first)], []
    primary = Primary([decision(first, "wrong")])
    adapter = HybridAdjudicationAdapter(primary, review)
    output = adapter.adjudicate_with_audio([first, second], clips(tmp_path, [first]))
    assert primary.calls[0][0] == [first.case_id]
    assert output[0]["verdict"] == "use_audio"
    assert output[1]["verdict"] == "keep_srt" and output[1]["confidence"] == 0
    assert adapter.route_report()["counts"] == {"primary": 0, "fallback": 1, "held": 1, "review_requested": 1}


@pytest.mark.parametrize("bounds", [{"start": None}, {"end": None}, {"start": float("nan")},
                                   {"end": float("inf")}, {"end": 1}, {"start": -1}])
def test_invalid_case_window_cannot_authorize_complete_clip(tmp_path, bounds):
    item = span()
    snippets = clips(tmp_path, [item])
    item = item.model_copy(update=bounds)
    primary = Primary()
    output = HybridAdjudicationAdapter(primary, forbidden_review).adjudicate_with_audio([item], snippets)
    assert output[0]["verdict"] == "keep_srt" and output[0]["confidence"] == 0
    assert primary.calls == []


def test_empty_input_makes_no_calls_and_can_close_unadorned_adapter():
    adapter = HybridAdjudicationAdapter(object(), forbidden_review)
    assert adapter.adjudicate([]) == []
    assert adapter.drain_usage_events() == []
    assert not adapter.audio_context_report()["enabled"]
    adapter.set_episode_context([])
    adapter.set_episode_words([])
    adapter.set_audio_context("unused", duration_seconds=1)
    adapter.close()


def test_primary_without_audio_method_uses_configured_fallback_once(tmp_path):
    item = span()
    seen = []
    def review(**kwargs):
        seen.append(kwargs)
        assert kwargs["reasons"] == {item.case_id: ["primary_provider_failure"]}
        return [decision(item)], []
    adapter = HybridAdjudicationAdapter(object(), review)
    assert adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))[0]["verdict"] == "use_audio"
    assert len(seen) == 1


# An extra decision for an unrelated id no longer holds a case that has exactly
# one valid decision of its own; see tests/test_hybrid_review_robustness.py.
@pytest.mark.parametrize("failure", ["missing", "duplicate", "invalid", "low_confidence", "provider"])
def test_invalid_or_failed_review_preserves_source_once(tmp_path, failure):
    item = span()
    calls = []
    def review(**kwargs):
        calls.append(kwargs)
        if failure == "provider":
            raise ProviderError("Synthetic provider error")
        data = {"missing": [], "duplicate": [decision(item), decision(item)],
                "invalid": [dict(decision(item), confidence=float("nan"))],
                "low_confidence": [decision(item, confidence=.6)]}[failure]
        return data, [{"usage_metadata": {"prompt_token_count": 2}}]
    primary = Primary([decision(item, "unsafe copied context")])
    adapter = HybridAdjudicationAdapter(primary, review)
    result = adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))[0]
    assert result["verdict"] == "keep_srt"
    assert result["final_text"] == item.srt_text
    assert result["confidence"] == 0
    assert result["reason"].startswith("[hybrid:held]")
    # A reply without one usable decision is asked once more (Fable review F21).
    assert len(primary.calls) == 1
    assert len(calls) == (2 if failure in {"missing", "duplicate", "invalid"} else 1)


def test_missing_review_id_holds_only_that_case(tmp_path):
    first, second = span(), span("case-2")
    def review(**kwargs):
        return [decision(first)], []
    adapter = HybridAdjudicationAdapter(Primary([decision(s, "wrong") for s in [first, second]]), review)
    result = adapter.adjudicate_with_audio([first, second], clips(tmp_path, [first, second]))
    assert result[0]["verdict"] == "use_audio"
    assert result[1]["verdict"] == "keep_srt"


@pytest.mark.parametrize("bad", ["missing", "wrong_id", "partial", "nan", "zero", "no_file"])
def test_unavailable_case_audio_cannot_authorize_primary_or_review(tmp_path, bad):
    item = span()
    snippets = clips(tmp_path, [item])
    if bad == "missing":
        snippets = {}
    elif bad == "no_file":
        snippets[item.case_id] = snippets[item.case_id].model_copy(update={"path": str(tmp_path / "absent.wav")})
    else:
        change = {"wrong_id": {"case_id": "other"}, "partial": {"start": 1.5},
                  "nan": {"end": float("nan")}, "zero": {"end": .9}}[bad]
        snippets[item.case_id] = snippets[item.case_id].model_copy(update=change)
    primary = Primary()
    result = HybridAdjudicationAdapter(primary, forbidden_review).adjudicate_with_audio([item], snippets)[0]
    assert result["verdict"] == "keep_srt" and result["confidence"] == 0
    assert "audio" in result["reason"]
    assert primary.calls == []


def test_text_only_entry_point_never_approves_without_case_audio():
    primary = Primary()
    result = HybridAdjudicationAdapter(primary, forbidden_review).adjudicate([span()])[0]
    assert result["verdict"] == "keep_srt" and result["confidence"] == 0
    assert primary.calls == []


def test_primary_provider_failure_routes_once_to_configured_fallback(tmp_path):
    item = span()
    seen = []
    def review(**kwargs):
        seen.append(kwargs)
        assert kwargs["primary_decisions"] == {}
        assert kwargs["reasons"] == {item.case_id: ["primary_provider_failure"]}
        return [decision(item)], [{"usage_metadata": {"prompt_token_count": 2}}]
    primary = Primary(error=ProviderError("Primary parser failed"))
    adapter = HybridAdjudicationAdapter(primary, review)
    assert adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))[0]["verdict"] == "use_audio"
    assert len(primary.calls) == len(seen) == 1
    assert [e["adjudication_route"] for e in adapter.drain_usage_events()] == ["primary", "fallback"]


@pytest.mark.parametrize("stage", ["primary", "review"])
def test_unexpected_or_budget_exceptions_are_not_hidden(tmp_path, stage):
    class BudgetStop(RuntimeError):
        pass
    item = span()
    primary = Primary([decision(item, "wrong")], error=BudgetStop("stop") if stage == "primary" else None)
    def review(**kwargs):
        raise BudgetStop("stop")
    adapter = HybridAdjudicationAdapter(primary, review)
    with pytest.raises(BudgetStop):
        adapter.adjudicate_with_audio([item], clips(tmp_path, [item]))
    assert len(adapter.drain_usage_events()) == 1


def test_reviewer_receives_only_selected_exact_clips_and_read_only_copies(tmp_path):
    first, second = span(), span("case-2", start=3, end=4)
    snippets = clips(tmp_path, [first, second])
    primary = Primary([decision(first), decision(second, "wrong")])
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["source context"])
    word = Word(text="spoken", start=1, end=1.5)
    seen = []
    def review(**kwargs):
        seen.append(kwargs)
        assert [s.case_id for s in kwargs["spans"]] == [second.case_id]
        assert set(kwargs["audio_snippets"]) == {second.case_id}
        assert kwargs["audio_snippets"][second.case_id].model_dump() == snippets[second.case_id].model_dump()
        assert [s.case_id for s in kwargs["batch_spans"]] == [first.case_id, second.case_id]
        assert kwargs["episode_context"][0].plain_text == "source context"
        assert kwargs["episode_words"][0].text == "spoken"
        for obj, field, value in [(kwargs["episode_context"][0], "index", 999),
                                  (kwargs["episode_words"][0], "text", "changed"),
                                  (kwargs["batch_spans"][0], "asr_text", "changed")]:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                setattr(obj, field, value)
        assert "audio_context" not in kwargs
        return [decision(second)], []
    adapter = HybridAdjudicationAdapter(primary, review)
    adapter.set_episode_context([cue])
    adapter.set_episode_words([word])
    cue.lines[0], word.text = "later mutation", "later mutation"
    adapter.set_audio_context(tmp_path / "episode.wav", duration_seconds=100, config={"enabled": False})
    adapter.adjudicate_with_audio([first, second], snippets)
    assert len(seen) == 1
    assert primary.audio_context_args[1]["duration_seconds"] == 100
    assert primary.context[0].plain_text == "source context"
    assert primary.words[0].text == "spoken"
    assert first.asr_text == "spoken words"
    report = adapter.audio_context_report()
    assert report["primary_report"] and report["hybrid_routes"]["counts"]["fallback"] == 1
    report["hybrid_routes"]["decisions"].clear()
    assert len(adapter.route_report()["decisions"]) == 2
    adapter.close()
    assert primary.closed


def test_concurrent_batches_drain_each_usage_event_once(tmp_path):
    items = [span(f"case-{i}", start=i * 3 + 1, end=i * 3 + 2) for i in range(4)]
    snippets = clips(tmp_path, items)
    barrier = Barrier(4)
    class ConcurrentPrimary(Primary):
        def adjudicate_with_audio(self, spans, audio_snippets):
            super().adjudicate_with_audio(spans, audio_snippets)
            barrier.wait(timeout=5)
            return [decision(spans[0], "wrong")]
    def review(**kwargs):
        item = kwargs["spans"][0]
        return [decision(item)], [{"usage_metadata": {"prompt_token_count": 2}, "id": item.case_id}]
    adapter = HybridAdjudicationAdapter(ConcurrentPrimary(), review)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda s: adapter.adjudicate_with_audio([s], {s.case_id: snippets[s.case_id]}), items))
    assert all(result[0]["verdict"] == "use_audio" for result in outputs)
    events = adapter.drain_usage_events()
    assert sum(e["adjudication_route"] == "primary" for e in events) == 4
    assert sum(e["adjudication_route"] == "fallback" for e in events) == 4
    assert adapter.drain_usage_events() == []
    assert adapter.route_report()["counts"] == {"primary": 0, "fallback": 4, "held": 0, "review_requested": 4}
