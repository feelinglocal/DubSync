"""A flaky or sloppy review response must not turn into permanent holds."""
from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dubsync import llm_providers, pipeline
from dubsync.hybrid_adjudication import HybridAdjudicationAdapter
from dubsync.models import AudioSnippet, DivergenceSpan
from dubsync.providers import ProviderError
from dubsync.srt_io import parse_srt_text


def _span(case_id="case-1", source="original words", asr="spoken words", start=1.0, end=2.0):
    return DivergenceSpan(case_id=case_id, cue_ids=[1], srt_text=source, asr_text=asr,
                          start=start, end=end, asr_word_indices=[0, 1])


def _decision(item, text=None, verdict="use_audio", confidence=.99):
    return dict(case_id=item.case_id, verdict=verdict,
                final_text=item.asr_text if text is None else text,
                confidence=confidence, reason="Recorded synthetic response")


def _clips(tmp_path, spans):
    result = {}
    for item in spans:
        path = tmp_path / f"{item.case_id}.wav"
        path.write_bytes(b"synthetic clip bytes")
        result[item.case_id] = AudioSnippet(case_id=item.case_id, path=str(path),
                                           start=item.start - .1, end=item.end + .1)
    return result


class _Primary:
    """A primary that always disagrees with the ASR, so every case is reviewed."""

    def adjudicate_with_audio(self, spans, snippets):
        return [_decision(item, "unsafe copied context") for item in spans]


def test_stray_case_id_in_a_review_response_does_not_hold_valid_decisions(tmp_path):
    # adjudication.md BUG-5: the review prompt lists read-only sibling case ids,
    # and one echoed sibling id held every case of the batch.
    first, second = _span(), _span("case-2", asr="other wording")

    def review(**_kwargs):
        return [_decision(first), _decision(second), _decision(_span("sibling-case"))], []

    adapter = HybridAdjudicationAdapter(_Primary(), review)
    output = adapter.adjudicate_with_audio([first, second], _clips(tmp_path, [first, second]))

    assert [(item["verdict"], item["final_text"]) for item in output] == [
        ("use_audio", "spoken words"), ("use_audio", "other wording"),
    ]
    report = adapter.route_report()
    assert report["counts"] == {"primary": 0, "fallback": 2, "held": 0, "review_requested": 2}
    assert all("review_stray_decision_ignored" in item["reasons"] for item in report["decisions"])


def test_stray_case_id_still_holds_the_case_without_its_own_decision(tmp_path):
    first, second = _span(), _span("case-2", asr="other wording")

    def review(**_kwargs):
        return [_decision(first), _decision(_span("sibling-case")), "not a decision"], []

    adapter = HybridAdjudicationAdapter(_Primary(), review)
    output = adapter.adjudicate_with_audio([first, second], _clips(tmp_path, [first, second]))

    assert output[0]["verdict"] == "use_audio"
    assert (output[1]["verdict"], output[1]["confidence"]) == ("keep_srt", 0.0)
    held = adapter.route_report()["decisions"][1]
    assert held["route"] == "held" and "review_missing_decision" in held["reasons"]


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


def test_review_provider_failure_is_not_cached_as_a_permanent_hold(tmp_path, monkeypatch):
    # adjudication.md BUG-4: the held result was cached, so every re-run reused it.
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:00,100 --> 00:00:00,800\nold source words\n", encoding="utf-8")
    audio = _wav(tmp_path / "episode.wav")
    wordstream = tmp_path / "words.json"
    wordstream.write_text(json.dumps({"words": [
        {"text": "different", "start": 0.1, "end": 0.3},
        {"text": "spoken", "start": 0.3, "end": 0.5},
        {"text": "wording", "start": 0.5, "end": 0.8},
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"adjudication": {
            "provider": "gemini", "model": "gemini-3.5-flash-lite", "thinking_level": "high", "api_key": "test-key",
            "audio_snippet_double_check": {"enabled": True}, "audio_context": {"enabled": False},
            "fallback": {"enabled": True, "provider": "gemini", "model": "gemini-3.8-flash", "thinking_level": "medium"},
        }},
    }), encoding="utf-8")
    review_calls: list[str] = []
    outage = {"active": True}

    def review(**context):
        review_calls.append("review")
        if outage["active"]:
            raise ProviderError("503 from the review model")
        return [dict(case_id=item.case_id, verdict="use_audio", final_text=item.asr_text,
                     confidence=0.99, reason="clear in the clip") for item in context["spans"]], []

    class Primary:
        def adjudicate_with_audio(self, spans, snippets):
            return [dict(case_id=item.case_id, verdict="keep_srt", final_text=item.srt_text,
                         confidence=1.0, reason="source") for item in spans]

    def extract(_audio, spans, directory, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        return [AudioSnippet(case_id=item.case_id, path=str(_wav(directory / f"{item.case_id}.wav")), start=0, end=1)
                for item in spans]

    monkeypatch.setattr(pipeline, "llm_adapter_from_config",
                        lambda *_args, **_kwargs: HybridAdjudicationAdapter(Primary(), review))
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    output = tmp_path / "output.srt"

    first = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=providers)

    assert parse_srt_text(output.read_text(encoding="utf-8"))[0].plain_text == "old source words"
    unavailable = [flag for flag in first.report["flags"] if flag["kind"] == "adjudication_review_unavailable"]
    assert len(unavailable) == 1
    assert "adjudication_review_unavailable" in pipeline._TRANSIENT_ADJUDICATION_FLAG_KINDS

    # The outage is over: the same job must ask again instead of replaying the hold.
    outage["active"] = False
    second = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=providers)

    assert review_calls == ["review", "review"]
    assert parse_srt_text(output.read_text(encoding="utf-8"))[0].plain_text == "different spoken wording"
    assert not [flag for flag in second.report["flags"] if flag["kind"] == "adjudication_review_unavailable"]


def test_an_unusable_review_reply_is_asked_once_more(tmp_path):
    # Fable review F21: an empty or invalid review reply carries no model
    # opinion. Like the direct route's schema retry, it is asked once more.
    item = _span()
    calls: list[list[str]] = []
    replies = [[], [_decision(item)]]

    def review(**kwargs):
        calls.append([span.case_id for span in kwargs["spans"]])
        return replies[len(calls) - 1], []

    adapter = HybridAdjudicationAdapter(_Primary(), review)
    output = adapter.adjudicate_with_audio([item], _clips(tmp_path, [item]))

    assert calls == [["case-1"], ["case-1"]]
    assert (output[0]["verdict"], output[0]["final_text"]) == ("use_audio", "spoken words")
    assert adapter.route_report()["counts"] == {"primary": 0, "fallback": 1, "held": 0, "review_requested": 1}


def test_the_review_retry_asks_only_cases_without_a_usable_decision(tmp_path):
    first, second = _span(), _span("case-2", asr="other wording")
    calls: list[list[str]] = []
    replies = [[_decision(first), dict(_decision(second), confidence="high")], [_decision(second)]]

    def review(**kwargs):
        calls.append([span.case_id for span in kwargs["spans"]])
        assert set(kwargs["audio_snippets"]) == set(calls[-1])
        assert set(kwargs["reasons"]) == set(calls[-1])
        return replies[len(calls) - 1], []

    adapter = HybridAdjudicationAdapter(_Primary(), review)
    output = adapter.adjudicate_with_audio([first, second], _clips(tmp_path, [first, second]))

    assert calls == [["case-1", "case-2"], ["case-2"]]
    assert [(item["verdict"], item["final_text"]) for item in output] == [
        ("use_audio", "spoken words"), ("use_audio", "other wording"),
    ]


_LITE, _FLASH = "gemini-3.5-flash-lite", "gemini-3.8-flash"
_UNUSABLE_REVIEW_REPLIES = {
    "truncated_json": '{"decisions": [{"case_id": "case-1", "verdict": "use_au',
    "empty_text": "",
    "omitted_case": json.dumps({"decisions": []}),
    "decision_without_evidence": json.dumps({"decisions": [{
        "case_id": "case-1", "verdict": "use_audio", "final_text": "different spoken wording",
        "speaker": None, "character": None, "reason": "clear"}]}),
}


@pytest.mark.parametrize("reply", sorted(_UNUSABLE_REVIEW_REPLIES))
def test_an_unusable_review_reply_is_a_transient_fault_and_never_a_cached_hold(tmp_path, monkeypatch, reply):
    # Fable review F21: the real hybrid factory turned these replies into a
    # cached confidence-0 hold that every later run replayed without asking.
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:00,100 --> 00:00:00,800\nold source words\n", encoding="utf-8")
    audio = _wav(tmp_path / "episode.wav")
    wordstream = tmp_path / "words.json"
    wordstream.write_text(json.dumps({"words": [
        {"text": "different", "start": 0.1, "end": 0.3},
        {"text": "spoken", "start": 0.3, "end": 0.5},
        {"text": "wording", "start": 0.5, "end": 0.8},
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"adjudication": {
            "provider": "gemini", "model": _LITE, "thinking_level": "high", "api_key": "test-key",
            "audio_snippet_double_check": {"enabled": True}, "audio_context": {"enabled": False},
            "fallback": {"enabled": True, "provider": "gemini", "model": _FLASH, "thinking_level": "medium"},
        }},
    }), encoding="utf-8")
    calls: list[str] = []
    broken = {"review": True}

    def generate(*, model, audio_snippets=None, **_kwargs):
        ids = [case_id for case_id in audio_snippets or {} if not case_id.endswith("-candidate")]
        role = "primary" if model == _LITE else "review"
        calls.append(role)
        usage = {"prompt_token_count": 10, "candidates_token_count": 5}
        if role == "review" and broken["review"]:
            return SimpleNamespace(text=_UNUSABLE_REVIEW_REPLIES[reply], usage_metadata=usage)
        text = "old source words" if role == "primary" else "different spoken wording"
        return SimpleNamespace(usage_metadata=usage, text=json.dumps({"decisions": [{
            "case_id": case_id, "verdict": "keep_srt" if role == "primary" else "use_audio",
            "final_text": text, "heard_text": text, "evidence": "heard_clearly",
            "speaker": None, "character": None, "reason": "heard in the clip"} for case_id in ids]}))

    def extract(_audio, spans, directory, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        return [AudioSnippet(case_id=item.case_id, path=str(_wav(directory / f"{item.case_id}.wav")), start=0, end=1)
                for item in spans]

    monkeypatch.setattr(llm_providers, "_gemini_generate_json", generate)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    output = tmp_path / "output.srt"

    first = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=providers)

    assert calls == ["primary", "review", "review"]
    assert parse_srt_text(output.read_text(encoding="utf-8"))[0].plain_text == "old source words"
    kinds = {flag["kind"] for flag in first.report["flags"]}
    assert "adjudication_review_unavailable" in kinds
    assert not list((first.episode_workdir / "llm-case-cache").glob("*.json"))
    assert not list((first.episode_workdir / "llm-cache").glob("*.json"))

    # The next run asks again and applies the actor's wording.
    broken["review"] = False
    calls.clear()
    second = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=providers)

    assert calls == ["primary", "review"]
    assert parse_srt_text(output.read_text(encoding="utf-8"))[0].plain_text == "different spoken wording"
    assert not [flag for flag in second.report["flags"] if flag["kind"] == "adjudication_review_unavailable"]
