"""A flaky or sloppy review response must not turn into permanent holds."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import yaml

from dubsync import pipeline
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
