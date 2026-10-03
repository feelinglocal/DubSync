from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import pytest

from dubsync.llm_providers import _adjudication_review_prompt, llm_adapter_from_config
from dubsync.models import AudioSnippet, Cue, DivergenceSpan, Word


@pytest.fixture(autouse=True)
def block_external_review_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Hybrid safety regressions must never make external requests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    try:
        from google import genai
    except ImportError:
        return
    monkeypatch.setattr(genai, "Client", forbidden)


def config():
    return {"llm": {"adjudication": {
        "provider": "gemini", "model": "gemini-3.5-flash-lite", "api_key": "synthetic-key",
        "thinking_level": "high", "audio_context": {"enabled": False},
        "audio_snippet_double_check": {"enabled": True},
        "fallback": {"enabled": True, "provider": "gemini", "model": "gemini-3.8-flash",
                     "thinking_level": "medium"},
    }}}


@pytest.mark.parametrize("stage", ["primary", "fallback"])
@pytest.mark.parametrize("invalid_confidence", [True, "0.99"])
def test_real_factory_does_not_turn_malformed_confidence_into_a_confident_approval(
    monkeypatch, tmp_path, stage, invalid_confidence,
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        primary = kwargs["model"] == "gemini-3.5-flash-lite"
        # A valid source keep deliberately selects fallback. An invalid primary
        # approval agrees with ASR and must still be escalated as malformed.
        source_keep = primary and stage == "fallback"
        reply = {"case_id": "one", "verdict": "keep_srt" if source_keep else "use_audio",
                 "final_text": "No" if source_keep else "Yes",
                 "heard_text": "No" if source_keep else "Yes", "evidence": "heard_clearly",
                 "reason": "Synthetic recorded evidence"}
        # v12 never requests confidence. A malformed legacy field cannot make
        # an otherwise native-looking response into a valid approval.
        if primary == (stage == "primary"):
            reply["confidence"] = invalid_confidence
        return SimpleNamespace(text=json.dumps({"decisions": [reply]}), usage_metadata={
            "prompt_token_count": 10, "candidates_token_count": 5,
        })

    monkeypatch.setattr("dubsync.llm_providers._gemini_generate_json", generate)
    adapter = llm_adapter_from_config(config(), "adjudication")
    item = DivergenceSpan(case_id="one", cue_ids=[1], srt_text="No", asr_text="Yes",
                          srt_token_indices=[0], asr_word_indices=[0], start=1, end=2)
    adapter.set_episode_context([Cue(index=1, start_ms=1000, end_ms=2000, lines=["No"])])
    adapter.set_episode_words([Word(text="Yes", start=1, end=2)])
    clip = tmp_path / "focused.wav"
    clip.write_bytes(b"Synthetic local fixture; no external upload")

    result = adapter.adjudicate_with_audio([item], {
        item.case_id: AudioSnippet(case_id=item.case_id, path=str(clip), start=0.8, end=2.2),
    })

    # A malformed review decision is asked once more (Fable review F21).
    assert len(calls) == (2 if stage == "primary" else 3)
    if stage == "primary":
        assert result[0]["reason"].startswith("[hybrid:fallback]")
        assert result[0]["final_text"] == "Yes"
    else:
        assert result[0]["reason"].startswith("[hybrid:held]")
        assert result[0]["verdict"] == "keep_srt"
        assert result[0]["final_text"] == "No"
        assert result[0]["confidence"] == 0
    # Rejected structured responses still incurred separately attributed usage.
    assert sorted(event["adjudication_route"] for event in adapter.drain_usage_events()) == (
        ["fallback", "primary"] if stage == "primary" else ["fallback", "fallback", "primary"])


def test_local_review_context_keeps_original_indices_for_repeated_source_words():
    cues = [Cue(index=index, start_ms=index * 1000, end_ms=index * 1000 + 900,
                lines=["Go go go now." if index == 5 else f"context{index}"])
            for index in range(1, 8)]
    item = DivergenceSpan(case_id="repeat", cue_ids=[5], srt_text="Go", asr_text="Oh",
                          srt_token_indices=[4], asr_word_indices=[0], start=5, end=5.2)
    payload = json.loads(_adjudication_review_prompt(
        spans=[item], audio_snippets={item.case_id: AudioSnippet(
            case_id=item.case_id, path="unused-prompt-only.wav", start=4.8, end=5.4,
        )}, reasons={item.case_id: ["wording_differs_from_owned_asr"]}, primary_decisions={},
        batch_spans=[item], episode_context=cues,
        episode_words=[Word(text="Oh", start=5, end=5.2)], confidence_gate=0.7,
    ))

    assert [cue["cue_id"] for cue in payload["episode_context"]] == [3, 4, 5, 6, 7]
    assert payload["spans"][0]["srt_token_indices"] == [4]
    assert payload["review_cases"][0]["source_token_ownership"] == [
        {"token_index": 4, "cue_id": 5, "text": "Go", "editable_here": True},
        {"token_index": 5, "cue_id": 5, "text": "go", "editable_here": False},
        {"token_index": 6, "cue_id": 5, "text": "go", "editable_here": False},
        {"token_index": 7, "cue_id": 5, "text": "now", "editable_here": False},
    ]
