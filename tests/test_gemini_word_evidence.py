from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dubsync.llm_providers import GeminiLLMAdapter, _adjudication_prompt, _adjudication_span_payload
from dubsync.models import AudioSnippet, DivergenceSpan, Word


def _words() -> list[Word]:
    return [
        Word(text="before", start=1490.0, end=1490.3, speaker_id="A"),
        Word(text="Eu", start=1491.8001, end=1492.2, speaker_id="A"),
        Word(text="quê?", start=1491.95, end=1492.4, speaker_id="B"),
    ]


def _span() -> DivergenceSpan:
    return DivergenceSpan(case_id="overlap", cue_ids=[548], srt_text="Eu", asr_text="Eu quê?",
                          asr_word_indices=[1, 2], speaker_ids=["A", "B"])


def test_span_word_evidence_preserves_local_order_overlap_and_individual_speakers():
    span = _span().model_copy(update={"asr_word_indices": [2, 0, 1], "prompt_scene_id": 3,
                                      "prompt_scene_position": 1})
    payload = _adjudication_span_payload(span, episode_words=_words())
    assert payload["asr_word_evidence"] == [
        {"word_index": 2, "text": "quê?", "start_seconds": 1491.95, "end_seconds": 1492.4, "speaker_id": "B"},
        {"word_index": 0, "text": "before", "start_seconds": 1490.0, "end_seconds": 1490.3, "speaker_id": "A"},
        {"word_index": 1, "text": "Eu", "start_seconds": 1491.8001, "end_seconds": 1492.2, "speaker_id": "A"},
    ]
    assert payload["scene_id"] == 3
    assert payload["scene_position"] == 1


def test_span_word_evidence_filters_invalid_indices_without_borrowing_neighbors():
    span = _span().model_copy(update={"asr_word_indices": [-1, True, "1", 1.2, 99, 2, 1]})
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        payload = _adjudication_span_payload(span, episode_words=_words())
    assert [item["word_index"] for item in payload["asr_word_evidence"]] == [2, 1]
    assert [item["text"] for item in payload["asr_word_evidence"]] == ["quê?", "Eu"]


def test_missing_words_preserve_existing_payload_and_empty_words_supply_no_invented_evidence():
    span = _span()
    assert _adjudication_span_payload(span) == span.model_dump()
    assert _adjudication_span_payload(span, episode_words=[])["asr_word_evidence"] == []
    assert "asr_word_evidence" not in json.loads(_adjudication_prompt([span]))["spans"][0]


def test_unknown_word_speaker_is_not_filled_from_span_speaker_list():
    words = _words()
    words[2].speaker_id = None
    evidence = _adjudication_span_payload(_span(), episode_words=words)["asr_word_evidence"]
    assert evidence[1]["speaker_id"] is None


@pytest.mark.parametrize("with_audio", [False, True])
def test_gemini_adjudication_uses_owned_word_snapshot_for_each_request(monkeypatch, with_audio):
    import dubsync.llm_providers as module

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"decisions": []}')

    monkeypatch.setattr(module, "_gemini_generate_json", generate)
    adapter = GeminiLLMAdapter(api_key="test", model="gemini-3.8-flash", thinking_level="medium")
    words = _words()
    adapter.set_episode_words(words)
    words[1].text = "caller changed this"
    words[1].speaker_id = "wrong speaker"
    words.clear()
    if with_audio:
        snippet = AudioSnippet(case_id="overlap", path="unused-by-stub.wav", start=1491.0, end=1493.0)
        adapter.adjudicate_with_audio([_span()], {"overlap": snippet})
    else:
        adapter.adjudicate([_span()])
    evidence = json.loads(calls[0]["prompt"])["spans"][0]["asr_word_evidence"]
    assert [item["text"] for item in evidence] == ["Eu", "quê?"]
    assert [item["speaker_id"] for item in evidence] == ["A", "B"]
    assert evidence[0]["start_seconds"] == 1491.8001
