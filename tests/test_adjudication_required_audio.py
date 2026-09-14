from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import wave

import pytest
import yaml

from dubsync.adjudication import AdjudicationEngine
from dubsync.models import AudioSnippet, DivergenceSpan
from dubsync import pipeline
from dubsync.providers import ProviderError


@pytest.fixture(autouse=True)
def _disable_unrelated_provider_passes(monkeypatch):
    # The pipeline has separate provider factories for its later passes.
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)
    try:
        from google import genai
    except ImportError:
        pass
    else:
        def forbidden_client(*_args, **_kwargs):
            raise AssertionError("Required-audio regression tests must never construct a real Gemini client")
        monkeypatch.setattr(genai, "Client", forbidden_client)


def _span(case_id="case-1", start=0.1, end=0.2):
    return DivergenceSpan(
        case_id=case_id, cue_ids=[int(case_id[-1]) if case_id.startswith("case-") else 1],
        srt_text="old source words",
        asr_text="different spoken wording", start=start, end=end, confidence=0.98,
    )


def _wav(path: Path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


def _snippet(tmp_path, span, *, case_id=None):
    return AudioSnippet(
        case_id=case_id or span.case_id,
        path=str(_wav(tmp_path / f"{span.case_id}.wav")),
        start=0, end=1,
    )


def _loader(snippets):
    @contextmanager
    def load(_spans):
        yield snippets
    return load


class _TextAdapter:
    def __init__(self):
        self.calls = []
        self.source_context = []

    @staticmethod
    def _responses(spans):
        return [dict(
            case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
            confidence=0.99, reason="test provider would approve the supplied wording",
        ) for span in spans]

    def adjudicate(self, spans):
        self.calls.append(("text", [span.case_id for span in spans]))
        return self._responses(spans)

    def set_episode_context(self, cues):
        self.source_context = list(cues)


class _AudioAdapter(_TextAdapter):
    def adjudicate_with_audio(self, spans, snippets):
        assert all(snippets[span.case_id].case_id == span.case_id for span in spans)
        self.calls.append(("audio", [span.case_id for span in spans]))
        return self._responses(spans)


def _assert_held(decision, flags, span):
    assert decision.case_id == span.case_id
    assert decision.verdict == "keep_srt"
    assert decision.final_text == span.srt_text
    assert decision.confidence == 0
    assert "audio" in decision.reason.lower()
    assert any(flag.kind == "adjudication_audio_unavailable" for flag in flags)


@pytest.mark.parametrize("case_id", ["case-1", "joint-case-3--case-4"])
def test_missing_required_clip_cannot_receive_text_only_approval(case_id):
    span = _span(case_id)
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=_loader({}), require_audio_snippets=True,
    ).adjudicate([span])

    _assert_held(decisions[0], flags, span)
    assert adapter.calls == []


def test_partial_audio_batch_approves_only_the_case_with_its_own_clip(tmp_path):
    first, second = _span(), _span("case-2", 0.3, 0.4)
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=_loader({first.case_id: _snippet(tmp_path, first)}),
        require_audio_snippets=True,
    ).adjudicate([first, second])

    assert decisions[0].verdict == "use_audio"
    _assert_held(decisions[1], flags, second)
    assert adapter.calls == [("audio", [first.case_id])]


def test_clip_for_a_different_case_cannot_authorize_the_requested_case(tmp_path):
    span = _span()
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=_loader({span.case_id: _snippet(tmp_path, span, case_id="another-case")}),
        require_audio_snippets=True,
    ).adjudicate([span])

    _assert_held(decisions[0], flags, span)
    assert adapter.calls == []


@pytest.mark.parametrize("bounds", [
    {"start": 0.15, "end": 1},
    {"start": 0, "end": 0.15},
    {"start": 0.1, "end": 0.1},
    {"start": 0, "end": float("nan")},
])
def test_required_clip_must_cover_the_complete_finite_case_window(tmp_path, bounds):
    span = _span()
    snippet = _snippet(tmp_path, span).model_copy(update=bounds)
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippets={span.case_id: snippet}, require_audio_snippets=True,
    ).adjudicate([span])

    _assert_held(decisions[0], flags, span)
    assert adapter.calls == []


def test_zero_confidence_gate_cannot_override_missing_audio_protection():
    span = _span()
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=_loader({}), confidence_gate=0,
        require_audio_snippets=True,
    ).adjudicate([span])

    _assert_held(decisions[0], flags, span)
    protected = pipeline._confidence_held_source_cue_ids(flags) | pipeline._timing_evidence_held_cue_ids(flags)
    assert set(span.cue_ids) <= protected
    assert adapter.calls == []


@pytest.mark.parametrize("error_type", [ProviderError, OSError])
def test_partial_batch_provider_failure_retains_the_missing_audio_hold(tmp_path, error_type):
    first, second = _span(), _span("case-2", 0.3, 0.4)

    class FailingAudioAdapter(_AudioAdapter):
        def adjudicate_with_audio(self, spans, snippets):
            self.calls.append(("audio", [span.case_id for span in spans]))
            raise error_type("synthetic provider failure; no external call")

    adapter = FailingAudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippets={first.case_id: _snippet(tmp_path, first)},
        confidence_gate=0, require_audio_snippets=True,
    ).adjudicate([first, second])

    _assert_held(decisions[1], flags, second)
    protected = pipeline._confidence_held_source_cue_ids(flags) | pipeline._timing_evidence_held_cue_ids(flags)
    assert set(second.cue_ids) <= protected
    assert adapter.calls == [("audio", [first.case_id])]


def test_adapter_without_audio_support_cannot_approve_required_audio_case(tmp_path):
    span = _span()
    adapter = _TextAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippets={span.case_id: _snippet(tmp_path, span)},
        require_audio_snippets=True,
    ).adjudicate([span])

    _assert_held(decisions[0], flags, span)
    assert adapter.calls == []


@pytest.mark.parametrize("require_audio", [False, True])
def test_valid_focused_audio_succeeds_with_either_full_context_policy(tmp_path, require_audio):
    span = _span()
    adapter = _AudioAdapter()
    decisions, flags = AdjudicationEngine(
        adapter, audio_snippet_batches=_loader({span.case_id: _snippet(tmp_path, span)}),
        require_audio_snippets=require_audio,
    ).adjudicate([span])

    assert decisions[0].verdict == "use_audio"
    assert flags == []
    assert adapter.calls == [("audio", [span.case_id])]


@pytest.mark.parametrize("full_audio", [False, True])
def test_pipeline_requires_focused_audio_only_when_full_audio_is_disabled(tmp_path, monkeypatch, full_audio):
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
        "llm": {"provider": "gemini", "adjudication": {
            "model": "gemini-3.5-flash-lite", "thinking_level": "high",
            "audio_context": {"enabled": full_audio},
            "audio_snippet_double_check": {"enabled": True},
        }},
        "punctuation": {"enabled": False},
    }), encoding="utf-8")
    adapter = _AudioAdapter()
    required_values = []
    original_engine = AdjudicationEngine

    def engine(*args, **kwargs):
        required_values.append(kwargs.get("require_audio_snippets", False))
        return original_engine(*args, **kwargs)

    def extract(_audio, spans, directory, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        return [_snippet(directory, span) for span in spans]

    monkeypatch.setattr(pipeline, "AdjudicationEngine", engine)
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    result = pipeline.sync_episode(
        source, audio, tmp_path / "output.srt", tmp_path / "work", providers_path=providers,
    )

    assert required_values == [not full_audio]
    assert adapter.calls == [("audio", ["case-1"])]
    assert [cue.plain_text for cue in adapter.source_context] == ["old source words"]
    assert "different spoken wording" in result.output_srt.read_text(encoding="utf-8")
