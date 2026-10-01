from __future__ import annotations

import sys
import wave
from types import SimpleNamespace

import httpx
import pytest

from dubsync.providers import ElevenLabsScribeAdapter, ProviderError


class FakeApiError(Exception):
    """Shape of elevenlabs.core.api_error.ApiError (status code, headers, private body)."""

    def __init__(self, status_code, headers=None, body="private-upstream-body test-key"):
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body


def _audio(tmp_path, seconds=2):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * int(seconds * 16000))
    return path


def _response(words=None):
    return SimpleNamespace(words=words if words is not None else [
        {"type": "word", "text": "Hallo", "start": 0.1, "end": 0.4, "speaker_id": "speaker_0", "logprob": -0.05},
    ])


def _fake_sdk(monkeypatch, outcomes):
    calls = []

    class FakeSpeechToText:
        def convert(self, **kwargs):
            calls.append({**kwargs, "file_bytes": kwargs["file"].read()})
            outcome = outcomes[len(calls) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class FakeElevenLabs:
        def __init__(self, api_key, **kwargs):
            del api_key, kwargs
            self.speech_to_text = FakeSpeechToText()

    monkeypatch.setitem(sys.modules, "elevenlabs", SimpleNamespace(ElevenLabs=FakeElevenLabs))
    delays = []
    monkeypatch.setattr("dubsync.providers.time.sleep", delays.append)
    return calls, delays


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("timed out"),
    httpx.RemoteProtocolError("server disconnected"),
    FakeApiError(408),
    FakeApiError(500),
    FakeApiError(503),
], ids=["read-timeout", "disconnect", "408", "500", "503"])
def test_transient_scribe_failure_is_retried_with_the_full_upload(monkeypatch, tmp_path, failure):
    calls, delays = _fake_sdk(monkeypatch, [failure, _response()])
    adapter = ElevenLabsScribeAdapter(api_key="test-key")

    words = adapter.transcribe(_audio(tmp_path))

    assert [word.text for word in words] == ["Hallo"]
    assert len(calls) == 2
    assert calls[0]["file_bytes"] == calls[1]["file_bytes"] and calls[1]["file_bytes"]
    assert len(delays) == 1 and 0 < delays[0] <= 10
    assert adapter.last_usage["request_count"] == 2
    # The failed attempt may have been processed and billed.
    assert adapter.last_usage["uncertain_request_count"] == 1
    assert adapter.last_usage["uncertain_seconds"] == pytest.approx(2.0)


def test_connection_failure_before_upload_is_retried_without_uncertain_billing(monkeypatch, tmp_path):
    calls, _delays = _fake_sdk(monkeypatch, [httpx.ConnectError("refused"), _response()])
    adapter = ElevenLabsScribeAdapter(api_key="test-key")

    adapter.transcribe(_audio(tmp_path))

    assert len(calls) == 2
    assert "uncertain_request_count" not in adapter.last_usage


def test_persistent_scribe_server_errors_stop_after_bounded_attempts(monkeypatch, tmp_path):
    calls, delays = _fake_sdk(monkeypatch, [FakeApiError(502)] * 5)

    with pytest.raises(ProviderError, match="HTTP 502") as caught:
        ElevenLabsScribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))

    assert len(calls) == 3
    assert len(delays) == 2
    assert "test-key" not in str(caught.value)
    assert "private-upstream-body" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("status,code", [(401, "authentication"), (403, "authentication"), (402, "credits")])
def test_scribe_account_failures_map_to_job_error_codes_without_retry(monkeypatch, tmp_path, status, code):
    calls, delays = _fake_sdk(monkeypatch, [FakeApiError(status)] * 3)

    with pytest.raises(ProviderError) as caught:
        ElevenLabsScribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))

    assert caught.value.code == code
    assert "private-upstream-body" not in str(caught.value)
    assert len(calls) == 1
    assert delays == []


def test_scribe_rate_limit_is_retried_once_then_reported(monkeypatch, tmp_path):
    calls, delays = _fake_sdk(monkeypatch, [FakeApiError(429, {"retry-after": "60"}), FakeApiError(429)])

    with pytest.raises(ProviderError) as caught:
        ElevenLabsScribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))

    assert caught.value.code == "rate_limit"
    assert len(calls) == 2
    assert delays == [5.0]


def test_scribe_client_errors_are_not_retried(monkeypatch, tmp_path):
    calls, _delays = _fake_sdk(monkeypatch, [FakeApiError(422)] * 3)

    with pytest.raises(ProviderError, match="HTTP 422"):
        ElevenLabsScribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))

    assert len(calls) == 1


def test_scribe_logprob_and_audio_events_are_kept_as_evidence_without_changing_word_confidence(monkeypatch, tmp_path):
    response = SimpleNamespace(
        language_code="deu", language_probability=0.98,
        words=[
            {"type": "audio_event", "text": "(lacht)", "start": 0.0, "end": 0.3, "speaker_id": "speaker_1", "logprob": 0.0},
            {"type": "word", "text": "Hallo", "start": 0.4, "end": 0.7, "speaker_id": "speaker_0", "logprob": -0.05},
            {"type": "spacing", "text": " ", "start": 0.7, "end": 0.8, "speaker_id": "speaker_0", "logprob": 0.0},
            {"type": "word", "text": "Welt.", "start": 0.8, "end": 1.2, "speaker_id": "speaker_0", "logprob": -1.25},
        ],
    )
    _fake_sdk(monkeypatch, [response])
    adapter = ElevenLabsScribeAdapter(api_key="test-key")

    words = adapter.transcribe(_audio(tmp_path))

    assert [(word.text, word.confidence) for word in words] == [("Hallo", 1.0), ("Welt.", 1.0)]
    assert adapter.last_evidence == {
        "provider": "elevenlabs",
        "language_code": "deu",
        "language_probability": 0.98,
        "word_logprobs": [
            {"text": "Hallo", "start": 0.4, "end": 0.7, "logprob": -0.05},
            {"text": "Welt.", "start": 0.8, "end": 1.2, "logprob": -1.25},
        ],
        "audio_events": [{"text": "(lacht)", "start": 0.0, "end": 0.3, "speaker_id": "speaker_1"}],
    }


def test_missing_scribe_key_is_a_configuration_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ProviderError) as caught:
        ElevenLabsScribeAdapter().transcribe(_audio(tmp_path))
    assert caught.value.code == "configuration"
