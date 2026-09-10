from __future__ import annotations

import base64
import io
import json
import socket
import wave
from urllib.error import HTTPError, URLError

import pytest

from dubsync.mai_transcribe import MAITranscribeAdapter
from dubsync.providers import ProviderError


def _audio(tmp_path, seconds=3, rate=16000, channels=1):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * int(seconds * rate * channels))
    return path


class _Response(io.BytesIO):
    def __init__(self, payload, generation_id="gen-test"):
        super().__init__(json.dumps(payload).encode())
        self.headers = {"X-Generation-Id": generation_id}


def _transport(monkeypatch, responses):
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return _Response(response, f"gen-{len(calls)}")

    monkeypatch.setattr("dubsync.mai_transcribe.urlopen", urlopen)
    return calls


def test_request_uses_audio_endpoint_with_word_timing_and_usage(monkeypatch, tmp_path):
    audio = _audio(tmp_path)
    calls = _transport(monkeypatch, [{
        "text": "Hello there",
        "words": [
            {"word": "Hello", "start": 0.2, "end": 0.6, "speaker": "speaker_0"},
            {"word": "there", "start": 0.7, "end": 1.1, "speaker": "speaker_0"},
        ],
        "usage": {"seconds": 3, "cost": 0.000075},
    }])
    adapter = MAITranscribeAdapter(api_key="test-secret", timeout_seconds=75)

    words = adapter.transcribe(audio)

    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert timeout == 75
    assert payload["model"] == "microsoft/mai-transcribe-2"
    assert payload["response_format"] == "verbose_json"
    assert payload["timestamp_granularities"] == ["segment", "word"]
    assert payload["provider"]["options"]["azure"]["diarization"] == {"enabled": True}
    assert payload["input_audio"]["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(payload["input_audio"]["data"]))) as clip:
        assert clip.getnframes() == 48000
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hello", 0.2, 0.6), ("there", 0.7, 1.1),
    ]
    assert words[0].speaker_id == words[1].speaker_id
    assert words[0].speaker_id is not None
    assert words[0].confidence is None
    assert adapter.last_usage == {
        "seconds": 3.0, "cost": 0.000075, "generation_ids": ["gen-1"], "request_count": 1,
        "reported_seconds": 3.0, "reported_cost": 0.000075,
    }


def test_chunks_preserve_absolute_timing_and_do_not_merge_speakers(monkeypatch, tmp_path):
    audio = _audio(tmp_path, seconds=9)
    # Chunks own [0, 4), [4, 8), [8, 9], with one second of context on each side.
    calls = _transport(monkeypatch, [
        {"words": [
            {"word": "one", "start": 0.5, "end": 0.8, "speaker": 0},
            {"word": "boundary", "start": 3.9, "end": 4.3, "speaker": 0},
        ], "usage": {"seconds": 5, "cost": 0.1}},
        {"words": [
            {"word": "boundary", "start": 0.9, "end": 1.3, "speaker": 0},
            {"word": "two", "start": 2, "end": 2.2, "speaker": 0},
            {"word": "last", "start": 4.9, "end": 5.3, "speaker": 0},
        ], "usage": {"seconds": 6, "cost": 0.2}},
        {"words": [
            {"word": "last", "start": 0.9, "end": 1.3, "speaker": 0},
        ], "usage": {"seconds": 2, "cost": 0.3}},
    ])
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)

    words = adapter.transcribe(audio)

    assert [word.text for word in words] == ["one", "boundary", "two", "last"]
    assert [word.start for word in words] == pytest.approx([0.5, 3.9, 5, 7.9])
    assert words[1].speaker_id == words[2].speaker_id
    assert len({word.speaker_id for word in words}) == 3
    assert len(calls) == 3
    assert adapter.last_usage["seconds"] == 13
    assert adapter.last_usage["cost"] == pytest.approx(0.6)
    assert adapter.last_usage["request_count"] == 3


@pytest.mark.parametrize("payload", [
    {"text": "Hello, world"},
    {"text": "Hello, world", "words": []},
    {"words": [{"word": "hello", "end": 1}]},
    {"words": [{"word": "hello", "start": 1, "end": 0.5}]},
    {"words": [{"word": "hello", "start": -1, "end": 0.5}]},
    {"words": [{"word": "hello", "start": 0, "end": 5}]},
    {"words": [{"word": "hello", "start": float("nan"), "end": 1}]},
    {"words": [{"word": "hello", "start": False, "end": 1}]},
    {"words": "hello"},
])
def test_missing_or_invalid_word_timing_fails_closed(monkeypatch, tmp_path, payload):
    _transport(monkeypatch, [payload])

    with pytest.raises(ProviderError, match="word|timing"):
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))


def test_empty_silence_transcript_is_valid(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"text": "", "words": [], "usage": {"seconds": 3, "cost": 0}}])
    assert MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path)) == []


def test_observed_ten_millisecond_end_rounding_is_clamped_and_auditable(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"words": [{"word": "para.", "start": 29.52, "end": 29.68}]}])
    adapter = MAITranscribeAdapter(api_key="test-key")
    words = adapter.transcribe(_audio(tmp_path, seconds=29.67))

    assert [(word.start, word.end) for word in words] == [(29.52, 29.67)]
    assert len(adapter.last_repair_flags) == 1
    flag = adapter.last_repair_flags[0]
    assert flag.kind == "asr_timestamp_rounding_clamped"
    assert flag.severity == "info"
    assert flag.start == 29.52
    assert flag.end == 29.67
    assert "start=29.52" in flag.message
    assert "end=29.68" in flag.message
    assert "end=29.67" in flag.message


def test_twenty_millisecond_rounding_limit_is_inclusive(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"words": [{"word": "last", "start": 2.8, "end": 3.02}]}])
    adapter = MAITranscribeAdapter(api_key="test-key")
    words = adapter.transcribe(_audio(tmp_path))
    assert words[0].end == 3
    assert len(adapter.last_repair_flags) == 1


@pytest.mark.parametrize("start,end", [(2.8, 3.020001), (3.0, 3.01), (3.001, 3.011)])
def test_end_rounding_does_not_allow_larger_overrun_or_start_at_eof(monkeypatch, tmp_path, start, end):
    _transport(monkeypatch, [{"words": [{"word": "last", "start": start, "end": end}]}])
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError, match="invalid word timing"):
        adapter.transcribe(_audio(tmp_path))
    assert adapter.last_repair_flags == []


def test_rounding_flag_has_absolute_range_and_resets_next_transcription(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": []},
        {"words": [{"word": "last", "start": 4.8, "end": 5.01}]},
        {"words": []},
    ])
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)
    words = adapter.transcribe(_audio(tmp_path, seconds=8))
    assert (words[0].start, words[0].end) == (7.8, 8.0)
    flag = adapter.last_repair_flags[0]
    assert (flag.start, flag.end) == (7.8, 8.0)
    assert "offset=3.0" in flag.message
    assert "start=4.8" in flag.message
    assert "end=5.01" in flag.message
    adapter.transcribe(_audio(tmp_path, seconds=3))
    assert adapter.last_repair_flags == []


def test_diarization_can_be_disabled(monkeypatch, tmp_path):
    calls = _transport(monkeypatch, [{"words": [{"word": "hello", "start": 0, "end": 1, "speaker": 2}]}])
    words = MAITranscribeAdapter(api_key="test-key", diarize=False).transcribe(_audio(tmp_path))
    payload = json.loads(calls[0][0].data)
    assert payload["provider"]["options"]["azure"]["diarization"] == {"enabled": False}
    assert words[0].speaker_id is None


def test_language_and_keyterms_use_supported_azure_options(monkeypatch, tmp_path):
    calls = _transport(monkeypatch, [{"words": []}])
    MAITranscribeAdapter(api_key="test-key", language_code="de", keyterms=["Luna", "SSS-Rang"]).transcribe(_audio(tmp_path))
    payload = json.loads(calls[0][0].data)
    assert payload["language"] == "de"
    assert "prompt" not in payload
    azure = payload["provider"]["options"]["azure"]
    assert azure["phraseList"] == {"phrases": ["Luna", "SSS-Rang"]}
    assert azure["enhancedMode"] == {"modelOptions": {"transcribeStyle": "verbatim"}}


def test_long_audio_reads_bounded_overlapping_chunks(monkeypatch, tmp_path):
    audio = _audio(tmp_path, seconds=610)
    calls = _transport(monkeypatch, [{"words": []}] * 3)
    reads = []
    original_read = wave.Wave_read.readframes

    def readframes(reader, count):
        reads.append(count)
        return original_read(reader, count)

    monkeypatch.setattr(wave.Wave_read, "readframes", readframes)
    MAITranscribeAdapter(api_key="test-key").transcribe(audio)

    assert reads == [301 * 16000, 302 * 16000, 11 * 16000]
    assert len(calls) == 3
    assert all(len(request.data) < 25 * 1024 * 1024 for request, _ in calls)


def test_paid_usage_survives_invalid_timing(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"text": "hello", "usage": {"seconds": 3, "cost": 0.01}}])
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError):
        adapter.transcribe(_audio(tmp_path))
    assert adapter.last_usage["cost"] == 0.01
    assert adapter.last_usage["seconds"] == 3


def test_timeout_does_not_claim_zero_bill(monkeypatch, tmp_path):
    _transport(monkeypatch, [socket.timeout()])
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError):
        adapter.transcribe(_audio(tmp_path))
    assert adapter.last_usage["cost"] is None
    assert adapter.last_usage["seconds"] is None


def test_partial_chunk_usage_with_missing_cost_stays_unknown(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [], "usage": {"seconds": 3}},
        {"words": [], "usage": {"seconds": 3, "cost": 0.1}},
    ])
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=2)
    adapter.transcribe(_audio(tmp_path, seconds=4))
    assert adapter.last_usage["seconds"] == 6
    assert adapter.last_usage["cost"] is None


def test_absent_cost_is_unknown_and_usage_is_reset(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [], "usage": {"seconds": 3, "cost": 0.25}},
        {"words": [], "usage": {"seconds": 3}},
    ])
    adapter = MAITranscribeAdapter(api_key="test-key")
    adapter.transcribe(_audio(tmp_path))
    assert adapter.last_usage["cost"] == 0.25
    adapter.transcribe(_audio(tmp_path))
    assert adapter.last_usage == {
        "seconds": 3.0, "cost": None, "generation_ids": ["gen-2"], "request_count": 1,
        "reported_seconds": 3.0, "reported_cost": 0.0,
    }


@pytest.mark.parametrize("left,right,expected_start", [
    ((3.8, 4.1), (0.9, 1.2), 3.8),  # Independent midpoint ownership duplicates this word.
    ((3.9, 4.2), (0.8, 1.1), 3.8),  # Independent midpoint ownership drops this word.
])
def test_chunk_overlap_retains_one_word_despite_timestamp_jitter(monkeypatch, tmp_path, left, right, expected_start):
    _transport(monkeypatch, [
        {"words": [{"word": "Boundary,", "start": left[0], "end": left[1], "speaker": 0}]},
        {"words": [{"word": "boundary", "start": right[0], "end": right[1], "speaker": 0}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))
    assert len(words) == 1
    assert words[0].start == expected_start
    assert (words[0].start, words[0].end) in [left, (right[0] + 3, right[1] + 3)]


def test_chunk_overlap_preserves_actual_repeated_words(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [
            {"word": "no", "start": 3.5, "end": 3.75},
            {"word": "no", "start": 3.9, "end": 4.1},
        ]},
        {"words": [
            {"word": "No,", "start": 0.6, "end": 0.85},
            {"word": "no", "start": 0.85, "end": 1.05},
        ]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))
    assert len(words) == 2
    assert [word.start for word in words] == [3.5, 3.85]


def test_same_word_at_distinct_times_is_not_merged(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "no", "start": 3.5, "end": 3.75}]},
        {"words": [{"word": "no", "start": 1.05, "end": 1.25}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))
    assert len(words) == 2
    assert [word.start for word in words] == [3.5, 4.05]


def test_paid_chunk_subtotal_survives_later_timeout(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [], "usage": {"seconds": 5, "cost": 0.1}},
        socket.timeout(),
    ])
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)
    with pytest.raises(ProviderError):
        adapter.transcribe(_audio(tmp_path, seconds=8))
    assert adapter.last_usage["cost"] is None
    assert adapter.last_usage["seconds"] is None
    assert adapter.last_usage["reported_cost"] == 0.1
    assert adapter.last_usage["reported_seconds"] == 5


@pytest.mark.parametrize("failure", [
    URLError("secret upstream detail test-key"),
    socket.timeout("secret upstream detail test-key"),
    HTTPError("https://openrouter.ai", 401, "test-key", {}, io.BytesIO(b"test-key")),
])
def test_errors_do_not_leak_request_credentials_or_upstream_body(monkeypatch, tmp_path, failure):
    calls = _transport(monkeypatch, [failure])
    with pytest.raises(ProviderError) as caught:
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))
    assert "test-key" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert len(calls) == 1


def test_rate_limit_retry_is_bounded_and_counted(monkeypatch, tmp_path):
    failure = HTTPError("https://openrouter.ai", 429, "rate limit", {"Retry-After": "1"}, io.BytesIO(b""))
    calls = _transport(monkeypatch, [failure, {"words": [], "usage": {"seconds": 3, "cost": 0.1}}])
    delays = []
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", delays.append)
    adapter = MAITranscribeAdapter(api_key="test-key")
    adapter.transcribe(_audio(tmp_path))
    assert delays == [1]
    assert len(calls) == 2
    assert adapter.last_usage["request_count"] == 2


def test_rate_limit_stops_after_one_retry_and_caps_delay(monkeypatch, tmp_path):
    failures = [
        HTTPError("https://openrouter.ai", 429, "rate limit", {"Retry-After": "99999"}, io.BytesIO(b"")),
        HTTPError("https://openrouter.ai", 429, "rate limit", {}, io.BytesIO(b"")),
    ]
    calls = _transport(monkeypatch, failures)
    delays = []
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", delays.append)
    with pytest.raises(ProviderError, match="rate limited") as caught:
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))
    assert caught.value.code == "rate_limit"
    assert delays == [5]
    assert len(calls) == 2


def test_insufficient_credit_error_is_actionable_and_does_not_retry(monkeypatch, tmp_path):
    failure = HTTPError("https://openrouter.ai", 402, "test-key", {}, io.BytesIO(b"test-key"))
    calls = _transport(monkeypatch, [failure])
    with pytest.raises(ProviderError, match="credits are insufficient") as caught:
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))
    assert "test-key" not in str(caught.value)
    assert caught.value.code == "credits"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failure_has_safe_job_error_code_without_retry(monkeypatch, tmp_path, status):
    failure = HTTPError("https://openrouter.ai", status, "test-key", {}, io.BytesIO(b"private-body"))
    calls = _transport(monkeypatch, [failure])
    with pytest.raises(ProviderError, match="authentication") as caught:
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))
    assert caught.value.code == "authentication"
    assert "test-key" not in str(caught.value)
    assert "private-body" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [302, 500, 502, 503, 504])
def test_unexpected_http_response_is_not_replayed_and_billing_is_unknown(monkeypatch, tmp_path, status):
    failure = HTTPError("https://openrouter.ai", status, "test-key", {}, io.BytesIO(b"test-key"))
    calls = _transport(monkeypatch, [failure])
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError, match=f"HTTP {status}") as caught:
        adapter.transcribe(_audio(tmp_path))
    assert caught.value.code is None
    assert adapter.last_usage["cost"] is None
    assert len(calls) == 1


@pytest.mark.parametrize("raw, message", [
    (b"not-json-test-key", "invalid JSON"),
    (b"[]", "invalid transcription response"),
    (b"x" * (8 * 1024 * 1024 + 1), "oversized response"),
], ids=["invalid-json", "wrong-shape", "too-large"])
def test_malformed_response_is_bounded_and_safe(monkeypatch, tmp_path, raw, message):
    response = _Response({})
    response.seek(0)
    response.truncate()
    response.write(raw)
    response.seek(0)
    monkeypatch.setattr("dubsync.mai_transcribe.urlopen", lambda request, timeout: response)
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError, match=message) as caught:
        adapter.transcribe(_audio(tmp_path))
    assert "test-key" not in str(caught.value)
    assert adapter.last_usage["cost"] is None


def test_auth_checked_before_opening_audio(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY") as caught:
        MAITranscribeAdapter().transcribe(tmp_path / "missing.wav")
    assert caught.value.code == "configuration"


def test_environment_auth_is_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
    calls = _transport(monkeypatch, [{"words": []}])
    MAITranscribeAdapter().transcribe(_audio(tmp_path))
    assert calls[0][0].get_header("Authorization") == "Bearer env-test-key"


def test_only_normalized_wav_is_accepted(monkeypatch, tmp_path):
    calls = _transport(monkeypatch, [])
    with pytest.raises(ProviderError, match="16 kHz|16-bit|mono|normalized"):
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path, rate=48000, channels=2))
    assert calls == []


@pytest.mark.parametrize("option,value", [("chunk_seconds", 0), ("chunk_seconds", 301), ("timeout_seconds", float("inf"))])
def test_unbounded_or_invalid_options_are_rejected(option, value):
    with pytest.raises(ValueError):
        MAITranscribeAdapter(api_key="test-key", **{option: value})
