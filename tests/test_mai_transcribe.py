from __future__ import annotations

import base64
import io
import json
import socket
import wave
from urllib.error import HTTPError, URLError

import pytest

from dubsync.speaker_evidence import speakers_known_different
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


def test_chunks_preserve_absolute_timing_and_link_speakers_through_overlap_words(monkeypatch, tmp_path):
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
    # Each cut has a word both chunks heard, so one voice keeps one id across chunks.
    assert {word.speaker_id for word in words} == {"chunk_1:0"}
    assert len(calls) == 3
    assert adapter.last_usage["seconds"] == 13
    assert adapter.last_usage["cost"] == pytest.approx(0.6)
    assert adapter.last_usage["request_count"] == 3


def test_sentence_straddling_a_chunk_boundary_keeps_one_speaker_id(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [
            {"word": "Ja.", "start": 1.0, "end": 1.3, "speaker": 1},
            {"word": "Ich", "start": 3.0, "end": 3.2, "speaker": 0},
            {"word": "gehe", "start": 3.3, "end": 3.6, "speaker": 0},
            {"word": "jetzt", "start": 3.7, "end": 4.0, "speaker": 0},
            {"word": "nach", "start": 4.1, "end": 4.3, "speaker": 0},
        ]},
        {"words": [
            # The second call labels the same actor 1, and a new actor 0.
            {"word": "gehe", "start": 0.3, "end": 0.6, "speaker": 1},
            {"word": "jetzt", "start": 0.7, "end": 1.0, "speaker": 1},
            {"word": "nach", "start": 1.1, "end": 1.3, "speaker": 1},
            {"word": "Hause.", "start": 1.4, "end": 1.8, "speaker": 1},
            {"word": "Tschüss.", "start": 3.0, "end": 3.4, "speaker": 0},
        ]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    by_text = {word.text: word.speaker_id for word in words}
    assert [word.text for word in words] == ["Ja.", "Ich", "gehe", "jetzt", "nach", "Hause.", "Tschüss."]
    assert {by_text[text] for text in ("Ich", "gehe", "jetzt", "nach", "Hause.")} == {"chunk_1:0"}
    # A label that no overlap word links keeps a name of its own, in the same
    # scope as its chunk-mates, so a turn inside the chunk stays provable.
    assert by_text["Tschüss."] == "chunk_1:chunk_2.0"
    assert speakers_known_different(by_text["Hause."], by_text["Tschüss."])
    assert by_text["Ja."] == "chunk_1:1"


def test_conflicting_overlap_speaker_evidence_keeps_labels_chunk_scoped(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [
            {"word": "Du", "start": 3.3, "end": 3.5, "speaker": 0},
            {"word": "nicht!", "start": 3.6, "end": 3.9, "speaker": 1},
        ]},
        {"words": [
            # The second call merges both voices into one label.
            {"word": "Du", "start": 0.3, "end": 0.5, "speaker": 0},
            {"word": "nicht!", "start": 0.6, "end": 0.9, "speaker": 0},
            {"word": "Doch.", "start": 1.5, "end": 1.8, "speaker": 0},
        ]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    assert [(word.text, word.speaker_id) for word in words] == [
        ("Du", "chunk_1:0"), ("nicht!", "chunk_1:1"), ("Doch.", "chunk_2:0"),
    ]


def test_speaker_links_chain_across_several_chunks(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "eins", "start": 3.5, "end": 3.8, "speaker": 2}]},
        {"words": [{"word": "eins", "start": 0.5, "end": 0.8, "speaker": 0}, {"word": "zwei", "start": 4.6, "end": 4.9, "speaker": 0}]},
        {"words": [{"word": "zwei", "start": 0.6, "end": 0.9, "speaker": 5}, {"word": "drei", "start": 2.0, "end": 2.3, "speaker": 5}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=12))

    assert [(word.text, word.speaker_id) for word in words] == [
        ("eins", "chunk_1:2"), ("zwei", "chunk_1:2"), ("drei", "chunk_1:2"),
    ]


def test_speakers_without_overlap_words_are_not_merged_between_chunks(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "vorher", "start": 1.0, "end": 1.4, "speaker": 0}]},
        {"words": [{"word": "nachher", "start": 3.0, "end": 3.4, "speaker": 0}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    assert [word.speaker_id for word in words] == ["chunk_1:0", "chunk_2:0"]


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


@pytest.mark.parametrize("segments", [[], [{"id": 0, "start": 0, "end": 18, "text": ""}]])
def test_observed_empty_transcription_without_words_preserves_usage(monkeypatch, tmp_path, segments):
    # The actual opening-song response omitted words and contained an empty segment.
    _transport(monkeypatch, [{
        "text": "", "language": "por", "duration": 18, "segments": segments,
        "usage": {"seconds": 18, "cost": 0.0005},
    }])
    adapter = MAITranscribeAdapter(api_key="test-key")

    assert adapter.transcribe(_audio(tmp_path, seconds=18)) == []
    assert adapter.last_usage["cost"] == 0.0005
    assert adapter.last_usage["seconds"] == 18
    assert adapter.last_usage["request_count"] == 1
    assert adapter.last_repair_flags == []


@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "", "segments": None},
    {"text": "", "segments": ["invalid"]},
    {"text": "", "segments": [{"start": 0, "end": 3}]},
    {"text": "", "segments": [{"text": "heard speech", "start": 0, "end": 3}]},
    {"text": "", "segments": [{"text": "", "start": 0, "end": 4}]},
    {"text": "", "segments": [{"text": "", "start": 2, "end": 1}]},
    {"text": "", "segments": [{"text": "", "start": 0, "end": float("nan")}]},
    {"text": "", "segments": [], "words": None},
    {"text": "", "segments": [], "error": {"message": "failed"}},
    {"text": "heard speech", "segments": []},
])
def test_missing_words_requires_explicit_valid_empty_transcription(monkeypatch, tmp_path, payload):
    _transport(monkeypatch, [payload])
    with pytest.raises(ProviderError, match="word|timing"):
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))


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


def test_optional_per_word_confidence_is_parsed_when_present(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"words": [
        {"word": "sure", "start": 0.2, "end": 0.5, "confidence": 0.87},
        {"word": "maybe", "start": 0.6, "end": 0.9, "confidence": 1.5},
        {"word": "odd", "start": 1.0, "end": 1.2, "confidence": "high"},
        {"word": "plain", "start": 1.3, "end": 1.6},
    ]}])
    words = MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))

    assert [(word.text, word.confidence) for word in words] == [
        ("sure", 0.87), ("maybe", None), ("odd", None), ("plain", None),
    ]


def test_chunk_language_and_diarization_are_kept_as_provider_evidence(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "Bom", "start": 3.5, "end": 3.8, "speaker": 0}], "language": "pt"},
        {"words": [{"word": "Bom", "start": 0.5, "end": 0.8, "speaker": 1}], "language": "ca"},
    ])
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)
    adapter.transcribe(_audio(tmp_path, seconds=8))

    assert adapter.last_evidence == {
        "provider": "openrouter",
        "chunks": [
            {"index": 1, "offset": 0.0, "duration": 5.0, "language": "pt", "diarized": True},
            {"index": 2, "offset": 3.0, "duration": 5.0, "language": "ca", "diarized": True},
        ],
        "speaker_links": [{"boundary": 4.0, "links": {"chunk_2:1": "chunk_1:0"}}],
    }


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
    _transport(monkeypatch, [socket.timeout()] * 3)
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError):
        adapter.transcribe(_audio(tmp_path))
    assert adapter.last_usage["cost"] is None
    assert adapter.last_usage["seconds"] is None
    assert adapter.last_usage["uncertain_request_count"] == 3


@pytest.mark.parametrize("failure", [
    socket.timeout("timed out"),
    URLError(socket.timeout("timed out")),
    URLError(ConnectionResetError("reset")),
    HTTPError("https://openrouter.ai", 408, "timeout", {}, io.BytesIO(b"")),
    HTTPError("https://openrouter.ai", 500, "error", {}, io.BytesIO(b"")),
    HTTPError("https://openrouter.ai", 502, "error", {}, io.BytesIO(b"")),
    HTTPError("https://openrouter.ai", 503, "error", {}, io.BytesIO(b"")),
    HTTPError("https://openrouter.ai", 504, "error", {}, io.BytesIO(b"")),
], ids=["socket-timeout", "url-timeout", "connection-reset", "408", "500", "502", "503", "504"])
def test_transient_chunk_failure_is_retried_and_its_possible_charge_recorded_as_uncertain(monkeypatch, tmp_path, failure):
    calls = _transport(monkeypatch, [failure, {
        "words": [{"word": "hello", "start": 0.2, "end": 0.6, "speaker": 0}], "usage": {"seconds": 3, "cost": 0.0001},
    }])
    delays = []
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", delays.append)
    adapter = MAITranscribeAdapter(api_key="test-key")

    words = adapter.transcribe(_audio(tmp_path))

    assert [word.text for word in words] == ["hello"]
    assert len(calls) == 2
    assert len(delays) == 1 and 0 < delays[0] <= 10
    # The failed attempt may have been billed: the total is unknown, the known part is kept.
    assert adapter.last_usage["cost"] is None
    assert adapter.last_usage["reported_cost"] == 0.0001
    assert adapter.last_usage["uncertain_request_count"] == 1
    assert adapter.last_usage["uncertain_seconds"] == 3.0
    assert adapter.last_usage["request_count"] == 2


def test_persistent_server_errors_stop_after_bounded_attempts(monkeypatch, tmp_path):
    failure = HTTPError("https://openrouter.ai", 503, "test-key", {}, io.BytesIO(b"test-key"))
    calls = _transport(monkeypatch, [failure] * 5)
    delays = []
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", delays.append)
    adapter = MAITranscribeAdapter(api_key="test-key")

    with pytest.raises(ProviderError, match="HTTP 503") as caught:
        adapter.transcribe(_audio(tmp_path))

    assert "test-key" not in str(caught.value)
    assert len(calls) == 3
    assert len(delays) == 2
    assert all(json.loads(request.data)["provider"]["options"]["azure"]["diarization"] == {"enabled": True} for request, _ in calls)


def test_chunk_that_keeps_timing_out_with_diarization_is_retried_once_without_it(monkeypatch, tmp_path):
    calls = _transport(monkeypatch, [
        HTTPError("https://openrouter.ai", 408, "timeout", {}, io.BytesIO(b"")),
        socket.timeout(),
        {"words": [{"word": "hello", "start": 0.2, "end": 0.6}], "usage": {"seconds": 3, "cost": 0.0001}},
    ])
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    adapter = MAITranscribeAdapter(api_key="test-key")

    words = adapter.transcribe(_audio(tmp_path))

    assert [json.loads(request.data)["provider"]["options"]["azure"]["diarization"]["enabled"] for request, _ in calls] == [
        True, True, False,
    ]
    assert [(word.text, word.speaker_id) for word in words] == [("hello", None)]
    flags = [flag for flag in adapter.last_repair_flags if flag.kind == "asr_diarization_unavailable"]
    assert len(flags) == 1
    assert flags[0].severity == "info"
    assert (flags[0].start, flags[0].end) == (0.0, 3.0)
    assert adapter.last_usage["uncertain_request_count"] == 2


def test_later_chunks_fall_back_after_one_diarized_timeout(monkeypatch, tmp_path):
    ok = {"words": [], "usage": {"seconds": 5, "cost": 0.0001}}
    calls = _transport(monkeypatch, [socket.timeout(), socket.timeout(), ok, socket.timeout(), ok])
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)

    adapter.transcribe(_audio(tmp_path, seconds=8))

    assert [json.loads(request.data)["provider"]["options"]["azure"]["diarization"]["enabled"] for request, _ in calls] == [
        True, True, False, True, False,
    ]


def test_persistent_timeout_without_diarization_fails_after_bounded_attempts(monkeypatch, tmp_path):
    calls = _transport(monkeypatch, [socket.timeout()] * 5)
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    with pytest.raises(ProviderError, match="timeout"):
        MAITranscribeAdapter(api_key="test-key", diarize=False).transcribe(_audio(tmp_path))
    assert len(calls) == 3


def test_one_invalid_word_record_is_dropped_with_a_flag(monkeypatch, tmp_path):
    _transport(monkeypatch, [{"words": [
        {"word": "eins", "start": 0.2, "end": 0.5},
        {"word": "kaputt", "start": 1.0, "end": 0.9},
        {"word": "drei", "start": 1.2, "end": 1.5},
    ]}])
    adapter = MAITranscribeAdapter(api_key="test-key")

    words = adapter.transcribe(_audio(tmp_path))

    assert [word.text for word in words] == ["eins", "drei"]
    assert [(flag.kind, flag.severity) for flag in adapter.last_repair_flags] == [("asr_invalid_word_dropped", "warning")]
    assert "kaputt" in adapter.last_repair_flags[0].message
    # The flag points at the dropped record, not at the whole chunk.
    assert (adapter.last_repair_flags[0].start, adapter.last_repair_flags[0].end) == (1.0, 1.0)


@pytest.mark.parametrize("records", [
    [{"word": "eins", "start": 0.2, "end": 0.5}, {"word": "zwei", "start": 1.0}, {"word": "drei", "start": 1.2, "end": 1.1}],
    [{"word": "eins", "start": 0.2, "end": 0.5}, "garbage", {"word": " ", "start": 1.2, "end": 1.5}],
], ids=["two-invalid-timings", "two-invalid-records"])
def test_more_than_one_invalid_word_record_in_a_chunk_still_fails_closed(monkeypatch, tmp_path, records):
    _transport(monkeypatch, [{"words": records}])
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError, match="invalid word"):
        adapter.transcribe(_audio(tmp_path))
    assert adapter.last_repair_flags == []


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


@pytest.mark.parametrize("left,right", [
    ((3.9, 4.2), (0.85, 1.1)),   # asr.md bug 6: crossed midpoints used to lose the word
    ((3.8, 4.1), (0.95, 1.25)),  # ... and these used to keep both spellings
])
def test_boundary_word_spelled_differently_by_the_two_chunks_is_kept_once(monkeypatch, tmp_path, left, right):
    _transport(monkeypatch, [
        {"words": [{"word": "Ok,", "start": 3.2, "end": 3.5}, {"word": "vamo", "start": left[0], "end": left[1]}]},
        {"words": [{"word": "vamos", "start": right[0], "end": right[1]}, {"word": "embora.", "start": 1.5, "end": 1.9}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    assert [word.text for word in words][0] == "Ok,"
    assert [word.text for word in words][-1] == "embora."
    assert len(words) == 3
    assert (words[1].text, words[1].start) in [("vamo", left[0]), ("vamos", pytest.approx(right[0] + 3))]


def test_boundary_pairing_prefers_exact_text_over_a_different_spelling(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "pra", "start": 3.7, "end": 3.9}, {"word": "casa.", "start": 3.95, "end": 4.3}]},
        {"words": [{"word": "para", "start": 0.68, "end": 0.92}, {"word": "casa.", "start": 0.96, "end": 1.31}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    assert [word.text for word in words] == ["pra", "casa."]


def test_weakly_overlapping_different_boundary_words_keep_midpoint_ownership(monkeypatch, tmp_path):
    _transport(monkeypatch, [
        {"words": [{"word": "Haus", "start": 3.7, "end": 3.95}]},
        {"words": [{"word": "jetzt", "start": 0.9, "end": 1.3}]},
    ])
    words = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4).transcribe(_audio(tmp_path, seconds=8))

    assert [word.text for word in words] == ["Haus", "jetzt"]


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
        socket.timeout(), socket.timeout(), socket.timeout(),
    ])
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    adapter = MAITranscribeAdapter(api_key="test-key", chunk_seconds=4)
    with pytest.raises(ProviderError):
        adapter.transcribe(_audio(tmp_path, seconds=8))
    assert adapter.last_usage["cost"] is None
    assert adapter.last_usage["seconds"] is None
    assert adapter.last_usage["reported_cost"] == 0.1
    assert adapter.last_usage["reported_seconds"] == 5


@pytest.mark.parametrize("failure,attempts", [
    (URLError("secret upstream detail test-key"), 3),
    (socket.timeout("secret upstream detail test-key"), 3),
    (HTTPError("https://openrouter.ai", 401, "test-key", {}, io.BytesIO(b"test-key")), 1),
])
def test_errors_do_not_leak_request_credentials_or_upstream_body(monkeypatch, tmp_path, failure, attempts):
    calls = _transport(monkeypatch, [failure] * 5)
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    with pytest.raises(ProviderError) as caught:
        MAITranscribeAdapter(api_key="test-key").transcribe(_audio(tmp_path))
    assert "test-key" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert len(calls) == attempts


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


@pytest.mark.parametrize("status,attempts", [(302, 1), (400, 1), (404, 1), (500, 3), (502, 3), (503, 3), (504, 3)])
def test_unexpected_http_response_billing_is_unknown_and_only_server_errors_are_retried(monkeypatch, tmp_path, status, attempts):
    failure = HTTPError("https://openrouter.ai", status, "test-key", {}, io.BytesIO(b"test-key"))
    calls = _transport(monkeypatch, [failure] * 5)
    monkeypatch.setattr("dubsync.mai_transcribe.time.sleep", lambda _seconds: None)
    adapter = MAITranscribeAdapter(api_key="test-key")
    with pytest.raises(ProviderError, match=f"HTTP {status}") as caught:
        adapter.transcribe(_audio(tmp_path))
    assert caught.value.code is None
    assert adapter.last_usage["cost"] is None
    assert len(calls) == attempts


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


def _raw_payload(raw, base):
    words = []
    for item in raw:
        text, start, end = item[:3]
        record = {"word": text, "start": round(start - base, 3), "end": round(end - base, 3)}
        if len(item) > 3:
            record["speaker"] = item[3]
        words.append(record)
    return {"words": words}


def _replay_raw(monkeypatch, tmp_path, raw, *, diarize=False):
    """Replay one raw MAI word list (absolute episode times) as a single short chunk."""
    base = raw[0][1] - 0.5
    _transport(monkeypatch, [_raw_payload(raw, base)])
    adapter = MAITranscribeAdapter(api_key="test-key", diarize=diarize)
    words = adapter.transcribe(_audio(tmp_path, seconds=max(item[2] for item in raw) - base + 1))
    return [(word.text, round(word.start + base, 3), round(word.end + base, 3)) for word in words], adapter


# Raw word lists copied from the ep11 / ep17 MAI chunk responses (2026-09-14 benchmark).
# MAI re-emits a run with (nearly) the same timestamps; the later copy continues the
# sentence in every observed case, so it is the one kept.
_REDECODED_RUNS = [
    pytest.param(
        [("comigo.", 244.92, 245.279), ("Você...", 253.52, 254.199), ("Você...", 253.52, 254.159), ("deu", 255.12, 255.259)],
        [("comigo.", 244.92, 245.279), ("Você...", 253.52, 254.159), ("deu", 255.12, 255.259)], id="ep11-253-voce"),
    pytest.param(
        [("carro.", 497.16, 497.44), ("Entra.", 500.88, 501.319), ("Entra.", 500.88, 501.28), ("Senhor", 507.68, 507.899)],
        [("carro.", 497.16, 497.44), ("Entra.", 500.88, 501.28), ("Senhor", 507.68, 507.899)], id="ep11-500-entra"),
    pytest.param(
        [("lindo.", 1212.76, 1213.28), ("Queria.", 1228.6, 1228.999), ("Queria", 1228.6, 1228.92),
         ("pular", 1229.04, 1229.28), ("nas", 1229.36, 1229.5), ("nuvens.", 1229.64, 1230.079)],
        [("lindo.", 1212.76, 1213.28), ("Queria", 1228.6, 1228.92), ("pular", 1229.04, 1229.28),
         ("nas", 1229.36, 1229.5), ("nuvens.", 1229.64, 1230.079)], id="ep11-1228-queria"),
    pytest.param(
        [("aqui?", 1323.46, 1323.579), ("Vamos,", 1323.6, 1323.78), ("vamos,", 1323.8, 1324.039),
         ("vamos", 1324.08, 1324.22), ("lá.", 1324.26, 1324.439), ("Vamos", 1324.08, 1324.22),
         ("lá.", 1324.26, 1324.439), ("Hoje,", 1332.56, 1332.819)],
        [("aqui?", 1323.46, 1323.579), ("Vamos,", 1323.6, 1323.78), ("vamos,", 1323.8, 1324.039),
         ("Vamos", 1324.08, 1324.22), ("lá.", 1324.26, 1324.439), ("Hoje,", 1332.56, 1332.819)], id="ep11-1324-vamos-la"),
    pytest.param(
        [("ver.", 1770.52, 1770.76), ("E", 1778.4, 1778.48), ("eu", 1778.56, 1778.699), ("e", 1778.72, 1778.779),
         ("ela.", 1778.82, 1779.0), ("E", 1778.4, 1778.48), ("eu", 1778.62, 1778.699), ("ia", 1778.76, 1778.88),
         ("lá", 1778.92, 1779.04), ("saber", 1779.12, 1779.279)],
        [("ver.", 1770.52, 1770.76), ("E", 1778.4, 1778.48), ("eu", 1778.62, 1778.699), ("ia", 1778.76, 1778.88),
         ("lá", 1778.92, 1779.04), ("saber", 1779.12, 1779.279)], id="ep11-1778-e-eu-ia-la"),
    pytest.param(
        [("namorada?", 1870.44, 1870.879), ("Sim.", 1875.16, 1875.519), ("Sim.", 1875.16, 1875.48), ("Tudo", 1881.4, 1881.619)],
        [("namorada?", 1870.44, 1870.879), ("Sim.", 1875.16, 1875.48), ("Tudo", 1881.4, 1881.619)], id="ep11-1875-sim"),
    pytest.param(
        [("porta.", 1970.48, 1970.8), ("Tá.", 1974.36, 1974.759), ("Tá.", 1974.4, 1974.759), ("Feliz", 1992.32, 1992.5)],
        [("porta.", 1970.48, 1970.8), ("Tá.", 1974.4, 1974.759), ("Feliz", 1992.32, 1992.5)], id="ep11-1974-ta"),
    pytest.param(
        [("está", 1998.82, 1999.0), ("melhor?", 1999.08, 1999.479), ("Melhor.", 1999.12, 1999.479), ("Vocês", 2008.72, 2009.02)],
        [("está", 1998.82, 1999.0), ("Melhor.", 1999.12, 1999.479), ("Vocês", 2008.72, 2009.02)], id="ep11-1999-melhor"),
    pytest.param(
        [("Hum.", 712.28, 712.659), ("Flora.", 727.44, 728.0), ("Flora.", 727.44, 728.039), ("Ele", 774.36, 774.519)],
        [("Hum.", 712.28, 712.659), ("Flora.", 727.44, 728.039), ("Ele", 774.36, 774.519)], id="ep17-727-flora"),
    pytest.param(
        [("Luki,", 2428.44, 2428.74), ("quando", 2428.8, 2428.999), ("Luke,", 2428.44, 2428.799),
         ("quando", 2428.88, 2429.079), ("você", 2429.12, 2429.28)],
        [("Luke,", 2428.44, 2428.799), ("quando", 2428.88, 2429.079), ("você", 2429.12, 2429.28)], id="ep17-2428-luke"),
]


@pytest.mark.parametrize("raw,expected", _REDECODED_RUNS)
def test_redecoded_word_run_keeps_only_the_later_copy(monkeypatch, tmp_path, raw, expected):
    words, adapter = _replay_raw(monkeypatch, tmp_path, raw)

    assert words == expected
    flags = [flag for flag in adapter.last_repair_flags if flag.kind == "asr_duplicate_words_dropped"]
    assert len(flags) == 1
    assert flags[0].severity == "info"


# ep11 1928.7: "Dez, nove, ..." was transcribed with a re-decoded "De-" and every
# number twice; the energy has one burst per spoken number.
_COUNTDOWN = [
    ("regressiva.", 1918.16, 1918.96), ("De-", 1928.72, 1929.0), ("10,", 1928.72, 1929.419), ("10,", 1929.44, 1929.819),
    ("9,", 1929.84, 1929.9), ("9,", 1929.919, 1930.579), ("8,", 1930.88, 1931.039), ("8,", 1931.08, 1931.779),
    ("7,", 1932.2, 1932.38), ("7,", 1932.4, 1932.979), ("6,", 1933.4, 1933.819), ("6,", 1933.84, 1934.059),
    ("5,", 1934.16, 1934.479), ("5,", 1934.6, 1935.139), ("4,", 1935.68, 1936.18), ("3,", 1936.32, 1936.6),
    ("3,", 1936.72, 1937.22), ("2,", 1937.44, 1937.56), ("2,", 1937.6, 1938.1), ("1.", 1938.56, 1939.159),
    ("Uou!", 1939.48, 1940.3),
]


def test_doubled_countdown_numbers_collapse_to_one_copy_each(monkeypatch, tmp_path):
    words, adapter = _replay_raw(monkeypatch, tmp_path, _COUNTDOWN)

    assert [text for text, _start, _end in words] == [
        "regressiva.", "10,", "9,", "8,", "7,", "6,", "5,", "4,", "3,", "2,", "1.", "Uou!",
    ]
    # The longer copy of each pair is kept; the short or pause-filling copy is dropped.
    assert words[1] == ("10,", 1928.72, 1929.419)
    assert words[2] == ("9,", 1929.919, 1930.579)
    assert words[8] == ("3,", 1936.72, 1937.22)
    assert all(left[2] <= right[1] for left, right in zip(words, words[1:]))
    kinds = sorted(flag.kind for flag in adapter.last_repair_flags)
    assert kinds == ["asr_doubled_words_collapsed", "asr_duplicate_words_dropped"]
    assert all(flag.severity == "info" for flag in adapter.last_repair_flags)


def test_doubled_chorus_words_are_kept_with_an_informational_flag(monkeypatch, tmp_path):
    raw = [
        ("Feliz", 1992.32, 1992.5), ("Feliz", 1992.639, 1992.819), ("Ano", 1992.84, 1992.939), ("Ano", 1992.96, 1993.199),
        ("Novo.", 1993.24, 1993.339), ("Novo.", 1993.36, 1993.739), ("Feliz", 1994.36, 1994.579),
        ("Ano", 1994.639, 1994.839), ("Novo.", 1994.92, 1995.239),
    ]
    words, adapter = _replay_raw(monkeypatch, tmp_path, raw)

    assert words == [(text, start, end) for text, start, end in raw]
    assert [(flag.kind, flag.severity) for flag in adapter.last_repair_flags] == [("asr_doubled_word_run_kept", "info")]
    # The replay shifts the clip so its first word starts at 0.5 s.
    assert adapter.last_repair_flags[0].start == pytest.approx(0.5)
    assert adapter.last_repair_flags[0].end == pytest.approx(1993.739 - 1991.82)


@pytest.mark.parametrize("raw", [
    pytest.param([("Nein,", 1.0, 1.3), ("nein,", 1.36, 1.7), ("bitte", 1.8, 2.1)], id="nein-nein"),
    pytest.param([("que", 1181.72, 1181.82), ("que", 1181.84, 1181.92), ("eu", 1182.0, 1182.1)], id="ep11-que-que"),
    pytest.param([("eu,", 1691.76, 1691.94), ("eu,", 1692.0, 1692.2), ("eu", 1692.48, 1692.56)], id="ep11-eu-eu-eu"),
    pytest.param([("Ja,", 1.0, 1.2), ("ja.", 1.26, 1.5), ("Nein,", 1.6, 1.9), ("nein.", 1.96, 2.3)], id="two-doubled-pairs"),
    pytest.param([("10,", 1.0, 1.4), ("10,", 1.46, 1.9), ("Fertig.", 2.0, 2.4)], id="single-number-pair"),
    pytest.param([("Rápido,", 1300.56, 1300.86), ("rápido.", 1300.88, 1301.1)], id="ep11-rapido"),
])
def test_genuine_repetitions_are_never_removed_or_flagged(monkeypatch, tmp_path, raw):
    words, adapter = _replay_raw(monkeypatch, tmp_path, raw)

    assert words == [(text, start, end) for text, start, end in raw]
    assert adapter.last_repair_flags == []


def test_overlapping_speech_of_different_diarized_speakers_is_not_treated_as_a_duplicate(monkeypatch, tmp_path):
    raw = [("Hallo", 1.0, 1.5, 0), ("du", 1.6, 1.9, 0), ("Nein!", 1.2, 1.6, 1), ("Geh.", 1.7, 2.0, 1)]
    words, adapter = _replay_raw(monkeypatch, tmp_path, raw, diarize=True)

    assert [text for text, _start, _end in words] == ["Hallo", "Nein!", "du", "Geh."]
    assert adapter.last_repair_flags == []


def test_redecoded_run_that_does_not_cover_the_earlier_words_is_kept(monkeypatch, tmp_path):
    # A rewind whose later words do not re-cover the earlier run is not a re-decode.
    raw = [("Eins", 1.0, 1.4), ("zwei", 2.0, 3.0), ("drei", 1.5, 1.7), ("vier", 3.4, 3.8)]
    words, adapter = _replay_raw(monkeypatch, tmp_path, raw)

    assert [text for text, _start, _end in words] == ["Eins", "drei", "zwei", "vier"]
    assert adapter.last_repair_flags == []
