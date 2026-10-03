from contextlib import contextmanager
from copy import deepcopy
import wave

import pytest

from dubsync.cache import CacheKey
from dubsync.models import AdjudicationDecision, AudioSnippet, DivergenceSpan


def _case(tmp_path):
    path = tmp_path / "heard.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 32000)
    span = DivergenceSpan(case_id="case-156", cue_ids=[44], srt_text="source", asr_text="heard",
                          start=.5, end=1.5)
    decision = AdjudicationDecision(case_id=span.case_id, verdict="use_audio", final_text="heard",
                                   heard_text="heard", evidence="heard_clearly", confidence=1, reason="Native hearing.")
    clip = AudioSnippet(case_id=span.case_id, path=str(path), start=0, end=2)
    context = {"strategy": "bounded_batches_v3", "audio_sha256": "a" * 64,
               "pad_seconds": 2, "max_duration_seconds": 20}
    key = CacheKey.from_payload({"span": span.model_dump(exclude={"case_id"}), "audio": context},
                                model="native-test", params={})
    return span, decision, clip, context, key


def _record(tmp_path):
    from dubsync.adjudication_audio_provenance import AudioProvenanceRecorder
    span, decision, clip, context, key = _case(tmp_path)
    @contextmanager
    def load(spans):
        assert spans == [span]
        yield {clip.case_id: clip}
    recorder = AudioProvenanceRecorder(load)
    with recorder.load([span]) as clips:
        assert clips == {span.case_id: clip}
    return span, decision, clip, context, key, recorder.manifest()


def test_recorded_native_clip_survives_partial_cache_reuse_and_case_renumbering(tmp_path):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance, cached_case_audio_snippet
    span, decision, clip, context, key, manifest = _record(tmp_path)
    receipt = bind_case_audio_provenance(key, span, decision, manifest, context)
    assert receipt is not None
    # The bounded source removes its transient files after the actual hearing.
    from pathlib import Path
    Path(clip.path).unlink()
    renamed = span.model_copy(update={"case_id": "case-157"})
    answer = decision.model_copy(update={"case_id": renamed.case_id})
    restored = cached_case_audio_snippet(receipt, key, renamed, answer, context)
    assert restored == {**manifest["snippets"][0], "case_id": renamed.case_id}
    assert "path" not in str(receipt)


@pytest.mark.parametrize("change", ["audio", "options", "key", "decision", "span", "missing_context", "invalid_hash"])
def test_cached_provenance_rejects_changed_evidence(tmp_path, change):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance, cached_case_audio_snippet
    span, decision, _, context, key, manifest = _record(tmp_path)
    receipt = bind_case_audio_provenance(key, span, decision, manifest, context)
    assert receipt
    if change == "audio":
        context = {**context, "audio_sha256": "b" * 64}
    elif change == "options":
        context = {**context, "pad_seconds": 1}
    elif change == "key":
        key = key.model_copy(update={"digest": "b" * 64})
    elif change == "decision":
        decision = decision.model_copy(update={"reason": "Different native evidence."})
    elif change == "span":
        span = span.model_copy(update={"srt_text": "other"})
    elif change == "missing_context":
        context = None
    else:
        context = {**context, "audio_sha256": "not-a-hash"}
    assert cached_case_audio_snippet(receipt, key, span, decision, context) is None


@pytest.mark.parametrize("change", ["legacy_manifest", "duplicate", "stale_id", "partial", "wrong_decision_id", "not_native"])
def test_bind_rejects_unbound_or_ambiguous_records(tmp_path, change):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance
    span, decision, _, context, key, manifest = _record(tmp_path)
    if change == "legacy_manifest":
        manifest.pop("capture_policy_version")
    elif change == "duplicate":
        manifest["snippets"].append(deepcopy(manifest["snippets"][0]))
    elif change == "stale_id":
        manifest["snippets"][0]["case_id"] = "unrelated"
    elif change == "partial":
        manifest["snippets"][0]["end"] = 1
    elif change == "wrong_decision_id":
        decision = decision.model_copy(update={"case_id": "unrelated"})
    else:
        decision = decision.model_copy(update={"evidence": None, "heard_text": None})
    assert bind_case_audio_provenance(key, span, decision, manifest, context) is None


@pytest.mark.parametrize("change", ["snippet", "case_id", "receipt", "legacy"])
def test_cached_receipt_rejects_metadata_tampering(tmp_path, change):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance, cached_case_audio_snippet
    span, decision, _, context, key, manifest = _record(tmp_path)
    receipt = bind_case_audio_provenance(key, span, decision, manifest, context)
    if change == "snippet":
        receipt["snippet"]["sha256"] = "c" * 64
    elif change == "case_id":
        receipt["snippet"]["case_id"] = "unrelated"
    elif change == "receipt":
        receipt["receipt_sha256"] = "c" * 64
    else:
        receipt = manifest
    assert cached_case_audio_snippet(receipt, key, span, decision, context) is None


@pytest.mark.parametrize("change", ["none", "bare_row", "snippet", "decision", "span", "audio", "key"])
def test_receipt_clip_is_only_the_one_bound_to_this_question_answer_and_audio(tmp_path, change):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance, receipt_audio_snippet
    span, decision, _, context, key, manifest = _record(tmp_path)
    receipt = bind_case_audio_provenance(key, span, decision, manifest, context)
    renamed = span.model_copy(update={"case_id": "case-157"})
    answer = decision.model_copy(update={"case_id": renamed.case_id})
    audio = context["audio_sha256"]
    if change == "bare_row":
        receipt = deepcopy(manifest["snippets"][0])
    elif change == "snippet":
        receipt["snippet"]["sha256"] = "c" * 64
    elif change == "decision":
        answer = answer.model_copy(update={"reason": "Different native evidence."})
    elif change == "span":
        renamed = renamed.model_copy(update={"srt_text": "other"})
    elif change == "audio":
        audio = "b" * 64
    elif change == "key":
        receipt.pop("case_key_sha256")
    clip = receipt_audio_snippet(receipt, renamed, answer, audio)
    assert clip == ({**manifest["snippets"][0], "case_id": "case-157"} if change == "none" else None)


@pytest.mark.parametrize("change", ["tamper_during_hearing", "truncated", "wrong_rate", "wrong_id", "duplicate_span", "exception"])
def test_recorder_rejects_invalid_or_changed_actual_clip_bytes(tmp_path, change):
    from dubsync.adjudication_audio_provenance import AudioProvenanceRecorder
    span, _, clip, _, _ = _case(tmp_path)
    if change in {"truncated", "wrong_rate"}:
        with wave.open(clip.path, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000 if change == "wrong_rate" else 16000)
            stream.writeframes(b"\0\0" * 8000)
    if change == "wrong_id":
        clip = clip.model_copy(update={"case_id": "unrelated"})
    @contextmanager
    def load(_spans):
        yield {span.case_id: clip}
    recorder = AudioProvenanceRecorder(load)
    try:
        with recorder.load([span, span] if change == "duplicate_span" else [span]):
            if change == "tamper_during_hearing":
                with open(clip.path, "r+b") as stream:
                    stream.seek(50)
                    stream.write(b"\x05\x05")
            if change == "exception":
                raise RuntimeError("provider interrupted")
    except RuntimeError:
        assert change == "exception"
    assert recorder.manifest()["snippets"] == []


def test_full_episode_audio_context_must_agree_with_focused_audio_hash(tmp_path):
    from dubsync.adjudication_audio_provenance import bind_case_audio_provenance, cached_case_audio_snippet
    span, decision, _, focused, key, manifest = _record(tmp_path)
    context = {"focused_snippets": focused, "episode_audio": {"source_sha256": "b" * 64,
               "normalized_sha256": focused["audio_sha256"], "duration_seconds": 2, "options": {"enabled": True}}}
    receipt = bind_case_audio_provenance(key, span, decision, manifest, context)
    assert receipt and cached_case_audio_snippet(receipt, key, span, decision, context)
    context["episode_audio"]["normalized_sha256"] = "c" * 64
    assert bind_case_audio_provenance(key, span, decision, manifest, context) is None


def test_recorder_does_not_claim_padding_beyond_actual_audio(tmp_path):
    from dubsync.adjudication_audio_provenance import AudioProvenanceRecorder
    span, _, clip, _, _ = _case(tmp_path)
    clip = clip.model_copy(update={"end": 3})
    @contextmanager
    def load(_spans):
        yield {span.case_id: clip}
    recorder = AudioProvenanceRecorder(load)
    with recorder.load([span]):
        pass
    assert recorder.manifest()["snippets"][0]["end"] == 2
