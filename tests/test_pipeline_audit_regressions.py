from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
import yaml

from dubsync.cache import CacheKey, JsonDiskCache
from dubsync.cache import _sha256_file
from dubsync.pipeline import sync_episode
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.style_profile import StyleProfile


def _audio(path: Path, sample: int = 1000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * 16000)


def _sync_fixture(tmp_path: Path):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:00,100 --> 00:00:00,600\nHello there.\n", encoding="utf-8")
    audio = tmp_path / "episode.wav"
    _audio(audio)
    wordstream = tmp_path / "words.json"
    wordstream.write_text(json.dumps({"words": [
        {"text": "Hello", "start": 0.1, "end": 0.3},
        {"text": "there.", "start": 0.3, "end": 0.6},
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({"asr": {"fixture_path": str(wordstream)}}), encoding="utf-8")
    options = dict(srt_path=source, audio_path=audio, output_path=tmp_path / "result.srt",
                   workdir=tmp_path / "work", providers_path=providers, no_llm=True)
    first = sync_episode(**options)
    return options, first


@pytest.mark.parametrize("stage", ["align", "adjudicate", "rebuild", "verify"])
def test_resume_rejects_changed_audio_before_overwriting_any_artifact(tmp_path, stage):
    options, first = _sync_fixture(tmp_path)
    paths = [first.output_srt, *first.episode_workdir.glob("*.json")]
    before = {path: path.read_bytes() for path in paths}
    _audio(options["audio_path"], sample=2000)

    with pytest.raises(ValueError, match="audio.*changed|different audio"):
        sync_episode(**options, resume=stage, fps=25)

    assert {path: path.read_bytes() for path in paths} == before


def test_resume_legacy_asr_explicitly_reports_unverified_audio_provenance(tmp_path):
    options, first = _sync_fixture(tmp_path)
    path = first.episode_workdir / "asr.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("metadata", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = sync_episode(**options, resume="align")

    assert any(flag["kind"] == "asr_audio_provenance_unverified" for flag in result.report["flags"])


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_resume_rejects_invalid_normalized_audio(tmp_path, mutation):
    options, first = _sync_fixture(tmp_path)
    normalized = first.episode_workdir / "audio.16k.wav"
    _audio(normalized)
    asr = first.episode_workdir / "asr.json"
    payload = json.loads(asr.read_text(encoding="utf-8"))
    payload["metadata"]["audio_provenance"] = {
        "source_sha256": _sha256_file(options["audio_path"]),
        "asr_input_sha256": _sha256_file(normalized),
        "normalized": True,
    }
    asr.write_text(json.dumps(payload), encoding="utf-8")
    if mutation == "changed":
        _audio(normalized, sample=2000)
    else:
        normalized.unlink()

    with pytest.raises(ValueError, match="normalized audio is missing or changed"):
        sync_episode(**options, resume="align")


def test_resume_ignores_unrelated_normalized_audio_when_asr_used_original(tmp_path, monkeypatch):
    options, first = _sync_fixture(tmp_path)
    _audio(first.episode_workdir / "audio.16k.wav", sample=2000)
    paths = []

    def activity_adapter(config):
        class Adapter:
            def detect(self, audio):
                paths.append(audio)
                return []
        return Adapter()

    monkeypatch.setattr("dubsync.pipeline.speech_activity_adapter_from_config", activity_adapter)
    sync_episode(**options, resume="align")

    assert paths == [options["audio_path"]]


def test_cache_failed_publication_preserves_previous_valid_result(tmp_path, monkeypatch):
    cache = JsonDiskCache(tmp_path / "cache")
    key = CacheKey.from_payload({"episode": "one"}, "fixture", {})
    cache.write(key, {"words": ["complete"]})
    previous = cache._path(key).read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        cache.write(key, {"words": ["replacement"]})

    assert cache._path(key).read_bytes() == previous
    assert cache.read(key) == {"words": ["complete"]}
    assert list(cache.root.iterdir()) == [cache._path(key)]


def test_cache_serialization_failure_preserves_previous_value(tmp_path):
    cache = JsonDiskCache(tmp_path / "cache")
    key = CacheKey.from_payload({}, "fixture", {})
    cache.write(key, {"words": ["complete"]})

    with pytest.raises(TypeError):
        cache.write(key, {"words": object()})

    assert cache.read(key) == {"words": ["complete"]}


def test_resume_verify_rejects_prior_rebuild_policy_before_overwriting(tmp_path):
    options, first = _sync_fixture(tmp_path)
    path = first.episode_workdir / "rebuild.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("policy_version")
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = first.output_srt.read_bytes()

    with pytest.raises(ValueError, match="resume from rebuild"):
        sync_episode(**options, resume="verify")

    assert first.output_srt.read_bytes() == before


def test_resume_verify_rechecks_new_confidence_gate_even_for_keep_srt(tmp_path):
    options, first = _sync_fixture(tmp_path)
    align_path = first.episode_workdir / "align.json"
    alignment = json.loads(align_path.read_text(encoding="utf-8"))
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="Hello there.",
                          asr_text="Hello again.", asr_word_indices=[1])
    alignment["divergence_spans"] = [span.model_dump()]
    align_path.write_text(json.dumps(alignment), encoding="utf-8")
    decision = AdjudicationDecision(case_id="case-1", verdict="keep_srt", final_text="Hello there.",
                                    confidence=0.8, reason="source judged right")
    (first.episode_workdir / "adjudicate.json").write_text(
        json.dumps({"decisions": [decision.model_dump()], "flags": []}), encoding="utf-8")
    providers = yaml.safe_load(options["providers_path"].read_text(encoding="utf-8"))
    providers["llm"] = {"adjudication": {"confidence_gate": 0.9}}
    options["providers_path"].write_text(yaml.safe_dump(providers), encoding="utf-8")

    with pytest.raises(ValueError, match="confidence gate.*resume from rebuild"):
        sync_episode(**options, resume="verify")


def test_low_confidence_hold_does_not_delete_overlapping_trusted_dialogue(tmp_path, monkeypatch):
    options, first = _sync_fixture(tmp_path)
    options["srt_path"].write_text(
        "1\n00:00:00,100 --> 00:00:00,600\nUncertain.\n\n"
        "2\n00:00:00,400 --> 00:00:00,900\nTrusted.\n", encoding="utf-8")
    words = [Word(text="Different.", start=.1, end=.6, speaker_id="A"),
             Word(text="Trusted.", start=.4, end=.9, speaker_id="B")]
    stream = tmp_path / "overlap-words.json"
    stream.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    options["providers_path"].write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(stream)},
        "llm": {"provider": "fixture", "responses": {"case-1": {
            "case_id": "case-1", "verdict": "use_audio", "final_text": "Different.",
            "confidence": .1, "reason": "uncertain audio", "speaker": "A", "character": "unknown",
        }}},
    }), encoding="utf-8")
    span = DivergenceSpan(case_id="case-1", cue_ids=[1], srt_text="Uncertain.", asr_text="Different.",
                          start=.1, end=.6, srt_token_indices=[0], asr_word_indices=[0])
    monkeypatch.setattr("dubsync.pipeline.align_cues_to_words", lambda *_args: AlignmentResult(
        cue_word_indices={1: [0], 2: [1]}, divergence_spans=[span], anchor_coverage=1))
    options["no_llm"] = False

    result = sync_episode(**options, style_profile=StyleProfile(overlap_policy="dash"))
    from dubsync.srt_io import parse_srt_text
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["Uncertain.", "Trusted."]
    assert (cues[0].start_ms, cues[0].end_ms) == (100, 600)


def test_uncertain_span_timing_hold_preserves_separate_approved_improvisation():
    from dubsync.pipeline import _restore_missing_audio_source_cues
    source = Cue(index=135, start_ms=549_800, end_ms=551_500,
                 lines=["Eu também posso dizer que meus anéis"])
    approved = source.model_copy(update={"start_ms": 549_810, "end_ms": 551_540,
                                         "lines": ["Eu também posso dizer que você roubou meus anéis"]})

    restored, _ = _restore_missing_audio_source_cues([approved], [source], {135}, reason="low_confidence")

    assert restored[0].lines == approved.lines
    assert (restored[0].start_ms, restored[0].end_ms) == (source.start_ms, source.end_ms)


def test_browser_fixture_selects_only_offline_language_adapters():
    from dubsync.adjudication import StaticLLMAdapter
    from dubsync.llm_providers import llm_adapter_from_config, punctuation_adapter_from_config
    config = yaml.safe_load(Path("web/e2e/fixtures/providers.yaml").read_text(encoding="utf-8"))

    assert isinstance(llm_adapter_from_config(config, pass_name="adjudication"), StaticLLMAdapter)
    assert punctuation_adapter_from_config(config) is None
