from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline


def _fixture(tmp_path, *, generated=False, model="scribe_v2"):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello." if generated else
                      "1\n00:00:01,000 --> 00:00:02,000\nHello friend.", encoding="utf-8")
    audio = tmp_path / "episode.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 10)
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"words": [
        {"text": "Hello." if generated else "Hello", "start": 1.04, "end": 1.15},
        {"text": "Surprise!" if generated else "there.", "start": 3.0 if generated else 1.2,
         "end": 3.3 if generated else 8.0},
    ]}), encoding="utf-8")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": (
        [{"start": 1.0, "end": 1.15}, {"start": 3.0, "end": 3.3}]
        if generated else [{"start": 1.0, "end": 1.4}]
    )}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(words), "model": model},
        "vad": {"fixture_path": str(regions)},
        "llm": {"provider": "fixture", "audio_snippet_double_check": {"enabled": not generated, "pad_seconds": 0},
                "responses": {"case-1": {"case_id": "case-1", "verdict": "use_audio", "confidence": 0.95,
                                            "final_text": "Surprise!" if generated else "there.",
                                            "reason": "Fixture confirms the spoken words."}}},
    }), encoding="utf-8")
    return dict(srt_path=source, audio_path=audio, output_path=tmp_path / "out.srt",
                workdir=tmp_path / "work", providers_path=providers)


@pytest.mark.parametrize("model", ["scribe_v2", "microsoft/mai-transcribe-2"])
def test_one_repair_supplies_alignment_snippets_text_rebuild_and_verify(tmp_path, monkeypatch, model):
    options = _fixture(tmp_path, model=model)
    observed = {}
    repairs = []
    original_evidence = pipeline.speech_evidence_for_words

    def evidence(adapter, words, *args, **kwargs):
        repairs.append([word.model_dump() for word in words])
        return original_evidence(adapter, words, *args, **kwargs)

    monkeypatch.setattr(pipeline, "speech_evidence_for_words", evidence)
    for name, position in [("align_cues_to_words", 1), ("rebuild_cues", 1)]:
        original = getattr(pipeline, name)

        def record(*args, _name=name, _position=position, _original=original, **kwargs):
            observed.setdefault(_name, []).append([word.model_dump() for word in args[_position]])
            return _original(*args, **kwargs)

        monkeypatch.setattr(pipeline, name, record)
    original_apply = pipeline.apply_adjudication_decisions
    original_verify = pipeline._run_verify_stage
    original_extract = pipeline.extract_audio_snippets

    def apply(*args, **kwargs):
        observed.setdefault("text", []).append([word.model_dump() for word in kwargs["words"]])
        return original_apply(*args, **kwargs)

    def verify(**kwargs):
        observed.setdefault("verify", []).append([word.model_dump() for word in kwargs["words"]])
        assert kwargs["words"] == kwargs["speech_evidence"].words
        return original_verify(**kwargs)

    def extract(audio, spans, *args, **kwargs):
        observed["snippets"] = [(span.start, span.end) for span in spans]
        return original_extract(audio, spans, *args, **kwargs)

    monkeypatch.setattr(pipeline, "apply_adjudication_decisions", apply)
    monkeypatch.setattr(pipeline, "_run_verify_stage", verify)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    result = pipeline.sync_episode(**options)

    assert len(repairs) == 1
    assert repairs[0][1]["end"] == 8.0
    for stage in ["align_cues_to_words", "rebuild_cues", "text", "verify"]:
        assert observed[stage]
        assert all(words[0]["start"] == 1.0 and words[1]["end"] == 1.4 for words in observed[stage])
    assert observed["snippets"] == [(1.2, 1.4)]
    raw = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))
    aligned = json.loads((result.episode_workdir / "align.json").read_text(encoding="utf-8"))
    assert raw["words"][1]["end"] == 8.0
    assert aligned["divergence_spans"][0]["end"] == 1.4
    assert aligned["word_timing"]["source_words_sha256"] != aligned["word_timing"]["words_sha256"]


@pytest.mark.parametrize("resume", ["align", "adjudicate", "rebuild", "verify"])
def test_resume_repairs_raw_asr_once_and_keeps_the_raw_artifact(tmp_path, monkeypatch, resume):
    options = _fixture(tmp_path)
    first = pipeline.sync_episode(**options)
    raw_path = first.episode_workdir / "asr.json"
    raw_bytes = raw_path.read_bytes()
    output = first.output_srt.read_bytes()
    inputs = []
    original = pipeline.speech_evidence_for_words

    def evidence(adapter, words, *args, **kwargs):
        inputs.append([word.model_dump() for word in words])
        return original(adapter, words, *args, **kwargs)

    monkeypatch.setattr(pipeline, "speech_evidence_for_words", evidence)
    resumed = pipeline.sync_episode(**options, resume=resume)

    assert len(inputs) == 1
    assert inputs[0][1]["end"] == 8.0
    assert raw_path.read_bytes() == raw_bytes
    assert resumed.output_srt.read_bytes() == output
    assert json.loads((first.episode_workdir / "align.json").read_text(encoding="utf-8"))["word_timing"]["policy_version"] > 0


@pytest.mark.parametrize("resume", ["adjudicate", "rebuild", "verify"])
@pytest.mark.parametrize("mutation", ["missing_provenance", "raw_words", "timing_config"])
def test_resume_rejects_stale_word_ownership_before_writing_artifacts(tmp_path, resume, mutation):
    options = _fixture(tmp_path)
    first = pipeline.sync_episode(**options)
    if mutation == "missing_provenance":
        path = first.episode_workdir / "align.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("word_timing", None)
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "raw_words":
        path = first.episode_workdir / "asr.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["words"][1]["end"] = 7.0
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        config = yaml.safe_load(options["providers_path"].read_text(encoding="utf-8"))
        config["timing"] = {"phrase_edge_snap": False}
        options["providers_path"].write_text(yaml.safe_dump(config), encoding="utf-8")
    paths = [first.output_srt, *first.episode_workdir.glob("*.json")]
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(ValueError, match="resume from align"):
        pipeline.sync_episode(**options, resume=resume, fps=25)

    assert {path: path.read_bytes() for path in paths} == before


def test_resume_align_refreshes_legacy_ownership_from_raw_words(tmp_path):
    options = _fixture(tmp_path)
    first = pipeline.sync_episode(**options)
    path = first.episode_workdir / "align.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("word_timing", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    pipeline.sync_episode(**options, resume="align")

    refreshed = json.loads(path.read_text(encoding="utf-8"))
    assert refreshed["word_timing"]["policy_version"] > 0
    assert refreshed["divergence_spans"][0]["end"] == 1.4


def test_verify_resume_uses_final_generated_cue_ownership(tmp_path):
    options = _fixture(tmp_path, generated=True)
    first = pipeline.sync_episode(**options)
    output = first.output_srt.read_bytes()
    checkpoint_path = first.episode_workdir / "rebuild.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert len(checkpoint["cues"]) == 2
    assert checkpoint["alignment"]["cue_word_indices"] == {"1": [0], "2": [1]}
    initial = json.loads((first.episode_workdir / "align.json").read_text(encoding="utf-8"))
    assert initial["cue_word_indices"] == {"1": [0]}

    resumed = pipeline.sync_episode(**options, resume="verify")

    after = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert resumed.output_srt.read_bytes() == output
    assert after["alignment"]["cue_word_indices"] == checkpoint["alignment"]["cue_word_indices"]
    assert after["word_timing"] == checkpoint["word_timing"]


@pytest.mark.parametrize("missing", ["alignment", "word_timing"])
def test_verify_resume_rejects_incomplete_final_ownership_checkpoint(tmp_path, missing):
    options = _fixture(tmp_path, generated=True)
    first = pipeline.sync_episode(**options)
    path = first.episode_workdir / "rebuild.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(missing, None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="resume from align"):
        pipeline.sync_episode(**options, resume="verify")
