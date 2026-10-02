import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
from test_missing_dialogue_reconciliation import _pipeline_case


def _native_case(tmp_path, monkeypatch):
    source = [Cue(index=600, start_ms=1900, end_ms=2134, lines=["né?"]),
              Cue(index=601, start_ms=8630, end_ms=8667, lines=["É."]),
              Cue(index=602, start_ms=8655, end_ms=9100, lines=["Tomam."])]
    words = [Word(text=text, start=start, end=end, speaker_id="primary") for text, start, end in
             [("né?", 1.93, 2.105), ("É.", 8.630, 8.631), ("Tomem.", 8.655, 9.045)]]
    secondary = [Word(text=text, start=start, end=end, speaker_id="secondary") for text, start, end in
                 [("né?", 1.92, 2.159), ("É.", 7.04, 7.32), ("Tomem.", 8.639, 9.079)]]
    alignment = AlignmentResult(cue_word_indices={600: [0], 601: [1], 602: [2]},
        token_matches=[TokenMatch(cue_id=600, srt_token_index=0, asr_word_index=0, score=1),
                       TokenMatch(cue_id=601, srt_token_index=1, asr_word_index=1, score=1)],
        divergence_spans=[DivergenceSpan(case_id="ordinary-tomem", cue_ids=[602], srt_text="Tomam",
            srt_token_indices=[2], asr_text="Tomem.", asr_word_indices=[2], start=8.655, end=9.045)],
        diagnostics={"missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION})
    regions = [SpeechRegion(start=start, end=end) for start, end in
               [(1.9, 2.105), (4.605, 5.305), (6.515, 6.585), (6.975, 7.245), (8.655, 9.045)]]
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=(source, words, alignment, regions))
    secondary_path = tmp_path / "secondary-asr-fixture.json"
    secondary_path.write_text(json.dumps({"words": [word.model_dump() for word in secondary]}), encoding="utf-8")
    config_path = tmp_path / "provider.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["asr"].update(provider="elevenlabs", model_id="scribe_v2", cross_check={
        "provider": "openrouter", "model": "microsoft/mai-transcribe-2", "fixture_path": str(secondary_path)})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def hear(spans, snippets):
        adapter.seen.extend(spans)
        assert all(snippets[span.case_id].start <= span.start and snippets[span.case_id].end >= span.end for span in spans)
        return [AdjudicationDecision(case_id=span.case_id,
            verdict="use_audio" if span.case_id == "ordinary-tomem" else "keep_srt",
            final_text="Tomem" if span.case_id == "ordinary-tomem" else span.srt_text,
            heard_text="Tomem" if span.case_id == "ordinary-tomem" else span.srt_text,
            evidence="heard_clearly", confidence=1, reason="Synthetic unit audio fixture.").model_dump() for span in spans]
    adapter.adjudicate_with_audio = hear
    return alignment, adapter, run, secondary


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_singleton_pipeline_hears_complete_word_and_replays_bound_secondary_proof(tmp_path, monkeypatch, mode):
    alignment, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    result = run()
    assert len([span for span in adapter.seen if span.case_id.startswith("collapsed-singleton-timing-")]) == 1
    original_output = result.output_srt.read_bytes()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
        assert result.output_srt.read_bytes() == original_output
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in rebuilt["cues"] if cue["index"] == 601)
    assert target["start_ms"] <= 6975 and 7245 <= target["end_ms"] < 8630
    assert target["lines"] == ["É."]
    assert rebuilt["alignment"]["cue_word_indices"] == {str(k): v for k, v in alignment.cue_word_indices.items()}
    receipt = json.loads((result.episode_workdir / "collapsed_singleton_timing.json").read_text(encoding="utf-8"))
    assert rebuilt["collapsed_singleton_receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_utterance"


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_singleton_replay_rejects_modified_receipt_without_native_retry(tmp_path, monkeypatch, mode):
    _, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "collapsed_singleton_timing.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0]["heard_text"] = "A different word"
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []


def test_singleton_disabled_hearing_never_releases_target(tmp_path, monkeypatch):
    _, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    result = run(no_llm=True)
    assert adapter.seen == []
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert not any(flag["kind"] == "collapsed_singleton_audio_reconciled" for flag in rebuilt["pre_output_flags"])


def test_singleton_negative_hearing_is_cached_and_keeps_source_hold(tmp_path, monkeypatch):
    _, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    native = adapter.adjudicate_with_audio
    def negative(spans, snippets):
        answers = native(spans, snippets)
        for item in answers:
            if item["case_id"].startswith("collapsed-singleton-timing-"):
                item.update(heard_text="", evidence="not_audible", confidence=1)
        return answers
    adapter.adjudicate_with_audio = negative
    first = run()
    assert any(span.case_id.startswith("collapsed-singleton-timing-") for span in adapter.seen)
    adapter.seen.clear()
    second = run()
    assert adapter.seen == [] and second.output_srt.read_bytes() == first.output_srt.read_bytes()
    payload = json.loads((second.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    target = next(cue for cue in payload["cues"] if cue["index"] == 601)
    assert target["start_ms"] == 8630


def test_singleton_verify_restores_the_exact_post_adjudication_decision_state(tmp_path, monkeypatch):
    _, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    original = pipeline._absorb_redecoded_insertions
    def transformed(*args, **kwargs):
        alignment, decisions, flags = original(*args, **kwargs)
        return alignment, [decision.model_copy(update={"reason": decision.reason + " Post-adjudication mapping."})
                           for decision in decisions], flags
    monkeypatch.setattr(pipeline, "_absorb_redecoded_insertions", transformed)
    first = run()
    original_output = first.output_srt.read_bytes()
    adapter.seen.clear()
    second = run(resume="verify")
    assert adapter.seen == [] and second.output_srt.read_bytes() == original_output


def test_singleton_verify_rejects_a_changed_ordinary_adjudication_artifact(tmp_path, monkeypatch):
    _, adapter, run, _ = _native_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "adjudicate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0]["reason"] += " Changed after the hearing."
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume="verify")
    assert adapter.seen == []
