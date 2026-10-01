import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import Cue
from dubsync.srt_io import write_srt


class RecordingAdapter:
    def __init__(self):
        self.seen = []
        self.context = []

    def set_adjudication_context(self, **kwargs):
        self.context.append(kwargs)

    def adjudicate(self, spans):
        self.seen.extend(spans)
        return [dict(case_id=s.case_id, verdict="use_audio", final_text=s.asr_text,
                     confidence=1, reason="clear audio") for s in spans]


def _run_inputs(tmp_path, monkeypatch, register="script"):
    source = tmp_path / "episode.srt"
    cues = [Cue(index=i, start_ms=i*2000, end_ms=i*2000+1500,
                lines=[("old orange anchor" if i == 1 else "old purple anchor" if i == 8 else f"middle cue number {i}")])
            for i in range(1, 9)]
    source.write_text(write_srt(cues), encoding="utf-8")
    words = []
    for cue in cues:
        for position, token in enumerate(cue.plain_text.replace("old", "fresh").split()):
            words.append(dict(text=token, start=cue.start_ms/1000+position*.3, end=cue.start_ms/1000+position*.3+.25))
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)},
        "llm": {"provider": "fixture"}, "adjudication": {"register_policy": register}}), encoding="utf-8")
    adapter = RecordingAdapter()
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **k: adapter)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *a, **k: None)
    def run():
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, language="en")
    return adapter, source, run


def test_pipeline_reuses_other_case_when_one_source_case_changes(tmp_path, monkeypatch):
    adapter, source, run = _run_inputs(tmp_path, monkeypatch)
    run()
    assert len(adapter.seen) == 2
    source.write_text(source.read_text(encoding="utf-8").replace("old orange", "stale orange"), encoding="utf-8")
    adapter.seen.clear()
    run()
    assert len(adapter.seen) == 1
    assert adapter.seen[0].srt_text == "stale"


@pytest.mark.parametrize("register", ["script", "spoken"])
def test_pipeline_binds_resolved_language_and_register(tmp_path, monkeypatch, register):
    adapter, _, run = _run_inputs(tmp_path, monkeypatch, register)
    run()
    assert adapter.context == [{"language": "en", "register_policy": register}]


def test_invalid_wording_policy_fails_before_asr(tmp_path, monkeypatch):
    _, _, run = _run_inputs(tmp_path, monkeypatch, "guess")
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **k: pytest.fail("ASR must not start"))
    with pytest.raises(ValueError, match="register_policy"):
        run()


def test_distant_authored_name_ambiguity_invalidates_case_cache():
    from dubsync.adjudication_policy import DeterministicAdjudicationPolicy
    from dubsync.models import DivergenceSpan
    cues = [Cue(index=i, start_ms=i*1000, end_ms=i*1000+800,
                lines=["Hello Donny" if i in (1, 2) else "Donny" if i == 3 else "Hello friend"])
            for i in range(1, 13)]
    span = DivergenceSpan(case_id="name", cue_ids=[3], srt_text="Donny", asr_text="Dony")
    config = {"asr": {"language_code": "en"}, "llm": {"provider": "fixture"}}
    before_policy = DeterministicAdjudicationPolicy(cues, "en")
    before = pipeline._adjudication_case_keys([span], config, cues, [], None)["name"]
    cues[-1] = cues[-1].model_copy(update={"lines": ["Hello Donnie"]})
    after_policy = DeterministicAdjudicationPolicy(cues, "en")
    assert before_policy.source_names == after_policy.source_names
    assert before_policy.decide(span) is not None and after_policy.decide(span) is None
    after = pipeline._adjudication_case_keys([span], config, cues, [], None)["name"]
    assert before.digest != after.digest


def test_one_review_outage_preserves_successful_case_cache(tmp_path, monkeypatch):
    adapter, _, run = _run_inputs(tmp_path, monkeypatch)
    adapter.route_report = lambda: {"counts": {"held": 1}, "decisions": [
        {"case_id": adapter.seen[-1].case_id, "route": "held", "reasons": ["review_provider_failure"]}
    ]}
    original = adapter.adjudicate
    def adjudicate(spans):
        result = original(spans)
        result[-1].update(verdict="keep_srt", final_text=spans[-1].srt_text, confidence=0)
        return result
    adapter.adjudicate = adjudicate
    run()
    assert len(adapter.seen) == 2
    adapter.seen.clear()
    run()
    assert len(adapter.seen) == 1
