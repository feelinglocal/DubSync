from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import DivergenceSpan, Word


MAI = "microsoft/mai-transcribe-2"


def _inputs(tmp_path, *, enabled=True, secondary_text=None):
    source = tmp_path / "episode.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello green apples goodbye.\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 5)
    primary = [dict(text=token, start=1 + i*.45, end=1.3 + i*.45)
               for i, token in enumerate("hello blue boats goodbye.".split())]
    secondary = [dict(word, start=word["start"] + .04, end=word["end"] + .04) for word in primary]
    if secondary_text:
        for word, token in zip(secondary, secondary_text.split()):
            word["text"] = token
    paths = {}
    for name, words in [("primary", primary), ("secondary", secondary)]:
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"words": words}), encoding="utf-8")
    config = {"asr": {"provider": "openrouter", "model": MAI, "fixture_path": str(paths["primary"])},
              "llm": {"provider": "fixture"}, "timing": {"phrase_edge_snap": False}}
    if enabled:
        config["asr"]["cross_check"] = {"provider": "elevenlabs", "model_id": "scribe_v2",
                                       "fixture_path": str(paths["secondary"])}
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    return dict(srt_path=source, audio_path=audio, output_path=tmp_path / "out.srt",
                workdir=tmp_path / "work", providers_path=providers, language="en")


def _config(options, update):
    path = options["providers_path"]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    update(config)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_web_override_off_never_constructs_secondary(tmp_path, monkeypatch):
    options = _inputs(tmp_path)
    _config(options, lambda config: config["asr"].update(cross_check={"bad": True}))
    calls = []
    original = pipeline.adapter_from_config
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda config, **kw: (calls.append(config["asr"]), original(config, **kw))[1])
    result = pipeline.sync_episode(**options, asr_cross_check=False, no_llm=True)
    assert len(calls) == 1
    assert "cross_check" not in calls[0]
    assert not (result.episode_workdir / "asr_cross_check.json").exists()


def test_same_model_rejected_before_primary_provider_is_constructed(tmp_path, monkeypatch):
    options = _inputs(tmp_path)
    _config(options, lambda config: config["asr"].update(cross_check={"provider": "openrouter"}))
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: pytest.fail("Primary ASR must not start"))
    with pytest.raises(ValueError, match="different"):
        pipeline.sync_episode(**options)


def test_missing_secondary_key_rejected_before_primary_provider_is_constructed(tmp_path, monkeypatch):
    options = _inputs(tmp_path)
    _config(options, lambda config: config["asr"]["cross_check"].pop("fixture_path"))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: pytest.fail("Primary ASR must not start"))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        pipeline.sync_episode(**options)


def test_agreement_can_preaccept_only_wording_and_preserves_primary_timing(tmp_path, monkeypatch):
    options = _inputs(tmp_path)
    observed = []
    original_align = pipeline.align_cues_to_words
    monkeypatch.setattr(pipeline, "align_cues_to_words", lambda cues, words, **kw: (observed.append(words), original_align(cues, words, **kw))[1])
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **kw: pytest.fail("Agreed safe span needs no LLM"))
    result = pipeline.sync_episode(**options)
    assert observed[0][1].start == pytest.approx(1.45)
    assert "blue boats" in result.output_srt.read_text(encoding="utf-8")
    secondary = _json(result.episode_workdir / "asr_cross_check.json")
    assert secondary["words"][1]["start"] == pytest.approx(1.49)
    assert _json(result.episode_workdir / "asr.json")["words"][1]["start"] == pytest.approx(1.45)
    analysis = _json(result.episode_workdir / "asr_cross_check_analysis.json")
    assert analysis["preaccepted_case_ids"] == ["case-1"]
    assert analysis["cases"][0]["label"] == "both_agree"
    assert secondary["metadata"]["cost_items"] == []


@pytest.mark.parametrize("resume", ["align", "adjudicate", "rebuild", "verify"])
def test_resume_loads_both_saved_streams_without_creating_provider(tmp_path, monkeypatch, resume):
    options = _inputs(tmp_path)
    result = pipeline.sync_episode(**options, no_llm=True)
    import dubsync.asr_crosscheck_runtime as runtime
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: pytest.fail("Resume must not create primary provider"))
    monkeypatch.setattr(runtime, "adapter_from_config", lambda *a, **kw: pytest.fail("Resume must not create secondary provider"))
    before = (result.episode_workdir / "asr_cross_check.json").read_bytes()
    resumed = pipeline.sync_episode(**options, resume=resume, no_llm=True)
    assert (resumed.episode_workdir / "asr_cross_check.json").read_bytes() == before
    assert resumed.cost_meter.total_usd == 0


@pytest.mark.parametrize("mutation", ["missing", "word", "fixture", "config"])
def test_resume_rejects_missing_or_changed_secondary_evidence_before_rewriting_artifacts(tmp_path, mutation):
    options = _inputs(tmp_path)
    result = pipeline.sync_episode(**options, no_llm=True)
    artifact = result.episode_workdir / "asr_cross_check.json"
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "word":
        content = _json(artifact)
        content["words"][1]["text"] = "tampered"
        artifact.write_text(json.dumps(content), encoding="utf-8")
    elif mutation == "fixture":
        fixture = tmp_path / "secondary.json"
        content = _json(fixture)
        content["words"][1]["text"] = "changed"
        fixture.write_text(json.dumps(content), encoding="utf-8")
    else:
        _config(options, lambda config: config["asr"]["cross_check"].update(diarize=False))
    paths = [result.output_srt, *result.episode_workdir.glob("*.json")]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="cross-check.*resume from asr"):
        pipeline.sync_episode(**options, resume="rebuild", fps=25, no_llm=True)
    assert {path: path.read_bytes() for path in paths} == before


def test_enabling_cross_check_on_old_resume_requires_explicit_asr_rerun(tmp_path, monkeypatch):
    options = _inputs(tmp_path, enabled=False)
    pipeline.sync_episode(**options, no_llm=True)
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: pytest.fail("Resume must not transcribe"))
    with pytest.raises(ValueError, match="cross-check.*resume from asr"):
        pipeline.sync_episode(**options, resume="rebuild", asr_cross_check=True, no_llm=True)


def test_secondary_evidence_is_in_both_adjudication_cache_identities():
    span = DivergenceSpan(case_id="sample", cue_ids=[1], srt_text="green", asr_text="blue", asr_word_indices=[0])
    words = [Word(text="blue", start=1, end=2)]
    config = {"llm": {"provider": "fixture"}}
    changed = {**config, "_asr_cross_check_context": {"words_sha256": "different"}}
    assert pipeline._adjudication_cache_key([span], config).digest != pipeline._adjudication_cache_key([span], changed).digest
    assert pipeline._adjudication_case_keys([span], config, [], words, None)["sample"].digest != pipeline._adjudication_case_keys([span], changed, [], words, None)["sample"].digest


def test_changed_secondary_settings_do_not_retranscribe_primary(tmp_path, monkeypatch):
    options = _inputs(tmp_path)
    result = pipeline.sync_episode(**options, no_llm=True)
    _config(options, lambda config: config["asr"]["cross_check"].update(diarize=False))
    pipeline.sync_episode(**options, no_llm=True)
    primary = _json(result.episode_workdir / "asr.json")
    assert primary["metadata"]["cache_hit"] is True
    assert len(list((result.episode_workdir / "asr-cache").glob("*.json"))) == 1
    assert len(list((result.episode_workdir / "asr-cross-check-cache").glob("*.json"))) == 2
    assert _json(result.episode_workdir / "asr_cross_check.json")["metadata"]["cache_hit"] is False


@pytest.mark.parametrize("evidence", [None, "heard_clearly"])
def test_secondary_script_agreement_holds_legacy_confidence_but_allows_clear_review(tmp_path, monkeypatch, evidence):
    options = _inputs(tmp_path, secondary_text="hello green apples goodbye.")
    class Reviewer:
        def adjudicate(self, spans):
            return [dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text, confidence=.99,
                         reason="Reviewed", **({"evidence": evidence, "heard_text": span.asr_text} if evidence else {}))
                    for span in spans]
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **kw: Reviewer())
    result = pipeline.sync_episode(**options)
    expected = "blue boats" if evidence else "green apples"
    assert expected in result.output_srt.read_text(encoding="utf-8")
    adjudication = _json(result.episode_workdir / "adjudicate.json")
    assert bool([flag for flag in adjudication["flags"] if flag["kind"] == "low_confidence_adjudication"]) is (evidence is None)


def test_both_costs_recorded_independently_and_cached_rerun_is_free(tmp_path, monkeypatch):
    import dubsync.asr_crosscheck_runtime as runtime
    options = _inputs(tmp_path)
    _config(options, lambda config: config["asr"]["cross_check"].pop("fixture_path"))
    primary_words = [Word.model_validate(word) for word in _json(tmp_path / "primary.json")["words"]]
    class Adapter:
        api_key = "fixture-key"
        def __init__(self, charge):
            self.last_usage = {"cost": charge, "seconds": 5}
            self.calls = 0
        def transcribe(self, path):
            self.calls += 1
            return primary_words
    primary, secondary = Adapter(.01), Adapter(.02)
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: primary)
    monkeypatch.setattr(runtime, "adapter_from_config", lambda *a, **kw: secondary)
    monkeypatch.setattr(runtime, "normalize_audio", lambda source, *a, **kw: source)
    first = pipeline.sync_episode(**options, no_llm=True)
    assert first.cost_meter.total_usd == pytest.approx(.03)
    assert {item.provider for item in first.cost_meter.items} == {MAI, "scribe_v2"}
    assert _json(first.episode_workdir / "cost.json")["total_usd"] == pytest.approx(.03)
    second = pipeline.sync_episode(**options, no_llm=True)
    assert (primary.calls, secondary.calls) == (1, 1)
    assert second.cost_meter.total_usd == 0
    assert _json(first.episode_workdir / "asr_cross_check.json")["metadata"]["cache_hit"] is True


def test_secondary_failure_keeps_primary_and_partial_secondary_charges(tmp_path, monkeypatch):
    import dubsync.asr_crosscheck_runtime as runtime
    from dubsync.providers import ProviderError
    options = _inputs(tmp_path)
    _config(options, lambda config: config["asr"]["cross_check"].pop("fixture_path"))
    primary_words = [Word.model_validate(word) for word in _json(tmp_path / "primary.json")["words"]]
    class Primary:
        last_usage = {"cost": .01, "seconds": 5}
        def transcribe(self, path):
            return primary_words
    class Secondary:
        api_key = "fixture-key"
        last_usage = {"cost": None, "reported_cost": .02, "reported_seconds": 3, "uncertain_seconds": 2}
        def transcribe(self, path):
            raise ProviderError("Temporary secondary provider failure", code="temporary")
    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *a, **kw: Primary())
    monkeypatch.setattr(runtime, "adapter_from_config", lambda *a, **kw: Secondary())
    monkeypatch.setattr(runtime, "normalize_audio", lambda source, *a, **kw: source)
    with pytest.raises(ProviderError, match="secondary provider failure"):
        pipeline.sync_episode(**options)
    workdir = options["workdir"] / "episode"
    cost = _json(workdir / "cost.json")
    assert cost == _json(workdir / "asr_cross_check_failure.json")["cost"]
    assert cost["total_usd"] > .03
    assert {item["provider"] for item in cost["items"]} == {MAI, "scribe_v2"}
    assert [item["kind"] for item in cost["items"]] == ["audio_billed", "audio_billed_partial", "audio_uncertain_estimate"]
    assert (workdir / "asr.json").exists()
    assert not (workdir / "asr_cross_check.json").exists()
    assert not options["output_path"].exists()


def test_default_off_keeps_existing_adjudication_cache_identity():
    span = DivergenceSpan(case_id="sample", cue_ids=[1], srt_text="green", asr_text="blue")
    config = {"llm": {"provider": "fixture"}}
    default = pipeline._adjudication_cache_key([span], config)
    assert default == pipeline._adjudication_cache_key([span], {**config, "_asr_cross_check_context": None})


def test_agreed_replacement_of_a_span_initial_name_is_reviewed(tmp_path, monkeypatch):
    # Fable review F24: 'Rafael' is a vocative, so the recurring-name lexicon
    # never holds it; both recognisers hear 'Gabriel' and the name was replaced
    # with no audio review.
    lines = ["Bom dia a todos.", "Rafael, vem cá agora.", "Ele saiu cedo hoje.", "Rafael, espera."]
    heard = ["Bom dia a todos.", "Gabriel, vai lá agora.", "Ele saiu cedo hoje.", "Rafael, espera."]
    starts = [1.0, 4.0, 8.0, 11.0]
    options = _inputs(tmp_path)
    options["language"] = "pt"
    options["srt_path"].write_text("".join(
        f"{index}\n00:00:{int(start):02d},000 --> 00:00:{int(start) + 2:02d},500\n{line}\n\n"
        for index, (start, line) in enumerate(zip(starts, lines), 1)), encoding="utf-8")
    with wave.open(str(options["audio_path"]), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 14)
    primary = [dict(text=token, start=round(start + position * .45, 3), end=round(start + position * .45 + .35, 3))
               for start, line in zip(starts, heard) for position, token in enumerate(line.split())]
    secondary = [dict(word, start=word["start"] + .04, end=word["end"] + .04) for word in primary]
    (tmp_path / "primary.json").write_text(json.dumps({"words": primary}), encoding="utf-8")
    (tmp_path / "secondary.json").write_text(json.dumps({"words": secondary}), encoding="utf-8")
    seen = []

    class Reviewer:
        def adjudicate(self, spans):
            seen.extend(span.srt_text for span in spans)
            return [dict(case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text, confidence=1.0,
                         reason="The reviewer keeps the customer's name.") for span in spans]

    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **kw: Reviewer())
    result = pipeline.sync_episode(**options)

    assert any("Rafael" in text for text in seen)
    assert _json(result.episode_workdir / "asr_cross_check_analysis.json")["preaccepted_case_ids"] == []
    assert "Rafael, vem cá agora." in result.output_srt.read_text(encoding="utf-8")
