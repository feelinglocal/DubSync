import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AdjudicationDecision
from test_accepted_anchor_omission import _case
from test_missing_dialogue_reconciliation import _pipeline_case


def _secondary_fixture(tmp_path, words):
    path = tmp_path / "secondary-asr-fixture.json"
    path.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    return str(path)


def _native_case(tmp_path, monkeypatch):
    case = _case()
    case["alignment"].diagnostics.missing_audio_guard_version = pipeline.MISSING_AUDIO_GUARD_VERSION
    initial_alignment = case["alignment"].model_copy(deep=True)
    initial_alignment.cue_word_indices[440] = [10, 11]
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=(
        case["source_cues"], case["words"], initial_alignment, case["regions"]))
    extract = pipeline.extract_audio_snippets
    def padded_extract(audio, spans, output, **kwargs):
        duration = pipeline.audio_seconds(audio)
        padded = [span.model_copy(update={"start": max(0, span.start - .1),
                                         "end": min(duration, span.end + .1)}) for span in spans]
        return extract(audio, padded, output, **kwargs)
    monkeypatch.setattr(pipeline, "extract_audio_snippets", padded_extract)
    config_path = tmp_path / "provider.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["asr"].update(provider="elevenlabs", model_id="scribe_v2", cross_check={
        "provider": "openrouter", "model": "microsoft/mai-transcribe-2", "language_code": "pt",
        "fixture_path": _secondary_fixture(tmp_path, case["secondary_words"])})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def hear(spans, snippets):
        adapter.seen.extend(spans)
        assert all(snippets[span.case_id].start <= span.start and snippets[span.case_id].end >= span.end for span in spans)
        return [AdjudicationDecision(case_id=span.case_id, verdict="use_audio",
            final_text=span.asr_text, heard_text=span.asr_text, evidence="heard_clearly", confidence=1,
            reason="Synthetic unit audio fixture only.").model_dump() for span in spans]
    adapter.adjudicate_with_audio = hear
    return case, adapter, run


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_accepted_current_anchors_reuse_native_absence_without_changing_primary_ownership(tmp_path, monkeypatch, mode):
    case, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run()
    output = result.output_srt.read_bytes()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
        assert result.output_srt.read_bytes() == output
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert 441 not in {cue["index"] for cue in rebuilt["cues"]}
    assert rebuilt["alignment"]["cue_word_indices"] == {str(k): v for k, v in case["alignment"].cue_word_indices.items()}
    receipt = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    outcome = next(item for item in receipt["outcomes"] if item.get("cue_id") == 441)
    assert outcome["outcome"] == "audio_confirmed_omission"
    assert outcome["accepted_anchor_omission_proof"]["raw_regions_in_primary_gap"] == []
    assert rebuilt["accepted_anchor_omission_sha256"]
    assert 441 not in rebuilt["accepted_anchor_omission_guards"]["protected_cue_ids"]


def test_accepted_anchor_route_keeps_source_when_native_hearing_disabled(tmp_path, monkeypatch):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run(no_llm=True)
    assert adapter.seen == []
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert 441 in {cue["index"] for cue in rebuilt["cues"]}


@pytest.mark.parametrize("provenance", ["valid", "legacy", "tampered"])
def test_partial_native_cache_reuse_preserves_only_bound_anchor_clip_provenance(tmp_path, monkeypatch, provenance):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    first = run()
    original_output = first.output_srt.read_bytes()
    original_manifest = json.loads((first.episode_workdir / "audio_snippets.json").read_text(encoding="utf-8"))
    missing_case = next(span for span in adapter.seen if span.case_id.startswith("missing-dialogue-")
                        and span.cue_ids == [441])
    cache_files = list((first.episode_workdir / "llm-case-cache").glob("*.json"))
    refreshed = []
    for path in cache_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload["value"]
        if value.get("decision", {}).get("case_id") == missing_case.case_id:
            path.unlink()
            refreshed.append(path)
        elif provenance != "valid":
            assert "audio_provenance" in value
            if provenance == "legacy":
                value.pop("audio_provenance")
            else:
                value["audio_provenance"]["snippet"]["sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(refreshed) == 1
    monkeypatch.setattr(pipeline, "_load_cached_adjudication", lambda *_a, **_k: None)
    adapter.seen.clear()
    second = run()
    assert [span.case_id for span in adapter.seen] == [missing_case.case_id]
    current_manifest = json.loads((second.episode_workdir / "audio_snippets.json").read_text(encoding="utf-8"))
    if provenance == "valid":
        assert second.output_srt.read_bytes() == original_output
        assert {row["case_id"]: row["sha256"] for row in current_manifest["snippets"]} == {
            row["case_id"]: row["sha256"] for row in original_manifest["snippets"]}
    else:
        assert second.output_srt.read_bytes() != original_output
        rebuilt = json.loads((second.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
        assert 441 in {cue["index"] for cue in rebuilt["cues"]}
        assert [row["case_id"] for row in current_manifest["snippets"]] == [missing_case.case_id]


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_accepted_anchor_resume_rejects_tampered_omission_proof(tmp_path, monkeypatch, mode):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "missing_dialogue_reconciliation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcome = next(item for item in payload["outcomes"] if item.get("cue_id") == 441)
    assert "accepted_anchor_omission_proof" in outcome
    outcome["accepted_anchor_omission_proof"]["primary_gap"]["end"] += .01
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []


def test_accepted_anchor_verify_rejects_modified_ordinary_hearing(tmp_path, monkeypatch):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "adjudicate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][1]["reason"] += " Changed after the proof."
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume="verify")
    assert adapter.seen == []


def test_accepted_anchor_route_preserves_an_already_confirmed_ordinary_omission(tmp_path, monkeypatch):
    case, _, run = _pipeline_case(tmp_path, monkeypatch)
    secondary = [word.model_copy(update={"speaker_id": "secondary"}) for word in case[1]]
    config_path = tmp_path / "provider.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["asr"].update(provider="elevenlabs", model_id="scribe_v2", cross_check={
        "provider": "openrouter", "model": "microsoft/mai-transcribe-2",
        "fixture_path": _secondary_fixture(tmp_path, secondary)})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run()
    receipt = json.loads((result.episode_workdir / "missing_dialogue_reconciliation.json").read_text(encoding="utf-8"))
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_omission"
    assert "accepted_anchor_omission_proof" not in receipt["outcomes"][0]
