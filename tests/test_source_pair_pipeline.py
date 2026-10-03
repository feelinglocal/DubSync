import json

import pytest

from dubsync import pipeline
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch
from dubsync.tokenize import alphanumeric_signature, tokenize_cues
from test_missing_dialogue_reconciliation import _pipeline_case
from test_source_pair_timing import _case


def _native_case(tmp_path, monkeypatch):
    _, source, alignment, words, regions = _case()
    alignment.diagnostics.missing_audio_guard_version = pipeline.MISSING_AUDIO_GUARD_VERSION
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=(source, words, alignment, regions))

    def hear(spans, snippets):
        adapter.seen.extend(spans)
        assert all(snippets[span.case_id].start <= span.start and snippets[span.case_id].end >= span.end for span in spans)
        return [AdjudicationDecision(
            case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text, heard_text=span.srt_text,
            evidence="heard_clearly", confidence=1,
            speaker=span.speaker_ids[0] if span.case_id.startswith("source-pair-timing-") else None,
            source_pair_evidence=({
                "first_text": span.srt_text.splitlines()[0], "second_text": span.srt_text.splitlines()[1],
                "sequence": "first_then_second", "voice_relation": "same", "intervening_speech": False,
                "candidate_complete": True, "candidate_start_clipped": False, "candidate_end_clipped": False,
                "laugh_outside_candidate": False, "candidate_audio_id": span.case_id + "-candidate",
            } if span.case_id.startswith("source-pair-timing-") else None),
            reason="Fixture confirms both parts in the supplied voice.",
        ).model_dump() for span in spans]

    adapter.adjudicate_with_audio = hear
    return alignment, adapter, run


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_pipeline_merges_complete_native_source_pair_and_replays_the_bound_receipt(tmp_path, monkeypatch, mode):
    alignment, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run()
    assert len([span for span in adapter.seen if span.case_id.startswith("source-pair-timing-")]) == 1
    first_output = result.output_srt.read_bytes()
    clip_manifest = result.episode_workdir / "source_pair_timing_audio_snippets.json"
    first_manifest = clip_manifest.read_bytes()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
        assert result.output_srt.read_bytes() == first_output
        if mode == "cache":
            # A cached hearing run rewrites the manifest for itself: the same bound clip, no audio loaded.
            assert json.loads(clip_manifest.read_bytes())["snippets"] == json.loads(first_manifest)["snippets"]
        else:
            assert clip_manifest.read_bytes() == first_manifest
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    output = {cue["index"]: cue for cue in rebuilt["cues"]}
    assert 11 not in output
    assert output[10]["lines"] == ["いいでしょう？", "ははは"]
    assert output[10]["start_ms"] <= 41360 and output[10]["end_ms"] >= 42465
    assert output[10]["end_ms"] <= output[12]["start_ms"]
    assert rebuilt["alignment"]["cue_word_indices"] == {str(k): v for k, v in alignment.cue_word_indices.items()}
    receipt = json.loads((result.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert rebuilt["source_pair_receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_source_pair"
    assert receipt["outcomes"][0]["source_cue_ids"] == [10, 11]


def _two_pairs():
    """The 1B pair and a copy 20 s later, so one episode asks two pair questions."""
    _, source_a, alignment_a, words_a, regions_a = _case()
    offset = len(words_a)
    source = [*source_a, *(Cue(index=cue.index + 20, start_ms=cue.start_ms + 20000, end_ms=cue.end_ms + 20000,
                               lines=list(cue.lines)) for cue in source_a)]
    words = [*words_a, *(word.model_copy(update={"start": word.start + 20, "end": word.end + 20}) for word in words_a)]
    regions = [*regions_a, *(SpeechRegion(start=region.start + 20, end=region.end + 20) for region in regions_a)]
    ownership = dict(alignment_a.cue_word_indices)
    ownership.update({cue_id + 20: [i + offset for i in indices] for cue_id, indices in alignment_a.cue_word_indices.items()})
    tokens = tokenize_cues(source)
    matches = []
    for cue_id, indices in ownership.items():
        own, cursor = [token for token in tokens if token.cue_id == cue_id], 0
        for word_index in indices:
            for normalized in alphanumeric_signature(words[word_index].text):
                token = next(token for token in own[cursor:] if token.normalized == normalized)
                cursor = own.index(token) + 1
                matches.append(TokenMatch(cue_id=cue_id, srt_token_index=token.token_index, asr_word_index=word_index, score=1))
    spans = [DivergenceSpan(case_id=f"parent-{laugh}", cue_ids=[laugh], srt_text="ははは", asr_text="？",
                            srt_token_indices=[token.token_index for token in tokens if token.cue_id == laugh],
                            asr_word_indices=[mark], start=words[mark].start, end=words[mark].end)
             for laugh, mark in ((11, 7), (31, 7 + offset))]
    alignment = AlignmentResult(cue_word_indices=ownership, token_matches=matches, divergence_spans=spans,
                                unmatched_cue_ids=[11, 31],
                                diagnostics={"missing_audio_cue_ids": [11, 31],
                                             "missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION})
    return source, words, alignment, regions


def test_partial_pair_rehearing_keeps_every_delivered_pair_clip_record(tmp_path, monkeypatch):
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=_two_pairs())
    def hear(spans, snippets):
        adapter.seen.extend(spans)
        return [AdjudicationDecision(
            case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text, heard_text=span.srt_text,
            evidence="heard_clearly", confidence=1, speaker=span.speaker_ids[0],
            source_pair_evidence={
                "first_text": span.srt_text.splitlines()[0], "second_text": span.srt_text.splitlines()[1],
                "sequence": "first_then_second", "voice_relation": "same", "intervening_speech": False,
                "candidate_complete": True, "candidate_start_clipped": False, "candidate_end_clipped": False,
                "laugh_outside_candidate": False, "candidate_audio_id": span.case_id + "-candidate"},
            reason="Fixture confirms both parts in the supplied voice.").model_dump() for span in spans]
    adapter.adjudicate_with_audio = hear
    first = run()
    path = first.episode_workdir / "source_pair_timing_audio_snippets.json"
    heard = {row["case_id"]: row["sha256"] for row in json.loads(path.read_text(encoding="utf-8"))["snippets"]}
    assert sorted(heard) == ["source-pair-timing-v2-10-11", "source-pair-timing-v2-30-31"]
    removed = []
    for item in (first.episode_workdir / "llm-case-cache").glob("*.json"):
        if json.loads(item.read_text(encoding="utf-8"))["value"]["decision"]["case_id"] == "source-pair-timing-v2-30-31":
            item.unlink()
            removed.append(item)
    assert len(removed) == 1
    adapter.seen.clear()
    second = run()
    assert [span.case_id for span in adapter.seen] == ["source-pair-timing-v2-30-31"]
    assert second.output_srt.read_bytes() == first.output_srt.read_bytes()
    assert {row["case_id"]: row["sha256"] for row in json.loads(path.read_text(encoding="utf-8"))["snippets"]} == heard


def test_disabled_hearing_does_not_cache_an_unasked_pair_as_complete(tmp_path, monkeypatch):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run(no_llm=True)
    assert adapter.seen == []
    run()
    assert len([span for span in adapter.seen if span.case_id.startswith("source-pair-timing-")]) == 1
    receipt = json.loads((result.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_source_pair"


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_replay_rejects_modified_pair_receipt_without_a_provider_call(tmp_path, monkeypatch, mode):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "source_pair_timing.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0]["speaker"] = "a different voice"
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []


def test_negative_same_voice_answer_is_cached_without_reasking_until_agreement(tmp_path, monkeypatch):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    native = adapter.adjudicate_with_audio

    def different_voice(spans, snippets):
        decisions = native(spans, snippets)
        for item in decisions:
            if item["case_id"].startswith("source-pair-timing-"):
                item["speaker"] = None
        return decisions

    adapter.adjudicate_with_audio = different_voice
    result = run()
    adapter.seen.clear()
    result = run()
    assert adapter.seen == []
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert any(cue["index"] == 11 for cue in payload["cues"])
    receipt = json.loads((result.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert receipt["outcomes"][0]["outcome"] == "wording_or_voice_unconfirmed"


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_native_two_voice_exchange_uses_separate_dash_lines_and_preserves_owners(tmp_path, monkeypatch, mode):
    alignment, adapter, run = _native_case(tmp_path, monkeypatch)
    native = adapter.adjudicate_with_audio
    def different(spans, snippets):
        decisions = native(spans, snippets)
        for item in decisions:
            if item["case_id"].startswith("source-pair-timing-"):
                assert item["case_id"] + "-candidate" in snippets
                item["speaker"] = None
                item["source_pair_evidence"]["voice_relation"] = "different"
        return decisions
    adapter.adjudicate_with_audio = different
    result = run()
    before = result.output_srt.read_bytes()
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == [] and result.output_srt.read_bytes() == before
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    cues = {cue["index"]: cue for cue in rebuilt["cues"]}
    assert 11 not in cues and cues[10]["lines"] == ["- いいでしょう？", "- ははは"]
    assert cues[10]["speaker_id"] is None and cues[10]["character"] is None
    assert cues[10]["end_ms"] <= cues[12]["start_ms"]
    assert rebuilt["alignment"]["cue_word_indices"] == {str(k): v for k, v in alignment.cue_word_indices.items()}
    receipt = json.loads((result.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_source_exchange"
    assert [item["source_cue_id"] for item in receipt["outcomes"][0]["line_provenance"]] == [10, 11]


def test_hybrid_review_outage_is_not_saved_as_a_final_pair_answer(tmp_path, monkeypatch):
    _, adapter, run = _native_case(tmp_path, monkeypatch)
    native = adapter.adjudicate_with_audio
    state = {"pair_calls": 0, "outages": []}

    def hear_with_one_outage(spans, snippets):
        decisions = native(spans, snippets)
        pair_ids = [span.case_id for span in spans if span.case_id.startswith("source-pair-timing-")]
        state["outages"] = []
        if pair_ids:
            state["pair_calls"] += 1
            if state["pair_calls"] == 1:
                state["outages"] = pair_ids
                for item in decisions:
                    if item["case_id"] in pair_ids:
                        item.update(evidence="heard_unclear", heard_text="", confidence=0)
        return decisions

    adapter.adjudicate_with_audio = hear_with_one_outage
    adapter.route_report = lambda: {"counts": {}, "decisions": [
        {"case_id": case_id, "route": "held", "reasons": ["review_provider_failure"]}
        for case_id in state["outages"]
    ]}
    first = run()
    assert state["pair_calls"] == 1
    receipt = json.loads((first.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "adjudication_review_unavailable" for flag in receipt["flags"])
    assert receipt["outcomes"][0]["outcome"] == "unconfirmed_audio"
    second = run()
    assert state["pair_calls"] == 2
    receipt = json.loads((second.episode_workdir / "source_pair_timing.json").read_text(encoding="utf-8"))
    assert receipt["outcomes"][0]["outcome"] == "audio_confirmed_source_pair"
