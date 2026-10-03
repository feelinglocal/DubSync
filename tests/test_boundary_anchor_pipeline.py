"""An expanded audio question cannot inherit a partial question's answer."""
import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
from dubsync.srt_io import parse_srt_text, write_srt
from test_boundary_anchor_regions import CUES_657, MAI_657, SCRIBE_657
from test_missing_dialogue_reconciliation import _pipeline_case


def _case():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1800, lines=["E você quer sair?"]),
        Cue(index=2, start_ms=2200, end_ms=3000, lines=["Que cheiro bom."]),
    ]
    words = [Word(text=text, start=start, end=end, speaker_id="voice") for text, start, end in [
        ("É", 1.0, 1.15), ("sério", 1.16, 1.4), ("isso.", 1.41, 1.65),
        ("Esse", 2.2, 2.35), ("cheiro", 2.36, 2.65), ("bom.", 2.66, 2.9),
    ]]
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1, 2], srt_text="você quer sair Que", asr_text="sério isso. Esse",
        srt_token_indices=[1, 2, 3, 4], asr_word_indices=[1, 2, 3], start=1.16, end=2.35,
        left_anchor_cue_id=1, left_anchor_end=1.15, left_anchor_speaker_id="voice",
        right_anchor_cue_id=2, right_anchor_start=2.36, right_anchor_speaker_id="voice",
        speaker_ids=["voice"],
    )
    alignment = AlignmentResult(
        token_matches=[TokenMatch(cue_id=cue, srt_token_index=token, asr_word_index=word, score=1)
                       for cue, token, word in [(1, 0, 0), (2, 5, 4), (2, 6, 5)]],
        cue_word_indices={1: [0], 2: [4, 5]}, divergence_spans=[span],
        diagnostics={"missing_audio_guard_version": pipeline.MISSING_AUDIO_GUARD_VERSION},
    )
    regions = [SpeechRegion(start=1.0, end=1.65), SpeechRegion(start=2.2, end=2.9)]
    return cues, words, alignment, regions


def _run_case(tmp_path, monkeypatch):
    return _pipeline_case(tmp_path, monkeypatch, case_override=_case(), heard="É sério isso. Esse")


@pytest.mark.parametrize("mode", ["fresh", "cache", "adjudicate", "rebuild", "verify"])
def test_complete_question_replaces_anchor_once_and_reuses_only_its_bound_answer(tmp_path, monkeypatch, mode):
    _, adapter, run = _run_case(tmp_path, monkeypatch)
    result = run()
    text = result.output_srt.read_text(encoding="utf-8")
    assert "E É" not in text and "E é" not in text
    assert "É sério isso." in text
    question = next(span for span in adapter.seen if span.case_id == "case-1")
    assert question.srt_token_indices == [0, 1, 2, 3, 4]
    assert question.asr_word_indices == [0, 1, 2, 3]
    assert question.left_anchor_cue_id is None
    if mode != "fresh":
        adapter.seen.clear()
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert adapter.seen == []
        assert result.output_srt.read_text(encoding="utf-8") == text
    rebuilt = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    assert rebuilt["alignment"]["cue_word_indices"] == {"1": [0, 1, 2], "2": [3, 4, 5]}
    assert len(rebuilt["alignment"]["token_matches"]) == 3


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
@pytest.mark.parametrize("fault", ["missing", "decision", "span", "unexpected"])
def test_resume_rejects_unbound_or_changed_expanded_scope_before_writing(tmp_path, monkeypatch, mode, fault):
    _, adapter, run = _run_case(tmp_path, monkeypatch)
    result = run()
    path = result.episode_workdir / "adjudicate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fault == "missing":
        payload.pop("boundary_anchor_bindings", None)
    elif fault == "decision":
        payload["decisions"][0]["final_text"] = "sério isso. Esse"
    elif fault == "unexpected":
        payload.setdefault("boundary_anchor_bindings", {})["unrelated"] = "0" * 64
    else:
        align_path = result.episode_workdir / "align.json"
        align = json.loads(align_path.read_text(encoding="utf-8"))
        align["divergence_spans"][0]["asr_text"] = "É sério isso. Esta"
        align_path.write_text(json.dumps(align), encoding="utf-8")
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = {name: (result.episode_workdir / name).read_bytes()
              for name in ("align.json", "adjudicate.json", "rebuild.json", "ingest.json")}
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []
    assert {name: (result.episode_workdir / name).read_bytes() for name in before} == before


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_old_partial_scope_cannot_resume_without_a_complete_question(tmp_path, monkeypatch, mode):
    _, adapter, run = _run_case(tmp_path, monkeypatch)
    with monkeypatch.context() as legacy:
        legacy.setattr(pipeline, "_alignment_with_boundary_anchor_regions", lambda alignment, *_a, **_k: alignment,
                       raising=False)
        result = run()
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []
    refreshed = run(resume="adjudicate")
    assert len(adapter.seen) == 1
    assert adapter.seen[0].asr_word_indices == [0, 1, 2, 3]
    assert "E É" not in refreshed.output_srt.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_interrupted_scope_refresh_cannot_promote_old_partial_answer(tmp_path, monkeypatch, mode):
    _, adapter, run = _run_case(tmp_path, monkeypatch)
    with monkeypatch.context() as legacy:
        legacy.setattr(pipeline, "_alignment_with_boundary_anchor_regions", lambda alignment, *_a, **_k: alignment,
                       raising=False)
        result = run()
    old_answer = (result.episode_workdir / "adjudicate.json").read_bytes()

    class InterruptedRefresh(BaseException):
        pass

    with monkeypatch.context() as interrupted:
        def stop(*_a, **_k):
            raise InterruptedRefresh()
        interrupted.setattr(pipeline, "llm_adapter_from_config", stop)
        with pytest.raises(InterruptedRefresh):
            run(resume="adjudicate")
    alignment = json.loads((result.episode_workdir / "align.json").read_text(encoding="utf-8"))
    assert alignment["divergence_spans"][0]["srt_token_indices"][0] == 0
    assert (result.episode_workdir / "adjudicate.json").read_bytes() == old_answer
    adapter.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.seen == []


def _sync_657(tmp_path, stream, responses, *, anchor_confidence=None):
    """EP11 cue 657 through the real aligner and pipeline, text-only fixture answers keyed by case."""
    source, audio, config, fixture = [tmp_path / name for name in ("source.srt", "audio.wav", "provider.yaml", "words.json")]
    source.write_text(write_srt([cue.model_copy(update={"index": number}) for number, cue in enumerate(CUES_657, 1)]),
                      encoding="utf-8")
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "speaker_id": "s",
         "confidence": anchor_confidence if (text, start) in {("E", 8.9), ("e", 9.35)} else None}
        for text, start, end in stream
    ]}, ensure_ascii=False), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": {
        case_id: {"case_id": case_id, "verdict": "keep_srt" if text is None else "use_audio", "final_text": text or "",
                  "confidence": 1.0, "reason": "heard"} for case_id, text in responses.items()
    }}}, allow_unicode=True), encoding="utf-8")
    result = pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work", providers_path=config,
                                   language="pt")
    delivered = {cue.index: " ".join(cue.plain_text.split())
                 for cue in parse_srt_text((tmp_path / "output.srt").read_text(encoding="utf-8"))}
    questions = json.loads((result.episode_workdir / "align.json").read_text(encoding="utf-8"))["divergence_spans"]
    return delivered, {span["case_id"]: span["srt_text"] for span in questions}, result.report["flags"]


@pytest.mark.parametrize("stream, responses, question", [
    (MAI_657, {"case-1": "E", "case-3": "ia lá saber"}, "Como é que"),
    (SCRIBE_657, {"case-1": "E eu ia lá saber"}, "Como é que eu sabia"),
])
def test_edits_around_a_retained_accent_anchor_are_heard_in_one_question(tmp_path, stream, responses, question):
    # EP11 cue 657 was delivered as "é eu ia lá saber ..." (MAI) and "E eu é ia lá saber ..." (Scribe).
    delivered, questions, flags = _sync_657(tmp_path, stream, responses)

    assert questions["case-1"] == question and "case-2" not in questions
    assert delivered[2] == "E eu ia lá saber que ele tinha namorada?"
    assert "adjudication_span_edit_held" not in [flag["kind"] for flag in flags]


def test_edit_beside_an_anchor_outside_every_question_holds_the_cue(tmp_path):
    # An uncertain anchor word cannot join a question; the approved edits around it would deliver
    # the script's "é" inside new wording, so the cue keeps its script wording for review.
    delivered, questions, flags = _sync_657(
        tmp_path, MAI_657, {"case-1": "", "case-2": "", "case-3": "ia lá saber"}, anchor_confidence=.5,
    )

    assert [questions[case] for case in ("case-1", "case-2", "case-3")] == ["Como", "que", "sabia"]
    assert delivered[2] == "Como é que eu sabia que ele tinha namorada?"
    held = [flag for flag in flags if flag["kind"] == "adjudication_span_edit_held"]
    assert held and {cue for flag in held for cue in flag["cue_ids"]} == {2}
    assert all(flag["severity"] == "error" for flag in held)
    assert not [flag for flag in flags if flag["kind"] == "text_changed" and 2 in flag["cue_ids"]]


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_ordinary_legacy_scope_remains_resumable(tmp_path, monkeypatch, mode):
    cues, words, alignment, regions = _case()
    cues[0] = cues[0].with_lines(["É você quer sair?"])
    _, adapter, run = _pipeline_case(tmp_path, monkeypatch, case_override=(cues, words, alignment, regions),
                                   heard="sério isso. Esse")
    result = run()
    path = result.episode_workdir / "adjudicate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("boundary_anchor_bindings", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter.seen.clear()
    run(resume=mode)
    assert adapter.seen == []
