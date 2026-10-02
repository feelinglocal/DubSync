"""An expanded audio question cannot inherit a partial question's answer."""
import json

import pytest

from dubsync import pipeline
from dubsync.models import AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
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
