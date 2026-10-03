"""Recovery mechanisms end to end with the real aligner, energy VAD and word repair (F32).

Each case writes a 16 kHz mono PCM WAV whose tone bursts sit at known times, a
primary word stream and, where the mechanism needs one, a secondary stream read
through ``asr.cross_check.fixture_path``. Only the hearing model is synthetic.
ffmpeg's cut is replaced by the same cut in pure Python, so the real clip
windows, budgets and source-pair candidate checks still run. Every case runs
fresh, then resume=rebuild and resume=verify, which must replay the saved
receipts without a new hearing and write the same output.
"""
from __future__ import annotations

import json
import math
import wave
from array import array

import pytest
import yaml

from dubsync import audio_snippets, pipeline
from dubsync.models import AdjudicationDecision, Cue, Word
from dubsync.srt_io import write_srt
from test_accepted_anchor_omission import _case as _accepted_anchor_case
from test_source_pair_timing import _case as _source_pair_case
from test_whole_utterance_secondary import _ambiguous_case
from test_whole_utterance_timing import _secondary_split

MAI = "microsoft/mai-transcribe-2"
RATE = 16000


def _write_tones(path, seconds, regions):
    """A 220 Hz tone per expected VAD region; the detector reports 5 ms past each tone edge."""
    pcm = array("h", bytes(2 * round(seconds * RATE)))
    for start, end in regions:
        for i in range(round((start + .005) * RATE), round((end - .005) * RATE)):
            pcm[i] = round(8000 * math.sin(2 * math.pi * 220 * i / RATE))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())


def _cut_wav(audio_path, output_path, start, end, _ffmpeg, _timeout, _cap):
    """``ffmpeg -ss start -t duration`` on the 16 kHz mono PCM source, without a subprocess."""
    with wave.open(str(audio_path), "rb") as source:
        source.setpos(min(round(start * RATE), source.getnframes()))
        frames = source.readframes(round((end - start) * RATE))
    with wave.open(str(output_path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(RATE)
        clip.writeframes(frames)


def _words(rows, speaker, confidence=1):
    return [Word(text=text, start=start, end=end, confidence=confidence, speaker_id=speaker) for text, start, end in rows]


class _Hearing:
    """Synthetic native hearing: confirms the source wording unless ``answer`` says otherwise."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def adjudicate(self, spans):
        pytest.fail("Recovery hearings require complete native audio.")

    def adjudicate_with_audio(self, spans, clips):
        assert all(clips[span.case_id].start <= span.start and clips[span.case_id].end >= span.end for span in spans)
        self.seen.extend(spans)
        return [AdjudicationDecision(**{
            "case_id": span.case_id, "verdict": "keep_srt", "final_text": span.srt_text, "heard_text": span.srt_text,
            "evidence": "heard_clearly", "confidence": 1, "reason": "Synthetic test hearing; no provider call.",
            **self.answer(span),
        }).model_dump() for span in spans]


def _episode(tmp_path, monkeypatch, cues, primary, regions, seconds, *, language, secondary=None,
             answer=lambda span: {}):
    source, audio, config_path = tmp_path / "episode.srt", tmp_path / "audio.wav", tmp_path / "providers.yaml"
    source.write_text(write_srt(cues), encoding="utf-8")
    _write_tones(audio, seconds, regions)
    config = {"asr": {"provider": "elevenlabs", "model_id": "scribe_v2", "fixture_path": str(tmp_path / "primary.json")},
              "vad": {"provider": "energy"}, "llm": {"provider": "fixture", "audio_snippet_double_check": True}}
    (tmp_path / "primary.json").write_text(json.dumps({"words": [w.model_dump() for w in primary]}), encoding="utf-8")
    if secondary is not None:
        (tmp_path / "secondary.json").write_text(json.dumps({"words": [w.model_dump() for w in secondary]}),
                                                 encoding="utf-8")
        config["asr"]["cross_check"] = {"provider": "openrouter", "model": MAI,
                                        "fixture_path": str(tmp_path / "secondary.json")}
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    hearing = _Hearing(answer)
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *_a, **_k: hearing)
    monkeypatch.setattr(audio_snippets, "_cut_wav_snippet", _cut_wav)

    def run(**kwargs):
        result = pipeline.sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work",
                                       providers_path=config_path, language=language, **kwargs)
        detected = _json(result, "vad.json")["regions"]
        assert [(region["start"], region["end"]) for region in detected] == list(regions)  # the real detector
        return result
    return run, hearing


def _json(result, name):
    return json.loads((result.episode_workdir / name).read_text(encoding="utf-8"))


def _timed(result):
    return {cue["index"]: (cue["start_ms"], cue["end_ms"], cue["lines"]) for cue in _json(result, "rebuild.json")["cues"]}


def _qc(result, kind):
    return [tuple(flag["cue_ids"]) for flag in _json(result, "qc_report.json")["flags"] if flag["kind"] == kind]


def _whole_utterance_questions(result):
    path = result.episode_workdir / "missing_dialogue_reconciliation.json"
    questions = json.loads(path.read_text(encoding="utf-8"))["questions"] if path.exists() else []
    return [q for q in questions if q["purpose"] == "whole_utterance_timing"]


def _three_modes(run, hearing, check):
    """Fresh hears once; rebuild and verify replay the receipts without hearing again."""
    result = run()
    check(result)
    asked = [span.case_id for span in hearing.seen]
    output = result.output_srt.read_bytes()
    for mode in ("rebuild", "verify"):
        hearing.seen.clear()
        result = run(resume=mode)
        assert hearing.seen == []
        assert result.output_srt.read_bytes() == output
        check(result)
    return asked


# --- whole-utterance timing ---------------------------------------------------

_MISMATCH_REGIONS = [(72.095, 72.905), (73.165, 73.555), (73.635, 74.225)]


def _mismatch_episode(tmp_path, monkeypatch, answer=lambda span: {}):
    """Source 'ああ' was heard by ASR as 'う','ん' inside one independent burst; the cue is mistimed."""
    labels = ["弁償しろだと？", "ああ", "いいだろ"]
    cues = [Cue(index=1, start_ms=72000, end_ms=72900, lines=[labels[0]]),
            Cue(index=2, start_ms=73560, end_ms=73960, lines=[labels[1]]),
            Cue(index=3, start_ms=74000, end_ms=75000, lines=[labels[2]])]
    words = _words([(labels[0], 72.12, 72.875), ("う", 73.22, 73.42), ("ん", 73.42, 73.54),
                    (labels[2], 73.635, 74.16)], "speaker_0")
    return _episode(tmp_path, monkeypatch, cues, words, _MISMATCH_REGIONS, 76.0, language="ja", answer=answer)


def test_whole_utterance_hearing_moves_the_cue_onto_its_independent_burst(tmp_path, monkeypatch):
    run, hearing = _mismatch_episode(tmp_path, monkeypatch)

    def check(result):
        assert _timed(result)[2] == (73134, 73600, ["ああ"])
        question, = _whole_utterance_questions(result)
        assert question["evidence_word_indices"] == [1, 2] and "secondary_acoustic_proof" not in question
        outcome, = _json(result, "missing_dialogue_reconciliation.json")["outcomes"]
        assert outcome["outcome"] == "audio_confirmed_utterance"
    assert "whole-utterance-timing-v2-cue-2" in _three_modes(run, hearing, check)


def test_whole_utterance_cue_keeps_its_source_timing_when_the_hearing_differs(tmp_path, monkeypatch):
    def answer(span):
        heard = {"verdict": "use_audio", "final_text": "うん", "heard_text": "うん"}
        return heard if span.case_id.startswith("whole-utterance-timing-") else {}
    run, hearing = _mismatch_episode(tmp_path, monkeypatch, answer)

    def check(result):
        assert _timed(result)[2] == (73560, 73960, ["ああ"])
        outcome, = _json(result, "missing_dialogue_reconciliation.json")["outcomes"]
        assert outcome["outcome"] == "wording_does_not_match_hearing"
    assert "whole-utterance-timing-v2-cue-2" in _three_modes(run, hearing, check)


@pytest.mark.parametrize("neighbour_heard", [True, False], ids=["single-cue-parent", "parent-covers-neighbour"])
def test_whole_utterance_question_needs_a_parent_span_owned_by_the_target_alone(tmp_path, monkeypatch, neighbour_heard):
    # 'dow' may be the unheard neighbour word 'come': with 'come' missing the
    # aligner's divergence span covers cues 2 and 3, so 'dow' cannot prove cue 2.
    cues = [Cue(index=1, start_ms=900, end_ms=1300, lines=["Wait here."]),
            Cue(index=2, start_ms=2600, end_ms=2900, lines=["Tao,"]),
            Cue(index=3, start_ms=3000, end_ms=3500, lines=["come closer."])]
    rows = [("Wait", 1.0, 1.15), ("here", 1.16, 1.3), ("dow", 1.62, 1.85),
            *([("come", 2.1, 2.2), ("closer", 2.2, 2.4)] if neighbour_heard else [("closer", 2.1, 2.4)])]
    run, hearing = _episode(tmp_path, monkeypatch, cues, _words(rows, "speaker_0"),
                            [(0.995, 1.305), (1.595, 1.865), (2.095, 2.405)], 4.0, language="en")

    def check(result):
        timed = _timed(result)
        assert timed[3] == (2067, 2467, ["come closer."])
        if neighbour_heard:
            assert timed[2] == (1567, 1934, ["Tao,"])
            assert [q["evidence_word_indices"] for q in _whole_utterance_questions(result)] == [[2]]
        else:
            assert timed[2] == (2600, 2900, ["Tao,"])
            assert _whole_utterance_questions(result) == []
    asked = _three_modes(run, hearing, check)
    assert ("whole-utterance-timing-v2-cue-2" in asked) is neighbour_heard


# --- secondary acoustic ownership -------------------------------------------------

def _japanese_2a():
    """Delivered 2A Scribe cue 50: the right neighbour's region is proved by the secondary stream."""
    cues = [Cue(index=1, start_ms=63600, end_ms=64200, lines=["三分だ"]),
            Cue(index=2, start_ms=64666, end_ms=65466, lines=["三分？"]),
            Cue(index=3, start_ms=70700, end_ms=72100, lines=["山下森彦に伝えろ"])]
    primary = _words([("三分だ", 63.74, 64.099), ("三", 64.739, 64.760), ("分", 71.14, 71.16), ("山", 71.28, 71.32),
                      ("下", 71.36, 71.361), ("森", 71.46, 71.48), ("彦", 71.48, 71.6), ("に", 71.6, 71.82),
                      ("伝", 71.82, 71.9), ("え", 71.9, 72.02), ("ろ", 72.02, 72.08)], "primary_1")
    regions = [(61.095, 64.125), (64.375, 65.135), (70.615, 72.095)]
    return cues, primary, _secondary_split(), regions, 73.0


def _japanese_2b():
    """Delivered 2B Scribe cue 17: its last primary word 'て' spans two bursts, so word repair keeps it ambiguous."""
    (cues, primary, _, _), secondary, _ = _ambiguous_case()
    regions = [(18.395, 19.105), (21.715, 23.075), (23.415, 25.475)]
    return cues, primary, secondary, regions, 26.0


_SECONDARY_CASES = {"2a-neighbour-proof": (_japanese_2a, (64367, 65200), (64666, 65466)),
                    "2b-ambiguous-tail": (_japanese_2b, (21700, 23134), (22300, 22933))}


@pytest.mark.parametrize("geometry", list(_SECONDARY_CASES))
@pytest.mark.parametrize("with_secondary", [True, False], ids=["secondary", "primary-only"])
def test_secondary_acoustic_proof_recovers_the_delivered_japanese_cues(tmp_path, monkeypatch, geometry, with_secondary):
    build, recovered, held = _SECONDARY_CASES[geometry]
    cues, primary, secondary, regions, seconds = build()
    run, hearing = _episode(tmp_path, monkeypatch, cues, primary, regions, seconds, language="ja",
                            secondary=secondary if with_secondary else None)

    def check(result):
        assert _timed(result)[2] == (*(recovered if with_secondary else held), cues[1].lines)
        ambiguous = _qc(result, "asr_word_timing_ambiguous")
        assert bool(ambiguous) is (geometry == "2b-ambiguous-tail")  # flagged by the real VAD and word repair
        questions = _whole_utterance_questions(result)
        if not with_secondary:
            assert questions == []
            assert (2,) in _qc(result, "timing_evidence_held")
            return
        question, = questions
        proof = question["secondary_acoustic_proof"]
        if geometry == "2a-neighbour-proof":
            assert proof["target"] is None
            assert [(item["side"], item["region"]) for item in proof["neighbor_regions"]] == [
                ("right", {"start": 70.615, "end": 72.095})]
        else:
            # The ambiguous word is proved by its unique secondary token, never retimed.
            assert proof["target"]["region"] == {"start": 21.715, "end": 23.075}
            assert proof["target"]["primary_mappings"] == [{"primary_word_index": 9, "secondary_word_index": 6}]
        outcome, = _json(result, "missing_dialogue_reconciliation.json")["outcomes"]
        assert outcome["outcome"] == "audio_confirmed_utterance"
    asked = _three_modes(run, hearing, check)
    assert ("whole-utterance-timing-v2-cue-2" in asked) is with_secondary


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_resume_rejects_a_secondary_proof_changed_after_the_hearing(tmp_path, monkeypatch, mode):
    cues, primary, secondary, regions, seconds = _japanese_2b()
    run, hearing = _episode(tmp_path, monkeypatch, cues, primary, regions, seconds, language="ja", secondary=secondary)
    path = run().episode_workdir / "missing_dialogue_reconciliation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["secondary_acoustic_proof"]["target"]["region"]["end"] += .01
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    hearing.seen.clear()
    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert hearing.seen == []


# --- source pairs ------------------------------------------------------------------

@pytest.mark.parametrize("intervening_speech", [False, True], ids=["confirmed", "intervening-speech"])
def test_source_pair_merges_only_a_confirmed_complete_pair(tmp_path, monkeypatch, intervening_speech):
    _, cues, _, words, regions = _source_pair_case()

    def answer(span):
        if not span.case_id.startswith("source-pair-timing-"):
            return {}
        first, second = span.srt_text.splitlines()
        return {"speaker": span.speaker_ids[0], "source_pair_evidence": {
            "first_text": first, "second_text": second, "sequence": "first_then_second", "voice_relation": "same",
            "intervening_speech": intervening_speech, "candidate_complete": True, "candidate_start_clipped": False,
            "candidate_end_clipped": False, "laugh_outside_candidate": False,
            "candidate_audio_id": span.case_id + "-candidate"}}
    run, hearing = _episode(tmp_path, monkeypatch, cues, words, [(r.start, r.end) for r in regions], 45.0,
                            language="ja", answer=answer)

    def check(result):
        timed = _timed(result)
        outcome, = _json(result, "source_pair_timing.json")["outcomes"]
        if intervening_speech:
            assert outcome["outcome"] == "pair_candidate_evidence_unconfirmed"
            assert timed[10] == (41334, 41800, ["いいでしょう？"]) and timed[11] == (42800, 43130, ["ははは"])
        else:
            assert outcome["outcome"] == "audio_confirmed_source_pair" and outcome["source_cue_ids"] == [10, 11]
            assert 11 not in timed and timed[10] == (41334, 42534, ["いいでしょう？", "ははは"])
    assert "source-pair-timing-v2-10-11" in _three_modes(run, hearing, check)


# --- collapsed singletons ------------------------------------------------------------

@pytest.mark.parametrize("with_secondary", [True, False], ids=["secondary", "primary-only"])
def test_collapsed_singleton_moves_only_with_secondary_proof(tmp_path, monkeypatch, with_secondary):
    # The primary 'É.' is a 1 ms placeholder at the next cue; the secondary stream places it on its own burst.
    cues = [Cue(index=600, start_ms=1900, end_ms=2134, lines=["né?"]),
            Cue(index=601, start_ms=8630, end_ms=8667, lines=["É."]),
            Cue(index=602, start_ms=8655, end_ms=9100, lines=["Tomam."])]
    primary = _words([("né?", 1.93, 2.105), ("É.", 8.630, 8.631), ("Tomem.", 8.655, 9.045)], "primary", None)
    secondary = _words([("né?", 1.92, 2.159), ("É.", 7.04, 7.32), ("Tomem.", 8.639, 9.079)], "secondary", None)
    regions = [(1.895, 2.105), (4.605, 5.305), (6.515, 6.585), (6.975, 7.245), (8.655, 9.045)]

    def answer(span):
        return {"verdict": "use_audio", "final_text": "Tomem", "heard_text": "Tomem"} if span.srt_text == "Tomam" else {}
    run, hearing = _episode(tmp_path, monkeypatch, cues, primary, regions, 10.0, language="pt",
                            secondary=secondary if with_secondary else None, answer=answer)

    def check(result):
        timed = _timed(result)
        assert timed[602][2] == ["Tomem."]
        if with_secondary:
            assert timed[601] == (6967, 7300, ["É."])
            outcome, = _json(result, "collapsed_singleton_timing.json")["outcomes"]
            assert outcome["outcome"] == "audio_confirmed_utterance"
        else:
            assert timed[601] == (8630, 8667, ["É."])
            assert (601,) in _qc(result, "timing_evidence_held")
            assert not (result.episode_workdir / "collapsed_singleton_timing.json").exists()
    asked = _three_modes(run, hearing, check)
    assert any(case_id.startswith("collapsed-singleton-timing-v1-cue-601-") for case_id in asked) is with_secondary


# --- accepted-anchor omission ------------------------------------------------------

@pytest.mark.parametrize("with_secondary", [True, False], ids=["secondary", "primary-only"])
def test_accepted_anchor_omission_removes_the_unspoken_cue_only_with_secondary_proof(tmp_path, monkeypatch,
                                                                                      with_secondary):
    case = _accepted_anchor_case()
    regions = [(r.start, r.end) for r in case["regions"]]
    run, hearing = _episode(tmp_path, monkeypatch, case["source_cues"], case["words"], regions, 7.0, language="pt",
                            secondary=case["secondary_words"] if with_secondary else None,
                            answer=lambda span: {"verdict": "use_audio", "final_text": span.asr_text,
                                                 "heard_text": span.asr_text})

    def check(result):
        timed = _timed(result)
        receipt = _json(result, "missing_dialogue_reconciliation.json")
        outcome = next(item for item in receipt["outcomes"] if item.get("cue_id") == 441)
        if with_secondary:
            assert 441 not in timed
            assert outcome["outcome"] == "audio_confirmed_omission"
            assert outcome["accepted_anchor_omission_proof"]["raw_regions_in_primary_gap"] == []
            assert _json(result, "rebuild.json")["accepted_anchor_omission_sha256"]
        else:
            assert timed[441] == (4067, 4310, ["Tao,"])
            assert outcome["outcome"] == "untranscribed_activity_remains"
            assert "accepted_anchor_omission_proof" not in outcome
            assert (441,) in _qc(result, "missing_audio_source_cue_held")
    assert "missing-dialogue-v1-case-2-cue-441" in _three_modes(run, hearing, check)
