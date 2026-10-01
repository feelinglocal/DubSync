"""Unknown ASR word confidence (None on every MAI word) is not "zero confidence".

Deterministic keep decisions are policies, not model opinions; a kept source
text never freezes its cue at source timing when the cue owns acoustic words.
"""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine, KeepSRTAdapter
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _mai_words(items: list[tuple[str, float, float]]) -> list[dict[str, object]]:
    # MAI-Transcribe 2 never reports a word confidence.
    return [{"text": text, "start": start, "end": end, "confidence": None} for text, start, end in items]


def _sync(tmp_path, srt: str, words: list[dict[str, object]], *, responses=None, no_llm=False, llm=None):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    config: dict[str, object] = {"asr": {"fixture_path": str(fixture)}}
    if not no_llm:
        config["llm"] = {"provider": "fixture", "responses": responses or {}, **(llm or {})}
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers, no_llm=no_llm,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.1),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


def _kinds(flags: list[dict[str, object]]) -> list[str]:
    return [str(flag["kind"]) for flag in flags]


def test_punctuation_only_keep_is_confident_when_asr_confidence_is_unknown():
    # aligner.py reports span confidence 0.0 when no word carries a confidence.
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[15], srt_text="hab s", asr_text="hab's",
        start=32.9, end=33.1, confidence=0.0,
    )

    decisions, flags = AdjudicationEngine(KeepSRTAdapter()).adjudicate([span])

    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].reason == "Punctuation/casing-only difference; preserved source SRT."
    assert decisions[0].confidence == 1.0
    assert flags == []


def test_disabled_llm_keep_is_a_policy_not_a_low_confidence_opinion():
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[3], srt_text="gehen", asr_text="laufen",
        start=1.7, end=1.9, confidence=0.0,
    )

    decisions, flags = AdjudicationEngine(KeepSRTAdapter()).adjudicate([span])

    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == "gehen"
    assert decisions[0].confidence == 1.0
    assert flags == []


_CONTRACTION_SRT = (
    "1\n00:00:32,800 --> 00:00:34,130\nIch hab's dir doch\ngesagt.\n\n"
    "2\n00:00:35,100 --> 00:00:36,370\nals wär's auf\nChinesisch\n"
)
_CONTRACTION_WORDS = _mai_words([
    ("Ich", 32.72, 32.86), ("hab's", 32.88, 33.10), ("dir", 33.12, 33.26), ("doch", 33.28, 33.46),
    ("gesagt.", 33.48, 33.72), ("als", 35.56, 35.70), ("wär's", 35.72, 35.94), ("auf", 35.96, 36.08),
    ("Chinesisch.", 36.10, 36.42),
])


@pytest.mark.parametrize("no_llm", [False, True])
def test_mai_contraction_tokenisation_does_not_freeze_cues_at_source_timing(tmp_path, no_llm):
    # runtime-data job ...5e14bd cues 15/17: identical words, different tokens.
    cues, flags = _sync(tmp_path, _CONTRACTION_SRT, _CONTRACTION_WORDS, no_llm=no_llm)

    assert [cue.plain_text for cue in cues] == ["Ich hab's dir doch gesagt.", "als wär's auf Chinesisch"]
    assert abs(cues[0].start_ms - 32720) <= 34 and abs(cues[0].end_ms - 33760) <= 34
    assert abs(cues[1].start_ms - 35560) <= 34 and abs(cues[1].end_ms - 36460) <= 34
    kinds = _kinds(flags)
    assert "low_confidence_adjudication" not in kinds
    assert "low_confidence_source_cue_restored" not in kinds
    assert "invalid_llm_response" not in kinds


def test_no_llm_mode_keeps_matched_word_timing_for_a_diverged_mai_cue(tmp_path):
    srt = "1\n00:00:01,000 --> 00:00:03,000\nWir gehen jetzt nach Hause.\n"
    words = _mai_words([
        ("Wir", 1.50, 1.62), ("laufen", 1.64, 1.90), ("jetzt", 1.92, 2.10),
        ("nach", 2.12, 2.26), ("Hause.", 2.28, 2.60),
    ])

    cues, flags = _sync(tmp_path, srt, words, no_llm=True)

    assert cues[0].plain_text == "Wir gehen jetzt nach Hause."
    assert abs(cues[0].start_ms - 1500) <= 34
    assert abs(cues[0].end_ms - 2640) <= 34
    kinds = _kinds(flags)
    assert kinds.count("divergence_unresolved") == 1
    assert "low_confidence_adjudication" not in kinds
    assert not any(kind.endswith("_source_cue_restored") for kind in kinds)


@pytest.mark.parametrize("verdict, final_text", [("keep_srt", "gehen"), ("use_audio", "laufen")])
def test_low_confidence_model_answer_keeps_source_text_but_not_source_timing(tmp_path, verdict, final_text):
    srt = "1\n00:00:01,000 --> 00:00:03,000\nWir gehen jetzt nach Hause.\n"
    words = _mai_words([
        ("Wir", 1.50, 1.62), ("laufen", 1.64, 1.90), ("jetzt", 1.92, 2.10),
        ("nach", 2.12, 2.26), ("Hause.", 2.28, 2.60),
    ])
    responses = {"case-1": {
        "case_id": "case-1", "verdict": verdict, "final_text": final_text,
        "confidence": 0.5, "reason": "unsure which verb was spoken",
    }}

    cues, flags = _sync(tmp_path, srt, words, responses=responses)

    # The uncertain wording stays reviewable; the cue still follows its speech.
    assert cues[0].plain_text == "Wir gehen jetzt nach Hause."
    assert abs(cues[0].start_ms - 1500) <= 34
    assert abs(cues[0].end_ms - 2640) <= 34
    low_confidence = [flag for flag in flags if flag["kind"] == "low_confidence_adjudication"]
    assert len(low_confidence) == 1
    assert low_confidence[0]["severity"] == "warning"
    assert low_confidence[0]["new_text"] == final_text
    assert not any(str(flag["kind"]).endswith("_source_cue_restored") for flag in flags)


@pytest.mark.parametrize("anchor_confidence", [None, 0.95])
def test_unknown_word_confidence_is_an_acceptable_lexical_anchor(anchor_confidence):
    # MAI ep11 cues 452-454 / 506-507: every anchor failed because None became 0.0.
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["old"]),
        Cue(index=2, start_ms=4000, end_ms=5000, lines=["other phrase"]),
    ]
    span = DivergenceSpan(
        case_id="anchor-confidence", cue_ids=[1, 2], srt_text="old other phrase",
        asr_text="anchor changed", srt_token_indices=[0, 1, 2],
        asr_word_indices=[0, 1], confidence=0.0,
    )
    words = [
        Word(text="anchor", start=0.2, end=0.4, confidence=anchor_confidence),
        Word(text="changed", start=4.2, end=4.4, confidence=anchor_confidence),
    ]
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="hybrid", final_text="anchor new words",
        confidence=0.95, reason="confirmed wording within the saved audio span",
    )

    aligned = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={1: [], 2: []}), [decision], [span], source_cues=cues, words=words,
    )

    assert aligned.cue_word_indices == {1: [0], 2: [1]}
    assert not aligned.flags


@pytest.mark.parametrize("word_confidence, shown", [(None, None), (0.4, 0.4)])
def test_adjudication_prompt_does_not_present_unknown_confidence_as_zero(word_confidence, shown):
    from dubsync.llm_providers import _adjudication_span_payload

    words = [Word(text="laufen", start=1.64, end=1.90, confidence=word_confidence)]
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1], srt_text="gehen", asr_text="laufen", start=1.64, end=1.90,
        confidence=0.0 if word_confidence is None else word_confidence, asr_word_indices=[0],
    )

    assert _adjudication_span_payload(span, episode_words=words)["confidence"] == shown


_SPARSE_SRT = "1\n00:00:01,000 --> 00:00:03,000\nWir gehen jetzt nach Hause.\n"
_SPARSE_WORDS = _mai_words([
    ("Ich", 1.50, 1.62), ("laufe", 1.64, 1.90), ("heute", 1.92, 2.10),
    ("weit", 2.12, 2.26), ("Hause.", 2.28, 2.60),
])


def test_unconfirmed_divergence_with_sparse_word_evidence_is_reported_once(tmp_path):
    cues, flags = _sync(tmp_path, _SPARSE_SRT, _SPARSE_WORDS, no_llm=True)

    # One matched word of five cannot time the complete source phrase.
    assert (cues[0].plain_text, cues[0].start_ms, cues[0].end_ms) == ("Wir gehen jetzt nach Hause.", 1000, 3000)
    kinds = _kinds(flags)
    assert kinds.count("divergence_unresolved") == 1
    assert "timing_evidence_held" not in kinds
    assert not any(kind.endswith("_source_cue_restored") for kind in kinds)


def test_confirmed_source_wording_still_reports_unusable_timing_evidence(tmp_path):
    responses = {"case-1": {
        "case_id": "case-1", "verdict": "keep_srt", "final_text": "Wir gehen jetzt nach",
        "confidence": 0.95, "reason": "the source wording is audible",
    }}

    cues, flags = _sync(tmp_path, _SPARSE_SRT, _SPARSE_WORDS, responses=responses)

    assert (cues[0].start_ms, cues[0].end_ms) == (1000, 3000)
    assert _kinds(flags).count("timing_evidence_held") == 1
