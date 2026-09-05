from __future__ import annotations

import json

import pytest
import yaml

from dubsync.adjudication import AdjudicationEngine, StaticLLMAdapter
from dubsync.changes import apply_adjudication_decisions, single_token_prefix_replacement_targets
from dubsync.cue_segmentation import (
    group_word_indices_for_cues,
    segment_generated_adlib_cues,
    split_overlong_existing_cues,
)
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.pipeline import sync_episode
from dubsync.punctuation import StaticPunctuationAdapter, apply_punctuation_pass
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


@pytest.mark.parametrize(
    ("source", "proposal"),
    [
        ("What the f**k", "What the f--k!"),
        ("What the f**k", "What the f*k!"),
        ("<i>Hello</i>", "<i>Hello!<i>"),
    ],
)
def test_punctuation_preserves_source_censor_masks_and_markup(source, proposal):
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=[source])

    result, flags = apply_punctuation_pass([cue], StaticPunctuationAdapter({1: proposal}))

    assert result == [cue]
    assert [flag.kind for flag in flags] == ["invalid_punctuation_change"]


def test_punctuation_accepts_valid_changes_around_preserved_editorial_content():
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["<i>what the f**k</i>"])

    result, flags = apply_punctuation_pass(
        [cue], StaticPunctuationAdapter({1: "<i>What the f**k!</i>"})
    )

    assert result[0].text == "<i>What the f**k!</i>"
    assert flags == []


def _profile():
    return StyleProfile(
        fps=25,
        min_cue_dur=0.1,
        max_chars_per_line=80,
        max_lines_per_cue=2,
        lead_in_ms=0,
        tail_ms=0,
    )


def test_sentence_similarity_does_not_hide_actor_lexical_replacement():
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1],
        srt_text="I will open the door before everyone arrives at the house tonight",
        asr_text="I will open the gate before everyone arrives at the house tonight",
        start=1, end=4, confidence=0.96,
    )
    adapter = StaticLLMAdapter({"case-1": {
        "case_id": "case-1", "verdict": "use_audio", "final_text": span.asr_text,
        "confidence": 0.96, "reason": "audio confirms the actor changed door to gate",
    }})

    decisions, flags = AdjudicationEngine(adapter).adjudicate([span])

    assert decisions[0].verdict == "use_audio"
    assert decisions[0].final_text == span.asr_text
    assert flags == []


@pytest.mark.parametrize("proposal", ["Different spoken words", ""])
@pytest.mark.parametrize("verdict", ["use_audio", "hybrid"])
def test_low_confidence_adjudication_preserves_source_instead_of_rewriting_or_deleting(proposal, verdict):
    source = Cue(index=1, start_ms=1000, end_ms=2000, lines=["Keep these source words"])
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1], srt_text=source.text,
        asr_text=proposal, start=1, end=2, confidence=0.9,
    )
    adapter = StaticLLMAdapter({"case-1": {
        "case_id": "case-1", "verdict": verdict, "final_text": proposal,
        "confidence": 0.61, "reason": "uncertain spoken wording",
    }})

    decisions, flags = AdjudicationEngine(adapter, confidence_gate=0.7).adjudicate([span])
    result, _ = apply_adjudication_decisions(
        [source], [span], decisions, _profile().model_copy(update={"drop_policy": "remove"})
    )

    assert result == [source]
    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == source.text
    assert flags[0].old_text == source.text
    assert flags[0].new_text == proposal
    assert flags[0].kind == "low_confidence_adjudication"


def test_confidence_at_gate_still_applies_verified_actor_rewrite():
    source = Cue(index=1, start_ms=1000, end_ms=2000, lines=["Keep these source words"])
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1], srt_text=source.text,
        asr_text="Different spoken words", start=1, end=2, confidence=0.9,
    )
    adapter = StaticLLMAdapter({"case-1": {
        "case_id": "case-1", "verdict": "use_audio", "final_text": span.asr_text,
        "confidence": 0.7, "reason": "verified spoken wording",
    }})

    decisions, flags = AdjudicationEngine(adapter, confidence_gate=0.7).adjudicate([span])
    result, _ = apply_adjudication_decisions([source], [span], decisions, _profile())

    assert result[0].plain_text == span.asr_text
    assert flags == []


def test_unknown_word_speaker_does_not_hide_known_speaker_change():
    words = [
        Word(text="Hello", start=1.0, end=1.2, speaker_id="A"),
        Word(text="there", start=1.2, end=1.4),
        Word(text="No", start=1.4, end=1.6, speaker_id="B"),
    ]

    groups = group_word_indices_for_cues(
        words, [0, 1, 2], _profile(), max_gap_seconds=0.8, max_cue_duration_seconds=5
    )

    assert groups == [[0, 1], [2]]


def test_source_line_split_retains_known_speaker_boundary_after_unknown_word():
    words = [
        Word(text="Hello", start=1.0, end=1.2, speaker_id="A"),
        Word(text="there", start=1.2, end=1.4),
        Word(text="No", start=1.4, end=1.6, speaker_id="B"),
        Word(text="thanks", start=1.6, end=1.8, speaker_id="B"),
    ]
    source = Cue(index=1, start_ms=1000, end_ms=1800, lines=["Hello there", "No", "thanks"])
    alignment = AlignmentResult(cue_word_indices={1: [0, 1, 2, 3]})

    result, updated, _, _ = split_overlong_existing_cues(
        [source], words, alignment, _profile(),
        max_gap_seconds=0.8, max_cue_duration_seconds=5,
    )

    assert [cue.plain_text for cue in result] == ["Hello there", "No thanks"]
    assert [updated.cue_word_indices[cue.index] for cue in result] == [[0, 1], [2, 3]]


def test_multi_token_asr_words_keep_generated_text_with_its_actual_speaker():
    words = [
        Word(text="I really", start=1.0, end=1.4, speaker_id="A"),
        Word(text="mean it.", start=1.4, end=1.8, speaker_id="A"),
        Word(text="No.", start=2.0, end=2.2, speaker_id="B"),
    ]
    source = Cue(index=1, start_ms=1000, end_ms=2200, lines=["I really mean it. No."])
    alignment = AlignmentResult(cue_word_indices={1: [0, 1, 2]})

    result, updated, _, _ = segment_generated_adlib_cues(
        [source], words, alignment, {1}, _profile(),
        max_gap_seconds=0.8, max_cue_duration_seconds=5,
    )

    assert [(cue.plain_text, cue.speaker_id) for cue in result] == [
        ("I really mean it.", "A"), ("No.", "B")
    ]
    for cue in result:
        spoken = " ".join(words[index].text for index in updated.cue_word_indices[cue.index])
        assert alphanumeric_signature(cue.plain_text) == alphanumeric_signature(spoken)
    assert alignment.cue_word_indices == {1: [0, 1, 2]}


def test_cross_cue_single_word_replacement_belongs_to_surviving_sentence_prefix():
    # Minimized from long-form corpus case-134, original cues 403/404.
    cues = [
        Cue(index=403, start_ms=1242960, end_ms=1243680, lines=["Divisão Comercial."]),
        Cue(index=404, start_ms=1243680, end_ms=1244830, lines=["Tem uma negociação comercial à tarde,"]),
    ]
    span = DivergenceSpan(
        case_id="case-134", cue_ids=[403, 404], srt_text="Divisão Comercial Tem",
        asr_text="Tenho", srt_token_indices=[0, 1, 2], asr_word_indices=[0],
        start=1243.072, end=1243.142, confidence=1, speaker_ids=["speaker_3"],
    )
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text="Tenho",
        confidence=1, reason="actor's spoken prefix verified in saved adjudication",
    )

    result, _ = apply_adjudication_decisions(cues, [span], [decision], _profile())

    assert [(cue.index, cue.plain_text) for cue in result] == [
        (404, "Tenho uma negociação comercial à tarde,")
    ]
    assert cues[0].plain_text == "Divisão Comercial."
    assert cues[1].plain_text == "Tem uma negociação comercial à tarde,"


@pytest.mark.parametrize(
    ("first", "second", "indices", "span_text"),
    [
        ("Hoje Divisão Comercial.", "Tem uma negociação", [1, 2, 3], "Divisão Comercial Tem"),
        ("Divisão Comercial.", "Tem", [0, 1, 2], "Divisão Comercial Tem"),
        ("Divisão Comercial.", "Tem uma negociação", [0, 2], "Divisão Tem"),
        ("[Divisão Comercial.]", "Tem uma negociação", [0], "Tem"),
    ],
)
def test_prefix_ownership_does_not_guess_without_complete_indexed_source_structure(first, second, indices, span_text):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1700, lines=[first]),
        Cue(index=2, start_ms=1700, end_ms=2800, lines=[second]),
    ]
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1, 2], srt_text=span_text, asr_text="Tenho",
        srt_token_indices=indices, asr_word_indices=[0],
    )
    decision = AdjudicationDecision(
        case_id="case-1", verdict="use_audio", final_text="Tenho", confidence=1,
        reason="verified replacement",
    )

    assert single_token_prefix_replacement_targets(cues, [span], [decision]) == {}


def test_pipeline_keeps_real_cross_cue_actor_prefix_with_following_sentence(tmp_path):
    source = tmp_path / "episode.srt"
    audio = tmp_path / "episode.wav"
    wordstream = tmp_path / "words.json"
    config = tmp_path / "providers.yaml"
    style = tmp_path / "style.yaml"
    output = tmp_path / "output.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nprocurar um novo local.\n\n"
        "2\n00:00:01,000 --> 00:00:01,720\nDivisão Comercial.\n\n"
        "3\n00:00:01,720 --> 00:00:02,870\nTem uma negociação comercial à tarde,\n\n"
        "4\n00:00:02,870 --> 00:00:04,430\nprepare logo o plano de cooperação.\n",
        encoding="utf-8",
    )
    audio.write_bytes(b"RIFF....WAVEfmt ")
    words = [
        Word(text=token, start=index * 0.2, end=index * 0.2 + 0.18, speaker_id="speaker_3")
        for index, token in enumerate("procurar um novo local.".split())
    ]
    words.extend([
        Word(text="Tenho", start=1.112, end=1.182, speaker_id="speaker_3"),
        Word(text="uma", start=1.192, end=1.292, speaker_id="speaker_3"),
    ])
    words.extend([
        Word(text=token, start=1.32 + index * 0.23, end=1.52 + index * 0.23, speaker_id="speaker_3")
        for index, token in enumerate("negociação comercial à tarde,".split())
    ])
    words.extend([
        Word(text=token, start=2.90 + index * 0.2, end=3.08 + index * 0.2, speaker_id="speaker_3")
        for index, token in enumerate("prepare logo o plano de cooperação.".split())
    ])
    wordstream.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"provider": "fixture", "responses": {"case-1": {
            "case_id": "case-1", "verdict": "use_audio", "final_text": "Tenho",
            "confidence": 1, "reason": "saved audio evidence confirms actor prefix",
        }}},
    }), encoding="utf-8")
    style.write_text(yaml.safe_dump(_profile().model_dump()), encoding="utf-8")

    sync_episode(source, audio, output, tmp_path / "work", style_path=style, providers_path=config)

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == [
        "procurar um novo local.", "Tenho uma negociação comercial à tarde,",
        "prepare logo o plano de cooperação.",
    ]
    assert cues[1].start_ms <= 1112
    assert cues[1].end_ms >= 2210
    assert cues[1].duration_ms > 1000
