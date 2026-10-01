"""An approved wording and its timing are applied together or held together."""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.changes import apply_adjudication_decisions
from dubsync.edit_consistency import hold_fragmenting_replacements, settle_edits_with_held_timing
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _sync(tmp_path, srt: str, words: list[tuple], responses: dict[str, dict[str, object]]):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "confidence": None} for text, start, end in words
    ]}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": responses},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


def _decide(case_id: str, final_text: str, verdict: str = "hybrid") -> dict[str, object]:
    return {"case_id": case_id, "verdict": verdict, "final_text": final_text, "confidence": 0.97,
            "reason": "heard in the case audio"}


def _kinds(flags) -> list[str]:
    return [flag["kind"] for flag in flags]


def test_approved_wording_is_timed_from_the_words_the_adjudicator_heard(tmp_path):
    # ep11 MAI cue 193: 'Sun Yu,' with the ASR 'Sonho. Hã?'. The approved text
    # 'Sun Yu. Hã?' was shown at 624.89-625.72 although "Hã?" is spoken at 626.12.
    srt = (
        "1\n00:00:02,000 --> 00:00:03,900\nVou colocar no vaso.\n\n"
        "2\n00:00:04,890 --> 00:00:05,720\nSun Yu,\n\n"
        "3\n00:00:08,000 --> 00:00:09,500\nVocê viu a empresa?\n"
    )
    words = [
        ("Vou", 2.60, 2.76), ("colocar", 3.16, 3.40), ("no", 3.44, 3.52), ("vaso.", 3.60, 3.84),
        ("Sonho.", 5.12, 5.68), ("Hã?", 6.12, 6.40),
        ("Você", 8.00, 8.18), ("viu", 8.20, 8.40), ("a", 8.42, 8.46), ("empresa?", 8.50, 9.20),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Sun Yu. Hã?")})

    edited = cues[1]
    assert edited.plain_text == "Sun Yu. Hã?"
    assert abs(edited.start_ms - 5120) <= 34 and abs(edited.end_ms - 6440) <= 34
    kinds = _kinds(flags)
    assert kinds.count("text_changed") == 1
    assert "timing_evidence_held" not in kinds
    assert not any(kind.endswith("_source_cue_restored") for kind in kinds)
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_wording_without_a_place_of_its_own_is_held_with_its_timing(tmp_path):
    # ep11 cue 761: 'no Ano-Novo.' became 'no Ano-Novo. Alô?' at source timing,
    # although "Alô?" is spoken 14 s later by another actor.
    srt = (
        "1\n00:00:01,000 --> 00:00:01,600\nRealizem seus desejos\n\n"
        "2\n00:00:01,640 --> 00:00:02,670\nno Ano-Novo.\n\n"
        "3\n00:00:17,000 --> 00:00:18,500\nQuem está falando agora?\n"
    )
    words = [
        ("Realizem", 0.72, 1.00), ("seus", 1.02, 1.20), ("desejos", 1.22, 1.46),
        ("no", 1.68, 1.76), ("ano", 1.84, 2.02), ("novo.", 2.12, 2.48),
        ("Alô?", 15.90, 16.30),
        ("Quem", 17.00, 17.20), ("está", 17.22, 17.50), ("falando", 17.52, 18.00), ("agora?", 18.02, 18.40),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Ano-Novo. Alô?", "use_audio")})

    held = cues[1]
    assert (held.plain_text, held.start_ms, held.end_ms) == ("no Ano-Novo.", 1640, 2670)
    kinds = _kinds(flags)
    assert "text_changed" not in kinds and "timing_evidence_held" not in kinds
    holds = [flag for flag in flags if flag["kind"] == "adjudication_replacement_ownership_held"]
    assert [(flag["cue_ids"], flag["severity"]) for flag in holds] == [([2], "warning")]
    assert not any(kind.endswith("_source_cue_restored") for kind in kinds)


_RESIDUE_SRT = (
    "1\n00:00:01,000 --> 00:00:02,400\nEu vi tudo ontem\n\n"
    "2\n00:00:02,830 --> 00:00:04,120\ne não deixei que ele conseguisse\n\n"
    "3\n00:00:06,000 --> 00:00:07,500\nEle tentou de novo.\n"
)


def _residue_words(letter_start: float) -> list[tuple]:
    return [
        ("Eu", 1.00, 1.10), ("vi", 1.12, 1.30), ("tudo", 1.32, 1.70), ("ontem", 1.72, 2.20),
        ("e", letter_start, letter_start + 0.10),  # the deleted words would follow here
        ("Ele", 6.00, 6.40), ("tentou", 6.42, 6.70), ("de", 6.72, 6.80), ("novo.", 6.82, 7.20),
    ]


def test_spoken_letter_left_by_a_deletion_joins_the_sentence_it_is_spoken_with(tmp_path):
    # ep17 cue 198: 'e não deixei que ele conseguisse' was exported as the cue
    # 'e' (824.00-824.10) directly before 'Ele ainda tentou me acalmar,' (824.10).
    cues, flags = _sync(tmp_path, _RESIDUE_SRT, _residue_words(5.80), {"case-1": _decide("case-1", "", "use_audio")})

    assert [cue.plain_text for cue in cues] == ["Eu vi tudo ontem", "e Ele tentou de novo."]
    assert abs(cues[1].start_ms - 5800) <= 34 and abs(cues[1].end_ms - 7240) <= 34
    changed = [flag for flag in flags if flag["kind"] == "text_changed"]
    assert [(flag["cue_ids"], flag["new_text"]) for flag in changed] == [([3], "e Ele tentou de novo.")]
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_deletion_that_would_leave_a_lone_one_letter_cue_is_held(tmp_path):
    cues, flags = _sync(tmp_path, _RESIDUE_SRT, _residue_words(4.90), {"case-1": _decide("case-1", "", "use_audio")})

    assert [cue.plain_text for cue in cues] == ["Eu vi tudo ontem", "e não deixei que ele conseguisse", "Ele tentou de novo."]
    assert (cues[1].start_ms, cues[1].end_ms) == (2830, 4120)
    kinds = _kinds(flags)
    assert "text_changed" not in kinds
    assert kinds.count("adjudication_replacement_ownership_held") == 1


def _case174():
    cues = [
        Cue(index=452, start_ms=1321160, end_ms=1321680, lines=["Vamos."]),
        Cue(index=453, start_ms=1321750, end_ms=1322310, lines=["Vamos rápido."]),
        Cue(index=454, start_ms=1322310, end_ms=1323030, lines=["Yuanzhu vai pagar."]),
    ]
    span = DivergenceSpan(
        case_id="case-174", cue_ids=[452, 453, 454],
        srt_text="Vamos Vamos rápido Yuanzhu", asr_text="E o Antzu",
        srt_token_indices=[0, 1, 2, 3], asr_word_indices=[0, 1, 2],
        start=1322.24, end=1322.74, speaker_ids=["speaker_4", "speaker_5"],
    )
    return cues, span


def _apply(cues, span):
    return lambda decisions: apply_adjudication_decisions(cues, [span], decisions, StyleProfile())


@pytest.mark.parametrize("final_text, held", [
    ("E o Yuanzhu", True),          # proportional pieces "E" / "o" / "Yuanzhu"
    ("Então olha o Yuanzhu", False),  # every cue receives real words
])
def test_replacement_leaving_one_letter_cues_is_held_as_a_whole(final_text, held):
    cues, span = _case174()
    decision = AdjudicationDecision(case_id=span.case_id, verdict="hybrid", final_text=final_text,
                                    confidence=0.95, reason="approved")

    decisions, flags = hold_fragmenting_replacements(cues, [span], [decision], _apply(cues, span))

    assert (decisions[0].verdict == "keep_srt") is held
    assert [flag.kind for flag in flags] == (["adjudication_replacement_ownership_held"] if held else [])
    changed, _ = apply_adjudication_decisions(cues, [span], decisions, StyleProfile())
    assert not [cue.plain_text for cue in changed if sum(ch.isalnum() for ch in cue.plain_text) <= 1]
    if held:
        assert changed == cues
        assert flags[0].cue_ids == [452, 453, 454] and flags[0].new_text == final_text


def test_complete_one_letter_answer_is_not_a_leftover():
    # ep11 cue 490: "Mas hoje não estou muito bem." is really answered with "É."
    cues = [Cue(index=490, start_ms=1000, end_ms=3000, lines=["Mas hoje não estou muito bem."])]
    span = DivergenceSpan(case_id="case-1", cue_ids=[490], srt_text="Mas hoje não estou muito bem", asr_text="É.",
                          srt_token_indices=[0, 1, 2, 3, 4, 5], asr_word_indices=[0], start=1.2, end=1.5)
    decision = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="É.", confidence=0.95, reason="heard")

    decisions, flags = hold_fragmenting_replacements(cues, [span], [decision], _apply(cues, span))

    assert decisions == [decision] and flags == []


def test_replacement_without_unique_word_ownership_keeps_text_and_timing_together(tmp_path):
    # One ASR word ("can't go") cannot be divided between two cues: the
    # mapping was held while the new text was still written into both cues.
    srt = (
        "1\n00:00:00,500 --> 00:00:01,400\nBefore that happened\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nold\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nother phrase\n\n"
        "4\n00:00:05,000 --> 00:00:06,000\nAfter that happened\n"
    )
    words = [
        ("Before", 0.50, 0.80), ("that", 0.82, 1.00), ("happened", 1.02, 1.40),
        ("can't go", 2.20, 3.30), ("now", 3.50, 3.80),
        ("After", 5.00, 5.30), ("that", 5.32, 5.50), ("happened", 5.52, 5.90),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "can't go now")})

    assert [cue.plain_text for cue in cues] == ["Before that happened", "old", "other phrase", "After that happened"]
    kinds = _kinds(flags)
    assert "text_changed" not in kinds
    assert kinds.count("adjudication_word_mapping_held") + kinds.count("adjudication_replacement_ownership_held") == 1


def test_cue_that_only_lost_words_keeps_its_approved_deletion_at_held_timing():
    # Scribe ep11 cue 513: the first turn was deleted; the remaining source
    # words are shown at source timing, which is consistent as it is.
    source = [Cue(index=513, start_ms=1_432_790, end_ms=1_434_680,
                  lines=["- Estou muito bonito. - Não consegui pegar ele."])]
    edited = [source[0].with_lines(["Não consegui pegar ele."])]
    recue_flags = [QCFlag(kind="timing_evidence_held", cue_ids=[513], severity="error", message="collapsed")]
    flags = [QCFlag(kind="text_changed", cue_ids=[513], message="deleted the unspoken turn",
                    old_text=source[0].text, new_text=edited[0].text)]

    def no_rebuild(_cues):
        raise AssertionError("a pure deletion needs no second rebuild")

    result = settle_edits_with_held_timing(
        edited, edited, recue_flags, flags, source_cues=source, words=[], alignment=AlignmentResult(),
        rebuild=no_rebuild, max_intra_cue_gap=1.5, max_word_duration=2.0,
    )

    assert result == (edited, recue_flags, flags)
