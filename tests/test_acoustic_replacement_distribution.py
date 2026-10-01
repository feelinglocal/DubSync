"""A replacement covering several cues gives each word to the cue it was spoken in."""
from __future__ import annotations

import json

import yaml

from dubsync import pipeline
from dubsync.changes import indexed_multi_cue_replacements
from dubsync.models import Cue, DivergenceSpan, Word
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


def _decide(case_id: str, final_text: str, verdict: str = "use_audio") -> dict[str, object]:
    return {"case_id": case_id, "verdict": verdict, "final_text": final_text, "confidence": 0.97,
            "reason": "heard in the case audio"}


def _near(value_ms: int, seconds: float, tolerance_ms: int = 45) -> bool:
    return abs(value_ms - round(seconds * 1000)) <= tolerance_ms


def test_words_after_a_pause_go_to_the_cue_spoken_at_that_time(tmp_path):
    # ep11 MAI cues 319/320: 'Ó.(993.36) | Aqui,(995.36) ó. Ah.' was cut by source
    # token share into 'água, Ó. Aqui,' / 'ó. Ah.': "Aqui," shown 1.7 s early.
    srt = (
        "1\n00:00:11,400 --> 00:00:12,470\nÁgua, água, água, toma.\n\n"
        "2\n00:00:15,200 --> 00:00:15,750\nÁgua.\n\n"
        "3\n00:00:15,960 --> 00:00:16,470\nToma logo isso.\n"
    )
    words = [
        ("Água,", 11.50, 11.80), ("água,", 11.84, 12.10), ("água,", 12.14, 12.64),
        ("Ó.", 13.36, 13.62), ("Aqui,", 15.36, 15.60), ("ó.", 15.64, 15.78), ("Ah.", 15.92, 16.16),
        ("Toma", 16.32, 16.50), ("logo", 16.52, 16.70), ("isso.", 16.72, 16.96),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Ó. Aqui, ó. Ah.")})

    assert [cue.plain_text for cue in cues] == ["Água, água, água, Ó.", "Aqui, ó. Ah.", "Toma logo isso."]
    assert _near(cues[0].end_ms, 13.66)
    assert _near(cues[1].start_ms, 15.36) and cues[1].end_ms <= 16330
    kinds = [flag["kind"] for flag in flags]
    assert "timing_outlier_trimmed" not in kinds and not [flag for flag in flags if flag["severity"] == "error"]


def test_interjection_before_the_next_line_is_not_appended_to_a_cue_spoken_long_before(tmp_path):
    # ep11 MAI cues 403/406: 'Ah,(1206.48) que(1208.12)' gave 'na nossa viagem anual Ah,'
    # at 1184.5 s, 21 s before "Ah," is spoken.
    srt = (
        "1\n00:00:03,990 --> 00:00:05,210\nna nossa viagem anual deste ano?\n\n"
        "2\n00:00:27,920 --> 00:00:29,200\nEssa vista é linda.\n"
    )
    words = [
        ("na", 4.53, 4.60), ("nossa", 4.62, 4.90), ("viagem", 4.94, 5.30), ("anual?", 5.34, 5.68),
        ("Ah,", 26.48, 27.12), ("que", 28.12, 28.24),
        ("vista", 28.44, 28.68), ("é", 28.70, 28.76), ("linda.", 28.84, 29.20),
    ]

    cues, _ = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Ah, que")})

    assert [cue.plain_text for cue in cues] == ["na nossa viagem anual?", "Ah, que vista é linda."]
    assert _near(cues[0].end_ms, 5.72)
    assert _near(cues[1].start_ms, 26.48)


def test_single_word_goes_to_the_cue_whose_words_it_touches(tmp_path):
    # ep17 cues 458/459: "Donnie" is spoken 40 ms before 'foi contratado' and
    # 360 ms after the previous cue, yet 'Dony' was appended to the previous cue.
    srt = (
        "1\n00:00:03,230 --> 00:00:05,510\nTambém houve outras irregularidades, como desvio de cargo.\n\n"
        "2\n00:00:05,560 --> 00:00:07,560\nAlém disso, o Dony foi contratado pela sede.\n"
    )
    words = [
        ("Também", 3.57, 3.90), ("houve", 3.92, 4.20), ("outras", 4.22, 4.50), ("irregularidades.", 4.52, 5.44),
        ("Donnie", 5.80, 6.08),
        ("foi", 6.12, 6.24), ("contratado", 6.26, 6.80), ("pela", 6.82, 6.98), ("sede.", 7.00, 7.40),
    ]

    cues, _ = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Dony", "hybrid")})

    assert "Dony" not in cues[0].plain_text
    assert cues[1].plain_text == "Dony foi contratado pela sede."
    assert _near(cues[0].end_ms, 5.48) and _near(cues[1].start_ms, 5.80)


def test_connector_exactly_at_the_attach_gap_joins_the_line_it_introduces(tmp_path):
    # ep17: "E" (234.56-234.64) before 'qual é o seu plano?' (234.84). In binary
    # floats 234.84 - 234.64 is a hair above 0.2 s, so "E" became a cue of its own.
    srt = (
        "1\n00:03:52,150 --> 00:03:53,400\nVocê é muito cuidadosa.\n\n"
        "2\n00:03:54,400 --> 00:03:55,800\nQual é o seu plano?\n"
    )
    words = [
        ("Você", 232.28, 232.46), ("é", 232.50, 232.56), ("muito", 232.60, 232.78), ("cuidadosa.", 232.84, 233.44),
        ("E", 234.56, 234.64),
        ("qual", 234.84, 234.96), ("é", 234.98, 235.02), ("o", 235.06, 235.08), ("seu", 235.12, 235.26),
        ("plano?", 235.36, 235.64),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "E")})

    assert [cue.plain_text for cue in cues] == ["Você é muito cuidadosa.", "E Qual é o seu plano?"]
    assert _near(cues[1].start_ms, 234.56)
    assert "adlib_inserted" not in [flag["kind"] for flag in flags]


def _case114():
    cues = [
        Cue(index=319, start_ms=991_400, end_ms=992_470, lines=["Água, água, água, toma."]),
        Cue(index=320, start_ms=995_200, end_ms=995_750, lines=["Água."]),
        Cue(index=321, start_ms=995_960, end_ms=996_470, lines=["Toma."]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("Ó.", 993.36, 993.62), ("Aqui,", 995.36, 995.60), ("ó.", 995.64, 995.78), ("Ah.", 995.92, 996.16),
    ]]
    span = DivergenceSpan(
        case_id="case-114", cue_ids=[319, 320], srt_text="toma Água", asr_text="Ó. Aqui, ó. Ah.",
        srt_token_indices=[3, 4], asr_word_indices=[0, 1, 2, 3], start=993.36, end=996.16,
        left_anchor_cue_id=319, right_anchor_cue_id=321, left_anchor_end=992.64, right_anchor_start=996.319,
    )
    return cues, words, span


def test_text_pieces_and_word_ownership_use_the_same_acoustic_cut():
    cues, words, span = _case114()
    ownership: dict[int, list[int]] = {}

    edits = indexed_multi_cue_replacements(cues, span, "Ó. Aqui, ó. Ah.", words=words, ownership=ownership)

    assert edits == {319: (3, 4, "Ó."), 320: (0, 1, "Aqui, ó. Ah.")}
    assert ownership == {319: [0], 320: [1, 2, 3]}


def test_stretched_word_after_a_short_pause_still_starts_the_later_cue():
    # Scribe ep11: 'Ó,(993.44-993.54) aqui(994.00-995.60) ó. Ah.' - the word
    # absorbed the silence before it, so only 0.46 s of pause is left.
    cues, _, span = _case114()
    words = [Word(text=text, start=start, end=end, confidence=1.0) for text, start, end in [
        ("Ó,", 993.44, 993.54), ("aqui", 994.00, 995.60), ("ó.", 995.66, 995.78), ("Ah.", 995.94, 996.20),
    ]]
    span = span.model_copy(update={
        "asr_text": "Ó, aqui ó. Ah.", "start": 993.44, "end": 996.20,
        "left_anchor_end": 993.322, "right_anchor_start": 996.262,
    })

    edits = indexed_multi_cue_replacements(cues, span, "Ó, aqui, ó. Ah.", words=words)

    assert edits == {319: (3, 4, "Ó,"), 320: (0, 1, "aqui, ó. Ah.")}


def test_without_word_timing_the_proportional_split_is_kept():
    cues, _, span = _case114()

    edits = indexed_multi_cue_replacements(cues, span, "Ó. Aqui, ó. Ah.")

    assert edits == {319: (3, 4, "Ó. Aqui,"), 320: (0, 1, "ó. Ah.")}


def test_continuous_speech_touching_both_cues_is_not_decided_by_timing():
    # 'a gente ficar velho, vai poder' runs from one cue's retained words into
    # the next cue's without a pause: timing does not say where the cut is.
    cues = [
        Cue(index=501, start_ms=1_417_510, end_ms=1_419_120, lines=["Assim, quando estivermos velhos,"]),
        Cue(index=502, start_ms=1_419_120, end_ms=1_420_990, lines=["podemos olhar para ela e lembrar desses momentos."]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("a", 1418.54, 1418.56), ("gente", 1418.58, 1418.72), ("ficar", 1418.80, 1419.00),
        ("velho,", 1419.08, 1419.38), ("vai", 1419.48, 1419.58), ("poder", 1419.60, 1419.76),
    ]]
    span = DivergenceSpan(
        case_id="case-194", cue_ids=[501, 502], srt_text="estivermos velhos podemos olhar para ela e",
        asr_text="a gente ficar velho, vai poder", srt_token_indices=[2, 3, 4, 5, 6, 7, 8],
        asr_word_indices=[0, 1, 2, 3, 4, 5], start=1418.54, end=1419.76,
        left_anchor_cue_id=501, right_anchor_cue_id=502, left_anchor_end=1418.52, right_anchor_start=1419.80,
    )
    final_text = "a gente ficar velho, vai poder"
    ownership: dict[int, list[int]] = {}

    with_timing = indexed_multi_cue_replacements(cues, span, final_text, words=words, ownership=ownership)

    assert with_timing == indexed_multi_cue_replacements(cues, span, final_text)
    assert ownership == {}
