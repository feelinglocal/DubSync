"""Words an ASR provider decoded twice never become inserted dialogue; a spoken repetition is reviewed."""
from __future__ import annotations

import json
import re

import pytest
import yaml

from dubsync import pipeline
from dubsync.aligner import align_cues_to_words
from dubsync.models import AdjudicationDecision, AlignmentResult, DivergenceSpan, Word
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import normalize_token


def _sync_episode(tmp_path, srt: str, words: list[tuple], responses: dict[str, dict[str, object]]):
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
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"], result.episode_workdir


def _sync(tmp_path, srt: str, words: list[tuple], responses: dict[str, dict[str, object]]):
    cues, flags, _workdir = _sync_episode(tmp_path, srt, words, responses)
    return cues, flags


def _decide(case_id: str, final_text: str, verdict: str = "use_audio") -> dict[str, object]:
    return {"case_id": case_id, "verdict": verdict, "final_text": final_text, "confidence": 0.97,
            "reason": "heard in the case audio"}


def _heard(case_id: str, text: str) -> dict[str, object]:
    return {"case_id": case_id, "verdict": "use_audio", "final_text": text, "heard_text": text,
            "evidence": "heard_clearly", "confidence": 1.0, "reason": "the actor says it twice"}


_FLOWERS_SRT = (
    "1\n00:00:19,900 --> 00:00:21,030\nSe quer ficar bem com ela,\n\n"
    "2\n00:00:21,200 --> 00:00:22,630\nmanda umas flores\n\n"
    "3\n00:00:22,830 --> 00:00:23,900\npra acalmar a garota.\n"
)


def _flowers_words(copy_gap: float) -> list[tuple]:
    # MAI ep11: 'manda umas flores.' (321.20-321.66) directly followed by
    # 'Manda umas flores' (321.68-322.48); the human reference has it once.
    second = 21.66 + copy_gap
    return [
        ("Se", 19.90, 20.00), ("quer", 20.02, 20.20), ("ficar", 20.28, 20.48), ("bem", 20.52, 20.64),
        ("com", 20.68, 20.82), ("ela,", 20.84, 21.02),
        ("manda", 21.20, 21.34), ("umas", 21.36, 21.50), ("flores.", 21.52, 21.66),
        ("Manda", second, second + 0.12), ("umas", second + 0.14, second + 0.28), ("flores", second + 0.48, second + 0.80),
        ("pra", second + 1.12, second + 1.24), ("acalmar", second + 1.28, second + 1.60),
        ("a", second + 1.62, second + 1.64), ("garota.", second + 1.76, second + 2.10),
    ]


# The source words match "manda" of the first copy and "umas flores" of the
# second, so the case holds the words in between: 'umas flores. Manda'.
_COPY_CASE = {"case-1": _decide("case-1", "umas flores. Manda")}
_COPY_KEPT = {"case-1": _decide("case-1", "", verdict="keep_srt")}


@pytest.mark.parametrize(("copy_gap", "responses"), [
    # The second copy starts before the first has ended: no actor says a phrase twice at one time.
    pytest.param(-0.06, _COPY_CASE, id="approved-copy-written-over-its-twin"),
    # ep11 MAI: the copy follows its twin and the adjudicator heard no second utterance.
    pytest.param(0.02, _COPY_KEPT, id="kept-copy-after-its-twin"),
])
def test_phrase_decoded_twice_is_shown_once_and_reported_nowhere(tmp_path, copy_gap, responses):
    cues, flags = _sync(tmp_path, _FLOWERS_SRT, _flowers_words(copy_gap), responses)

    assert [cue.plain_text for cue in cues] == [
        "Se quer ficar bem com ela,", "manda umas flores", "pra acalmar a garota.",
    ]
    # Both copies are the cue's speech: it starts with the first and ends with the second.
    assert abs(cues[1].start_ms - 21200) <= 34 and abs(cues[1].end_ms - round((22.50 + copy_gap) * 1000)) <= 45
    kinds = [flag["kind"] for flag in flags]
    assert not {"adlib_inserted", "text_changed", "timing_outlier_trimmed"} & set(kinds)
    assert not [flag for flag in flags if flag["severity"] != "info" and flag["kind"] != "fps_detection_low_confidence"]


def test_phrase_repeated_after_a_pause_is_real_speech_and_stays(tmp_path):
    cues, flags = _sync(tmp_path, _FLOWERS_SRT, _flowers_words(0.45), _COPY_CASE)

    assert " ".join(cue.plain_text for cue in cues).casefold().count("manda umas flores") == 2
    assert {"adlib_inserted", "text_changed"} & {flag["kind"] for flag in flags}


def _alignment(span: DivergenceSpan) -> AlignmentResult:
    return AlignmentResult(divergence_spans=[span], cue_word_indices={1: [0], 2: [2, 3], 3: [5]})


def _words(copy_start: float) -> list[Word]:
    return [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("Olha.", 9.00, 9.40), ("a", copy_start, copy_start + 0.02), ("A", 10.64, 10.66), ("garota.", 10.68, 11.04),
        ("Hum.", 12.00, 12.30), ("Depois.", 14.00, 14.40),
    ]]


def test_overlapping_copy_is_dropped_and_its_words_time_the_cue_that_owns_the_twin():
    # 'a garota.' (323.30-323.62) / 'A garota.' (323.64-324.04): the copy here
    # is the single word before its twin, written without a duration of its own.
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="a", asr_word_indices=[1],
                          start=10.60, end=10.62, left_anchor_cue_id=1, right_anchor_cue_id=2)
    decision = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="a", confidence=0.95, reason="heard")

    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [decision], _words(10.60))

    assert [(item.verdict, item.final_text) for item in decisions] == [("keep_srt", "")]
    assert alignment.cue_word_indices[2] == [1, 2, 3]
    assert alignment.flags == [] and flags == []


def test_insertion_with_other_wording_or_far_from_its_twin_is_left_alone():
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="a", asr_word_indices=[1],
                          start=10.60, end=10.62, left_anchor_cue_id=1, right_anchor_cue_id=2)
    changed = AdjudicationDecision(case_id="case-1", verdict="hybrid", final_text="ah, a", confidence=0.95, reason="heard")
    same = changed.model_copy(update={"final_text": "a"})

    for decision, words in ((changed, _words(10.60)), (same, _words(10.20))):
        alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [decision], words)
        assert decisions == [decision] and flags == []
        assert alignment.cue_word_indices[2] == [2, 3]


def test_clearly_heard_insertion_is_never_taken_back_as_a_provider_copy():
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="a", asr_word_indices=[1],
                          start=10.60, end=10.62, left_anchor_cue_id=1, right_anchor_cue_id=2)
    heard = AdjudicationDecision.model_validate(_heard("case-1", "a"))

    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [heard], _words(10.60))

    assert decisions == [heard] and flags == []
    assert alignment.cue_word_indices[2] == [2, 3]


def _come_words(copy: tuple[float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("Olha.", 9.00, 9.40), ("Vem", *copy), ("Vem", 10.12, 10.22), ("cá.", 10.24, 10.40),
        ("Hum.", 12.00, 12.30), ("Depois.", 14.00, 14.40),
    ]]


def test_approved_copy_is_only_taken_back_when_it_shares_the_time_of_its_twin():
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="Vem", asr_word_indices=[1],
                          start=10.00, end=10.10, left_anchor_cue_id=1, right_anchor_cue_id=2)
    approved = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="Vem", confidence=0.95, reason="heard")
    heard = AdjudicationDecision.model_validate(_heard("case-1", "Vem"))

    # Spoken one after the other (20 ms apart) and heard clearly: the approved repetition stays.
    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [heard], _come_words((10.00, 10.10)))
    assert decisions == [heard] and flags == []
    assert alignment.cue_word_indices[2] == [2, 3]

    # The same copy approved without audio evidence: timing cannot tell a repetition from a
    # re-decode, so the source is kept, the hold is reported and the words time the twin's cue.
    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [approved], _come_words((10.00, 10.10)))
    assert [(item.verdict, item.final_text) for item in decisions] == [("keep_srt", "")]
    assert "no audio evidence" in decisions[0].reason and "Proposed use_audio: 'Vem'" in decisions[0].reason
    assert [(flag.kind, flag.severity, flag.cue_ids, flag.new_text, flag.confidence) for flag in flags] == [
        ("low_confidence_adjudication", "warning", [], "Vem", 0.95),
    ]
    assert (flags[0].start, flags[0].end) == (10.00, 10.10)
    assert alignment.cue_word_indices[2] == [1, 2, 3]

    # Written over its twin: one utterance decoded twice, taken back without a review item.
    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [approved], _come_words((10.04, 10.16)))
    assert [(item.verdict, item.final_text) for item in decisions] == [("keep_srt", "")]
    assert "decoded one utterance twice" in decisions[0].reason and flags == []
    assert alignment.cue_word_indices[2] == [1, 2, 3]


def test_hold_of_an_approval_without_audio_evidence_names_no_internal_cue_id():
    # The hold text reaches the customer as the review item's detail. Review items name the
    # delivered cue numbers (the output is renumbered in time order), so the text must not carry
    # the internal id of the twin's cue; and evidence=None also covers answers from before the
    # evidence field, so it must not claim that nobody listened.
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="Vem", asr_word_indices=[1],
                          start=10.00, end=10.10, left_anchor_cue_id=1, right_anchor_cue_id=2)
    approved = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="Vem", confidence=0.95, reason="heard")

    _, decisions, flags = pipeline._absorb_redecoded_insertions(_alignment(span), [approved], _come_words((10.00, 10.10)))

    assert decisions[0].verdict == "keep_srt" and [flag.kind for flag in flags] == ["low_confidence_adjudication"]
    for text in (flags[0].message, decisions[0].reason):
        assert "no audio evidence" in text
        assert re.search(r"\bcue \d+", text) is None
        assert "without hearing" not in text


def test_touching_copy_the_adjudicator_did_not_approve_still_times_the_cue_of_its_twin():
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="Vem", asr_word_indices=[1],
                          start=10.00, end=10.10, left_anchor_cue_id=1, right_anchor_cue_id=2)
    kept = AdjudicationDecision(case_id="case-1", verdict="keep_srt", final_text="", confidence=0.95, reason="one utterance")

    for decisions in ([kept], []):
        alignment, returned, flags = pipeline._absorb_redecoded_insertions(_alignment(span), decisions, _come_words((10.00, 10.10)))
        assert returned == decisions and flags == []
        assert alignment.cue_word_indices[2] == [1, 2, 3]


@pytest.mark.parametrize(("spoken", "heard"), [
    # ep11 MAI 1691.76 (minus 1682 s): "eu, | eu, eu não sou". The case holds the
    # copy that touches the cue's "eu," and a later repetition.
    ([("É,", 9.48, 9.68), ("eu,", 9.76, 9.94), ("eu,", 10.00, 10.199), ("eu", 10.48, 10.56),
      ("não", 10.60, 10.699), ("sou", 10.72, 10.819)], "eu, eu"),
    # 2A MAI 104.6 (minus 95 s): a further "は" of a laugh and the "、" after it are one case.
    ([("と", 8.44, 8.519), ("は", 9.615, 9.755), ("は", 9.84, 9.919), ("、", 10.00, 10.079),
      ("命", 10.52, 10.599), ("知", 10.88, 10.959)], "は"),
])
def test_unapproved_case_that_begins_with_a_touching_copy_still_times_the_cue_of_its_twin(spoken, heard):
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in spoken]
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text=f"{spoken[2][0]} {spoken[3][0]}",
                          asr_word_indices=[2, 3], start=spoken[2][1], end=spoken[3][2],
                          left_anchor_cue_id=2, right_anchor_cue_id=3)
    aligned = AlignmentResult(divergence_spans=[span], cue_word_indices={1: [0], 2: [1], 3: [4, 5]})
    kept = AdjudicationDecision(case_id="case-1", verdict="keep_srt", final_text="", confidence=0.0, reason="held")
    approved = AdjudicationDecision.model_validate(_heard("case-1", heard))

    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(aligned, [kept], words)
    assert decisions == [kept] and flags == []
    assert alignment.cue_word_indices == {1: [0], 2: [1, 2], 3: [4, 5]}

    # Approved, the words of the case belong to the inserted wording.
    alignment, decisions, flags = pipeline._absorb_redecoded_insertions(aligned, [approved], words)
    assert decisions == [approved] and flags == []
    assert alignment.cue_word_indices == aligned.cue_word_indices


# ep11 cue 680 (times minus 1890 s). MAI and Scribe both heard "por que que ela"
# and the human reference keeps it; the script has one "que".
_QUE_SRT = (
    "1\n00:00:03,900 --> 00:00:04,880\nEu só não entendo\n\n"
    "2\n00:00:04,880 --> 00:00:06,920\npor que ela não veio passar a virada com você.\n"
)
_QUE = (5.039, 5.119)
_MAI_QUE_COPY = (5.160, 5.220)


def _que_words(copy: tuple[float, float], twin: tuple[float, float] = _QUE) -> list[tuple]:
    return [
        ("Eu", 3.940, 4.020), ("só", 4.100, 4.300), ("não", 4.400, 4.499), ("entendo", 4.560, 4.840),
        ("por", 4.920, 5.019), ("que", *twin), ("que", *copy), ("ela", 5.240, 5.359), ("não", 5.380, 5.460),
        ("veio", 5.500, 5.640), ("passar", 5.720, 5.900), ("a", 5.940, 5.960), ("virada", 6.039, 6.319),
        ("com", 6.360, 6.480), ("você.", 6.520, 6.835),
    ]


def _align(srt: str, words: list[tuple]) -> AlignmentResult:
    return align_cues_to_words(parse_srt_text(srt), [
        Word(text=text, start=start, end=end, confidence=None) for text, start, end in words
    ])


@pytest.mark.parametrize("copy", [
    pytest.param(_MAI_QUE_COPY, id="ep11-mai-41ms-after"),
    pytest.param((5.179, 5.230), id="60ms-after"),
    pytest.param((5.139, 5.199), id="ep11-scribe-20ms-after"),
    pytest.param((5.119, 5.199), id="starts-where-the-twin-ends"),
])
def test_word_spoken_twice_in_a_row_is_a_case_for_the_adjudicator(copy):
    result = _align(_QUE_SRT, _que_words(copy))

    span, = result.divergence_spans
    assert (span.cue_ids, span.srt_text, span.asr_text) == ([], "", "que")
    owned = [index for indices in result.cue_word_indices.values() for index in indices]
    assert sorted([*owned, *span.asr_word_indices]) == list(range(15))


@pytest.mark.parametrize(("copy", "twin"), [
    pytest.param((5.079, 5.159), _QUE, id="copy-over-its-twin"),
    pytest.param((5.139, 5.159), _QUE, id="copy-without-duration"),
    pytest.param((5.100, 5.220), (5.039, 5.059), id="twin-without-duration"),
])
def test_word_decoded_twice_at_one_time_joins_the_cue_of_its_twin(copy, twin):
    result = _align(_QUE_SRT, _que_words(copy, twin))

    assert result.divergence_spans == []
    assert result.cue_word_indices == {1: [0, 1, 2, 3], 2: list(range(4, 15))}


_GERMAN_SRT = (
    "1\n00:00:09,000 --> 00:00:10,000\nWir müssen los.\n\n"
    "2\n00:00:10,000 --> 00:00:11,000\n{line}\n\n"
    "3\n00:00:11,200 --> 00:00:12,500\nKomm jetzt mit.\n"
)
_GERMAN_BEFORE = [("Wir", 9.0, 9.2), ("müssen", 9.22, 9.5), ("los.", 9.52, 9.8)]
_GERMAN_AFTER = [("Komm", 11.2, 11.4), ("jetzt", 11.42, 11.7), ("mit.", 11.72, 12.0)]


@pytest.mark.parametrize(("line", "spoken", "repeated"), [
    ("Schnell!", [("Schnell,", 10.00, 10.30), ("schnell!", 10.34, 10.70)], ["schnell"]),
    ("Nein.", [("Nein,", 10.00, 10.25), ("nein,", 10.28, 10.52), ("nein.", 10.55, 10.85)], ["nein", "nein"]),
    # ep11 MAI 1691.76 has "eu, eu, eu não sou" with these times against a script with one "Eu...".
    ("Ich...", [("ich,", 10.00, 10.18), ("ich,", 10.24, 10.44), ("ich", 10.72, 10.80)], ["ich", "ich"]),
])
def test_every_further_copy_of_a_repeated_word_is_reviewable(line, spoken, repeated):
    result = _align(_GERMAN_SRT.format(line=line), [*_GERMAN_BEFORE, *spoken, *_GERMAN_AFTER])

    assert {(tuple(span.cue_ids), span.srt_text) for span in result.divergence_spans} == {((), "")}
    assert [
        normalize_token(word) for span in result.divergence_spans for word in span.asr_text.split()
    ] == repeated
    assert len(result.cue_word_indices[2]) == 1


@pytest.mark.parametrize(("line", "spoken"), [
    # ep17 727.4 and 2428.8 (effective times): MAI rewound and wrote the word again over itself.
    ("Flora.", [("Flora.", 10.385, 10.885), ("Flora.", 10.440, 10.885)]),
    ("Wann kommst du?", [("wann", 10.800, 10.999), ("wann", 10.880, 11.079), ("kommst", 11.10, 11.14), ("du?", 11.15, 11.18)]),
    ("Schnell!", [("Schnell!", 10.00, 10.40), ("Schnell!", 10.20, 10.60)]),
])
def test_copy_written_over_its_twin_is_not_reviewed(line, spoken):
    result = _align(_GERMAN_SRT.format(line=line), [*_GERMAN_BEFORE, *spoken, *_GERMAN_AFTER])

    assert result.divergence_spans == []
    assert result.cue_word_indices[2] == list(range(3, 3 + len(spoken)))


def _word_owners(workdir, word_index: int) -> list[str]:
    alignment = json.loads((workdir / "rebuild.json").read_text(encoding="utf-8"))["alignment"]
    return [cue_id for cue_id, indices in alignment["cue_word_indices"].items() if word_index in indices]


def test_repeat_the_model_heard_clearly_is_delivered_and_owned_by_one_cue(tmp_path):
    cues, flags, workdir = _sync_episode(
        tmp_path, _QUE_SRT, _que_words(_MAI_QUE_COPY), {"case-1": _heard("case-1", "que")},
    )

    assert [cue.plain_text for cue in cues] == [
        "Eu só não entendo", "por que que ela não veio passar a virada com você.",
    ]
    assert "text_changed" in [flag["kind"] for flag in flags]
    assert _word_owners(workdir, 5) == ["2"] and _word_owners(workdir, 6) == ["2"]


def test_repeat_without_an_answer_keeps_the_source_wording_and_is_flagged(tmp_path):
    cues, flags, workdir = _sync_episode(tmp_path, _QUE_SRT, _que_words(_MAI_QUE_COPY), {})

    assert [cue.plain_text for cue in cues] == [
        "Eu só não entendo", "por que ela não veio passar a virada com você.",
    ]
    assert "invalid_llm_response" in [flag["kind"] for flag in flags if flag["severity"] == "error"]
    assert _word_owners(workdir, 5) == ["2"] and _word_owners(workdir, 6) == ["2"]


_COME_SRT = (
    "1\n00:00:01,000 --> 00:00:01,700\nRápido, rápido.\n\n"
    "2\n00:00:01,700 --> 00:00:02,500\nVem cá.\n\n"
    "3\n00:00:03,500 --> 00:00:04,500\nMais perto.\n"
)
# ep11 MAI 1301.12 (minus 1300 s, moved behind cue 1): the second phrase starts
# 21 ms after the first and the human reference keeps "Vem cá, vem cá."
_COME_WORDS = [
    ("Rápido,", 1.000, 1.259), ("rápido.", 1.280, 1.499), ("Vem", 1.520, 1.619), ("cá,", 1.640, 1.739),
    ("vem", 1.760, 1.820), ("cá.", 1.880, 1.979), ("Mais", 3.600, 3.740), ("perto.", 3.800, 4.080),
]


def test_phrase_the_adjudicator_heard_clearly_right_after_its_twin_is_delivered(tmp_path):
    cues, flags, workdir = _sync_episode(tmp_path, _COME_SRT, _COME_WORDS, {"case-1": _heard("case-1", "Vem cá,")})

    assert " ".join(cue.plain_text for cue in cues).casefold().count("vem cá") == 2
    assert {"adlib_inserted", "text_changed"} & {flag["kind"] for flag in flags}
    assert all(len(_word_owners(workdir, index)) == 1 for index in range(2, 6))


def _hold_flags(flags: list[dict]) -> list[dict]:
    return [flag for flag in flags
            if flag["kind"] == "low_confidence_adjudication" and "no audio evidence" in flag["message"]]


def test_phrase_approved_without_audio_evidence_right_after_its_twin_is_held_and_reported(tmp_path):
    # A text-only route (or an answer from before the evidence field) approved the repetition
    # without hearing it: the script is kept, the customer is asked to listen, both copies time cue 2.
    cues, flags, workdir = _sync_episode(tmp_path, _COME_SRT, _COME_WORDS, {"case-1": _decide("case-1", "Vem cá,")})

    assert [cue.plain_text for cue in cues] == ["Rápido, rápido.", "Vem cá.", "Mais perto."]
    assert not {"adlib_inserted", "text_changed"} & {flag["kind"] for flag in flags}
    assert [(flag["severity"], flag["new_text"]) for flag in _hold_flags(flags)] == [("warning", "Vem cá,")]
    assert all(_word_owners(workdir, index) == ["2"] for index in range(2, 6))
    assert cues[1].start_ms <= 1520 and cues[1].end_ms >= 1979


def test_golden_rewind_copy_approved_by_a_text_only_answer_is_shown_once_and_reported(tmp_path):
    # ep11 MAI 321.2 s: 'manda umas flores.' then 'Manda umas flores' 20 ms later; the hybrid-v8
    # text-only stage approved the copy (evidence None); the human reference has it once.
    cues, flags = _sync(tmp_path, _FLOWERS_SRT, _flowers_words(0.02), _COPY_CASE)

    assert [cue.plain_text for cue in cues] == [
        "Se quer ficar bem com ela,", "manda umas flores", "pra acalmar a garota.",
    ]
    assert abs(cues[1].start_ms - 21200) <= 34 and abs(cues[1].end_ms - round(22.52 * 1000)) <= 45
    assert not {"adlib_inserted", "text_changed", "timing_outlier_trimmed"} & {flag["kind"] for flag in flags}
    assert [flag["new_text"] for flag in _hold_flags(flags)] == ["umas flores. Manda"]
