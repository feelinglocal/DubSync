"""Words an ASR provider decoded twice never become inserted dialogue."""
from __future__ import annotations

import json

import yaml

from dubsync import pipeline
from dubsync.models import AdjudicationDecision, AlignmentResult, DivergenceSpan, Word
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


_FLOWERS_SRT = (
    "1\n00:00:19,900 --> 00:00:21,030\nSe quer ficar bem com ela,\n\n"
    "2\n00:00:21,200 --> 00:00:22,630\nmanda umas flores\n\n"
    "3\n00:00:22,830 --> 00:00:23,900\npra acalmar a garota.\n"
)


def _flowers_words(copy_gap: float) -> list[tuple]:
    # MAI ep11: 'manda umas flores.' (321.20-321.66) directly followed by
    # 'Manda umas flores' (321.68-322.48): one utterance decoded twice.
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


def test_phrase_decoded_twice_is_shown_once_and_reported_nowhere(tmp_path):
    cues, flags = _sync(tmp_path, _FLOWERS_SRT, _flowers_words(0.02), _COPY_CASE)

    assert [cue.plain_text for cue in cues] == [
        "Se quer ficar bem com ela,", "manda umas flores", "pra acalmar a garota.",
    ]
    # Both copies are the cue's speech: it starts with the first and ends with the second.
    assert abs(cues[1].start_ms - 21200) <= 34 and abs(cues[1].end_ms - 22520) <= 45
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
    # is the single word before its twin.
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="a", asr_word_indices=[1],
                          start=10.60, end=10.62, left_anchor_cue_id=1, right_anchor_cue_id=2)
    decision = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="a", confidence=0.95, reason="heard")

    alignment, decisions = pipeline._absorb_redecoded_insertions(_alignment(span), [decision], _words(10.60))

    assert [(item.verdict, item.final_text) for item in decisions] == [("keep_srt", "")]
    assert alignment.cue_word_indices[2] == [1, 2, 3]
    assert alignment.flags == []


def test_insertion_with_other_wording_or_far_from_its_twin_is_left_alone():
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="a", asr_word_indices=[1],
                          start=10.60, end=10.62, left_anchor_cue_id=1, right_anchor_cue_id=2)
    changed = AdjudicationDecision(case_id="case-1", verdict="hybrid", final_text="ah, a", confidence=0.95, reason="heard")
    same = changed.model_copy(update={"final_text": "a"})

    for decision, words in ((changed, _words(10.60)), (same, _words(10.20))):
        alignment, decisions = pipeline._absorb_redecoded_insertions(_alignment(span), [decision], words)
        assert decisions == [decision]
        assert alignment.cue_word_indices[2] == [2, 3]
