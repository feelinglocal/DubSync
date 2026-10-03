import pytest

from dubsync.adjudication import confidence_gated_decision
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.qc_review import build_review


@pytest.mark.parametrize("evidence,heard", [("heard_unclear", "maybe"), ("not_audible", "")])
def test_uncertain_audio_is_held_even_when_numeric_gate_is_disabled(evidence, heard):
    span = DivergenceSpan(case_id="case", cue_ids=[1], srt_text="original", asr_text="maybe")
    decision = AdjudicationDecision(case_id="case", verdict="use_audio", final_text=heard,
        confidence=1, reason="uncertain", evidence=evidence, heard_text=heard)
    selected, flag = confidence_gated_decision(span, decision, 0)
    assert selected.verdict == "keep_srt" and selected.final_text == "original"
    assert flag is not None and flag.kind == "low_confidence_adjudication"


_SOURCE = [Cue(index=2, start_ms=26_500, end_ms=27_365, lines=["いいかしら？"]),
           Cue(index=3, start_ms=32_300, end_ms=32_630, lines=["あなたは…"])]


def _insertion_answer(source, asr, evidence, heard, verdict="keep_srt"):
    # Delivered 1B-mai case-47: '。' between two cues, answered "no spoken dialogue exists".
    span = DivergenceSpan(case_id="case-47", cue_ids=[3] if source else [], srt_text=source, asr_text=asr,
                          start=27.44, end=27.52)
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict=verdict, final_text=source if verdict == "keep_srt" else "",
        confidence=0, reason="ASR proposes punctuation where no spoken dialogue exists.",
        evidence=evidence, heard_text=heard,
    )
    return span, decision


@pytest.mark.parametrize("verdict", ["keep_srt", "use_audio"])
@pytest.mark.parametrize("evidence", ["not_audible", "heard_unclear"])
@pytest.mark.parametrize("asr", ["。", "、", "?"])
def test_agreed_absence_on_a_punctuation_only_insertion_is_not_a_hold(asr, evidence, verdict):
    span, decision = _insertion_answer("", asr, evidence, "", verdict)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert flag is None
    assert (selected.verdict, selected.final_text) == ("keep_srt", "")


@pytest.mark.parametrize("source,asr,evidence,heard", [
    ("あなたは", "。", "not_audible", ""),  # source words are not confirmed absent
    ("", "Oh.", "not_audible", ""),  # the ASR heard a word the model could not recover
    ("", "う", "heard_unclear", ""),
    ("", "。", "heard_unclear", "あ"),  # the model heard something
])
def test_uncertain_hearing_with_a_spoken_side_stays_customer_review(source, asr, evidence, heard):
    span, decision = _insertion_answer(source, asr, evidence, heard)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert (selected.verdict, selected.final_text) == ("keep_srt", source)
    assert flag is not None and flag.kind == "low_confidence_adjudication"
    review = build_review([flag], [], _SOURCE, source_cues=_SOURCE)
    assert [item.raw_flags for item in review.review] == [[0]]


_2B_CUE = [Cue(index=79, start_ms=109_100, end_ms=110_900, lines=["大井周治 いい度胸だな！"])]


def _2b_scribe_case_36(final_text, *, srt_text="大井周治", asr_text="おい秀二"):
    # Delivered 2B-scribe case-36: the MAI run of the same audio hears 大井周治 clearly.
    span = DivergenceSpan(case_id="case-36", cue_ids=[79], srt_text=srt_text, asr_text=asr_text,
                          start=109.16, end=109.9, srt_token_indices=[547, 548, 549, 550],
                          asr_word_indices=[582, 583, 585, 586])
    decision = AdjudicationDecision(
        case_id="case-36", verdict="hybrid", final_text=final_text, confidence=1.0,
        reason="Speaker says interjection 'おい' followed by the character's name '周治'.",
        evidence="heard_clearly", heard_text=final_text,
    )
    return span, decision


def test_a_clear_hearing_that_respells_source_kanji_in_kana_is_held_for_review():
    # W3R-3 / F10: 大井 and おい sound alike; the hearing cannot tell the customer's name from the
    # interjection, so the confident rewrite 大井周治 -> おい周治 is held, not applied silently.
    span, decision = _2b_scribe_case_36("おい周治")

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert (selected.verdict, selected.final_text) == ("keep_srt", "大井周治")
    assert flag is not None and flag.kind == "low_confidence_adjudication"
    assert (flag.old_text, flag.new_text, flag.cue_ids) == ("大井周治", "おい周治", [79])
    review = build_review([flag], [], _2B_CUE, source_cues=_2B_CUE)
    assert [item.raw_flags for item in review.review] == [[0]]


@pytest.mark.parametrize("srt_text,final_text", [
    ("山下様", "山下さん"),  # one kanji: a real honorific change (delivered 1B 様 -> さん)
    ("いい度胸", "いい根性"),  # kanji to kanji stays a model decision
    ("そうだね", "そうだよ"),  # kana only
    ("大井周治", "大井周治さん"),  # an addition keeps every kanji
])
def test_other_confident_rewrites_still_apply(srt_text, final_text):
    span, decision = _2b_scribe_case_36(final_text, srt_text=srt_text, asr_text=final_text)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert flag is None
    assert selected.final_text == final_text
