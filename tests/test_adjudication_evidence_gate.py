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
