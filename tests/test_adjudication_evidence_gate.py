import pytest

from dubsync.adjudication import confidence_gated_decision
from dubsync.models import AdjudicationDecision, DivergenceSpan


@pytest.mark.parametrize("evidence,heard", [("heard_unclear", "maybe"), ("not_audible", "")])
def test_uncertain_audio_is_held_even_when_numeric_gate_is_disabled(evidence, heard):
    span = DivergenceSpan(case_id="case", cue_ids=[1], srt_text="original", asr_text="maybe")
    decision = AdjudicationDecision(case_id="case", verdict="use_audio", final_text=heard,
        confidence=1, reason="uncertain", evidence=evidence, heard_text=heard)
    selected, flag = confidence_gated_decision(span, decision, 0)
    assert selected.verdict == "keep_srt" and selected.final_text == "original"
    assert flag is not None and flag.kind == "low_confidence_adjudication"
