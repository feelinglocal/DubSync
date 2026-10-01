"""A detector's 10 ms grid must not strand a connector before its continuation."""
import pytest

from dubsync.changes import _anchored_prefix_replacement_target
from dubsync.models import Cue, DivergenceSpan


def _case(**updates):
    cues = {
        214: Cue(index=214, start_ms=853000, end_ms=855000, lines=["Vamos falar disso depois."]),
        215: Cue(index=215, start_ms=856000, end_ms=856300, lines=["Agora"]),
        216: Cue(index=216, start_ms=856400, end_ms=857500, lines=["você sabe disso."]),
    }
    span = DivergenceSpan(
        case_id="connector", cue_ids=[214, 215], srt_text="depois. Agora", asr_text="E",
        srt_token_indices=[3, 4], asr_word_indices=[10], start=856.36, end=856.42,
        left_anchor_cue_id=214, left_anchor_end=854.8,
        right_anchor_cue_id=216, right_anchor_start=856.415,
        speaker_ids=["a"], right_anchor_speaker_id="a",
    ).model_copy(update=updates)
    return cues, {214: (3, 4), 215: (0, 1)}, span


@pytest.mark.parametrize("end", [856.415, 856.42, 856.425])
def test_connector_shares_continuation_despite_one_hop_edge_overlap(end):
    cues, bounds, span = _case(end=end)
    assert _anchored_prefix_replacement_target(cues, bounds, span, "E") == 216
    # This is ownership only: neither acoustic edge may be rewritten.
    assert span.start == 856.36 and span.end == end


@pytest.mark.parametrize("updates", [
    {"end": 856.436}, {"start": 856.416},
    {"right_anchor_speaker_id": "b"}, {"speaker_ids": ["a", "b"]},
    {"left_anchor_end": 856.3}, {"right_anchor_start": 857.0},
])
def test_overlap_tolerance_does_not_remove_evidence_guards(updates):
    cues, bounds, span = _case(**updates)
    assert _anchored_prefix_replacement_target(cues, bounds, span, "E") is None
