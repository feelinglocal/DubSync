from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.style_profile import StyleProfile
import pytest


@pytest.mark.parametrize("source,removed,expected", [
    ("- Vamos tirar. - Vem cá.", "Vamos tirar", "Vem cá."),
    ("- Estou muito bonito. - Não consegui pegar ele.", "Estou muito bonito", "Não consegui pegar ele."),
    ("– Vamos tirar.\n– Vem cá.", "Vamos tirar", "Vem cá."),
])
def test_confirmed_turn_deletion_removes_its_empty_dialogue_marker(source, removed, expected):
    cue = Cue(index=1, start_ms=1000, end_ms=2500, lines=source.splitlines())
    span = DivergenceSpan(case_id="one", cue_ids=[1], srt_text=removed, asr_text="",
                          srt_token_indices=list(range(len(removed.split()))))
    decision = AdjudicationDecision(case_id="one", verdict="use_audio", final_text="", confidence=.95,
                                    reason="the complete first turn is absent here")
    changed, flags = apply_adjudication_decisions([cue], [span], [decision], StyleProfile())
    assert changed[0].plain_text == expected
    assert flags[0].new_text == expected
    assert cue.text == source


def test_unedited_dialogue_markers_are_preserved():
    cue = Cue(index=1, start_ms=1000, end_ms=2500, lines=["- Yes. - No."])
    changed, _ = apply_adjudication_decisions([cue], [], [], StyleProfile())
    assert changed == [cue]


def test_deleted_opening_reaction_does_not_leave_its_comma_at_cue_start():
    cue = Cue(index=1, start_ms=1000, end_ms=2500, lines=["Ah, meu Deus."])
    span = DivergenceSpan(case_id="one", cue_ids=[1], srt_text="Ah", asr_text="",
                          srt_token_indices=[0])
    decision = AdjudicationDecision(case_id="one", verdict="use_audio", final_text="", confidence=.95,
                                    reason="The opening reaction belongs to the previous actor's cue.")
    changed, flags = apply_adjudication_decisions([cue], [span], [decision], StyleProfile())
    assert changed[0].plain_text == "meu Deus."
    assert flags[0].new_text == "meu Deus."
    assert cue.text == "Ah, meu Deus."
