from __future__ import annotations

from dubsync.models import Cue
from dubsync.output_order import finalize_cues_for_output
from dubsync.style_profile import StyleProfile


def test_finalize_cues_for_output_reports_unchanged_text_source_order_inversion():
    cues_in_source_narrative_order = [
        Cue(
            index=28,
            start_ms=2000,
            end_ms=2600,
            lines=["Sie haben vor Kurzem Ihre Verlobung"],
        ),
        Cue(
            index=29,
            start_ms=1000,
            end_ms=1600,
            lines=["mit Frau Morris bekannt gegeben."],
        ),
    ]

    _, flags = finalize_cues_for_output(
        cues_in_source_narrative_order,
        StyleProfile(fps=30.0),
        no_overlaps=True,
    )

    inversion_flags = [flag for flag in flags if flag.kind == "output_order_inversion"]
    assert inversion_flags
    assert inversion_flags[0].severity == "error"
    assert set(inversion_flags[0].cue_ids) == {28, 29}


def test_finalize_output_cannot_merge_or_shift_a_missing_audio_cue():
    missing = Cue(index=2, start_ms=1000, end_ms=2000, lines=["same line"])
    cues = [
        Cue(index=1, start_ms=900, end_ms=1500, lines=["same line"]),
        missing,
        Cue(index=3, start_ms=1900, end_ms=2400, lines=["next line"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues,
        StyleProfile(fps=30.0),
        no_overlaps=True,
        protected_cue_ids={2},
    )

    protected = next(cue for cue in finalized if cue.index == 2)
    assert protected == missing
    assert {cue.index for cue in finalized} == {1, 2, 3}
    assert not any(2 in flag.cue_ids and flag.kind == "duplicate_cue_merged" for flag in flags)
