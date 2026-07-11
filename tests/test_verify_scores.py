from __future__ import annotations

from dubsync.models import AlignmentResult, Cue, ForcedAlignmentCue, Word
from dubsync.verify import cps_sanity_flags, score_cues


def test_score_cues_uses_average_asr_confidence_and_forced_alignment_override():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1500, lines=["hello there"]),
        Cue(index=2, start_ms=2000, end_ms=2500, lines=["quiet"]),
    ]
    words = [
        Word(text="hello", start=1.0, end=1.2, confidence=0.8),
        Word(text="there", start=1.2, end=1.5, confidence=1.0),
        Word(text="quiet", start=2.0, end=2.3, confidence=0.4),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0, 1], 2: [2]})
    forced = [ForcedAlignmentCue(cue_id=2, start=2.0, end=2.4, score=0.95)]

    scores = score_cues(cues, words, alignment, forced)

    assert scores[0].cue_id == 1
    assert scores[0].score == 0.9
    assert scores[0].source == "asr_confidence"
    assert scores[1].cue_id == 2
    assert scores[1].score == 0.95
    assert scores[1].source == "forced_alignment"


def test_cps_sanity_flags_fire_for_fast_and_slow_cues():
    borderline = Cue(index=1, start_ms=0, end_ms=1000, lines=["x" * 30])
    impossible = Cue(index=2, start_ms=2000, end_ms=2500, lines=["x" * 40])
    slow = Cue(index=3, start_ms=3000, end_ms=6000, lines=["hey"])

    flags = cps_sanity_flags([borderline, impossible, slow], max_cps=30, min_cps=2)

    assert [flag.kind for flag in flags] == ["impossible_cps_fast", "impossible_cps_slow"]
    assert flags[0].cue_ids == [2]
    assert flags[1].cue_ids == [3]
