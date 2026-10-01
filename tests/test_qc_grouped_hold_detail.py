from dubsync.models import Cue, QCFlag
from dubsync.qc_review import build_review


def test_grouped_overlap_does_not_describe_the_spoken_neighbor_as_missing_audio():
    spoken = Cue(index=12, start_ms=42767, end_ms=43700, lines=["お嬢さんよ"])
    held = Cue(index=11, start_ms=42800, end_ms=43130, lines=["ははは"])
    flags = [
        QCFlag(kind="missing_audio_timing_held", severity="error", cue_ids=[11],
               message="No matching speech; source timing was kept."),
        QCFlag(kind="output_overlap_unresolved", severity="error", cue_ids=[11, 12],
               message="Held source cue overlaps its spoken neighbor.", start=42.8, end=43.13),
    ]
    review = build_review(flags, [], [spoken, held], source_cues=[held, spoken])
    item, = review.review
    assert item.srt_numbers == [1, 2]
    assert item.cue_ids == [12, 11]
    assert item.raw_flags == [0, 1]
    assert "Cue #2 has no matching speech" in item.detail
    assert "2 cues" not in item.detail
    assert "Cues overlap" in item.detail


def test_grouped_hold_counts_only_missing_cues_in_a_larger_overlap_group():
    cues = [
        Cue(index=10, start_ms=1000, end_ms=2400, lines=["Spoken words."]),
        Cue(index=20, start_ms=2000, end_ms=2200, lines=["Missing one."]),
        Cue(index=30, start_ms=2300, end_ms=2600, lines=["Missing two."]),
    ]
    flags = [
        QCFlag(kind="missing_audio_timing_held", severity="error", cue_ids=[20, 30],
               message="No matching speech; source timing was kept."),
        QCFlag(kind="output_overlap_unresolved", severity="error", cue_ids=[10, 20],
               message="Held source overlaps speech.", start=2.0, end=2.2),
        QCFlag(kind="output_overlap_unresolved", severity="error", cue_ids=[10, 30],
               message="Held source overlaps speech.", start=2.3, end=2.4),
    ]
    item, = build_review(flags, [], cues, source_cues=cues).review
    assert item.srt_numbers == [1, 2, 3]
    assert "2 cues (00:00:02,000–00:00:02,600) have no matching speech" in item.detail
    assert "3 cues" not in item.detail
