from __future__ import annotations

from .models import Cue, QCFlag


def sort_cues_chronologically(cues: list[Cue]) -> tuple[list[Cue], list[QCFlag]]:
    indexed = list(enumerate(cues))
    sorted_indexed = sorted(indexed, key=lambda item: (item[1].start_ms, item[1].index))
    sorted_cues = [cue for _, cue in sorted_indexed]
    if [cue.index for cue in sorted_cues] == [cue.index for cue in cues]:
        return cues, []

    original_positions = {cue.index: position for position, cue in enumerate(cues)}
    moved_ids = [
        cue.index
        for position, cue in enumerate(sorted_cues)
        if original_positions.get(cue.index) != position
    ]
    moved_cues = [cue for cue in sorted_cues if cue.index in set(moved_ids)]
    return sorted_cues, [
        QCFlag(
            kind="source_out_of_order",
            cue_ids=moved_ids,
            message="Source cues were not chronological; cues were sorted by start time before alignment.",
            severity="warning",
            old_text="\n\n".join(cue.text for cue in moved_cues),
            start=moved_cues[0].start_ms / 1000.0 if moved_cues else None,
            end=moved_cues[-1].end_ms / 1000.0 if moved_cues else None,
        )
    ]
