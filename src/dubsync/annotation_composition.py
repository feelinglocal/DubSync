from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from .models import Cue
from .subtitle_annotations import is_bracketed_screen_text_cue


@dataclass(frozen=True)
class AnnotationComposition:
    cues: list[Cue]
    cue_word_indices: dict[int, list[int]]
    cue_annotations: dict[int, list[int]]
    tracks: dict[int, dict[str, object]]

    def artifact(self) -> dict[str, object]:
        """JSON-ready continuous-caption provenance for final display QC."""
        return deepcopy({"policy_version": 1, "cue_annotations": self.cue_annotations, "tracks": self.tracks})


def _overlaps(left: Cue, right: Cue) -> bool:
    return left.start_ms < right.end_ms and right.start_ms < left.end_ms


def compose_bracketed_annotations(
    cues: list[Cue], cue_word_indices: Mapping[int, list[int]] | None = None,
) -> AnnotationComposition:
    """Compose screen captions around exact incoming speech intervals.

    Only pure bracketed text without owned words is a caption track. Lyrics
    and mixed dialogue stay timed speech. Caption text may repeat continuously
    across adjoining displays; spoken text, IDs and ownership never repeat or
    move. Short caption fragments must be evaluated using the returned tracks.
    Reapplying to composed cues is a no-op for their content and ownership.
    """
    ownership = {index: list(words) for index, words in (cue_word_indices or {}).items()}
    annotations = sorted((cue for cue in cues if cue.start_ms < cue.end_ms
                          and not any(mark in cue.text for mark in "♪♫")
                          and is_bracketed_screen_text_cue(cue)
                          and not ownership.get(cue.index)), key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    annotation_ids = {cue.index for cue in annotations}
    speech = [cue for cue in cues if cue.index not in annotation_ids]
    affected = {cue.index for cue in annotations if any(_overlaps(cue, spoken) for spoken in speech)}
    # Include intersecting caption neighbors so residual caption pieces also
    # compose cleanly; isolated authored captions remain the original objects.
    while True:
        expanded = affected | {cue.index for cue in annotations
                               if any(other.index in affected and _overlaps(cue, other) for other in annotations)}
        if expanded == affected:
            break
        affected = expanded
    if not affected:
        return AnnotationComposition(list(cues), ownership, {}, {})
    tracks = [cue for cue in annotations if cue.index in affected]
    result: list[Cue] = []
    cue_annotations: dict[int, list[int]] = {}
    for cue in cues:
        if cue.index in affected:
            continue
        added = [track for track in tracks if cue.index not in annotation_ids and _overlaps(track, cue)]
        result.append(cue.with_lines(cue.lines + [line for track in added for line in track.lines]) if added else cue)
        if added:
            cue_annotations[cue.index] = [track.index for track in added]

    # Outside spoken intervals, sweep caption endpoints and speech endpoints
    # into nonoverlapping prefix/gap/suffix pieces, combining active tracks.
    boundaries = sorted({time for cue in tracks + speech for time in (cue.start_ms, cue.end_ms)})
    residuals: list[tuple[int, int, list[Cue]]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [track for track in tracks if track.start_ms <= start and track.end_ms >= end]
        if not active or any(cue.start_ms < end and cue.end_ms > start for cue in speech):
            continue
        if residuals and residuals[-1][1] == start and residuals[-1][2] == active:
            residuals[-1] = (residuals[-1][0], end, active)
        else:
            residuals.append((start, end, active))
    used = {cue.index for cue in result}
    next_id = max([0, *ownership, *(cue.index for cue in cues)]) + 1
    for track in tracks:
        ownership.pop(track.index, None)  # Eligible tracks have no owned words.
    for start, end, active in residuals:
        index = next((track.index for track in active if track.index not in used), None)
        if index is None:
            index, next_id = next_id, next_id + 1
        used.add(index)
        result.append(active[0].model_copy(update={
            "index": index, "start_ms": start, "end_ms": end,
            "lines": [line for track in active for line in track.lines],
        }))
        ownership[index] = []
        cue_annotations[index] = [track.index for track in active]
    result.sort(key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    metadata: dict[int, dict[str, object]] = {}
    for track in tracks:
        displayed = [cue for cue in result if track.index in cue_annotations.get(cue.index, ())]
        intervals: list[list[int]] = []
        for cue in displayed:
            if intervals and cue.start_ms <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], cue.end_ms)
            else:
                intervals.append([cue.start_ms, cue.end_ms])
        metadata[track.index] = {
            "original_start_ms": track.start_ms, "original_end_ms": track.end_ms, "lines": list(track.lines),
            "display_cue_ids": [cue.index for cue in displayed], "display_intervals": intervals,
            "early_extension_ms": track.start_ms - intervals[0][0],
            "late_extension_ms": intervals[-1][1] - track.end_ms,
        }
    return AnnotationComposition(result, ownership, cue_annotations, metadata)
