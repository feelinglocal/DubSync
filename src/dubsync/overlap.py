from __future__ import annotations

from .models import Cue, QCFlag
from .speaker_evidence import speakers_known_different
from .subtitle_annotations import is_bracketed_screen_text_cue

_POLICY_OVERLAP_FLAG_KINDS = frozenset({"overlap_stacked", "overlap_flag_only"})
_FINAL_OVERLAP_FLAG_KINDS = frozenset({"output_overlap_unresolved", "output_overlap_preserved"})


def apply_overlap_policy(
    cues: list[Cue],
    policy: str = "stack",
    *,
    protected_cue_ids: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    if policy == "stack":
        return cues, _overlap_flags(cues, "overlap_stacked")
    if policy == "flag_only":
        return cues, _overlap_flags(cues, "overlap_flag_only")
    if policy != "dash":
        raise ValueError(f"unsupported overlap policy: {policy}")

    protected = protected_cue_ids or set()
    merged: list[Cue] = []
    flags: list[QCFlag] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        if index + 1 < len(cues):
            nxt = cues[index + 1]
            if (
                current.index not in protected
                and nxt.index not in protected
                and _can_dash_merge(current, nxt)
            ):
                merged_cue = Cue(
                    index=current.index,
                    start_ms=min(current.start_ms, nxt.start_ms),
                    end_ms=max(current.end_ms, nxt.end_ms),
                    lines=[f"- {current.plain_text}", f"- {nxt.plain_text}"],
                    speaker_id=None,
                )
                merged.append(merged_cue)
                flags.append(
                    QCFlag(
                        kind="overlap_dash_merge",
                        cue_ids=[current.index, nxt.index],
                        message="Overlapping speakers merged into a dashed two-line cue.",
                        old_text=f"{current.text}\n{nxt.text}",
                        new_text=merged_cue.text,
                        start=_overlap_start(current, nxt),
                        end=_overlap_end(current, nxt),
                    )
                )
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged, [*flags, *_overlap_flags(merged, "overlap_flag_only")]


def _overlap_flags(cues: list[Cue], kind: str) -> list[QCFlag]:
    flags: list[QCFlag] = []
    # Compare with the earlier cue that is still on screen, not only with the
    # list neighbour, and leave on-screen text annotations out: a sign shown
    # during dialogue is not two speakers talking over each other.
    latest_ending: Cue | None = None
    for cue in cues:
        if is_bracketed_screen_text_cue(cue):
            continue
        if latest_ending is not None and cue.start_ms < latest_ending.end_ms:
            flags.append(
                QCFlag(
                    kind=kind,
                    cue_ids=[latest_ending.index, cue.index],
                    message="Overlapping speaker cues require QC review.",
                    old_text=f"{latest_ending.text}\n{cue.text}",
                    start=_overlap_start(latest_ending, cue),
                    end=_overlap_end(latest_ending, cue),
                )
            )
        if latest_ending is None or cue.end_ms > latest_ending.end_ms:
            latest_ending = cue
    return flags


def reconcile_overlap_flags(flags: list[QCFlag], final_cues: list[Cue], final_flags: list[QCFlag]) -> list[QCFlag]:
    """Keep one finding per overlap that is really in the exported cues.

    ``overlap_stacked`` / ``overlap_flag_only`` are raised after rebuild. A pair
    that was separated later is stale, and a pair that final ordering reports
    itself would be counted twice; both are dropped here.
    """
    cues_by_id = {cue.index: cue for cue in final_cues}
    reported = {
        frozenset(flag.cue_ids)
        for flag in final_flags
        if flag.kind in _FINAL_OVERLAP_FLAG_KINDS
    }

    def still_needed(flag: QCFlag) -> bool:
        if flag.kind not in _POLICY_OVERLAP_FLAG_KINDS or len(flag.cue_ids) != 2:
            return True
        left, right = (cues_by_id.get(cue_id) for cue_id in flag.cue_ids)
        if left is None or right is None or frozenset(flag.cue_ids) in reported:
            return False
        return left.start_ms < right.end_ms and right.start_ms < left.end_ms

    return [flag for flag in flags if still_needed(flag)]


def _can_dash_merge(left: Cue, right: Cue) -> bool:
    # Only provably different actors talk over each other; labels of unrelated
    # scopes (two MAI chunks) may name the same one.
    return left.end_ms > right.start_ms and speakers_known_different(left.speaker_id, right.speaker_id)


def _overlap_start(left: Cue, right: Cue) -> float:
    return max(left.start_ms, right.start_ms) / 1000.0


def _overlap_end(left: Cue, right: Cue) -> float:
    return min(left.end_ms, right.end_ms) / 1000.0
