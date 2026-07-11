from __future__ import annotations

from .models import Cue, QCFlag


def apply_overlap_policy(cues: list[Cue], policy: str = "stack") -> tuple[list[Cue], list[QCFlag]]:
    if policy == "stack":
        return cues, _overlap_flags(cues, "overlap_stacked")
    if policy == "flag_only":
        return cues, _overlap_flags(cues, "overlap_flag_only")
    if policy != "dash":
        raise ValueError(f"unsupported overlap policy: {policy}")

    merged: list[Cue] = []
    flags: list[QCFlag] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        if index + 1 < len(cues):
            nxt = cues[index + 1]
            if _can_dash_merge(current, nxt):
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
    previous: Cue | None = None
    for cue in cues:
        if previous is not None and cue.start_ms < previous.end_ms:
            flags.append(
                QCFlag(
                    kind=kind,
                    cue_ids=[previous.index, cue.index],
                    message="Overlapping speaker cues require QC review.",
                    old_text=f"{previous.text}\n{cue.text}",
                    start=_overlap_start(previous, cue),
                    end=_overlap_end(previous, cue),
                )
            )
        previous = cue
    return flags


def _can_dash_merge(left: Cue, right: Cue) -> bool:
    return (
        left.end_ms > right.start_ms
        and left.speaker_id is not None
        and right.speaker_id is not None
        and left.speaker_id != right.speaker_id
    )


def _overlap_start(left: Cue, right: Cue) -> float:
    return max(left.start_ms, right.start_ms) / 1000.0


def _overlap_end(left: Cue, right: Cue) -> float:
    return min(left.end_ms, right.end_ms) / 1000.0
