from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from .models import Cue, QCFlag, Word
from .semantic_output import _dialogue_turns, compact_lines, paginate_annotation_lines, split_crowded_output_cues
from .srt_io import format_timestamp
from .style_profile import StyleProfile
from .subtitle_annotations import cue_has_bracketed_screen_text, is_bracketed_screen_text_cue


@dataclass(frozen=True)
class AnnotationComposition:
    cues: list[Cue]
    cue_word_indices: dict[int, list[int]]
    cue_annotations: dict[int, list[int]]
    tracks: dict[int, dict[str, object]]
    flags: list[QCFlag] = field(default_factory=list)
    expansions: dict[int, list[int]] = field(default_factory=dict)
    policy_version: int = 1

    def artifact(self) -> dict[str, object]:
        """JSON-ready caption-track and page provenance for final display QC."""
        return deepcopy({"policy_version": self.policy_version, "cue_annotations": self.cue_annotations, "tracks": self.tracks})


def _overlaps(left: Cue, right: Cue) -> bool:
    return left.start_ms < right.end_ms and right.start_ms < left.end_ms


def compose_bracketed_annotations(
    cues: list[Cue], cue_word_indices: Mapping[int, list[int]] | None = None,
    *, words: list[Word] | None = None, profile: StyleProfile | None = None,
    protected_cue_ids: set[int] | None = None, enforce_width: bool = True,
    reserved_cue_ids: set[int] | None = None, media_end_ms: int | None = None,
) -> AnnotationComposition:
    """Compose screen captions around exact incoming speech intervals.

    Only pure bracketed text without owned words is a caption track. Lyrics
    and mixed dialogue stay timed speech. With a style profile, crowded
    captions become ordered pages and reliably owned speech may acquire
    children at exact word boundaries. Caption wording is retained, spoken
    words never repeat, and actual page intervals are recorded. Legacy calls
    retain continuous full-caption composition. With ``enforce_width`` false
    (a source-derived width) only the line count crowds a display. A new
    display cue never takes one of ``reserved_cue_ids`` (removed source cues
    or cues a finding still names). ``media_end_ms`` (the known media
    length) bounds the free time after the last display.
    """
    if profile is not None:
        return _compose_bounded_annotations(cues, cue_word_indices or {}, words or [], profile, protected_cue_ids,
                                            enforce_width, reserved_cue_ids, media_end_ms)
    return _compose_unbounded_annotations(cues, cue_word_indices, reserved_cue_ids)


def _compose_unbounded_annotations(
    cues: list[Cue], cue_word_indices: Mapping[int, list[int]] | None, reserved: set[int] | None = None,
) -> AnnotationComposition:
    ownership = {index: list(indices) for index, indices in (cue_word_indices or {}).items()}
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
    next_id = max([0, *ownership, *(cue.index for cue in cues), *(reserved or ())]) + 1
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


def _merged_intervals(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _coverage_gaps(start: int, end: int, intervals: list[list[int]]) -> list[list[int]]:
    gaps, cursor = [], start
    for left, right in intervals:
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            gaps.append([cursor, min(end, left)])
        cursor = max(cursor, min(end, right))
    if cursor < end:
        gaps.append([cursor, end])
    return gaps


def _caption_pages(track: Cue, profile: StyleProfile, limit: int, enforce_width: bool) -> list[list[str]]:
    if not enforce_width and len(track.text.splitlines()) <= limit:
        return [list(track.lines)]  # The customer's caption lines already fit the available lines.
    return paginate_annotation_lines(track.lines, profile.max_chars_per_line, limit)


def _speech_can_join(slot: dict[str, object], limit: int, enforce_width: bool) -> bool:
    """Whether full plain speech may give up a line by joining its own lines.

    Only under the customer's advisory width: never across dialogue turns,
    never screen text with dialogue, never past an enforced width.
    """
    cue = slot["cue"]
    return (cue is not None and limit > 1 and not enforce_width and slot["spoken"] >= limit
            and _dialogue_turns(cue.text.splitlines()) is None and not cue_has_bracketed_screen_text(cue))


def _interval_beside(slots: list[dict[str, object]], full: list[dict[str, object]], track: Cue,
                     min_ms: int, media_end_ms: int | None = None) -> tuple[int, int] | None:
    """The nearest free interval just before or after full speech displays.

    It holds at least the minimum display and lies between displays that
    exist, the media start or a known media end, so it never reaches past
    known timing. A tie goes to the later side: a two-line display delays a
    visual page.
    """
    start, end = min(slot["start"] for slot in full), max(slot["end"] for slot in full)
    duration = max(track.end_ms - track.start_ms, min_ms)
    options: list[tuple[int, int, int, int]] = []
    following = [slot["start"] for slot in slots if slot["start"] >= end]
    if media_end_ms is not None:
        following.append(media_end_ms)
    if following and not any(slot["start"] < end < slot["end"] for slot in slots) and min(following) - end >= min_ms:
        options.append((end - track.end_ms, 0, end, min(min(following), end + duration)))
    previous = max((slot["end"] for slot in slots if slot["end"] <= start), default=0)
    if not any(slot["start"] < start < slot["end"] for slot in slots) and start - previous >= min_ms:
        options.append((track.start_ms - start, 1, max(previous, start - duration), start))
    if not options:
        return None
    _, _, first, last = min(options)
    return first, last


def _compose_bounded_annotations(cues: list[Cue], incoming: Mapping[int, list[int]], words: list[Word],
                                 profile: StyleProfile, protected: set[int] | None,
                                 enforce_width: bool = True,
                                 reserved: set[int] | None = None,
                                 media_end_ms: int | None = None) -> AnnotationComposition:
    """Paginate crowded visual tracks while keeping each spoken word once.

    A caption page can occupy a whole spoken child or a known visual gap.
    Its actual interval, delay and uncovered original intervals are recorded;
    sequential pages never claim continuous display of the full parent text.
    """
    legacy = _compose_unbounded_annotations(cues, incoming, reserved)
    if not legacy.tracks:
        segmented = split_crowded_output_cues(cues, words, incoming, profile, protected_cue_ids=protected,
                                              enforce_width=enforce_width, reserved_cue_ids=reserved)
        return AnnotationComposition(segmented.cues, segmented.cue_word_indices, {}, {}, segmented.flags,
                                     segmented.expansions, 2)
    limit = min(2, profile.max_lines_per_cue)
    tracks = sorted((cue for cue in cues if cue.index in legacy.tracks),
                    key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    track_ids = {track.index for track in tracks}
    ownership = {index: list(indices) for index, indices in incoming.items()}
    # Reserve every source identity, including captions with no ownership key,
    # before the speech splitter assigns any new child identities.
    for cue in cues:
        ownership.setdefault(cue.index, [])
    speech, flags, expansions = [], [], {}
    for cue in cues:
        if cue.index in track_ids:
            continue
        overlapping = [track for track in tracks if _overlaps(track, cue)]
        if overlapping:
            budget = max(1, limit - 1)
            contained = [track for track in overlapping
                         if cue.start_ms <= track.start_ms and track.end_ms <= cue.end_ms]
            requested = max(1, sum(len(_caption_pages(track, profile, budget, enforce_width))
                                   for track in contained))
            # Speech that can be neither divided nor fitted to the line left
            # beside the caption keeps its own lines (no over-wide line or
            # joined dialogue turns) while the caption has another slot; the
            # track loop below decides when it has none.
            segmented = split_crowded_output_cues([cue], words, ownership, profile, max_lines=budget,
                                                  min_parts=requested, protected_cue_ids=protected,
                                                  enforce_width=enforce_width, reserved_cue_ids=reserved,
                                                  hold_unfitting=True)
            speech.extend(segmented.cues)
            ownership = segmented.cue_word_indices
            flags.extend(segmented.flags)
            expansions.update(segmented.expansions)
        else:
            speech.append(cue)
    speech.sort(key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    # A slot is an entire speech envelope or a gap known to contain visual
    # text. Never create a new speech boundary by dividing reading time.
    slots: list[dict[str, object]] = [
        {"start": cue.start_ms, "end": cue.end_ms, "cue": cue, "lines": list(cue.lines), "spoken": len(cue.lines),
         "annotations": [], "page_refs": [], "capacity": max(0, limit - len(cue.lines))}
        for cue in speech
    ]
    boundaries = sorted({time for cue in tracks + speech for time in (cue.start_ms, cue.end_ms)})
    residuals: list[dict[str, object]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [track for track in tracks if track.start_ms <= start and track.end_ms >= end]
        if not active or any(cue.start_ms < end and cue.end_ms > start for cue in speech):
            continue
        if residuals and residuals[-1]["end"] == start and residuals[-1]["active"] == [track.index for track in active]:
            residuals[-1]["end"] = end
        else:
            residuals.append({"start": start, "end": end, "cue": None, "template": active[0],
                              "active": [track.index for track in active], "lines": [], "spoken": 0,
                              "annotations": [], "page_refs": [], "capacity": limit})
    slots.extend(residuals)
    slots.sort(key=lambda slot: (slot["start"], slot["end"], slot["cue"].index if slot["cue"] else -1))
    track_pages: dict[int, list[dict[str, object]]] = {}
    # Captions that full speech displays moved out of their own time or hid.
    beside: set[int] = set()
    min_display_ms = max(1, round(profile.min_cue_dur * 1000))
    for track_position, track in enumerate(tracks):
        eligible = [slot for slot in slots if slot["start"] < track.end_ms and slot["end"] > track.start_ms]
        available = [slot for slot in eligible if slot["capacity"] > 0]
        # Without a free line, a caption shares the line(s) that fewer spoken
        # lines than the limit leave with the captions already shown there.
        shared = None if available else next((slot for slot in eligible if slot["spoken"] < limit), None)
        if not available and shared is None:
            joined = next((slot for slot in eligible if _speech_can_join(slot, limit, enforce_width)), None)
            if joined is not None:
                # The width is the customer's advisory one: plain speech joins
                # its lines rather than leave the caption no time of its own.
                cue = joined["cue"]
                lines = compact_lines(cue.text, limit - 1, profile.max_chars_per_line)
                joined.update(lines=list(lines), spoken=len(lines), capacity=limit - len(lines))
                flags.append(QCFlag(kind="output_line_limit_reflow", cue_ids=[cue.index], severity="info",
                                    message="A safe spoken split was unavailable. Text was reflowed within its existing interval; width overflow remains visible to style QC.",
                                    old_text=cue.text, new_text="\n".join(lines),
                                    start=cue.start_ms / 1000, end=cue.end_ms / 1000))
                available = [joined]
        if not available and shared is None:
            # Spoken lines fill the display for the caption's whole time. The
            # two-line ceiling holds and speech keeps its timing: the caption
            # is shown on its own in the nearest free interval beside them, or
            # it is not delivered, as an error and a removed line.
            pages = _caption_pages(track, profile, limit, enforce_width)
            page = pages[0] if len(pages) == 1 else compact_lines(track.text, limit, profile.max_chars_per_line)
            track_pages[track.index] = [{"page": 1, "lines": list(page), "display_cue_ids": [], "display_intervals": []}]
            beside.add(track.index)
            interval = _interval_beside(slots, eligible, track, min_display_ms, media_end_ms)
            if interval is not None:
                slots.append({"start": interval[0], "end": interval[1], "cue": None, "template": track,
                              "active": [track.index], "lines": list(page), "spoken": 0,
                              "annotations": [track.index], "page_refs": [(track.index, 0)],
                              "capacity": max(0, limit - len(page))})
            continue
        page_limit = min((slot["capacity"] for slot in available), default=1)
        pages = _caption_pages(track, profile, page_limit, enforce_width)
        if len(pages) > len(available) or not available:
            # A single word, held cue or simultaneous speech cannot always
            # supply enough child envelopes. Keep all visual wording and the
            # hard line cap; the existing width linter exposes the compromise.
            pages = [compact_lines(track.text, page_limit, profile.max_chars_per_line)]
            flags.append(QCFlag(kind="annotation_line_limit_reflow", cue_ids=[track.index], severity="info",
                                message="All visual caption wording was retained within the two-line display limit; any width overflow remains visible to style QC.",
                                old_text=track.text, new_text="\n".join(pages[0]),
                                start=track.start_ms / 1000, end=track.end_ms / 1000))
        records = [{"page": position + 1, "lines": list(page), "display_cue_ids": [], "display_intervals": []}
                   for position, page in enumerate(pages)]
        track_pages[track.index] = records
        if not available:
            slot = shared
            # Screen text never shares a line with dialogue: captions compact
            # into the lines the speech leaves.
            spoken = slot["spoken"]
            shown = compact_lines("\n".join([*slot["lines"][spoken:], *pages[0]]), limit - spoken,
                                  profile.max_chars_per_line)
            slot["lines"] = [*slot["lines"][:spoken], *shown]
            slot["annotations"].append(track.index)
            slot["page_refs"].append((track.index, 0))
            slot["capacity"] = 0
            continue
        for position, slot in enumerate(available):
            page_position = min(position, len(pages) - 1)
            if position >= len(pages) and any(
                slot["start"] < later.end_ms and slot["end"] > later.start_ms
                for later in tracks[track_position + 1:]
            ):
                continue  # Leave space for the next visual message.
            page = pages[page_position]
            slot["lines"].extend(page)
            slot["annotations"].append(track.index)
            slot["page_refs"].append((track.index, page_position))
            slot["capacity"] -= len(page)
    used = {cue.index for cue in speech}
    next_id = max([0, *ownership, *(cue.index for cue in cues), *(cue.index for cue in speech),
                   *(reserved or ())]) + 1
    for track in tracks:
        ownership.pop(track.index, None)
    result, cue_annotations = [], {}
    for slot in slots:
        if not slot["lines"]:
            continue
        original = slot["cue"]
        if original is None:
            index = next((track_id for track_id in slot["annotations"] if track_id not in used), None)
            if index is None:
                index, next_id = next_id, next_id + 1
            used.add(index)
            ownership[index] = []
            cue = slot["template"].model_copy(update={"index": index, "start_ms": slot["start"],
                                                       "end_ms": slot["end"], "lines": list(slot["lines"])})
        else:
            cue = original.with_lines(list(slot["lines"])) if slot["annotations"] else original
        result.append(cue)
        if slot["annotations"]:
            cue_annotations[cue.index] = list(slot["annotations"])
        for track_id, page_position in slot["page_refs"]:
            record = track_pages[track_id][page_position]
            record["display_cue_ids"].append(cue.index)
            record["display_intervals"].append([cue.start_ms, cue.end_ms])
    result.sort(key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    metadata: dict[int, dict[str, object]] = {}
    for track in tracks:
        pages = track_pages[track.index]
        for page in pages:
            page["display_intervals"] = _merged_intervals(page["display_intervals"])
            page["delay_ms"] = (max(0, page["display_intervals"][0][0] - track.start_ms)
                                if page["display_intervals"] else None)
        intervals = _merged_intervals([interval for page in pages for interval in page["display_intervals"]])
        metadata[track.index] = {
            "original_start_ms": track.start_ms, "original_end_ms": track.end_ms, "lines": list(track.lines),
            "display_cue_ids": list(dict.fromkeys(index for page in pages for index in page["display_cue_ids"])),
            "display_intervals": intervals,
            "early_extension_ms": max(0, track.start_ms - intervals[0][0]) if intervals else 0,
            "late_extension_ms": max(0, intervals[-1][1] - track.end_ms) if intervals else 0,
            "coverage_gaps_ms": _coverage_gaps(track.start_ms, track.end_ms, intervals), "pages": pages,
            "pagination_policy": "ordered_visual_pages",
        }
        original = f"{format_timestamp(track.start_ms)} --> {format_timestamp(track.end_ms)}"
        if track.index in beside and not intervals:
            # The customer's screen text is not in the delivery. It is never
            # lost silently: an error review item at its own time (named by
            # the caption's cue id), and QC logs it as a removed line.
            flags.append(QCFlag(
                kind="annotation_display_full", severity="error", cue_ids=[track.index],
                message=(f'The screen text "{" ".join(track.lines)}" could not be displayed: spoken lines fill the '
                         f"two-line display for its whole time ({original}) and no free interval of at least "
                         f"{min_display_ms / 1000:g} s beside that speech can hold it, so it is not in the delivered "
                         "subtitle. Add it back where the picture needs it."),
                old_text=track.text, new_text=None, start=track.start_ms / 1000, end=track.end_ms / 1000))
        elif track.index in beside:
            # Shown on its own beside the full speech: a display change whose
            # page lineage and change-log entry say where it went.
            side = "before" if intervals[-1][1] <= track.start_ms else "after"
            shown = f"{format_timestamp(intervals[0][0])} --> {format_timestamp(intervals[-1][1])}"
            flags.append(QCFlag(
                kind="annotation_line_limit_pagination", severity="info",
                cue_ids=metadata[track.index]["display_cue_ids"],
                message=(f"Spoken lines fill the two-line display for this screen text's whole time ({original}), "
                         f"so it is shown on its own just {side} that speech instead ({shown}); its interval and "
                         "coverage gap are recorded in caption provenance."),
                old_text=track.text, new_text="\n\n".join("\n".join(page["lines"]) for page in pages),
                start=intervals[0][0] / 1000, end=intervals[-1][1] / 1000))
        elif len(pages) > 1 or metadata[track.index]["coverage_gaps_ms"]:
            flags.append(QCFlag(kind="annotation_line_limit_pagination", cue_ids=metadata[track.index]["display_cue_ids"], severity="info",
                                message="Visual caption wording was shown as ordered pages to keep the display within two lines; actual page intervals and any delayed visual onset are recorded in caption provenance.",
                                old_text=track.text, new_text="\n\n".join("\n".join(page["lines"]) for page in pages),
                                start=intervals[0][0] / 1000, end=intervals[-1][1] / 1000))
    return AnnotationComposition(result, ownership, cue_annotations, metadata, flags, expansions, 2)
