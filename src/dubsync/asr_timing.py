from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .models import QCFlag, SpeechRegion, Word

# A speech burst must cover this much of a word (or half of a shorter word)
# before the word's edges are moved onto it.
MIN_OWNED_OVERLAP_SECONDS = 0.05
# Speech at the start of a stretched interval is the word itself only when it
# is at least this long; a shorter remnant is the previous word's tail.
MIN_WHOLE_WORD_SECONDS = 0.1
MIN_SECONDS_PER_LETTER = 0.03
# Neighbouring words may touch a burst edge by this much without owning it.
_EDGE_TOLERANCE_SECONDS = 0.01


@dataclass(frozen=True)
class PhraseEdgeSnap:
    """How far a phrase edge inside a speech burst may move onto the burst edge.

    ASR starts lag soft consonant onsets and sit on a coarse grid, and ASR ends
    stop before or after the voice does. When no other word lies in between, a
    phrase-initial start within ``start_advance`` seconds after the burst onset
    moves back to it and a phrase-final end within ``end_extension`` seconds
    before the burst offset moves forward to it. Edges that lie in silence are
    always moved onto the burst, whatever the distance.
    """

    start_advance: float = 0.2
    end_extension: float = 0.3

    def __post_init__(self) -> None:
        if self.start_advance < 0 or self.end_extension < 0:
            raise ValueError("phrase edge snap limits must be non-negative")


NO_PHRASE_EDGE_SNAP = PhraseEdgeSnap(start_advance=0.0, end_extension=0.0)


def phrase_edge_snap_from_config(
    provider_config: dict[str, object],
    asr_model: str | None = None,
    *,
    default_end_extension: float | None = None,
) -> PhraseEdgeSnap:
    """Read ``timing.phrase_edge_snap``; ``models.<asr model id>`` entries override the shared limits."""
    defaults = PhraseEdgeSnap() if default_end_extension is None else PhraseEdgeSnap(end_extension=default_end_extension)
    timing_config = provider_config.get("timing", {}) if isinstance(provider_config, dict) else {}
    options = timing_config.get("phrase_edge_snap") if isinstance(timing_config, dict) else None
    if options is None:
        return defaults
    if options is False:
        return NO_PHRASE_EDGE_SNAP
    if not isinstance(options, dict):
        raise ValueError("timing.phrase_edge_snap must be a mapping or false")
    merged = {key: value for key, value in options.items() if key != "models"}
    models = options.get("models")
    if models is not None and not isinstance(models, dict):
        raise ValueError("timing.phrase_edge_snap.models must be a mapping")
    if models and asr_model is not None and isinstance(models.get(asr_model), dict):
        merged.update(models[asr_model])
    return PhraseEdgeSnap(
        start_advance=_snap_seconds(merged, "start_advance_ms", defaults.start_advance),
        end_extension=_snap_seconds(merged, "end_extension_ms", defaults.end_extension),
    )


def _snap_seconds(options: dict[str, object], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    try:
        seconds = float(value) / 1000.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timing.phrase_edge_snap.{key} must be numeric") from exc
    if not isfinite(seconds) or seconds < 0:
        raise ValueError(f"timing.phrase_edge_snap.{key} must be finite and non-negative")
    return seconds


def asr_model_from_artifact(asr_artifact_path: Path | None) -> str | None:
    """ASR model id recorded with the word stream, when the artifact exists."""
    if asr_artifact_path is None or not asr_artifact_path.exists():
        return None
    try:
        payload = json.loads(asr_artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    model = metadata.get("model") if isinstance(metadata, dict) else None
    return str(model) if model else None


def ambiguous_word_indices(words: list[Word], flags: list[QCFlag]) -> set[int]:
    """Locate retained ambiguous words using exact provider edges and their text.

    Indices are derived for the supplied stream, so callers do not rely on a
    mutable marker on Word or on rounded display timestamps alone.
    """
    evidence = {
        (flag.start, flag.end, flag.old_text)
        for flag in flags if flag.kind == "asr_word_timing_ambiguous"
    }
    return {
        index for index, word in enumerate(words)
        if (word.start, word.end, f"{word.text} {word.start:.3f} --> {word.end:.3f}") in evidence
    }


def ambiguous_word_indices_from_regions(
    words: list[Word], regions: list[SpeechRegion], *, max_region_gap: float = 0.2,
) -> set[int]:
    """Detect ambiguity for standalone timing callers that lack repair flags."""
    ordered_regions, region_starts, prefix_max_ends = _region_lookup(regions)
    return {
        index for index, word in enumerate(words)
        if isfinite(word.start) and isfinite(word.end) and word.end > word.start
        and len(_anchor_candidates(
            word, _speech_chains(word, ordered_regions, region_starts, prefix_max_ends, max_region_gap),
        )) > 1
    }


def _region_lookup(
    regions: list[SpeechRegion],
) -> tuple[list[SpeechRegion], list[float], list[float]]:
    ordered_regions = sorted(regions, key=lambda region: (region.start, region.end))
    region_starts = [region.start for region in ordered_regions]
    prefix_max_ends: list[float] = []
    maximum = float("-inf")
    for region in ordered_regions:
        maximum = max(maximum, region.end)
        prefix_max_ends.append(maximum)
    return ordered_regions, region_starts, prefix_max_ends


def repair_asr_word_edges(
    words: list[Word],
    regions: list[SpeechRegion],
    *,
    max_word_duration: float = 2.0,
    max_region_overrun: float = 0.3,
    max_region_gap: float = 0.2,
    snap: PhraseEdgeSnap | None = None,
) -> tuple[list[Word], list[QCFlag]]:
    """Move ASR word edges that lie in silence onto the speech they belong to.

    Both ASR models place some word edges outside the audible word. Scribe
    stretches a word across the pause before or after it, MAI ends words after
    the voice has stopped and starts phrases on a 40 ms grid. Each word is
    anchored on the speech bursts it overlaps: an edge in silence moves to the
    burst edge (from the side that has speech, so a start-stretched word keeps
    its real end), and phrase edges inside a burst are snapped as described by
    ``PhraseEdgeSnap``. Without acoustic evidence only the duration limit
    applies. When several separate bursts can each contain the word, its raw
    interval is retained for downstream uncertainty handling; energy cannot
    identify which burst contains that word. The returned list keeps the order
    and length of ``words``.

    A word whose edge moved by more than ``max_region_overrun`` seconds, or that
    needed the duration limit, is reported as ``asr_word_clamped``. Retained
    ambiguous intervals are reported as ``asr_word_timing_ambiguous`` instead.
    """
    if max_word_duration <= 0:
        raise ValueError("timing.max_word_duration must be positive")
    if max_region_overrun < 0 or max_region_gap < 0:
        raise ValueError("speech-region timing limits must be non-negative")
    snap = snap or PhraseEdgeSnap()

    ordered_regions, region_starts, prefix_max_ends = _region_lookup(regions)

    # First pass: edges in silence and the duration limit. These depend only
    # on the word itself, so neighbours can be consulted afterwards.
    repaired: list[tuple[float, float]] = []
    ambiguous: dict[int, list[tuple[float, float]]] = {}
    for index, word in enumerate(words):
        start, end = word.start, word.end
        if isfinite(start) and isfinite(end) and end > start and ordered_regions:
            candidates = _anchor_candidates(
                word,
                _speech_chains(word, ordered_regions, region_starts, prefix_max_ends, max_region_gap),
            )
            if len(candidates) > 1:
                # A duration cap or phrase snap would silently pick one of the
                # possible utterances too. Preserve both provider edges.
                ambiguous[index] = candidates
                repaired.append((start, end))
                continue
            if candidates:
                anchor = candidates[0]
                anchored_start = max(start, anchor[0])
                anchored_end = min(end, anchor[1])
                if anchored_end > anchored_start:
                    start, end = anchored_start, anchored_end
        if isfinite(start) and isfinite(end) and end - start > max_word_duration:
            end = start + max_word_duration
        repaired.append((start, end))

    result: list[Word] = []
    flags: list[QCFlag] = []
    for index, word in enumerate(words):
        if index in ambiguous:
            candidates = ambiguous[index]
            windows = ", ".join(f"{start:.3f} --> {end:.3f}" for start, end in candidates[:4])
            if len(candidates) > 4:
                windows += f", and {len(candidates) - 4} more"
            result.append(word)
            flags.append(QCFlag(
                kind="asr_word_timing_ambiguous",
                cue_ids=[],
                severity="warning",
                message=(
                    "ASR word overlaps several plausible speech bursts; provider timing was retained "
                    f"because energy alone cannot identify the spoken word. Bursts: {windows}."
                ),
                old_text=f"{word.text} {word.start:.3f} --> {word.end:.3f}",
                start=word.start,
                end=word.end,
            ))
            continue
        start, end = repaired[index]
        if isfinite(start) and isfinite(end) and end > start and ordered_regions:
            previous_end = repaired[index - 1][1] if index > 0 else float("-inf")
            next_start = repaired[index + 1][0] if index + 1 < len(words) else float("inf")
            onset = _containing_region(start, ordered_regions, region_starts, prefix_max_ends)
            if (
                onset is not None
                and onset.start < start <= onset.start + snap.start_advance
                and previous_end <= onset.start + _EDGE_TOLERANCE_SECONDS
            ):
                start = onset.start
            offset = _containing_region(end, ordered_regions, region_starts, prefix_max_ends)
            if (
                offset is not None
                and end < offset.end <= end + snap.end_extension
                and next_start >= offset.end - _EDGE_TOLERANCE_SECONDS
            ):
                end = offset.end
        # Only moved edges are rounded; untouched provider values stay exact.
        start = word.start if start == word.start else round(start, 3)
        end = word.end if end == word.end else round(end, 3)
        if start == word.start and end == word.end:
            result.append(word)
            continue
        if end <= start:
            # Rounding may not erase a real word; keep the provider's interval.
            result.append(word)
            continue
        next_word = word.model_copy(update={"start": start, "end": end})
        result.append(next_word)
        duration = word.end - word.start
        if (
            duration > max_word_duration
            or word.end - end > max_region_overrun
            or start - word.start > max_region_overrun
        ):
            flags.append(
                QCFlag(
                    kind="asr_word_clamped",
                    cue_ids=[],
                    message="ASR word endpoint exceeded the configured duration or speech region bounds and was clamped.",
                    old_text=f"{word.text} {word.start:.3f} --> {word.end:.3f}",
                    new_text=f"{next_word.text} {next_word.start:.3f} --> {next_word.end:.3f}",
                    start=next_word.start,
                    end=next_word.end,
                    confidence=round(duration, 3),
                )
            )
    return result, flags


def clamp_asr_word_durations(
    words: list[Word],
    regions: list[SpeechRegion],
    *,
    max_word_duration: float = 2.0,
    max_region_overrun: float = 0.3,
    max_region_gap: float = 0.2,
) -> tuple[list[Word], list[QCFlag]]:
    """Compatibility entry point for ``repair_asr_word_edges`` with default snapping."""
    return repair_asr_word_edges(
        words,
        regions,
        max_word_duration=max_word_duration,
        max_region_overrun=max_region_overrun,
        max_region_gap=max_region_gap,
    )


def _speech_chains(
    word: Word,
    regions: list[SpeechRegion],
    region_starts: list[float],
    prefix_max_ends: list[float],
    max_region_gap: float,
) -> list[tuple[float, float]]:
    """Speech bursts overlapping the word, joined across pauses shorter than ``max_region_gap``."""
    chains: list[tuple[float, float]] = []
    left = bisect_right(prefix_max_ends, word.start)
    right = bisect_left(region_starts, word.end)
    for region in regions[left:right]:
        if region.end <= word.start or region.start >= word.end:
            continue
        if chains and region.start - chains[-1][1] < max_region_gap:
            chains[-1] = (chains[-1][0], max(chains[-1][1], region.end))
        else:
            chains.append((region.start, region.end))
    return chains


def _anchor_candidates(word: Word, chains: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return plausible speech locations without choosing between utterances.

    A stretched word often touches speech on one side only, or includes a short
    tail of the previous word. Multiple bursts long enough to contain the whole
    word are ambiguous: a preferred provider edge is not lexical evidence.
    """
    owned = [chain for chain in chains if has_sufficient_speech_overlap(word, *chain)]
    if not owned:
        return []
    first, last = owned[0], owned[-1]
    if first == last:
        return [first]
    letters = sum(character.isalnum() for character in word.text)
    plausible = max(MIN_WHOLE_WORD_SECONDS, MIN_SECONDS_PER_LETTER * letters)
    candidates = [chain for chain in owned if _overlap(word, chain) >= plausible]
    if candidates:
        return candidates
    # Several short overlaps may be separate utterances or split phonemes.
    # Choosing the longest would still guess the word's acoustic ownership.
    return owned


def has_sufficient_speech_overlap(word: Word, start: float, end: float) -> bool:
    """Whether a burst covers enough of a word to replace its provider edge."""
    needed = min(MIN_OWNED_OVERLAP_SECONDS, (word.end - word.start) / 2)
    return _overlap(word, (start, end)) >= needed


def _overlap(word: Word, chain: tuple[float, float]) -> float:
    return min(word.end, chain[1]) - max(word.start, chain[0])


def _containing_region(
    timestamp: float,
    regions: list[SpeechRegion],
    region_starts: list[float],
    prefix_max_ends: list[float],
) -> SpeechRegion | None:
    left = bisect_left(prefix_max_ends, timestamp)
    right = bisect_right(region_starts, timestamp)
    for region in regions[left:right]:
        if region.start <= timestamp <= region.end:
            return region
    return None
