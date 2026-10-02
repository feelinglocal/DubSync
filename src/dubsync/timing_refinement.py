from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from .asr_timing import (
    PhraseEdgeSnap,
    ambiguous_word_indices_from_regions,
    asr_model_from_artifact,
    clamp_asr_word_durations,
    has_sufficient_speech_overlap,
    phrase_edge_snap_from_config,
    repair_asr_word_edges,
)
from .models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from .recue import ambiguous_word_cue_ids, ambiguous_word_timing_flags, select_cue_word_window, source_fragment_boundary_word_ids
from .region_index import SpeechRegionIndex
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue
from .vad import SpeechActivityAdapter, SpeechLevels


# Another word may begin this close to a burst offset without sharing the burst.
_SHARED_BURST_TOLERANCE_SECONDS = 0.01
MIN_DURATION_POLICIES = ("extend_into_silence", "acoustic")


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    enabled: bool = True
    start_pad_ms: int = 40
    end_pad_ms: int = 40
    max_end_extension_ms: int = 300
    max_leading_silence_ms: int = 150
    max_trailing_silence_ms: int = 300
    max_word_duration_ms: int = 2000
    # Shared with rebuild (timing.max_intra_cue_gap, timing.min_duration_policy).
    max_intra_cue_gap_ms: int = 1500
    min_duration_policy: str = "extend_into_silence"


def min_duration_policy_from_config(provider_config: dict[str, object]) -> str:
    """How a cue shorter than the minimum display duration is finished.

    ``extend_into_silence`` keeps it on screen up to the minimum while nothing
    else is heard and no other cue starts; ``acoustic`` ends it with its speech.
    """
    timing_config = provider_config.get("timing", {}) if isinstance(provider_config, dict) else {}
    value = timing_config.get("min_duration_policy", MIN_DURATION_POLICIES[0]) if isinstance(timing_config, dict) else MIN_DURATION_POLICIES[0]
    if value not in MIN_DURATION_POLICIES:
        raise ValueError(f"timing.min_duration_policy must be one of: {', '.join(MIN_DURATION_POLICIES)}")
    return str(value)


def boundary_refinement_config_from_config(provider_config: dict[str, object]) -> BoundaryRefinementConfig:
    """Read the same acoustic-boundary policy for synchronization and generation."""
    vad_config = provider_config.get("vad", {}) if isinstance(provider_config, dict) else {}
    if not isinstance(vad_config, dict):
        return BoundaryRefinementConfig(enabled=False)
    value = vad_config.get("boundary_refinement", False)
    if value in (False, None):
        return BoundaryRefinementConfig(enabled=False)
    if value is not True and not isinstance(value, dict):
        raise ValueError("vad.boundary_refinement must be a mapping or boolean")
    options = {} if value is True else value
    enabled = options.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("vad.boundary_refinement.enabled must be boolean")
    timing_config = provider_config.get("timing", {})
    return BoundaryRefinementConfig(
        enabled=enabled,
        start_pad_ms=_boundary_milliseconds(options, "start_pad_ms", 40),
        end_pad_ms=_boundary_milliseconds(options, "end_pad_ms", 40),
        max_end_extension_ms=_boundary_milliseconds(options, "max_end_extension_ms", 300),
        max_leading_silence_ms=_boundary_milliseconds(options, "max_leading_silence_ms", 150),
        max_trailing_silence_ms=_boundary_milliseconds(options, "max_trailing_silence_ms", 300),
        max_word_duration_ms=int(_timing_seconds(timing_config, "max_word_duration", 2.0) * 1000),
        max_intra_cue_gap_ms=int(_timing_seconds(timing_config, "max_intra_cue_gap", 1.5) * 1000),
        min_duration_policy=min_duration_policy_from_config(provider_config),
    )


def _timing_seconds(timing_config: object, key: str, default: float) -> float:
    value = timing_config.get(key, default) if isinstance(timing_config, dict) else default
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timing.{key} must be numeric") from exc
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError(f"timing.{key} must be finite and positive")
    return seconds


@dataclass(frozen=True)
class SpeechEvidence:
    """Speech bursts of one recording and the ASR words repaired against them.

    Detected once before cues are timed so rebuild and verification use the
    same word edges.
    """

    words: list[Word]
    regions: list[SpeechRegion] = field(default_factory=list)
    word_flags: list[QCFlag] = field(default_factory=list)
    detected: bool = False
    fallback_used: bool = False
    # How far (seconds) a phrase-initial word start was moved back onto its burst onset.
    start_snap: float = PhraseEdgeSnap().start_advance
    # The same limit for a recording whose phrase starts lag, and the level track that decided it.
    lagging_start_snap: float = PhraseEdgeSnap().lagging_start_advance
    levels: SpeechLevels | None = None


def speech_evidence_for_words(
    adapter: SpeechActivityAdapter | None,
    words: list[Word],
    audio_path: Path,
    provider_config: dict[str, object],
    *,
    max_word_duration: float = 2.0,
    asr_artifact_path: Path | None = None,
) -> SpeechEvidence:
    """Run the configured VAD once and repair ASR word edges against its bursts."""
    if adapter is None:
        return SpeechEvidence(words=words)
    regions = adapter.detect(audio_path)
    boundary = boundary_refinement_config_from_config(provider_config)
    snap = phrase_edge_snap_from_config(
        provider_config,
        asr_model_from_artifact(asr_artifact_path),
        # Refinement follows a burst past the last word by the same limit;
        # a different word-level limit would only make the stages disagree.
        default_end_extension=boundary.max_end_extension_ms / 1000.0,
    )
    # Only the energy detector measures levels; other detectors leave the fixed start window.
    levels = getattr(adapter, "last_levels", None)
    repaired, word_flags = repair_asr_word_edges(
        words,
        regions,
        max_word_duration=max_word_duration,
        max_region_overrun=boundary.max_trailing_silence_ms / 1000.0,
        snap=snap,
        levels=levels,
    )
    return SpeechEvidence(
        words=repaired,
        regions=regions,
        word_flags=word_flags,
        detected=True,
        fallback_used=bool(getattr(adapter, "fallback_used", False)),
        start_snap=snap.start_advance,
        lagging_start_snap=snap.lagging_start_advance,
        levels=levels,
    )


def _boundary_milliseconds(options: dict[str, object], key: str, default: int) -> int:
    try:
        number = int(options.get(key, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"vad.boundary_refinement.{key} must be an integer") from exc
    if number < 0:
        raise ValueError(f"vad.boundary_refinement.{key} must be non-negative")
    return number


def refine_cues_to_speech_activity(
    cues: list[Cue],
    regions: list[SpeechRegion],
    profile: StyleProfile,
    config: BoundaryRefinementConfig | None = None,
    *,
    words: list[Word] | None = None,
    alignment: AlignmentResult | None = None,
    protected_cue_ids: set[int] | None = None,
    fixed_cue_ids: set[int] | None = None,
    ambiguous_word_indices: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    options = config or BoundaryRefinementConfig()
    ambiguous_indices = set(ambiguous_word_indices or ())
    if ambiguous_word_indices is None and words and alignment is not None and options.enabled and regions:
        # The pipeline passes its once-computed repair evidence. Standalone
        # callers still must not select a burst after retaining an unsure word.
        ambiguous_indices = ambiguous_word_indices_from_regions(words, regions)
    ambiguous = ambiguous_word_cue_ids(alignment, ambiguous_indices) if alignment is not None else set()
    if not options.enabled or not regions:
        return cues, ambiguous_word_timing_flags(cues, ambiguous - (protected_cue_ids or set()))

    protected = (
        set(alignment.diagnostics.missing_audio_cue_ids)
        if alignment is not None
        else set()
    ) | (protected_cue_ids or set())
    ambiguity_flags = ambiguous_word_timing_flags(cues, ambiguous - protected)
    protected |= ambiguous
    # Acoustic retiming can put dialogue before a source-held cue that was
    # earlier in the script. Caps follow playback order, not source order.
    dialogue_cues = sorted(
        (cue for cue in cues if not is_bracketed_screen_text_cue(cue)),
        key=lambda cue: (cue.start_ms, cue.end_ms, cue.index),
    )
    refined: list[Cue] = []
    flags: list[QCFlag] = list(ambiguity_flags)
    word_repair_flags: list[QCFlag] = []
    if words and any(
        index not in ambiguous_indices and _is_word_duration_outlier(word, options)
        for index, word in enumerate(words)
    ):
        # Repair a corrupt endpoint before choosing a lexical word cluster;
        # otherwise a real final word can be discarded as a separate cluster.
        clamped_words, word_repair_flags = clamp_asr_word_durations(
            words, regions, max_word_duration=options.max_word_duration_ms / 1000.0,
        )
        # A duration-only fallback is still uncertain. Only adopt a shortened
        # word when the speech region supplies a tighter endpoint. Compare the
        # exact rounded fallback used by the clamp, not float subtraction:
        # arbitrary sub-millisecond starts can otherwise appear shorter.
        words = [
            clamped
            if clamped.end < round(original.start + options.max_word_duration_ms / 1000.0, 3)
            else original
            for original, clamped in zip(words, clamped_words)
        ]
        retained_repairs = {(word.start, word.end) for word in words}
        word_repair_flags = [
            flag for flag in word_repair_flags if (flag.start, flag.end) in retained_repairs
        ]
    region_index = SpeechRegionIndex(regions)
    region_starts = [region.start for region in region_index.regions]
    word_starts = sorted(word.start for word in words) if words else []

    for index, cue in enumerate(dialogue_cues):
        # Held dialogue and accepted per-cue alignment remain neighbor caps,
        # but their own timing is preserved.
        if cue.index in protected or cue.index in (fixed_cue_ids or set()):
            refined.append(cue)
            continue
        word_window = _word_window_for_cue(
            cue,
            words,
            alignment,
            max_word_duration_seconds=options.max_word_duration_ms / 1000.0,
            max_intra_cue_gap_seconds=options.max_intra_cue_gap_ms / 1000.0,
        )
        cue_regions = (
            _regions_from_word_window(word_window, region_index, options)
            if word_window is not None
            else _regions_overlapping_cue(cue, region_index)
        )
        if cue_regions is None:
            refined.append(cue)
            continue

        start_region, end_region = cue_regions
        start_ms = _refined_start_ms(cue, start_region, profile, options)
        last_word_is_outlier = word_window is not None and _is_word_duration_outlier(word_window[-1], options)
        # The cue's own voice ends with the burst that holds its last word.
        acoustic_end = (
            _acoustic_end_seconds(word_window[-1], end_region, options, word_starts)
            if word_window is not None
            else end_region.end
        )
        if word_window is None:
            end_ms = _refined_end_ms(cue, end_region, profile, options)
        elif last_word_is_outlier:
            end_ms = profile.snap_ceil(acoustic_end * 1000 + options.end_pad_ms)
        else:
            end_ms = _word_refined_end_ms(cue, acoustic_end, profile, options)
        end_cap_ms = None
        if index + 1 < len(dialogue_cues):
            # The following cue can limit display padding, but cannot erase
            # speech from a simultaneous speaker or collapse an inverted source
            # cue to zero length. Output policy handles real overlaps separately.
            next_start_ms = dialogue_cues[index + 1].start_ms
            acoustic_floor_ms = min(cue.end_ms, end_ms)
            if word_window is not None and not last_word_is_outlier:
                acoustic_floor_ms = profile.snap_ceil(acoustic_end * 1000)
                if next_start_ms > start_ms and next_start_ms >= acoustic_end * 1000 - profile.frame_ms:
                    # Words less than a frame apart share the next start as
                    # their boundary, exactly as rebuild decided.
                    acoustic_floor_ms = min(acoustic_floor_ms, next_start_ms)
            end_cap_ms = max(next_start_ms, acoustic_floor_ms)
            end_ms = min(end_ms, end_cap_ms)
        speech_end_ms = profile.snap_ceil(acoustic_end * 1000 + options.end_pad_ms)
        minimum_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000)
        if options.min_duration_policy == "acoustic":
            # Minimum display duration cannot add a silence tail beyond the
            # acoustic endpoint. Preserve already accepted short tails, but do
            # not manufacture more silence merely to reach the readability floor.
            end_ms = min(max(end_ms, minimum_end_ms), max(end_ms, speech_end_ms))
        elif start_ms < speech_end_ms < minimum_end_ms:
            # A cue shorter than the readability floor stays up while nothing
            # else is heard: up to the floor, the next sound or the next cue.
            quiet_until = _quiet_until_seconds(
                acoustic_end, end_region, region_index.regions, region_starts, word_starts,
            )
            quiet_limit_ms = profile.snap_floor(quiet_until * 1000) if isfinite(quiet_until) else minimum_end_ms
            end_ms = max(speech_end_ms, min(minimum_end_ms, quiet_limit_ms))
            if end_cap_ms is not None:
                end_ms = min(end_ms, end_cap_ms)
        if end_cap_ms is not None and end_ms > end_cap_ms:
            end_ms = max(start_ms, end_cap_ms)

        if end_ms <= start_ms:
            # Conflicting cue order or word ownership must be repaired upstream.
            # A speech cap is not permission to export reversed/zero duration or
            # to manufacture a new endpoint merely to satisfy readability.
            refined.append(cue)
            flags.append(
                QCFlag(
                    kind="timing_refinement_held",
                    cue_ids=[cue.index],
                    severity="error",
                    message=(
                        "Speech evidence conflicts with this cue's placement; "
                        "kept its prior timing instead of creating a non-positive duration. "
                        "Review the cue's word ownership and neighboring dialogue."
                    ),
                    old_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
                    new_text=f"{start_ms / 1000.0:.3f} --> {end_ms / 1000.0:.3f}",
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        # On the frame grid a cue within one frame of the minimum is at the
        # minimum; only a real shortfall is reported.
        minimum_unattainable = end_ms - start_ms < profile.min_cue_dur * 1000 - profile.frame_ms

        if start_ms == cue.start_ms and end_ms == cue.end_ms:
            refined.append(cue)
            if minimum_unattainable:
                flags.append(_min_duration_unattainable_flag(cue, cue, profile))
            continue

        next_cue = cue.with_timing(start_ms, end_ms)
        refined.append(next_cue)
        flags.append(
            QCFlag(
                kind="timing_refined",
                cue_ids=[cue.index],
                message="Cue boundary adjusted to the detected speech activity envelope.",
                old_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
                new_text=f"{next_cue.start_ms / 1000.0:.3f} --> {next_cue.end_ms / 1000.0:.3f}",
                start=next_cue.start_ms / 1000.0,
                end=next_cue.end_ms / 1000.0,
            )
        )
        if minimum_unattainable:
            flags.append(_min_duration_unattainable_flag(cue, next_cue, profile))

    refined_by_id = {cue.index: cue for cue in refined}
    merged = [
        cue
        if cue.index in protected or is_bracketed_screen_text_cue(cue)
        else refined_by_id[cue.index]
        for cue in cues
    ]
    return merged, [*flags, *word_repair_flags]


def _min_duration_unattainable_flag(old_cue: Cue, cue: Cue, profile: StyleProfile) -> QCFlag:
    return QCFlag(
        kind="min_duration_unattainable",
        cue_ids=[cue.index],
        message=(
            f"Cue could not reach the {profile.min_cue_dur:.3f}s minimum display duration "
            "without exceeding the speech envelope or crossing the following cue boundary."
        ),
        severity="error",
        old_text=f"{old_cue.start_ms / 1000.0:.3f} --> {old_cue.end_ms / 1000.0:.3f}",
        new_text=f"{cue.start_ms / 1000.0:.3f} --> {cue.end_ms / 1000.0:.3f}",
        start=cue.start_ms / 1000.0,
        end=cue.end_ms / 1000.0,
    )


def _regions_overlapping_cue(cue: Cue, region_index: SpeechRegionIndex) -> tuple[SpeechRegion, SpeechRegion] | None:
    cue_start = cue.start_ms / 1000.0
    cue_end = cue.end_ms / 1000.0
    overlapping = region_index.overlapping(cue_start, cue_end)
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def _word_window_for_cue(
    cue: Cue,
    words: list[Word] | None,
    alignment: AlignmentResult | None,
    *,
    max_word_duration_seconds: float,
    max_intra_cue_gap_seconds: float,
) -> list[Word] | None:
    if words is None or alignment is None:
        return None
    matched = [
        words[index]
        for index in alignment.cue_word_indices.get(cue.index, [])
        if 0 <= index < len(words)
    ]
    if not matched:
        return None
    # The same selection as rebuild, with the same configured gap, so the two
    # stages cannot time one cue from different words.
    selected, _ = select_cue_word_window(
        cue,
        sorted(matched, key=lambda word: (word.start, word.end)),
        max_word_duration=max_word_duration_seconds,
        max_intra_cue_gap=max_intra_cue_gap_seconds,
        preserve_boundary_word_ids=source_fragment_boundary_word_ids(cue.index, words, alignment),
    )
    return selected


def _regions_from_word_window(
    word_window: list[Word],
    region_index: SpeechRegionIndex,
    config: BoundaryRefinementConfig,
) -> tuple[SpeechRegion, SpeechRegion] | None:
    first_word = word_window[0]
    last_word = word_window[-1]
    start_region = _region_containing_timestamp(first_word.start, region_index)
    if start_region is None:
        start_region = _region_overlapping_word(first_word, region_index)
    end_probe = last_word.start if _is_word_duration_outlier(last_word, config) else last_word.end
    end_region = _region_containing_timestamp(end_probe, region_index)
    if end_region is None:
        end_region = _region_containing_timestamp(last_word.start, region_index) or _region_overlapping_word(last_word, region_index)
    if start_region is None or end_region is None:
        return None
    return start_region, end_region


def _region_containing_timestamp(timestamp: float, region_index: SpeechRegionIndex) -> SpeechRegion | None:
    match = region_index.first_containing(timestamp)
    return match[1] if match is not None else None


def _region_overlapping_word(word: Word, region_index: SpeechRegionIndex) -> SpeechRegion | None:
    overlapping = region_index.overlapping(word.start, word.end)
    return overlapping[0] if overlapping else None


def _is_word_duration_outlier(word: Word, config: BoundaryRefinementConfig) -> bool:
    return (word.end - word.start) * 1000 > config.max_word_duration_ms


def _refined_start_ms(
    cue: Cue,
    first_region: SpeechRegion,
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    first_speech_ms = int(first_region.start * 1000)
    leading_silence_ms = first_speech_ms - cue.start_ms
    if leading_silence_ms <= config.max_leading_silence_ms:
        return cue.start_ms
    return max(0, profile.snap_floor(first_speech_ms - config.start_pad_ms))


def _refined_end_ms(
    cue: Cue,
    last_region: SpeechRegion,
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    speech_end_ms = int(last_region.end * 1000)
    padded_end_ms = profile.snap_ceil(speech_end_ms + config.end_pad_ms)
    end_overrun_ms = speech_end_ms - cue.end_ms
    if config.end_pad_ms < end_overrun_ms <= config.max_end_extension_ms:
        return padded_end_ms
    if cue.end_ms - padded_end_ms > config.max_trailing_silence_ms:
        return padded_end_ms
    return cue.end_ms


def _acoustic_end_seconds(
    last_word: Word,
    last_region: SpeechRegion,
    config: BoundaryRefinementConfig,
    word_starts: list[float],
) -> float:
    """Where the cue's own voice stops: the offset of the burst holding its last word.

    A word that runs past a burst ends there only when the burst owns enough
    of the word to replace its edge, as in word repair. A burst that runs
    past the word is the same voice only while it is short and no other word
    begins inside it; a longer or shared burst is another sound, and a later
    separate burst (a breath) is never considered.
    """
    if _is_word_duration_outlier(last_word, config):
        return last_region.end
    if last_region.end <= last_word.start:
        return last_word.end
    if last_word.end >= last_region.end:
        return (
            last_region.end
            if has_sufficient_speech_overlap(last_word, last_region.start, last_region.end)
            else last_word.end
        )
    following = bisect_right(word_starts, last_word.start)
    next_word_start = word_starts[following] if following < len(word_starts) else float("inf")
    if (
        (last_region.end - last_word.end) * 1000 <= config.max_end_extension_ms
        and next_word_start >= last_region.end - _SHARED_BURST_TOLERANCE_SECONDS
    ):
        return last_region.end
    return last_word.end


def _quiet_until_seconds(
    acoustic_end: float,
    end_region: SpeechRegion,
    regions: tuple[SpeechRegion, ...],
    region_starts: list[float],
    word_starts: list[float],
) -> float:
    """When the silence after the cue's own voice ends (``inf`` when nothing follows).

    Silence lasts until the next burst or the next ASR word, whichever begins
    first. When the cue's burst keeps sounding, the next word spoken in it marks
    the limit; a burst that continues without any word is another sound and
    leaves no silence to extend into.
    """
    threshold = acoustic_end - _SHARED_BURST_TOLERANCE_SECONDS
    following_word = bisect_left(word_starts, threshold)
    next_word_start = word_starts[following_word] if following_word < len(word_starts) else float("inf")
    if end_region.end - acoustic_end > _SHARED_BURST_TOLERANCE_SECONDS:
        return next_word_start if next_word_start <= end_region.end else acoustic_end
    next_region = bisect_left(region_starts, threshold)
    while next_region < len(regions) and regions[next_region] is end_region:
        next_region += 1
    next_region_start = region_starts[next_region] if next_region < len(regions) else float("inf")
    return min(next_region_start, next_word_start)


def _word_refined_end_ms(
    cue: Cue,
    acoustic_end_seconds: float,
    profile: StyleProfile,
    config: BoundaryRefinementConfig,
) -> int:
    speech_end_ms = profile.snap_ceil(acoustic_end_seconds * 1000 + config.end_pad_ms)
    if cue.end_ms < speech_end_ms:
        return speech_end_ms
    if cue.end_ms - speech_end_ms > config.max_trailing_silence_ms:
        # Display time far beyond the cue's own voice is an ASR overrun or
        # padding; another speaker or sound in the tail cannot keep it.
        return speech_end_ms
    return cue.end_ms
