from __future__ import annotations

import json
import math
import wave
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate
from operator import mul
from pathlib import Path
from statistics import median
from typing import Protocol

from .region_index import SpeechRegionIndex
from .silence import _dbfs, _mono_pcm16, _validate_pcm16
from .models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from .subtitle_annotations import cue_has_spoken_text

# Direct word-energy check used before a generated cue is deleted as silent.
WORD_ENERGY_THRESHOLD_DBFS = -45.0
WORD_ENERGY_MIN_ACTIVE_MS = 30
WORD_ENERGY_PAD_SECONDS = 0.05
# Another word may reach this far into a burst without owning its onset
# (the tolerance word repair applies before it snaps a phrase start).
_LEAD_EDGE_TOLERANCE_SECONDS = 0.01
# The sound between a burst onset and a later phrase start is that phrase's own
# voice only when at least this share of it is within this many dB of the speech
# that follows. A breath before the phrase is 15-25 dB quieter than the phrase.
LEAD_SPEECH_MARGIN_DB = 10.0
MIN_LEAD_SPEECH_SHARE = 0.5
# A lead that is quiet as a whole still ends in the voice when at least
# MIN_TRAILING_SPEECH_HOPS of the last TRAILING_LEAD_HOPS hops before the phrase
# start are within the margin: a fricative or breathy onset rising into its
# vowel, or a burst the detector opened before the voice (EP11 "Cinco.", "Um.").
TRAILING_LEAD_HOPS = 5
MIN_TRAILING_SPEECH_HOPS = 4

# Energy VAD defaults (measured on clean dub stems; see EnergySpeechActivityAdapter).
DEFAULT_HOP_MS = 10
DEFAULT_MIN_REGION_MS = 30
DEFAULT_MERGE_GAP_MS = 80
DEFAULT_HYSTERESIS_DB = 6.0
DEFAULT_EDGE_RISE_DB = 9.0
# A soft consonant may lead into a burst this long before the voice proper.
MAX_SOFT_ONSET_MS = 150
ADAPTIVE_FLOOR_BLOCK_MS = 3000
ADAPTIVE_LEVEL_PERCENTILE = 95.0
ADAPTIVE_FLOOR_MARGIN_DB = 12.0
ADAPTIVE_LEVEL_MARGIN_DB = 32.0
ADAPTIVE_ON_MIN_DBFS = -62.0
ADAPTIVE_ON_MAX_DBFS = -40.0
# Frames at or below this level are digital silence and say nothing about the
# recording's noise floor or speech level.
DIGITAL_SILENCE_DBFS = -110.0
_HISTOGRAM_BIN_DB = 0.25
_STREAM_CHUNK_HOPS = 100
_FULL_SCALE_DB = 20.0 * math.log10(32767)


class SpeechActivityAdapter(Protocol):
    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        raise NotImplementedError


class FixtureSpeechActivityAdapter:
    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        del audio_path
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        rows = payload.get("regions", payload)
        return [SpeechRegion.model_validate(row) for row in rows]


@dataclass(frozen=True)
class SpeechLevels:
    """Level track of one recording: the dBFS values the energy VAD measured in its single pass.

    Value ``i`` describes the ``hop_seconds`` long slice that begins at
    ``i * hop_seconds + offset_seconds``, the grid the speech regions lie on.
    """

    levels: array
    hop_seconds: float
    offset_seconds: float = 0.0

    def between(self, start: float, end: float) -> array:
        first = max(0, round((start - self.offset_seconds) / self.hop_seconds))
        last = min(len(self.levels), round((end - self.offset_seconds) / self.hop_seconds))
        return self.levels[first:last]

    def lead_is_speech(self, onset: float, start: float, burst_end: float) -> bool:
        """Whether a burst sounds like the phrase itself from ``onset`` up to the phrase start at ``start``."""
        lead = self.between(onset, start)
        body = self.between(start, burst_end)
        if not lead or not body:
            return False
        floor = median(body) - LEAD_SPEECH_MARGIN_DB
        return sum(level >= floor for level in lead) >= MIN_LEAD_SPEECH_SHARE * len(lead)

    def lead_ends_in_speech(self, onset: float, start: float, burst_end: float) -> bool:
        """Whether the phrase's own voice is already sounding right before its start at ``start``.

        Weaker than :meth:`lead_is_speech`, which asks the whole lead to sound
        like the phrase: here only the hops immediately before the start count,
        so a start inside a rising onset is recognised although the lead as a
        whole is quiet. Right for a review flag, not for a move onto the onset.
        """
        lead = self.between(onset, start)
        body = self.between(start, burst_end)
        if not lead or not body:
            return False
        floor = median(body) - LEAD_SPEECH_MARGIN_DB
        trailing = lead[-TRAILING_LEAD_HOPS:]
        return sum(level >= floor for level in trailing) >= min(len(trailing), MIN_TRAILING_SPEECH_HOPS)


@dataclass(frozen=True)
class EnergyThresholds:
    """Levels used for one file: detect above ``on``, bridge above ``off``, place edges at ``edge``."""

    on_dbfs: float
    off_dbfs: float
    edge_dbfs: float
    adaptive: bool
    floor_dbfs: float | None = None
    level_dbfs: float | None = None


class EnergySpeechActivityAdapter:
    """Streaming energy VAD that reports speech bursts at 10 ms resolution.

    By default the thresholds follow the file: the on-threshold sits 32 dB under
    the loud speech level (95th percentile of the sounding frames) and at least
    12 dB over the noise floor (typical quietest frame per 3 s). Activity
    continues while the level stays within ``hysteresis_db`` below it, silences
    shorter than ``merge_gap_ms`` are bridged and activity shorter than
    ``min_region_ms`` is dropped. A region is the voiced part of that activity:
    the frames ``edge_rise_db`` above the on-threshold, so it ends with the
    voice rather than with its decay or a breath, plus a soft consonant that
    leads straight into it. Activity that never gets that loud is kept as
    quiet speech.

    ``threshold_dbfs`` forces an absolute on-threshold (both edges use it) and
    ``window_ms`` restores non-overlapping analysis windows of that size, which
    keeps configurations written for the first VAD working unchanged.
    """

    def __init__(
        self,
        threshold_dbfs: float | None = None,
        window_ms: int | None = None,
        min_region_ms: int | None = None,
        *,
        hysteresis_db: float = DEFAULT_HYSTERESIS_DB,
        merge_gap_ms: int = DEFAULT_MERGE_GAP_MS,
        edge_rise_db: float = DEFAULT_EDGE_RISE_DB,
    ):
        if window_ms is not None and window_ms <= 0:
            raise ValueError("vad.window_ms must be positive")
        if min_region_ms is not None and min_region_ms < 0:
            raise ValueError("vad.min_region_ms must be non-negative")
        if hysteresis_db < 0 or merge_gap_ms < 0 or edge_rise_db < 0:
            raise ValueError("vad hysteresis, merge gap and edge rise must be non-negative")
        self.threshold_dbfs = threshold_dbfs
        self.window_ms = window_ms
        self.min_region_ms = DEFAULT_MIN_REGION_MS if min_region_ms is None else min_region_ms
        self.hysteresis_db = hysteresis_db
        self.merge_gap_ms = merge_gap_ms
        self.edge_rise_db = edge_rise_db
        self.last_thresholds: EnergyThresholds | None = None
        self.last_levels: SpeechLevels | None = None

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        self.last_thresholds = None
        self.last_levels = None
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            total_frames = wav.getnframes()
            _validate_pcm16(sample_width)
            if total_frames <= 0 or frame_rate <= 0:
                return []
            legacy_windows = self.window_ms is not None
            hop_frames = max(1, int(frame_rate * (self.window_ms if legacy_windows else DEFAULT_HOP_MS) / 1000.0))
            levels = _hop_levels(wav, channels, hop_frames, overlap=not legacy_windows)

        hop_ms = self.window_ms if legacy_windows else DEFAULT_HOP_MS
        thresholds = self._thresholds(levels, max(1, round(ADAPTIVE_FLOOR_BLOCK_MS / hop_ms)))
        self.last_thresholds = thresholds
        if thresholds is None:
            return []
        # Frame i of the overlapping analysis describes the 10 ms slice centred
        # in its 20 ms window; legacy windows describe themselves.
        offset_frames = 0 if legacy_windows else hop_frames // 2
        self.last_levels = SpeechLevels(levels, hop_frames / frame_rate, offset_frames / frame_rate)
        regions: list[SpeechRegion] = []
        pending: tuple[int, int] | None = None
        bursts = _active_runs(
            levels,
            thresholds,
            split_gap_hops=max(1, math.ceil(self.merge_gap_ms / hop_ms)),
            max_soft_onset_hops=round(MAX_SOFT_ONSET_MS / hop_ms),
        )
        for first, last in bursts:
            start_frame = 0 if first == 0 else first * hop_frames + offset_frames
            end_frame = total_frames if last == len(levels) - 1 else (last + 1) * hop_frames + offset_frames
            end_frame = min(total_frames, end_frame)
            if pending is not None and (start_frame - pending[1]) * 1000 < self.merge_gap_ms * frame_rate:
                pending = (pending[0], end_frame)
                continue
            if pending is not None:
                _append_region(regions, pending[0], pending[1], frame_rate, self.min_region_ms)
            pending = (start_frame, end_frame)
        if pending is not None:
            _append_region(regions, pending[0], pending[1], frame_rate, self.min_region_ms)
        return regions

    def _thresholds(self, levels: array, floor_block_hops: int) -> EnergyThresholds | None:
        if self.threshold_dbfs is not None:
            on = float(self.threshold_dbfs)
            return EnergyThresholds(on_dbfs=on, off_dbfs=on - self.hysteresis_db, edge_dbfs=on, adaptive=False)
        level = _signal_level_percentile(levels, ADAPTIVE_LEVEL_PERCENTILE)
        if level is None:
            return None
        floor = _noise_floor(levels, floor_block_hops)
        on = min(
            ADAPTIVE_ON_MAX_DBFS,
            max(ADAPTIVE_ON_MIN_DBFS, floor + ADAPTIVE_FLOOR_MARGIN_DB, level - ADAPTIVE_LEVEL_MARGIN_DB),
        )
        return EnergyThresholds(
            on_dbfs=on,
            off_dbfs=on - self.hysteresis_db,
            edge_dbfs=on + self.edge_rise_db,
            adaptive=True,
            floor_dbfs=floor,
            level_dbfs=level,
        )


class SileroSpeechActivityAdapter:  # pragma: no cover - optional local model path
    def __init__(
        self,
        threshold_dbfs: float | None = None,
        window_ms: int | None = None,
        min_region_ms: int | None = None,
        sampling_rate: int = 16000,
        **energy_options: float,
    ):
        self.fallback = EnergySpeechActivityAdapter(
            threshold_dbfs=threshold_dbfs,
            window_ms=window_ms,
            min_region_ms=min_region_ms,
            **energy_options,
        )
        self.sampling_rate = sampling_rate
        self.min_region_ms = 100 if min_region_ms is None else min_region_ms
        self.fallback_used = False
        self.last_levels: SpeechLevels | None = None

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        self.fallback_used = False
        self.last_levels = None
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False,
            )
            get_speech_timestamps, _, read_audio, _, _ = utils
            waveform = read_audio(str(audio_path), sampling_rate=self.sampling_rate)
            raw_regions = get_speech_timestamps(
                waveform,
                model,
                sampling_rate=self.sampling_rate,
                min_speech_duration_ms=self.min_region_ms,
                return_seconds=True,
            )
        except Exception:
            self.fallback_used = True
            regions = self.fallback.detect(audio_path)
            self.last_levels = self.fallback.last_levels
            return regions
        return [
            SpeechRegion(
                start=round(float(region["start"]), 3),
                end=round(float(region["end"]), 3),
                confidence=None,
            )
            for region in raw_regions
            if float(region["end"]) > float(region["start"])
        ]


def speech_activity_adapter_from_config(config: dict[str, object]) -> SpeechActivityAdapter | None:
    vad_config = config.get("vad", {}) if isinstance(config, dict) else {}
    if vad_config is None:
        return None
    if not isinstance(vad_config, dict):
        raise ValueError("providers.yaml vad section must be a mapping")
    if not vad_config:
        return None
    fixture_path = vad_config.get("fixture_path")
    if fixture_path:
        return FixtureSpeechActivityAdapter(Path(str(fixture_path)))
    provider = str(vad_config.get("provider", "energy")).lower()
    if provider == "energy":
        return EnergySpeechActivityAdapter(**_energy_options(vad_config))
    if provider == "silero":
        return SileroSpeechActivityAdapter(
            sampling_rate=int(vad_config.get("sampling_rate", 16000)),
            **_energy_options(vad_config),
        )
    raise ValueError(f"Unsupported VAD provider: {provider}")


def _energy_options(vad_config: dict[str, object]) -> dict[str, object]:
    """Read energy VAD keys; an absent key selects the adaptive 10 ms default."""
    options: dict[str, object] = {}
    for key, convert in (
        ("threshold_dbfs", float),
        ("window_ms", int),
        ("min_region_ms", int),
        ("hysteresis_db", float),
        ("merge_gap_ms", int),
        ("edge_rise_db", float),
    ):
        value = vad_config.get(key)
        if value is None:
            continue
        try:
            options[key] = convert(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vad.{key} must be numeric") from exc
    return options


def speech_activity_flags_for_cues(
    cues: list[Cue],
    regions: list[SpeechRegion],
    min_coverage: float = 0.2,
    *,
    min_cue_duration_ms: int = 0,
) -> list[QCFlag]:
    """Flag cues that show little speech activity.

    A cue no longer than ``min_cue_duration_ms`` was only kept on screen for
    readability: a short interjection inside it is real speech even when it
    covers less than ``min_coverage`` of the padded display time.
    """
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
        if not cue_has_spoken_text(cue):
            continue
        cue_start = cue.start_ms / 1000.0
        cue_end = cue.end_ms / 1000.0
        duration = cue_end - cue_start
        if duration <= 0:
            continue
        covered = region_index.covered_seconds(cue_start, cue_end)
        coverage = covered / duration
        if cue.duration_ms <= min_cue_duration_ms and covered * 1000 >= DEFAULT_MIN_REGION_MS:
            continue
        if coverage < min_coverage:
            flags.append(
                QCFlag(
                    kind="cue_without_speech_activity",
                    cue_ids=[cue.index],
                    message=f"Cue overlaps speech activity for only {coverage:.0%} of its duration.",
                    confidence=round(coverage, 3),
                    old_text=cue.text,
                    start=cue_start,
                    end=cue_end,
                )
            )
    return flags


def trailing_silence_flags_for_cues(
    cues: list[Cue],
    regions: list[SpeechRegion],
    max_trailing_silence_ms: int = 300,
    *,
    min_cue_duration_ms: int = 0,
) -> list[QCFlag]:
    """Flag cues that stay visible long after their speech.

    A cue no longer than ``min_cue_duration_ms`` is exempt: its tail is the
    minimum display time of a short utterance, not an overrun.
    """
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
        if not cue_has_spoken_text(cue) or cue.duration_ms <= min_cue_duration_ms:
            continue
        cue_start = cue.start_ms / 1000.0
        cue_end = cue.end_ms / 1000.0
        overlapping = region_index.overlapping(cue_start, cue_end)
        if not overlapping:
            continue
        last_speech_end = min(cue_end, max(region.end for region in overlapping))
        trailing_silence_ms = round((cue_end - last_speech_end) * 1000)
        if trailing_silence_ms <= max_trailing_silence_ms:
            continue
        flags.append(
            QCFlag(
                kind="cue_with_excessive_trailing_silence",
                cue_ids=[cue.index],
                message=(
                    f"Cue remains visible for {trailing_silence_ms} ms after "
                    f"the last detected speech activity; limit is {max_trailing_silence_ms} ms."
                ),
                old_text=cue.text,
                start=last_speech_end,
                end=cue_end,
            )
        )
    return flags


def late_start_flags_for_cues(
    cues: list[Cue],
    regions: list[SpeechRegion],
    words: list[Word],
    cue_word_indices: Mapping[int, list[int]],
    spoken_spans: Mapping[int, tuple[int, int]],
    *,
    max_onset_lead_ms: float,
    frame_ms: float,
    end_pad_ms: float = 0.0,
    cue_ids: set[int] | None = None,
    excluded_cue_ids: set[int] | None = None,
    levels: SpeechLevels | None = None,
) -> list[QCFlag]:
    """Flag cues that start after their own speech burst has begun.

    A cue starts on its first owned word (``spoken_spans`` holds that word's
    onset in ms). When the ASR places the word more than ``max_onset_lead_ms``
    after the onset of the burst that contains it, word repair leaves it there
    unless the recording shows that its phrase starts lag, and the cue begins
    inside its own speech. That is reported when the cue still starts that late
    (frame flooring allowed) and the speech before it belongs to nobody else:
    no other word reaches into it and no other spoken cue is on screen there
    for longer than its display padding (``end_pad_ms`` plus one frame). A
    punctuation-only ASR token is not a word here: its duration is no speech.
    There is no upper bound on the lead: a long speech-level lead that nobody
    owns is at least unrecognised sound the customer should hear. The only
    lead not reported is one that ``levels`` shows to be quieter than the
    phrase as a whole and still quiet right before the cue start (a breath,
    then the word on the cue start); a lead that ends in the voice is the
    cue's own onset however quiet it began. ``cues`` is the delivered list;
    ``cue_ids`` limits which of them are checked.
    """
    region_index = SpeechRegionIndex(regions)
    if not region_index.regions:
        return []
    excluded = excluded_cue_ids or set()
    spoken_cues = [cue for cue in cues if cue_has_spoken_text(cue)]
    timed_words = sorted(
        (word.start, word.end, index) for index, word in enumerate(words)
        if math.isfinite(word.start) and math.isfinite(word.end)
        and any(character.isalnum() for character in word.text)
    )
    word_starts = [start for start, _, _ in timed_words]
    latest_word_ends = list(accumulate((end for _, end, _ in timed_words), max))
    flags: list[QCFlag] = []
    for cue in spoken_cues:
        span = spoken_spans.get(cue.index)
        if span is None or cue.index in excluded or (cue_ids is not None and cue.index not in cue_ids):
            continue
        burst = region_index.first_containing(span[0] / 1000.0)
        if burst is None:
            continue
        onset_ms = burst[1].start * 1000.0
        lead_ms = cue.start_ms - onset_ms
        if span[0] - onset_ms <= max_onset_lead_ms or lead_ms <= max_onset_lead_ms - frame_ms:
            continue
        if levels is not None and not (
            levels.lead_is_speech(burst[1].start, span[0] / 1000.0, burst[1].end)
            or levels.lead_ends_in_speech(burst[1].start, span[0] / 1000.0, burst[1].end)
        ):
            continue
        own = set(cue_word_indices.get(cue.index, ()))
        # A neighbouring word may touch the onset by this much without owning it, as in word repair.
        lead_start = burst[1].start + _LEAD_EDGE_TOLERANCE_SECONDS
        in_lead = timed_words[
            bisect_right(latest_word_ends, lead_start):bisect_left(word_starts, cue.start_ms / 1000.0)
        ]
        if any(index not in own and end > lead_start for _, end, index in in_lead):
            continue
        if any(
            other.index != cue.index
            and min(other.end_ms, cue.start_ms) - max(other.start_ms, onset_ms) > end_pad_ms + frame_ms
            for other in spoken_cues
        ):
            continue
        flags.append(
            QCFlag(
                kind="cue_starts_after_speech_onset",
                cue_ids=[cue.index],
                message=(
                    f"Cue starts {round(lead_ms)} ms after the detected speech onset; "
                    "no other cue or recognised word covers that speech."
                ),
                old_text=cue.text,
                start=burst[1].start,
                end=cue.start_ms / 1000.0,
            )
        )
    return flags


def dropped_line_flags_for_unmatched_cues(
    cues: list[Cue],
    unmatched_cue_ids: list[int],
    regions: list[SpeechRegion],
    min_coverage: float = 0.2,
    *,
    cue_word_indices: dict[int, list[int]] | None = None,
) -> list[QCFlag]:
    """Flag unmatched source cues that sit on silence.

    ``unmatched_cue_ids`` is the aligner's list. A cue that received ASR words
    afterwards (``cue_word_indices``) was spoken somewhere else than its source
    time and is not a dropped line.
    """
    unmatched = {
        cue_id for cue_id in unmatched_cue_ids
        if not (cue_word_indices or {}).get(cue_id)
    }
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
        if cue.index not in unmatched:
            continue
        if not cue_has_spoken_text(cue):
            continue
        cue_start = cue.start_ms / 1000.0
        cue_end = cue.end_ms / 1000.0
        duration = cue_end - cue_start
        if duration <= 0:
            continue
        coverage = region_index.covered_seconds(cue_start, cue_end) / duration
        if coverage < min_coverage:
            flags.append(
                QCFlag(
                    kind="dropped_line_candidate",
                    cue_ids=[cue.index],
                    message=f"Unmatched source cue overlaps speech activity for only {coverage:.0%}; actor may have dropped this line.",
                    confidence=round(coverage, 3),
                    old_text=cue.text,
                    start=cue_start,
                    end=cue_end,
                )
            )
    return flags


def min_coverage_from_config(config: dict[str, object]) -> float:
    vad_config = config.get("vad", {}) if isinstance(config, dict) else {}
    if not isinstance(vad_config, dict):
        return 0.2
    return float(vad_config.get("min_coverage", 0.2))


def cue_ids_with_audible_words(
    audio_path: Path,
    activity_flags: list[QCFlag],
    words: list[Word],
    alignment: AlignmentResult,
    *,
    threshold_dbfs: float = WORD_ENERGY_THRESHOLD_DBFS,
    min_active_ms: int = WORD_ENERGY_MIN_ACTIVE_MS,
) -> set[int]:
    """Find zero-coverage cues whose own ASR words still sit on audible energy.

    Speech regions can miss a short interjection. Before a generated cue is
    deleted as silent, its word intervals are measured directly at 10 ms
    resolution. Undecodable audio yields no evidence, never an exception.
    """
    candidates = {
        cue_id
        for flag in activity_flags
        if flag.kind == "cue_without_speech_activity" and flag.confidence is not None and flag.confidence <= 0.0
        for cue_id in flag.cue_ids
        if alignment.cue_word_indices.get(cue_id)
    }
    if not candidates:
        return set()
    audible: set[int] = set()
    try:
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            frame_rate = wav.getframerate()
            total_frames = wav.getnframes()
            if wav.getsampwidth() != 2 or frame_rate <= 0:
                return set()
            hop_frames = max(1, frame_rate // 100)
            needed_hops = max(1, -(-min_active_ms * frame_rate // (1000 * hop_frames)))
            for cue_id in sorted(candidates):
                for word_index in alignment.cue_word_indices[cue_id]:
                    if not 0 <= word_index < len(words):
                        continue
                    word = words[word_index]
                    first = max(0, int((word.start - WORD_ENERGY_PAD_SECONDS) * frame_rate))
                    last = min(total_frames, int((word.end + WORD_ENERGY_PAD_SECONDS) * frame_rate) + 1)
                    if last <= first:
                        continue
                    wav.setpos(first)
                    pcm = _mono_pcm16(wav.readframes(last - first), channels)
                    active_hops = sum(
                        1
                        for offset in range(0, len(pcm), hop_frames)
                        if _dbfs(pcm[offset:offset + hop_frames], 32767) > threshold_dbfs
                    )
                    if active_hops >= needed_hops:
                        audible.add(cue_id)
                        break
    except (wave.Error, EOFError, OSError):
        return set()
    return audible


def _hop_levels(wav: wave.Wave_read, channels: int, hop_frames: int, *, overlap: bool) -> array:
    """Stream the file once and return one dBFS level per hop.

    With ``overlap`` each level is the RMS of two consecutive hops (a 20 ms
    window at the default 10 ms hop). Only the compact level track is kept, so
    memory stays bounded for multi-hour recordings.
    """
    levels = array("f")
    previous: tuple[int, int] | None = None
    while True:
        pcm = _mono_pcm16(wav.readframes(hop_frames * _STREAM_CHUNK_HOPS), channels)
        if not pcm:
            break
        for offset in range(0, len(pcm), hop_frames):
            block = pcm[offset:offset + hop_frames]
            current = (sum(map(mul, block, block)), len(block))
            if not overlap:
                levels.append(_energy_dbfs(*current))
            elif previous is not None:
                levels.append(_energy_dbfs(previous[0] + current[0], previous[1] + current[1]))
            previous = current
    if overlap and previous is not None:
        levels.append(_energy_dbfs(*previous))
    return levels


def _energy_dbfs(square_sum: int, sample_count: int) -> float:
    if square_sum <= 0 or sample_count <= 0:
        return -math.inf
    return 10.0 * math.log10(square_sum / sample_count) - _FULL_SCALE_DB


def _signal_level_percentile(levels: array, percentile: float) -> float | None:
    """Percentile of the frames that carry any signal, from a fixed-size histogram."""
    bin_count = int(-DIGITAL_SILENCE_DBFS / _HISTOGRAM_BIN_DB)
    histogram = [0] * bin_count
    total = 0
    for level in levels:
        if level <= DIGITAL_SILENCE_DBFS:
            continue
        histogram[min(bin_count - 1, int((level - DIGITAL_SILENCE_DBFS) / _HISTOGRAM_BIN_DB))] += 1
        total += 1
    if total == 0:
        return None
    needed = max(1, math.ceil(total * percentile / 100.0))
    seen = 0
    for index, count in enumerate(histogram):
        seen += count
        if seen >= needed:
            return DIGITAL_SILENCE_DBFS + (index + 0.5) * _HISTOGRAM_BIN_DB
    return 0.0


def _noise_floor(levels: array, block_hops: int) -> float:
    """Typical level the recording falls to between sounds.

    The quietest frame of each ~3 s block follows a changing bed, and the
    median of those minima is unaffected by how much of the file is speech.
    Digital silence counts as the lowest measurable floor.
    """
    minima = sorted(
        max(DIGITAL_SILENCE_DBFS, min(levels[offset:offset + block_hops]))
        for offset in range(0, len(levels), block_hops)
    )
    if not minima:
        return DIGITAL_SILENCE_DBFS
    return minima[len(minima) // 2]


def _active_runs(
    levels: array,
    thresholds: EnergyThresholds,
    *,
    split_gap_hops: int,
    max_soft_onset_hops: int,
) -> Iterator[tuple[int, int]]:
    """Yield inclusive (first, last) frame indices of each speech burst.

    Hysteresis groups the frames above the off-threshold into runs. Inside a
    run the voiced bursts are the frames above the edge level; a quieter
    stretch of at least ``split_gap_hops`` separates two bursts, so a breath
    that follows a word without a real silence does not prolong it. A burst
    begins up to ``max_soft_onset_hops`` early when frames above the
    on-threshold lead straight into it (a soft consonant). A run that never
    reaches the edge level but does reach the on-threshold is quiet speech and
    is reported from its first to its last on-frame.
    """
    on, off, edge = thresholds.on_dbfs, thresholds.off_dbfs, thresholds.edge_dbfs
    run_start = -1
    for index, level in enumerate(levels):
        if level > off:
            if run_start < 0:
                run_start = index
        elif run_start >= 0:
            yield from _bursts_in_run(levels, run_start, index, on, edge, split_gap_hops, max_soft_onset_hops)
            run_start = -1
    if run_start >= 0:
        yield from _bursts_in_run(levels, run_start, len(levels), on, edge, split_gap_hops, max_soft_onset_hops)


def _bursts_in_run(
    levels: array,
    first: int,
    stop: int,
    on: float,
    edge: float,
    split_gap_hops: int,
    max_soft_onset_hops: int,
) -> Iterator[tuple[int, int]]:
    loud = [index for index in range(first, stop) if levels[index] > edge]
    if not loud:
        audible = [index for index in range(first, stop) if levels[index] > on]
        if audible:
            yield audible[0], audible[-1]
        return
    earliest_onset = first
    burst_start = previous = loud[0]
    for index in [*loud[1:], stop + split_gap_hops]:
        if index - previous - 1 < split_gap_hops:
            previous = index
            continue
        onset = burst_start
        while (
            onset > earliest_onset
            and burst_start - onset < max_soft_onset_hops
            and levels[onset - 1] > on
        ):
            onset -= 1
        yield onset, previous
        # The next burst must stay a full gap away or the two would be merged.
        earliest_onset = previous + 1 + split_gap_hops
        burst_start = previous = index


def _append_region(
    regions: list[SpeechRegion], start_frame: int, end_frame: int, frame_rate: int, min_region_ms: int,
) -> None:
    if (end_frame - start_frame) * 1000 >= min_region_ms * frame_rate:
        regions.append(
            SpeechRegion(start=round(start_frame / frame_rate, 3), end=round(end_frame / frame_rate, 3), confidence=None)
        )


def _covered_seconds(start: float, end: float, regions: list[SpeechRegion]) -> float:
    return SpeechRegionIndex(regions).covered_seconds(start, end)
