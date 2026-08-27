from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Protocol

from .models import Cue, QCFlag, SpeechRegion
from .region_index import SpeechRegionIndex
from .silence import _dbfs, _mono_pcm16, _validate_pcm16


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


class EnergySpeechActivityAdapter:
    def __init__(self, threshold_dbfs: float = -45.0, window_ms: int = 100, min_region_ms: int = 100):
        self.threshold_dbfs = threshold_dbfs
        self.window_ms = window_ms
        self.min_region_ms = min_region_ms

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            total_frames = wav.getnframes()
            _validate_pcm16(sample_width)
            if total_frames <= 0 or frame_rate <= 0:
                return []
            window_frames = max(1, int(frame_rate * self.window_ms / 1000.0))
            min_region_seconds = self.min_region_ms / 1000.0
            regions: list[SpeechRegion] = []
            active_start: float | None = None
            active_end: float | None = None
            start_frame = 0

            while start_frame < total_frames:
                pcm = _mono_pcm16(wav.readframes(window_frames), channels)
                if not pcm:
                    break
                end_frame = min(total_frames, start_frame + len(pcm))
                is_active = _dbfs(pcm, 32767) > self.threshold_dbfs
                start_seconds = start_frame / frame_rate
                end_seconds = end_frame / frame_rate
                if is_active:
                    if active_start is None:
                        active_start = start_seconds
                    active_end = end_seconds
                elif active_start is not None and active_end is not None:
                    _append_region(regions, active_start, active_end, min_region_seconds)
                    active_start = None
                    active_end = None
                start_frame = end_frame

        if active_start is not None and active_end is not None:
            _append_region(regions, active_start, active_end, min_region_seconds)
        return regions


class SileroSpeechActivityAdapter:  # pragma: no cover - optional local model path
    def __init__(
        self,
        threshold_dbfs: float = -45.0,
        window_ms: int = 100,
        min_region_ms: int = 100,
        sampling_rate: int = 16000,
    ):
        self.fallback = EnergySpeechActivityAdapter(
            threshold_dbfs=threshold_dbfs,
            window_ms=window_ms,
            min_region_ms=min_region_ms,
        )
        self.sampling_rate = sampling_rate
        self.min_region_ms = min_region_ms
        self.fallback_used = False

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        self.fallback_used = False
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
            return self.fallback.detect(audio_path)
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
        return EnergySpeechActivityAdapter(
            threshold_dbfs=float(vad_config.get("threshold_dbfs", -45.0)),
            window_ms=int(vad_config.get("window_ms", 100)),
            min_region_ms=int(vad_config.get("min_region_ms", 100)),
        )
    if provider == "silero":
        return SileroSpeechActivityAdapter(
            threshold_dbfs=float(vad_config.get("threshold_dbfs", -45.0)),
            window_ms=int(vad_config.get("window_ms", 100)),
            min_region_ms=int(vad_config.get("min_region_ms", 100)),
            sampling_rate=int(vad_config.get("sampling_rate", 16000)),
        )
    raise ValueError(f"Unsupported VAD provider: {provider}")


def speech_activity_flags_for_cues(
    cues: list[Cue],
    regions: list[SpeechRegion],
    min_coverage: float = 0.2,
) -> list[QCFlag]:
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
        cue_start = cue.start_ms / 1000.0
        cue_end = cue.end_ms / 1000.0
        duration = cue_end - cue_start
        if duration <= 0:
            continue
        coverage = region_index.covered_seconds(cue_start, cue_end) / duration
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
) -> list[QCFlag]:
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
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


def dropped_line_flags_for_unmatched_cues(
    cues: list[Cue],
    unmatched_cue_ids: list[int],
    regions: list[SpeechRegion],
    min_coverage: float = 0.2,
) -> list[QCFlag]:
    unmatched = set(unmatched_cue_ids)
    flags: list[QCFlag] = []
    region_index = SpeechRegionIndex(regions)
    for cue in cues:
        if cue.index not in unmatched:
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


def _append_region(regions: list[SpeechRegion], start: float, end: float, min_region_seconds: float) -> None:
    if end - start >= min_region_seconds:
        regions.append(SpeechRegion(start=round(start, 3), end=round(end, 3), confidence=None))


def _covered_seconds(start: float, end: float, regions: list[SpeechRegion]) -> float:
    return SpeechRegionIndex(regions).covered_seconds(start, end)
