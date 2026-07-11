from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .models import Cue, OverlapRegion, QCFlag
from .providers import ProviderError


class OverlapDetectionAdapter(Protocol):
    def detect(self, audio_path: Path) -> list[OverlapRegion]:
        raise NotImplementedError


class FixtureOverlapDetectionAdapter:
    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def detect(self, audio_path: Path) -> list[OverlapRegion]:
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return [OverlapRegion.model_validate(item) for item in payload.get("regions", [])]


class PyannoteOverlapDetectionAdapter:  # pragma: no cover - optional local dependency path
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        token: str | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.token = token
        self.device = device

    def detect(self, audio_path: Path) -> list[OverlapRegion]:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ProviderError("pyannote.audio is required for overlap_detection.provider=pyannote.") from exc

        pipeline = Pipeline.from_pretrained(self.model, token=self.token)
        if self.device:
            try:
                import torch
            except ImportError as exc:
                raise ProviderError("torch is required to set a pyannote processing device.") from exc
            pipeline.to(torch.device(self.device))

        output = pipeline(str(audio_path))
        speaker_turns = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, speaker in output.speaker_diarization
        ]
        return _overlap_regions_from_speaker_turns(speaker_turns)


def overlap_detection_adapter_from_config(config: dict[str, object]) -> OverlapDetectionAdapter | None:
    detection_config = config.get("overlap_detection", {}) if isinstance(config, dict) else {}
    if detection_config is None:
        return None
    if not isinstance(detection_config, dict):
        raise ValueError("providers.yaml overlap_detection section must be a mapping")
    if not detection_config:
        return None

    fixture_path = detection_config.get("fixture_path")
    if fixture_path:
        return FixtureOverlapDetectionAdapter(Path(str(fixture_path)))

    provider = str(detection_config.get("provider", "pyannote")).lower()
    if provider == "pyannote":
        token = detection_config.get("token") or os.getenv("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
        return PyannoteOverlapDetectionAdapter(
            model=str(detection_config.get("model", "pyannote/speaker-diarization-community-1")),
            token=str(token) if token else None,
            device=str(detection_config["device"]) if "device" in detection_config else None,
        )

    raise ValueError(f"Unsupported overlap detection provider: {provider}")


def overlap_flags_for_regions(cues: list[Cue], regions: list[OverlapRegion]) -> list[QCFlag]:
    flags: list[QCFlag] = []
    for region in regions:
        cue_ids = [
            cue.index
            for cue in cues
            if _intersects(cue.start_ms / 1000.0, cue.end_ms / 1000.0, region.start, region.end)
        ]
        if not cue_ids:
            continue
        flags.append(
            QCFlag(
                kind="overlap_detected",
                cue_ids=cue_ids,
                message="Overlap detector found simultaneous speech in this time range.",
                confidence=region.confidence,
                start=region.start,
                end=region.end,
                old_text="\n".join(cue.text for cue in cues if cue.index in cue_ids),
            )
        )
    return flags


def _overlap_regions_from_speaker_turns(turns: list[tuple[float, float, str]]) -> list[OverlapRegion]:
    regions: list[OverlapRegion] = []
    sorted_turns = sorted(turns, key=lambda item: (item[0], item[1], item[2]))
    for left_index, left in enumerate(sorted_turns):
        for right in sorted_turns[left_index + 1 :]:
            if right[0] >= left[1]:
                break
            if left[2] == right[2]:
                continue
            start = max(left[0], right[0])
            end = min(left[1], right[1])
            if end > start:
                regions.append(OverlapRegion(start=start, end=end))
    return _merge_regions(regions)


def _merge_regions(regions: list[OverlapRegion]) -> list[OverlapRegion]:
    merged: list[OverlapRegion] = []
    for region in sorted(regions, key=lambda item: (item.start, item.end)):
        if not merged or region.start > merged[-1].end:
            merged.append(region)
            continue
        previous = merged[-1]
        merged[-1] = previous.model_copy(update={"end": max(previous.end, region.end)})
    return merged


def _intersects(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_start < right_end and right_start < left_end
