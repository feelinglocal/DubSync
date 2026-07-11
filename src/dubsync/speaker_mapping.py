from __future__ import annotations

from typing import Protocol

from .models import Cue, QCFlag
from .llm_providers import llm_adapter_from_config


class SpeakerMappingAdapter(Protocol):
    def map_speakers(self, cues: list[Cue]) -> dict[str, str]:
        raise NotImplementedError


class FixtureSpeakerMappingAdapter:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def map_speakers(self, cues: list[Cue]) -> dict[str, str]:
        present = {cue.speaker_id for cue in cues if cue.speaker_id}
        return {speaker_id: character for speaker_id, character in self.mapping.items() if speaker_id in present}


def speaker_mapping_adapter_from_config(config: dict[str, object]) -> SpeakerMappingAdapter | None:
    mapping_config = config.get("speaker_mapping", {}) if isinstance(config, dict) else {}
    if mapping_config is None:
        return None
    if not isinstance(mapping_config, dict):
        raise ValueError("providers.yaml speaker_mapping section must be a mapping")
    if not mapping_config:
        return None
    fixture = mapping_config.get("fixture")
    if isinstance(fixture, dict):
        return FixtureSpeakerMappingAdapter({str(key): str(value) for key, value in fixture.items()})
    provider = str(mapping_config.get("provider", "")).lower()
    if provider == "llm":
        adapter = llm_adapter_from_config(config, pass_name="speaker_mapping")
        if not hasattr(adapter, "map_speakers"):
            raise ValueError("configured LLM adapter does not support speaker mapping")
        return adapter
    raise ValueError("speaker_mapping.fixture must be a mapping of speaker_id to character name, or provider must be llm")


def speaker_mapping_flags(mapping: dict[str, str]) -> list[QCFlag]:
    return [
        QCFlag(
            kind="speaker_character_mapped",
            cue_ids=[],
            message=f"Speaker cluster {speaker_id} mapped to character {character}.",
            old_text=speaker_id,
            new_text=character,
            severity="info",
        )
        for speaker_id, character in sorted(mapping.items())
    ]
