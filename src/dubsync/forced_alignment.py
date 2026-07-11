from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .models import Cue, ForcedAlignmentCue, QCFlag
from .style_profile import StyleProfile
from .tokenize import alphanumeric_signature


class ForcedAlignmentAdapter(Protocol):
    def align(self, audio_path: Path, cues: list[Cue]) -> list[ForcedAlignmentCue]:
        raise NotImplementedError


class FixtureForcedAlignmentAdapter:
    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def align(self, audio_path: Path, cues: list[Cue]) -> list[ForcedAlignmentCue]:
        del audio_path, cues
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        rows = payload.get("cues", payload)
        return [ForcedAlignmentCue.model_validate(row) for row in rows]


class MMSForcedAlignmentAdapter:
    def __init__(
        self,
        language: str | None = None,
        romanize: bool = True,
        batch_size: int = 4,
        device: str | None = None,
    ):
        self.language = language or "eng"
        self.romanize = romanize
        self.batch_size = batch_size
        self.device = device

    def align(self, audio_path: Path, cues: list[Cue]) -> list[ForcedAlignmentCue]:
        transcript = " ".join(cue.plain_text for cue in cues).strip()
        if not transcript:
            return []
        try:
            import torch
            from ctc_forced_aligner import (
                generate_emissions,
                get_alignments,
                get_spans,
                load_alignment_model,
                load_audio,
                postprocess_results,
                preprocess_text,
            )
        except ImportError as exc:
            raise RuntimeError("Install dubsync[precision] to use MMS forced alignment.") from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=dtype)
        audio_waveform = load_audio(str(audio_path), alignment_model.dtype, alignment_model.device)
        emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=self.batch_size)
        tokens_starred, text_starred = preprocess_text(transcript, romanize=self.romanize, language=self.language)
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        return _cue_alignments_from_word_timestamps(cues, list(word_timestamps))


def forced_alignment_adapter_from_config(config: dict[str, object]) -> ForcedAlignmentAdapter | None:
    fa_config = config.get("forced_alignment", {}) if isinstance(config, dict) else {}
    if fa_config is None:
        return None
    if not isinstance(fa_config, dict):
        raise ValueError("providers.yaml forced_alignment section must be a mapping")
    if not fa_config:
        return None
    fixture_path = fa_config.get("fixture_path")
    if fixture_path:
        return FixtureForcedAlignmentAdapter(Path(str(fixture_path)))
    if fa_config.get("provider", "mms") == "mms":
        language = fa_config.get("language")
        romanize = bool(fa_config.get("romanize", True))
        batch_size = int(fa_config.get("batch_size", 4))
        device = fa_config.get("device")
        return MMSForcedAlignmentAdapter(
            str(language) if language else None,
            romanize=romanize,
            batch_size=batch_size,
            device=str(device) if device else None,
        )
    raise ValueError(f"Unsupported forced alignment provider: {fa_config.get('provider')}")


def apply_forced_alignment(
    cues: list[Cue],
    alignments: list[ForcedAlignmentCue],
    profile: StyleProfile,
) -> tuple[list[Cue], list[QCFlag]]:
    if not alignments:
        return cues, []

    by_cue = {alignment.cue_id: alignment for alignment in alignments}
    updated: list[Cue] = []
    flags: list[QCFlag] = []
    min_duration_ms = int(profile.min_cue_dur * 1000)

    for cue in cues:
        alignment = by_cue.get(cue.index)
        if alignment is None:
            updated.append(cue)
            continue
        start_ms = max(0, profile.snap_floor(alignment.start * 1000))
        end_ms = max(start_ms, profile.snap_ceil(alignment.end * 1000))
        if end_ms - start_ms < min_duration_ms:
            end_ms = profile.snap_ceil(start_ms + min_duration_ms)
        refined = cue.with_timing(start_ms, end_ms)
        updated.append(refined)
        if refined.start_ms != cue.start_ms or refined.end_ms != cue.end_ms:
            flags.append(
                QCFlag(
                    kind="forced_alignment_refined",
                    cue_ids=[cue.index],
                    message="Forced alignment refined cue timing.",
                    confidence=alignment.score,
                    start=refined.start_ms / 1000,
                    end=refined.end_ms / 1000,
                )
            )

    return updated, flags


def _cue_alignments_from_word_timestamps(cues: list[Cue], word_timestamps: list[object]) -> list[ForcedAlignmentCue]:
    alignments: list[ForcedAlignmentCue] = []
    cursor = 0
    for cue in cues:
        token_count = len(alphanumeric_signature(cue.plain_text))
        if token_count <= 0:
            continue
        chunk = word_timestamps[cursor : cursor + token_count]
        cursor += token_count
        if not chunk:
            continue
        alignments.append(
            ForcedAlignmentCue(
                cue_id=cue.index,
                start=_float_field(chunk[0], "start"),
                end=_float_field(chunk[-1], "end"),
                score=_mean_score(chunk),
            )
        )
    return alignments


def _mean_score(segments: list[object]) -> float:
    scores = [_float_field(segment, "score") for segment in segments if _field(segment, "score") is not None]
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _float_field(source: object, name: str) -> float:
    value = _field(source, name)
    if value is None:
        return 0.0
    return float(value)


def _field(source: object, name: str) -> Any | None:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)
