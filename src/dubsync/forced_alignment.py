from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .models import Cue, ForcedAlignmentCue, QCFlag
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue, speech_text_for_alignment
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
        transcript = " ".join(speech_text_for_alignment(cue) for cue in cues).strip()
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


def usable_forced_alignments_by_cue(
    cues: list[Cue],
    alignments: list[ForcedAlignmentCue],
    *,
    protected_cue_ids: set[int] | None = None,
) -> dict[int, ForcedAlignmentCue]:
    """Share exactly the evidence accepted for retiming with confidence scoring."""
    by_cue = {alignment.cue_id: alignment for alignment in alignments}
    alignment_counts = Counter(alignment.cue_id for alignment in alignments)
    protected = protected_cue_ids or set()
    return {
        cue.index: by_cue[cue.index]
        for cue in cues
        if cue.index not in protected
        and not is_bracketed_screen_text_cue(cue)
        and alignment_counts[cue.index] == 1
        and _has_valid_interval(by_cue[cue.index])
    }


def apply_forced_alignment(
    cues: list[Cue],
    alignments: list[ForcedAlignmentCue],
    profile: StyleProfile,
    *,
    protected_cue_ids: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    updated: list[Cue] = []
    flags: list[QCFlag] = []
    min_duration_ms = int(profile.min_cue_dur * 1000)
    protected = protected_cue_ids or set()
    usable_by_cue = usable_forced_alignments_by_cue(cues, alignments, protected_cue_ids=protected)
    trusted_starts = sorted(alignment.start for alignment in usable_by_cue.values())

    for cue in cues:
        if cue.index in protected or is_bracketed_screen_text_cue(cue):
            updated.append(cue)
            continue
        alignment = usable_by_cue.get(cue.index)
        if alignment is None:
            updated.append(cue)
            flags.append(
                QCFlag(
                    kind="forced_alignment_unresolved",
                    cue_ids=[cue.index],
                    message=(
                        "Forced alignment did not provide a unique, valid acoustic interval "
                        "for this cue; its incoming text and timing were retained for review."
                    ),
                    severity="error",
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue
        start_ms = max(0, profile.snap_floor(alignment.start * 1000))
        end_ms = max(start_ms, profile.snap_ceil(alignment.end * 1000))
        if end_ms - start_ms < min_duration_ms:
            duration_end_ms = profile.snap_ceil(start_ms + min_duration_ms)
            next_start_index = bisect_right(trusted_starts, alignment.start)
            if next_start_index < len(trusted_starts):
                # Limit only display padding, never cut an acoustically aligned
                # endpoint to satisfy another cue's readability requirement.
                next_start_ms = max(0, profile.snap_floor(trusted_starts[next_start_index] * 1000))
                duration_end_ms = min(duration_end_ms, next_start_ms)
            end_ms = max(end_ms, duration_end_ms)
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
        if end_ms - start_ms < min_duration_ms:
            flags.append(
                QCFlag(
                    kind="min_duration_unattainable",
                    cue_ids=[cue.index],
                    message="Minimum display duration would cross the next aligned speech onset; acoustic timing was retained.",
                    severity="error",
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                )
            )

    return updated, flags


def _has_valid_interval(alignment: ForcedAlignmentCue) -> bool:
    return (
        math.isfinite(alignment.start)
        and math.isfinite(alignment.end)
        and alignment.end > max(0.0, alignment.start)
    )


def _cue_alignments_from_word_timestamps(cues: list[Cue], word_timestamps: list[object]) -> list[ForcedAlignmentCue]:
    cue_tokens = [alphanumeric_signature(speech_text_for_alignment(cue)) for cue in cues]
    expected_tokens = [token for tokens in cue_tokens for token in tokens]
    observed_tokens: list[str] = []
    token_rows: list[int] = []
    for row_index, row in enumerate(word_timestamps):
        # MMS returns the original text, including with romanization enabled.
        # One provider row may contain several of our lexical tokens (e.g.
        # "don't", "50%", or a Korean word); Japanese/Chinese may use chars.
        tokens = alphanumeric_signature(str(_field(row, "text") or ""))
        observed_tokens.extend(tokens)
        token_rows.extend([row_index] * len(tokens))
    if observed_tokens != expected_tokens:
        # A missing or changed word makes positional assignment unsafe. The
        # caller reports unresolved cues instead of borrowing another sentence.
        return []

    alignments: list[ForcedAlignmentCue] = []
    cursor = 0
    for cue, tokens in zip(cues, cue_tokens):
        token_count = len(tokens)
        if token_count <= 0:
            continue
        first_row = token_rows[cursor]
        last_row = token_rows[cursor + token_count - 1]
        if cursor and first_row == token_rows[cursor - 1]:
            # A provider span crossing cue text boundaries has no evidence for
            # a per-cue split inside that span.
            return []
        chunk = word_timestamps[first_row : last_row + 1]
        cursor += token_count
        try:
            starts = [_float_field(row, "start") for row in chunk]
            ends = [_float_field(row, "end") for row in chunk]
            score = _mean_score(chunk)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            any(_field(row, "start") is None or _field(row, "end") is None for row in chunk)
            or any(not math.isfinite(start) or not math.isfinite(end) or end <= start for start, end in zip(starts, ends))
            or any(right < left for left, right in zip(starts, starts[1:]))
            or any(right < left for left, right in zip(ends, ends[1:]))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            continue
        alignments.append(
            ForcedAlignmentCue(
                cue_id=cue.index,
                start=starts[0],
                end=ends[-1],
                score=score,
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
