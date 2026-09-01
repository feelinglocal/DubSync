from __future__ import annotations

import math
import sys
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

from .models import Cue, QCFlag
from .subtitle_annotations import cue_has_spoken_text


def silence_flags_for_cues(audio_path: Path, cues: list[Cue], threshold_dbfs: float = -45.0) -> list[QCFlag]:
    flags: list[QCFlag] = []
    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        total_frames = wav.getnframes()
        _validate_pcm16(sample_width)
        if frame_rate <= 0:
            return flags
        for cue in cues:
            if not cue_has_spoken_text(cue):
                continue
            start_frame = min(total_frames, max(0, int(cue.start_ms / 1000.0 * frame_rate)))
            end_frame = min(total_frames, max(0, int(cue.end_ms / 1000.0 * frame_rate)))
            if end_frame <= start_frame:
                continue
            wav.setpos(start_frame)
            pcm = _mono_pcm16(wav.readframes(end_frame - start_frame), channels)
            if _dbfs(pcm, 32767) <= threshold_dbfs:
                flags.append(
                    QCFlag(
                        kind="cue_on_silence",
                        cue_ids=[cue.index],
                        message=f"Cue sits on audio below {threshold_dbfs:.1f} dBFS.",
                        old_text=cue.text,
                        start=cue.start_ms / 1000.0,
                        end=cue.end_ms / 1000.0,
                    )
                )
    return flags


def _read_mono_pcm(audio_path: Path) -> tuple[array, int, int]:
    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    _validate_pcm16(sample_width)
    return _mono_pcm16(raw, channels), frame_rate, 32767


def _validate_pcm16(sample_width: int) -> None:
    if sample_width != 2:
        raise ValueError("silence gate currently supports 16-bit PCM WAV")


def _mono_pcm16(raw: bytes, channels: int) -> array:
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if channels > 1:
        samples = samples[::channels]
    return samples


def _dbfs(samples: Sequence[int], max_value: int) -> float:
    if not samples:
        return -math.inf
    square_sum = sum(sample * sample for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    if rms == 0:
        return -math.inf
    return 20.0 * math.log10(rms / max_value)
