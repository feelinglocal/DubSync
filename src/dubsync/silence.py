from __future__ import annotations

import math
import sys
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

from .models import Cue, QCFlag


def silence_flags_for_cues(audio_path: Path, cues: list[Cue], threshold_dbfs: float = -45.0) -> list[QCFlag]:
    pcm, frame_rate, max_value = _read_mono_pcm(audio_path)
    flags: list[QCFlag] = []
    for cue in cues:
        start_frame = max(0, int(cue.start_ms / 1000.0 * frame_rate))
        end_frame = min(len(pcm), int(cue.end_ms / 1000.0 * frame_rate))
        if end_frame <= start_frame:
            continue
        dbfs = _dbfs(pcm[start_frame:end_frame], max_value)
        if dbfs <= threshold_dbfs:
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

    if sample_width != 2:
        raise ValueError("silence gate currently supports 16-bit PCM WAV")

    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if channels > 1:
        samples = samples[::channels]
    return samples, frame_rate, 32767


def _dbfs(samples: Sequence[int], max_value: int) -> float:
    if not samples:
        return -math.inf
    square_sum = sum(sample * sample for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    if rms == 0:
        return -math.inf
    return 20.0 * math.log10(rms / max_value)
