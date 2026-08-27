from __future__ import annotations

import math
import tracemalloc
import wave
from array import array

import pytest

from dubsync.models import Cue
from dubsync.silence import _dbfs, _read_mono_pcm, silence_flags_for_cues
from dubsync.vad import EnergySpeechActivityAdapter


def _write_pcm16_wav(path, *, channels: int, frame_rate: int, frames: list[tuple[int, ...]]) -> None:
    raw = b"".join(
        sample.to_bytes(2, byteorder="little", signed=True)
        for frame in frames
        for sample in frame
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(raw)


def _write_silent_pcm16_wav(path, *, duration_seconds: int, frame_rate: int = 16000) -> None:
    one_second = b"\x00\x00" * frame_rate
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        for _ in range(duration_seconds):
            wav.writeframesraw(one_second)


@pytest.mark.parametrize(
    ("channels", "frames"),
    [
        (1, [(0,), (32767,), (-32768,), (16384,)]),
        (2, [(0, 111), (32767, -222), (-32768, 333), (16384, -444)]),
    ],
)
def test_read_mono_pcm_uses_compact_16_bit_samples_and_keeps_first_channel(
    tmp_path,
    channels,
    frames,
):
    audio = tmp_path / f"{channels}-channel.wav"
    _write_pcm16_wav(audio, channels=channels, frame_rate=8000, frames=frames)

    pcm, frame_rate, max_value = _read_mono_pcm(audio)

    expected = [frame[0] for frame in frames]
    assert isinstance(pcm, array)
    assert pcm.typecode == "h"
    assert memoryview(pcm).nbytes <= len(pcm) * 2
    assert list(pcm) == expected
    assert frame_rate == 8000
    assert max_value == 32767
    assert _dbfs(pcm, max_value) == pytest.approx(_dbfs(expected, max_value))


def test_compact_pcm_keeps_silence_gate_and_energy_vad_semantics(tmp_path):
    audio = tmp_path / "stereo-activity.wav"
    quiet = [(0, 28000)] * 4
    speech = [(12000, 0)] * 4
    _write_pcm16_wav(
        audio,
        channels=2,
        frame_rate=1000,
        frames=quiet + speech,
    )

    flags = silence_flags_for_cues(
        audio,
        [
            Cue(index=1, start_ms=0, end_ms=4, lines=["quiet"]),
            Cue(index=2, start_ms=4, end_ms=8, lines=["speech"]),
        ],
        threshold_dbfs=-45.0,
    )
    regions = EnergySpeechActivityAdapter(
        threshold_dbfs=-45.0,
        window_ms=4,
        min_region_ms=4,
    ).detect(audio)

    assert [flag.cue_ids for flag in flags] == [[1]]
    assert len(regions) == 1
    assert regions[0].start == pytest.approx(0.004)
    assert regions[0].end == pytest.approx(0.008)
    assert not math.isinf(_dbfs(array("h", [12000]), 32767))


def test_pcm_consumers_stream_long_audio_with_bounded_peak_memory(tmp_path):
    audio = tmp_path / "thirty-minutes.wav"
    _write_silent_pcm16_wav(audio, duration_seconds=30 * 60)
    cues = [
        Cue(index=index + 1, start_ms=index * 60_000, end_ms=index * 60_000 + 2_000, lines=["quiet"])
        for index in range(30)
    ]

    tracemalloc.start()
    try:
        flags = silence_flags_for_cues(audio, cues, threshold_dbfs=-45.0)
        regions = EnergySpeechActivityAdapter(
            threshold_dbfs=-45.0,
            window_ms=100,
            min_region_ms=100,
        ).detect(audio)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(flags) == len(cues)
    assert regions == []
    assert peak_bytes <= 32 * 1024 * 1024


def test_streaming_silence_gate_keeps_overlapping_cues_independent(tmp_path):
    audio = tmp_path / "overlap.wav"
    _write_pcm16_wav(
        audio,
        channels=1,
        frame_rate=1000,
        frames=[(0,)] * 4 + [(12000,)] * 4,
    )

    flags = silence_flags_for_cues(
        audio,
        [
            Cue(index=1, start_ms=0, end_ms=4, lines=["quiet"]),
            Cue(index=2, start_ms=0, end_ms=8, lines=["overlap"]),
            Cue(index=3, start_ms=4, end_ms=8, lines=["speech"]),
        ],
        threshold_dbfs=-45.0,
    )

    assert [flag.cue_ids for flag in flags] == [[1]]
