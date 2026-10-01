from __future__ import annotations

import json
import tracemalloc
import wave
from array import array

import pytest

from dubsync import pipeline
from dubsync.cost import CostMeter
from dubsync.models import AlignmentResult, Cue, QCFlag, Word
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.vad import (
    EnergySpeechActivityAdapter,
    cue_ids_with_audible_words,
    speech_activity_adapter_from_config,
)


def _write_wav(path, *, frame_rate: int, duration_seconds: float, bursts: list[tuple[float, float, int]]) -> None:
    samples = array("h", [0]) * int(round(duration_seconds * frame_rate))
    for start, end, amplitude in bursts:
        for index in range(int(round(start * frame_rate)), int(round(end * frame_rate))):
            # A square wave keeps the RMS at the amplitude without sine tables.
            samples[index] = amplitude if index % 2 else -amplitude
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(frame_rate)
        stream.writeframes(samples.tobytes())


# --- Task 1: single-window speech bursts and generated ad-libs -------------------------------


@pytest.mark.parametrize("window_index", [2, 3, 7, 58, 7907])
def test_single_loud_window_is_a_region_regardless_of_float_rounding(tmp_path, window_index):
    # timing.md B1: 0.2 -> 0.3 gave 0.0999999 s and the interjection was dropped.
    audio = tmp_path / "burst.wav"
    start = window_index / 10
    _write_wav(audio, frame_rate=1000, duration_seconds=start + 1, bursts=[(start, start + 0.1, 10000)])

    regions = EnergySpeechActivityAdapter(threshold_dbfs=-45.0, window_ms=100, min_region_ms=100).detect(audio)

    assert len(regions) == 1
    assert regions[0].start == pytest.approx(start)
    assert regions[0].end == pytest.approx(start + 0.1)


def test_audible_word_evidence_only_reports_cues_with_energy_under_their_own_words(tmp_path):
    audio = tmp_path / "audio.wav"
    _write_wav(audio, frame_rate=16000, duration_seconds=4, bursts=[(1.02, 1.12, 3000)])
    words = [Word(text="Hã?", start=1.0, end=1.15), Word(text="Ei,", start=3.0, end=3.2)]
    alignment = AlignmentResult(cue_word_indices={7: [0], 8: [1]})
    flags = [
        QCFlag(kind="cue_without_speech_activity", cue_ids=[cue_id], message="no activity", confidence=0.0)
        for cue_id in (7, 8)
    ]

    assert cue_ids_with_audible_words(audio, flags, words, alignment) == {7}


def test_audible_word_evidence_is_empty_when_audio_cannot_be_decoded(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    flags = [QCFlag(kind="cue_without_speech_activity", cue_ids=[7], message="no activity", confidence=0.0)]

    assert cue_ids_with_audible_words(
        audio, flags, [Word(text="Hã?", start=1.0, end=1.15)], AlignmentResult(cue_word_indices={7: [0]})
    ) == set()


@pytest.mark.parametrize("spoken", [True, False])
def test_generated_adlib_is_only_removed_when_its_own_words_are_silent(tmp_path, spoken):
    # Episode 11 "Hã?" (790.69 s): the coarse VAD missed a real interjection and
    # the generated cue was deleted from the subtitle file.
    source = Cue(index=1, start_ms=2000, end_ms=3000, lines=["Depois a gente fala."])
    rebuilt = [
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["Hã?"]),
        Cue(index=1, start_ms=2000, end_ms=3000, lines=["Depois a gente fala."]),
    ]
    words = [
        Word(text="Hã?", start=1.0, end=1.15),
        Word(text="Depois", start=2.0, end=2.3),
        Word(text="a", start=2.32, end=2.4),
        Word(text="gente", start=2.42, end=2.7),
        Word(text="fala.", start=2.72, end=3.0),
    ]
    audio = tmp_path / "audio.wav"
    bursts = [(2.0, 3.0, 3000)] + ([(1.02, 1.12, 3000)] if spoken else [])
    _write_wav(audio, frame_rate=16000, duration_seconds=4, bursts=bursts)
    vad = tmp_path / "vad-fixture.json"
    vad.write_text(json.dumps({"regions": [{"start": 2.0, "end": 3.0}]}), encoding="utf-8")
    output = tmp_path / "output.srt"

    result = pipeline._run_verify_stage(
        episode_workdir=tmp_path, output_path=output, audio_path=audio, audio_for_asr=audio,
        provider_config={"vad": {"fixture_path": str(vad)}},
        profile=StyleProfile(fps=30), source_cues=[source], rebuilt=rebuilt, words=words,
        alignment=AlignmentResult(cue_word_indices={2: [0], 1: [1, 2, 3, 4]}),
        flags=[QCFlag(kind="adlib_inserted", cue_ids=[2], message="Generated from an audio-only insertion.", new_text="Hã?")],
        cost_meter=CostMeter(), include_dropped_line_flags=False,
    )

    texts = [cue.plain_text for cue in parse_srt_text(output.read_text(encoding="utf-8"))]
    removed = any(flag["kind"] == "adlib_removed_without_speech_activity" for flag in result.report["flags"])
    if spoken:
        assert texts == ["Hã?", "Depois a gente fala."]
        assert not removed
    else:
        assert texts == ["Depois a gente fala."]
        assert removed


# --- Task 2: 10 ms adaptive energy VAD ---------------------------------------------------------

LOUD = 3277  # about -20 dBFS as a square wave
SOFT_ONSET = 130  # about -48 dBFS: above the adaptive on-threshold, below the edge level
DECAY = 58  # about -55 dBFS: between the adaptive off- and on-thresholds


def test_default_energy_vad_places_burst_edges_within_two_hops(tmp_path):
    audio = tmp_path / "burst.wav"
    _write_wav(audio, frame_rate=16000, duration_seconds=3, bursts=[(0.523, 0.871, LOUD)])

    adapter = EnergySpeechActivityAdapter()
    regions = adapter.detect(audio)

    assert len(regions) == 1
    assert regions[0].start == pytest.approx(0.523, abs=0.02)
    assert regions[0].end == pytest.approx(0.871, abs=0.02)
    assert adapter.last_thresholds is not None and adapter.last_thresholds.adaptive
    assert adapter.last_thresholds.off_dbfs < adapter.last_thresholds.on_dbfs < adapter.last_thresholds.edge_dbfs


def test_default_energy_vad_bridges_short_gaps_and_drops_blips(tmp_path):
    audio = tmp_path / "gaps.wav"
    _write_wav(
        audio, frame_rate=16000, duration_seconds=5,
        bursts=[
            (0.5, 0.8, LOUD), (0.85, 1.2, LOUD),  # 50 ms stop closure: one burst
            (2.0, 2.3, LOUD), (2.45, 2.8, LOUD),  # 150 ms pause: two bursts
            (4.0, 4.005, LOUD),  # click
        ],
    )

    regions = EnergySpeechActivityAdapter().detect(audio)

    assert [(round(region.start, 1), round(region.end, 1)) for region in regions] == [
        (0.5, 1.2), (2.0, 2.3), (2.4, 2.8),
    ]


def test_default_energy_vad_keeps_soft_onsets_and_cuts_decay_tails(tmp_path):
    audio = tmp_path / "hysteresis.wav"
    _write_wav(
        audio, frame_rate=16000, duration_seconds=6,
        bursts=[
            (0.5, 1.0, LOUD), (1.0, 1.2, DECAY), (1.2, 1.6, LOUD),  # level dips but stays above off: one burst
            (2.5, 3.0, LOUD), (3.0, 3.3, DECAY),  # decay tail after the voice
            (4.0, 4.1, SOFT_ONSET), (4.1, 4.5, LOUD),  # soft consonant before the vowel
        ],
    )

    regions = EnergySpeechActivityAdapter().detect(audio)

    assert len(regions) == 3
    assert regions[0].start == pytest.approx(0.5, abs=0.02)
    assert regions[0].end == pytest.approx(1.6, abs=0.02)
    assert regions[1].end == pytest.approx(3.0, abs=0.02)
    assert regions[2].start == pytest.approx(4.0, abs=0.02)


def test_adaptive_threshold_follows_a_quiet_delivery_and_absolute_override_still_works(tmp_path):
    audio = tmp_path / "quiet.wav"
    quiet_speech = 104  # about -50 dBFS: the whole stem was delivered 30 dB low
    _write_wav(audio, frame_rate=16000, duration_seconds=3, bursts=[(1.0, 1.6, quiet_speech)])

    adaptive = EnergySpeechActivityAdapter().detect(audio)
    absolute = EnergySpeechActivityAdapter(threshold_dbfs=-45.0).detect(audio)
    lowered = EnergySpeechActivityAdapter(threshold_dbfs=-60.0).detect(audio)

    assert len(adaptive) == 1
    assert adaptive[0].start == pytest.approx(1.0, abs=0.02)
    assert adaptive[0].end == pytest.approx(1.6, abs=0.02)
    assert absolute == []
    assert len(lowered) == 1


def test_vad_config_defaults_to_adaptive_and_keeps_legacy_keys_working():
    default = speech_activity_adapter_from_config({"vad": {"provider": "energy"}})
    legacy = speech_activity_adapter_from_config(
        {"vad": {"provider": "energy", "threshold_dbfs": -45.0, "window_ms": 100, "min_region_ms": 100}}
    )

    assert isinstance(default, EnergySpeechActivityAdapter)
    assert (default.threshold_dbfs, default.window_ms, default.min_region_ms) == (None, None, 30)
    assert (legacy.threshold_dbfs, legacy.window_ms, legacy.min_region_ms) == (-45.0, 100, 100)
    with pytest.raises(ValueError, match="vad.window_ms"):
        speech_activity_adapter_from_config({"vad": {"provider": "energy", "window_ms": "wide"}})


def test_default_energy_vad_streams_long_audio_with_bounded_memory(tmp_path):
    audio = tmp_path / "thirty-minutes.wav"
    one_second = b"\x00\x00" * 16000
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        for _ in range(30 * 60):
            stream.writeframesraw(one_second)

    tracemalloc.start()
    try:
        regions = EnergySpeechActivityAdapter().detect(audio)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert regions == []
    assert peak_bytes <= 8 * 1024 * 1024
