from __future__ import annotations

import json
import tracemalloc
import wave
from array import array

import pytest
import yaml

from dubsync import pipeline
from dubsync.asr_timing import PhraseEdgeSnap, phrase_edge_snap_from_config, repair_asr_word_edges
from dubsync.cost import CostMeter
from dubsync.models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from dubsync.recue import rebuild_cues
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import boundary_refinement_config_from_config, refine_cues_to_speech_activity
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


# --- Task 3: ASR word edges repaired from the speech bursts before rebuild ---------------------


def test_start_stretched_word_keeps_its_real_end_and_moves_to_the_burst_onset():
    # timing.md B2: Scribe stretched "Obrigada," back over an 11.7 s pause.
    words = [
        Word(text="aqui.", start=2115.50, end=2115.83),
        Word(text="Obrigada,", start=2115.892, end=2127.592),
        Word(text="Luke.", start=2127.652, end=2127.912),
    ]
    regions = [SpeechRegion(start=2115.20, end=2115.90), SpeechRegion(start=2127.08, end=2127.93)]

    repaired, flags = repair_asr_word_edges(words, regions)

    assert (repaired[1].start, repaired[1].end) == (2127.08, 2127.592)
    assert repaired[2].start == 2127.652
    assert [flag.kind for flag in flags] == ["asr_word_clamped"]
    assert flags[0].old_text == "Obrigada, 2115.892 --> 2127.592"
    assert flags[0].new_text == "Obrigada, 2127.080 --> 2127.592"


def test_end_stretched_word_is_cut_at_the_end_of_its_own_burst():
    words = [Word(text="o", start=1341.90, end=1342.10), Word(text="público.", start=1342.118, end=1346.978)]
    regions = [SpeechRegion(start=1341.0, end=1342.60), SpeechRegion(start=1346.99, end=1348.0)]

    repaired, flags = repair_asr_word_edges(words, regions)

    assert (repaired[1].start, repaired[1].end) == (1342.118, 1342.60)
    assert [flag.kind for flag in flags] == ["asr_word_clamped"]


def test_stretch_that_reaches_the_next_phrase_keeps_the_side_that_can_hold_the_word():
    regions = [SpeechRegion(start=48.2, end=48.9), SpeechRegion(start=67.0, end=69.0)]
    end_stretched = [Word(text="Los", start=48.299, end=48.5), Word(text="geht's.", start=48.599, end=67.379)]
    start_stretched = [Word(text="aqui.", start=48.3, end=48.8), Word(text="Obrigada,", start=48.84, end=67.5)]

    repaired_end, _ = repair_asr_word_edges(end_stretched, regions)
    repaired_start, _ = repair_asr_word_edges(start_stretched, regions)

    # 0.3 s of speech after "Los" is the whole word "geht's."
    assert (repaired_end[1].start, repaired_end[1].end) == (48.599, 48.9)
    # 60 ms after "aqui." cannot be "Obrigada,": it was spoken where it ends.
    assert (repaired_start[1].start, repaired_start[1].end) == (67.0, 67.5)


def test_phrase_edges_snap_to_the_burst_without_per_word_flags():
    # MAI: phrase starts on a 40 ms grid and ends after the voice has stopped.
    words = [
        Word(text="Se", start=9.975, end=10.10, confidence=None),  # starts 25 ms before the onset
        Word(text="arruma,", start=10.16, end=10.66, confidence=None),  # ends 60 ms after the offset
        Word(text="Vamos", start=12.045, end=12.30, confidence=None),  # starts 45 ms after the onset
        Word(text="logo", start=12.36, end=12.55, confidence=None),
        Word(text="embora.", start=12.60, end=12.95, confidence=None),  # ends 50 ms before the offset
    ]
    regions = [SpeechRegion(start=10.0, end=10.6), SpeechRegion(start=12.0, end=13.0)]

    repaired, flags = repair_asr_word_edges(words, regions)

    assert [(word.start, word.end) for word in repaired] == [
        (10.0, 10.10), (10.16, 10.6), (12.0, 12.30), (12.36, 12.55), (12.60, 13.0),
    ]
    assert flags == []


def test_burst_edges_owned_by_another_word_or_too_far_away_are_not_borrowed():
    words = [
        Word(text="Uau,", start=1.05, end=1.30),
        Word(text="lindo!", start=1.55, end=1.80),
        Word(text="Outra", start=2.0, end=2.2, speaker_id="speaker_2"),
        Word(text="Hã?", start=5.40, end=5.56),
    ]
    regions = [SpeechRegion(start=1.0, end=3.0), SpeechRegion(start=5.0, end=6.0)]

    repaired, _ = repair_asr_word_edges(words, regions)

    assert repaired[1] == words[1]  # mid-burst word between two other words
    assert repaired[2].end == 2.2  # the burst continues for 0.8 s: another sound, not this word
    assert repaired[3] == words[3]  # 0.4 s after the onset and 0.44 s before the offset


def test_word_without_acoustic_evidence_only_gets_the_duration_limit():
    words = [Word(text="soft", start=1.0, end=1.4), Word(text="stretched", start=5.0, end=23.0)]

    untouched, no_flags = repair_asr_word_edges(words[:1], [SpeechRegion(start=3.0, end=4.0)])
    limited, flags = repair_asr_word_edges(words[1:], [])

    assert untouched == words[:1] and no_flags == []
    assert (limited[0].start, limited[0].end) == (5.0, 7.0)
    assert [flag.kind for flag in flags] == ["asr_word_clamped"]


def test_word_edge_repair_is_idempotent_and_keeps_word_count_and_order():
    words = [
        Word(text="aqui.", start=2115.50, end=2115.83),
        Word(text="Obrigada,", start=2115.892, end=2127.592),
        Word(text="Luke.", start=2127.652, end=2127.912),
        Word(text="público.", start=2130.118, end=2134.978),
    ]
    regions = [
        SpeechRegion(start=2115.20, end=2115.90), SpeechRegion(start=2127.08, end=2127.93),
        SpeechRegion(start=2130.0, end=2130.6),
    ]

    once, _ = repair_asr_word_edges(words, regions)
    twice, flags = repair_asr_word_edges(once, regions)

    assert [word.text for word in once] == [word.text for word in words]
    assert twice == once
    assert flags == []


def test_phrase_edge_snap_config_supports_per_model_overrides():
    config = {"timing": {"phrase_edge_snap": {
        "start_advance_ms": 120,
        "models": {"scribe_v2": {"end_extension_ms": 80}},
    }}}

    assert phrase_edge_snap_from_config({}) == PhraseEdgeSnap()
    assert phrase_edge_snap_from_config(config, "microsoft/mai-transcribe-2") == PhraseEdgeSnap(start_advance=0.12)
    assert phrase_edge_snap_from_config(config, "scribe_v2") == PhraseEdgeSnap(start_advance=0.12, end_extension=0.08)
    assert phrase_edge_snap_from_config({"timing": {"phrase_edge_snap": False}}) == PhraseEdgeSnap(0.0, 0.0)
    with pytest.raises(ValueError, match="start_advance_ms"):
        phrase_edge_snap_from_config({"timing": {"phrase_edge_snap": {"start_advance_ms": -1}}})


def test_sync_starts_a_cue_at_the_real_onset_of_its_stretched_first_word(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:01,700\nBom dia.\n\n"
        "2\n00:00:12,000 --> 00:00:13,000\nObrigada, Luke.\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": "Bom", "start": 1.0, "end": 1.2, "speaker_id": "A"},
        {"text": "dia.", "start": 1.25, "end": 1.6, "speaker_id": "A"},
        {"text": "Obrigada,", "start": 1.66, "end": 12.592, "speaker_id": "B"},
        {"text": "Luke.", "start": 12.652, "end": 12.912, "speaker_id": "B"},
    ]}), encoding="utf-8")
    vad = tmp_path / "vad.json"
    vad.write_text(
        json.dumps({"regions": [{"start": 1.0, "end": 1.62}, {"start": 12.08, "end": 12.93}]}), encoding="utf-8"
    )
    config = tmp_path / "provider.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)},
        "vad": {"fixture_path": str(vad), "boundary_refinement": True},
    }), encoding="utf-8")
    output = tmp_path / "output.srt"
    profile = StyleProfile(fps=30, min_cue_dur=0.5)

    result = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=config,
                                   no_llm=True, style_profile=profile)

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["Bom dia.", "Obrigada, Luke."]
    assert cues[1].start_ms == profile.snap_floor(12080)
    kinds = [flag["kind"] for flag in result.report["flags"]]
    assert "timing_outlier_trimmed" not in kinds
    assert kinds.count("asr_word_clamped") == 1


# --- Task 4: one word-window rule for rebuild and refinement ------------------------------------


def _words(*rows: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in rows]


def test_cue_with_a_mid_sentence_pause_starts_on_its_first_spoken_word():
    # Episode 11 cue 479: the first "Hum." is 1.72 s before the rest and was
    # dropped, so the cue appeared 1.9 s after the actor started.
    cue = Cue(index=479, start_ms=1_376_000, end_ms=1_380_000, lines=["Hum. Hum, Não precisa esperar."])
    words = _words(
        ("Hum.", 1376.325, 1376.495), ("Hum,", 1378.215, 1378.415), ("não", 1379.155, 1379.278),
        ("precisa", 1379.298, 1379.578), ("esperar.", 1379.618, 1379.935),
    )
    profile = StyleProfile(fps=30, min_cue_dur=0.5)

    rebuilt, flags = rebuild_cues([cue], words, AlignmentResult(cue_word_indices={479: [0, 1, 2, 3, 4]}), profile)

    assert rebuilt[0].start_ms == profile.snap_floor(1_376_325)
    assert rebuilt[0].end_ms == profile.snap_ceil(1_379_935 + 40)
    assert flags == []


def test_far_away_word_does_not_stretch_the_cue_even_when_its_text_is_in_the_cue():
    cue = Cue(index=403, start_ms=1_184_000, end_ms=1_186_000, lines=["na nossa viagem anual? Ah,"])
    words = _words(
        ("na", 1184.538, 1184.618), ("nossa", 1184.638, 1184.818), ("viagem", 1184.898, 1185.218),
        ("anual?", 1185.258, 1185.735), ("Ah,", 1206.375, 1206.925),
    )
    profile = StyleProfile(fps=30, min_cue_dur=0.5)

    rebuilt, flags = rebuild_cues([cue], words, AlignmentResult(cue_word_indices={403: [0, 1, 2, 3, 4]}), profile)

    assert rebuilt[0].start_ms == profile.snap_floor(1_184_538)
    assert rebuilt[0].end_ms == profile.snap_ceil(1_185_735 + 40)
    assert [flag.kind for flag in flags] == ["timing_outlier_trimmed"]


@pytest.mark.parametrize(("second_start", "expected_end"), [(4.0, 4.4), (9.0, 1.3)])
def test_two_part_cue_is_never_timed_to_an_arbitrary_half(second_start, expected_end):
    # rebuild.md BUG-12: two one-word groups tied and the shorter word won.
    cue = Cue(index=1, start_ms=1000, end_ms=5000, lines=["Sim. Vamos."])
    words = _words(("Sim.", 1.0, 1.3), ("Vamos.", second_start, second_start + 0.4))
    profile = StyleProfile(fps=30, min_cue_dur=0.1, tail_ms=0)

    rebuilt, flags = rebuild_cues([cue], words, AlignmentResult(cue_word_indices={1: [0, 1]}), profile)

    assert rebuilt[0].start_ms == 1000
    assert rebuilt[0].end_ms == profile.snap_ceil(expected_end * 1000)
    assert [flag.kind for flag in flags] == ([] if second_start == 4.0 else ["timing_outlier_trimmed"])


def test_refinement_uses_the_configured_intra_cue_gap_like_rebuild():
    # timing.md B6: with timing.max_intra_cue_gap 2.0 rebuild kept "A B",
    # refinement used a hard-coded 1.5 s and moved the start to "C".
    cue = Cue(index=1, start_ms=1000, end_ms=4333, lines=["Unrelated script wording here."])
    words = _words(("A", 1.0, 1.2), ("B", 1.25, 1.5), ("C", 3.2, 3.5), ("D", 3.55, 3.9), ("E", 3.95, 4.3))
    regions = [SpeechRegion(start=1.0, end=1.5), SpeechRegion(start=3.2, end=4.3)]
    configured = boundary_refinement_config_from_config(
        {"vad": {"boundary_refinement": True}, "timing": {"max_intra_cue_gap": 2.0}}
    )

    refined, _ = refine_cues_to_speech_activity(
        [cue], regions, StyleProfile(fps=30, min_cue_dur=0.5), configured,
        words=words, alignment=AlignmentResult(cue_word_indices={1: [0, 1, 2, 3, 4]}),
    )

    assert configured.max_intra_cue_gap_ms == 2000
    assert refined[0].start_ms == 1000


# --- Task 5: the cue ends with the burst that holds its last word -------------------------------


@pytest.mark.parametrize(
    ("burst_end", "expected_speech_end_ms"),
    [
        (1.42, 1420),  # the ASR word runs 80 ms past the voice (typical MAI end)
        (1.62, 1620),  # the voice runs 120 ms past the ASR word (typical Scribe end)
        (2.40, 1500),  # 0.9 s more sound in the burst is not this word
    ],
)
def test_cue_end_is_the_offset_of_the_burst_holding_its_last_word(burst_end, expected_speech_end_ms):
    cue = Cue(index=1, start_ms=1000, end_ms=2500, lines=["Oi."])
    profile = StyleProfile(fps=30, min_cue_dur=0.1)

    refined, _ = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=burst_end)], profile,
        words=_words(("Oi.", 1.0, 1.5)), alignment=AlignmentResult(cue_word_indices={1: [0]}),
    )

    assert refined[0].start_ms == 1000
    assert refined[0].end_ms == profile.snap_ceil(expected_speech_end_ms + 40)


def test_cue_end_is_not_extended_over_another_speakers_word_in_the_same_burst():
    # The old rule padded to the region end whenever it was within 300 ms.
    cue = Cue(index=1, start_ms=1000, end_ms=1866, lines=["Uau, que lindo!"])
    words = _words(("Uau,", 1.05, 1.3), ("que", 1.35, 1.5), ("lindo!", 1.55, 1.8), ("Outra", 1.85, 2.05))

    refined, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=2.05)], StyleProfile(fps=30, min_cue_dur=0.5),
        words=words, alignment=AlignmentResult(cue_word_indices={1: [0, 1, 2]}),
    )

    assert refined == [cue]
    assert flags == []


def test_separate_breath_burst_after_the_last_word_does_not_extend_the_cue():
    profile = StyleProfile(fps=30, min_cue_dur=0.5)
    cue = Cue(index=1, start_ms=1000, end_ms=profile.snap_ceil(1540), lines=["Oi."])

    refined, flags = refine_cues_to_speech_activity(
        [cue], [SpeechRegion(start=1.0, end=1.5), SpeechRegion(start=1.6, end=1.9)], profile,
        words=_words(("Oi.", 1.0, 1.5)), alignment=AlignmentResult(cue_word_indices={1: [0]}),
    )

    assert refined == [cue]
    assert not any(flag.kind == "timing_refined" for flag in flags)
