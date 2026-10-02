"""A late phrase start moves onto its speech onset only where the recording's own evidence supports it."""
from __future__ import annotations

import json
import sys
import wave
from array import array
from math import sin, tau
from pathlib import Path

import pytest
import yaml

from dubsync import pipeline, vad
from dubsync.asr_timing import (
    NO_PHRASE_EDGE_SNAP, PhraseEdgeSnap, phrase_edge_snap_from_config, repair_asr_word_edges,
)
from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.recue import cue_spoken_spans, rebuild_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity
from dubsync.transcription import generate_srt_from_audio

KIND = "cue_starts_after_speech_onset"
PROFILE = StyleProfile(fps=30.0)
# The shape of 2B-scribe cue 51: the burst starts at 70.835 s, Scribe packs the phrase into its end.
BURST = SpeechRegion(start=70.835, end=72.285)
PHRASE = "山下森彦に伝えろ"
# Scribe Japanese: most phrase starts lag their onset, one in five by more than the 200 ms snap window.
LAGGING = (0.15, 0.15, 0.25, 0.15, 0.12) * 5
# MAI, German, Portuguese: phrase starts sit on their onset and a late one is an exception.
ON_TIME = (0.03, 0.05, 0.0, 0.08, 0.04) * 5
SPEECH_DB, BREATH_DB = -25.0, -45.0


def _characters(text: str, start: float, end: float) -> list[Word]:
    step = (end - start) / len(text)
    return [
        Word(text=character, start=round(start + index * step, 3), end=round(start + (index + 1) * step, 3))
        for index, character in enumerate(text)
    ]


def _recording(lags, first_word_start: float = 71.5, *, lead_db: float = SPEECH_DB, intruder: Word | None = None):
    """One-phrase bursts with the given first-word lags, then the phrase under test in ``BURST``."""
    words: list[Word] = []
    regions: list[SpeechRegion] = []
    for index, lag in enumerate(lags):
        onset = 1.0 + 2.0 * index
        regions.append(SpeechRegion(start=onset, end=onset + 1.0))
        words.extend(_characters("そうだな", onset + lag, onset + 0.98))
    regions.append(BURST)
    if intruder is not None:
        words.append(intruder)
    first = len(words)
    words.extend(_characters(PHRASE, first_word_start, 72.26))
    hops = array("f", [-90.0]) * 7400
    for region in regions:
        for hop in range(round((region.start - 0.005) * 100), round((region.end - 0.005) * 100)):
            hops[hop] = SPEECH_DB
    for hop in range(round((BURST.start - 0.005) * 100), round((first_word_start - 0.005) * 100)):
        hops[hop] = lead_db
    return words, regions, vad.SpeechLevels(levels=hops, hop_seconds=0.01, offset_seconds=0.005), first


def _timed(raw_words, regions, levels, first, *, snap: PhraseEdgeSnap | None = None):
    """Word repair, rebuild, refinement and the late-start check as the pipeline runs them."""
    words, _ = repair_asr_word_edges(raw_words, regions, max_region_overrun=0.3, snap=snap, levels=levels)
    cue = Cue(index=1, start_ms=70700, end_ms=72133, lines=[PHRASE])
    alignment = AlignmentResult(
        cue_word_indices={1: list(range(first, first + len(PHRASE)))}, anchor_coverage=1.0,
    )
    rebuilt, _ = rebuild_cues([cue], words, alignment, PROFILE)
    refined, _ = refine_cues_to_speech_activity(
        rebuilt, regions, PROFILE, BoundaryRefinementConfig(), words=words, alignment=alignment,
    )
    flags = vad.late_start_flags_for_cues(
        refined, regions, words, alignment.cue_word_indices, cue_spoken_spans(refined, words, alignment),
        max_onset_lead_ms=200, max_review_lead_ms=700, frame_ms=PROFILE.frame_ms, end_pad_ms=40, levels=levels,
    )
    return words, refined[0], flags


@pytest.mark.parametrize("first_word_start", [71.036, 71.5])
def test_late_first_word_of_a_lagging_recording_moves_onto_its_speech_onset(first_word_start):
    raw, regions, levels, first = _recording(LAGGING, first_word_start)
    provider_words = [word.model_copy() for word in raw]

    words, cue, flags = _timed(raw, regions, levels, first)

    assert raw == provider_words
    assert (words[first].start, words[first].end) == (BURST.start, raw[first].end)
    assert [word.start for word in words[first + 1:]] == [word.start for word in raw[first + 1:]]
    assert cue.start_ms == PROFILE.snap_floor(BURST.start * 1000)
    assert flags == []
    # Every stage sees the moved start: repairing the repaired words changes nothing.
    assert repair_asr_word_edges(words, regions, max_region_overrun=0.3, levels=levels)[0] == words


def test_late_first_word_stays_and_is_reported_when_the_recording_does_not_lag():
    raw, regions, levels, first = _recording(ON_TIME)

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first] == raw[first]
    assert cue.start_ms == 71500
    assert [(flag.kind, flag.cue_ids, flag.severity) for flag in flags] == [(KIND, [1], "warning")]
    assert (flags[0].start, flags[0].end) == (pytest.approx(BURST.start), pytest.approx(71.5))


def test_too_few_phrase_starts_say_nothing_about_the_recording():
    raw, regions, levels, first = _recording(LAGGING[:15])

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first] == raw[first]
    assert [flag.kind for flag in flags] == [KIND]


@pytest.mark.parametrize("intruder", [
    Word(text="え", start=70.9, end=71.2),      # a word of another cue, or of nobody, inside the lead
    Word(text="だ", start=70.3, end=70.95),     # the previous word reaches into the burst
])
def test_lead_owned_by_another_word_is_not_taken(intruder):
    raw, regions, levels, first = _recording(LAGGING, intruder=intruder)

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first].start == 71.5
    assert cue.start_ms == 71500
    assert flags == []


def test_punctuation_token_reaching_into_the_burst_blocks_the_move_and_keeps_the_review_item():
    raw, regions, levels, first = _recording(LAGGING, intruder=Word(text="？", start=70.5, end=70.86))

    words, cue, flags = _timed(raw, regions, levels, first)

    # The token is no speech, but its interval makes the ownership of the onset unclear.
    assert words[first].start == 71.5
    assert cue.start_ms == 71500
    assert [(flag.kind, flag.cue_ids) for flag in flags] == [(KIND, [1])]


def test_quiet_lead_is_neither_moved_onto_nor_reported():
    raw, regions, levels, first = _recording(LAGGING, lead_db=BREATH_DB)

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first] == raw[first]
    assert cue.start_ms == 71500
    assert flags == []


def test_lead_beyond_the_plausible_lag_is_neither_moved_onto_nor_reported():
    raw, regions, levels, first = _recording(LAGGING, 71.55)

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first].start == 71.55
    assert cue.start_ms == PROFILE.snap_floor(71550)
    assert flags == []


def test_without_a_level_track_the_late_start_is_reported_not_moved():
    raw, regions, _, first = _recording(LAGGING)

    words, cue, flags = _timed(raw, regions, None, first)

    assert words[first] == raw[first]
    assert cue.start_ms == 71500
    assert [flag.kind for flag in flags] == [KIND]


@pytest.mark.parametrize("snap", [
    NO_PHRASE_EDGE_SNAP,
    phrase_edge_snap_from_config({"timing": {"phrase_edge_snap": False}}),
    phrase_edge_snap_from_config({"timing": {"phrase_edge_snap": {"lagging_start_advance_ms": 0}}}),
    phrase_edge_snap_from_config(
        {"timing": {"phrase_edge_snap": {"models": {"scribe_v2": {"lagging_start_advance_ms": 300}}}}}, "scribe_v2",
    ),
])
def test_configured_off_or_narrower_window_leaves_the_late_start(snap):
    raw, regions, levels, first = _recording(LAGGING)

    words, cue, flags = _timed(raw, regions, levels, first, snap=snap)

    assert words[first] == raw[first]
    assert cue.start_ms == 71500
    assert [flag.kind for flag in flags] == [KIND]


def test_lagging_window_is_read_from_the_phrase_edge_snap_configuration():
    config = {"timing": {"phrase_edge_snap": {
        "lagging_start_advance_ms": 500,
        "models": {"scribe_v2": {"lagging_start_advance_ms": 900}},
    }}}

    assert PhraseEdgeSnap().lagging_start_advance == 0.7
    assert phrase_edge_snap_from_config({}) == PhraseEdgeSnap()
    assert phrase_edge_snap_from_config(config, "microsoft/mai-transcribe-2").lagging_start_advance == 0.5
    assert phrase_edge_snap_from_config(config, "scribe_v2") == PhraseEdgeSnap(lagging_start_advance=0.9)
    with pytest.raises(ValueError, match="lagging_start_advance_ms"):
        phrase_edge_snap_from_config({"timing": {"phrase_edge_snap": {"lagging_start_advance_ms": -1}}})


def test_a_start_inside_the_snap_window_still_snaps_without_any_recording_evidence():
    raw, regions, _, first = _recording(ON_TIME, 71.03)

    words, cue, flags = _timed(raw, regions, None, first)

    assert words[first].start == BURST.start
    assert cue.start_ms == PROFILE.snap_floor(BURST.start * 1000)
    assert flags == []


# --- Real audio: the energy VAD's own level track decides, in synchronization and in generation ---------

TOKENS = [first + second for first in ("ka", "mo", "ri", "su", "te", "no", "ha", "mi")
          for second in ("lan", "den", "vik", "tor", "mes", "pol", "gar", "bin", "zul", "fen")]


def _write_tone_audio(path: Path, seconds: float, segments: list[tuple[float, float, int]]) -> None:
    samples = array("h", [0]) * round(seconds * 16000)
    for start, end, amplitude in segments:
        first, last = round(start * 16000), round(end * 16000)
        samples[first:last] = array("h", (round(amplitude * sin(tau * 400 * index / 16000))
                                           for index in range(last - first)))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def _episode(tmp_path, lags, *, last_lag: float, lead_amplitude: int = 6000, generation: bool = False):
    """25 one-phrase bursts 2 s apart; the last phrase starts ``last_lag`` seconds into its burst."""
    lags = [*lags[:-1], last_lag]
    audio, wordstream, providers, source = (
        tmp_path / name for name in ("episode.wav", "words.json", "providers.yaml", "episode.srt")
    )
    onsets = [1.0 + 2.0 * index for index in range(len(lags))]
    segments = [(onset, onset + 1.2, 6000) for onset in onsets]
    # The sound before the last phrase: as loud as the phrase, or 16 dB below it.
    segments.append((onsets[-1], onsets[-1] + last_lag, lead_amplitude))
    _write_tone_audio(audio, onsets[-1] + 3.0, segments)
    words, cues = [], []
    for index, (onset, lag) in enumerate(zip(onsets, lags)):
        tokens = TOKENS[3 * index:3 * index + 3]
        start = onset + lag
        step = (onset + 1.18 - start) / 3
        words.extend({"text": token, "start": round(start + position * step, 3),
                      "end": round(start + (position + 1) * step - 0.02, 3), "confidence": 0.98, "speaker_id": "A"}
                     for position, token in enumerate(tokens))
        cues.append(Cue(index=index + 1, start_ms=round(onset * 1000) + 300, end_ms=round(onset * 1000) + 1500,
                        lines=[" ".join(tokens)]))
    wordstream.write_text(json.dumps({"words": words}), encoding="utf-8")
    source.write_text(write_srt(cues), encoding="utf-8")
    config: dict[str, object] = {
        "asr": {"fixture_path": str(wordstream), "model_id": "scribe_v2"},
        "vad": {"provider": "energy", "boundary_refinement": {"enabled": True}},
    }
    if generation:
        config["generation"] = {"max_gap_seconds": 0.5}
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    regions = vad.EnergySpeechActivityAdapter().detect(audio)
    assert len(regions) == len(lags)
    return source, audio, providers, regions


def _late_flags(report: dict[str, object]) -> list[dict[str, object]]:
    return [flag for flag in report["flags"] if flag["kind"] == KIND]


@pytest.mark.parametrize("mode", ["fresh", "rebuild", "verify"])
def test_sync_starts_the_cues_of_a_lagging_recording_on_their_speech_onsets(tmp_path, mode):
    source, audio, providers, regions = _episode(tmp_path, LAGGING, last_lag=0.665)

    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=providers, no_llm=True, fps=30.0, **kwargs)

    result = run()
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        result = run(resume=mode)
        assert result.output_srt.read_bytes() == first

    delivered = parse_srt_text(first.decode("utf-8"))
    assert [cue.start_ms for cue in delivered] == [PROFILE.snap_floor(region.start * 1000) for region in regions]
    assert _late_flags(result.report) == []
    assert not [item for item in result.report["review"] if item["kind"] == KIND]
    # The provider's words are kept as received.
    saved = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))["words"]
    assert saved[-3]["start"] == pytest.approx(regions[-1].start + 0.665, abs=0.011)


@pytest.mark.parametrize(("lead_amplitude", "reported"), [(6000, True), (1000, False)])
def test_sync_keeps_a_late_start_when_the_recording_does_not_lag(tmp_path, lead_amplitude, reported):
    source, audio, providers, regions = _episode(tmp_path, ON_TIME, last_lag=0.41, lead_amplitude=lead_amplitude)

    result = pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, fps=30.0)

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert delivered[-1].start_ms == PROFILE.snap_floor(49410)
    # A speech-level lead is the customer's to check; a quiet one (a breath) is not.
    assert [flag["cue_ids"] for flag in _late_flags(result.report)] == ([[len(regions)]] if reported else [])


def test_sync_with_the_snap_configured_off_moves_nothing(tmp_path):
    source, audio, providers, regions = _episode(tmp_path, LAGGING, last_lag=0.665)
    config = yaml.safe_load(providers.read_text(encoding="utf-8"))
    providers.write_text(yaml.safe_dump({**config, "timing": {"phrase_edge_snap": False}}), encoding="utf-8")

    result = pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, fps=30.0)

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert delivered[-1].start_ms == PROFILE.snap_floor(49665)
    assert [len(regions)] in [flag["cue_ids"] for flag in _late_flags(result.report)]


def test_generation_starts_the_cues_of_a_lagging_recording_on_their_speech_onsets(tmp_path):
    _, audio, providers, regions = _episode(tmp_path, LAGGING, last_lag=0.665, generation=True)

    result = generate_srt_from_audio(audio, tmp_path / "output.srt", tmp_path / "work", providers_path=providers,
                                     no_llm=True, style_profile=PROFILE)

    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.start_ms for cue in cues] == [PROFILE.snap_floor(region.start * 1000) for region in regions]
    assert _late_flags(result.report) == []


@pytest.mark.parametrize(("lead_amplitude", "reported"), [(6000, True), (1000, False)])
def test_generation_reports_a_late_start_it_cannot_move(tmp_path, lead_amplitude, reported):
    _, audio, providers, regions = _episode(
        tmp_path, ON_TIME, last_lag=0.41, lead_amplitude=lead_amplitude, generation=True,
    )

    result = generate_srt_from_audio(audio, tmp_path / "output.srt", tmp_path / "work", providers_path=providers,
                                     no_llm=True, style_profile=PROFILE)

    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert cues[-1].start_ms == PROFILE.snap_floor(49410)
    flags = _late_flags(result.report)
    assert len(flags) == (1 if reported else 0)
    if reported:
        assert (flags[0]["severity"], flags[0]["start"]) == ("warning", pytest.approx(regions[-1].start))
        assert [item["title"] for item in result.report["review"] if item["kind"] == KIND] == [
            "Cue starts after the speech begins",
        ]

