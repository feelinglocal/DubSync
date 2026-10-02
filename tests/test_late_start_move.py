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
# Scribe Japanese: most phrase starts lag their onset, one in five by more than the 200 ms snap window
# (exactly the share the rule asks for); the second sample lags well inside the rule (two in five).
LAGGING = (0.15, 0.15, 0.25, 0.15, 0.12) * 5
CLEARLY_LAGGING = (0.15, 0.25, 0.25, 0.15, 0.12) * 5
# MAI, German, Portuguese: phrase starts sit on their onset and a late one is an exception.
ON_TIME = (0.03, 0.05, 0.0, 0.08, 0.04) * 5
SPEECH_DB, BREATH_DB = -25.0, -45.0


def _characters(text: str, start: float, end: float) -> list[Word]:
    step = (end - start) / len(text)
    return [
        Word(text=character, start=round(start + index * step, 3), end=round(start + (index + 1) * step, 3))
        for index, character in enumerate(text)
    ]


def _recording(lags, first_word_start: float = 71.5, *, lead_db: float = SPEECH_DB, intruder: Word | None = None,
               lead_levels: list[float] | None = None):
    """One-phrase bursts with the given first-word lags, then the phrase under test in ``BURST``.

    The lead before the phrase is ``lead_db`` throughout, or ends with the 10 ms
    ``lead_levels`` (a measured shape) right before the first word.
    """
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
    last_lead_hop = round((first_word_start - 0.005) * 100)
    for hop in range(round((BURST.start - 0.005) * 100), last_lead_hop):
        hops[hop] = lead_db
    for position, level in enumerate(reversed(lead_levels or [])):
        hops[last_lead_hop - 1 - position] = level
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
        max_onset_lead_ms=200, frame_ms=PROFILE.frame_ms, end_pad_ms=40, levels=levels,
    )
    return words, refined[0], flags


@pytest.mark.parametrize("lags", [LAGGING, CLEARLY_LAGGING], ids=["share-on-threshold", "clearly-lagging"])
@pytest.mark.parametrize("first_word_start", [71.036, 71.5])
def test_late_first_word_of_a_lagging_recording_moves_onto_its_speech_onset(first_word_start, lags):
    raw, regions, levels, first = _recording(lags, first_word_start)
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


# Measured 10 ms levels (dBFS) of the lead before two delivered EP11 cue starts the human editor moved
# 100-134 ms earlier. Both leads are quiet as a whole and end in the voice.
# EP11 MAI "Cinco.": a fricative onset 12-18 dB below the vowel, then 80 ms of vowel before the cue start.
CINCO_LEAD = [-52, -49, -46, -41, -39, -40, -40, -36, -33, -33, -35, -35, -34, -29, -24, -22, -24, -25, -23, -22, -22]
# EP11 Scribe "Um.": the detector opened the burst 110 ms before the voice, then 130 ms of loud voice.
UM_LEAD = [-54, -52, -53, -53, -53, -56, -57, -59, -63, -63, -56, -44, -35, -25, -19, -18, -17, -17, -17, -18, -20,
           -19, -19, -19]


@pytest.mark.parametrize("lead_levels", [CINCO_LEAD, UM_LEAD], ids=["cinco", "um"])
def test_a_start_inside_a_rising_onset_is_reported_but_not_moved_onto_the_onset(lead_levels):
    raw, regions, levels, first = _recording(LAGGING, lead_db=BREATH_DB, lead_levels=lead_levels)

    words, cue, flags = _timed(raw, regions, levels, first)

    # The lead as a whole is not the phrase: nothing moves onto the burst onset.
    assert words[first] == raw[first]
    assert cue.start_ms == 71500
    # But the voice is already sounding when the cue starts: the customer must hear it.
    assert [(flag.kind, flag.cue_ids) for flag in flags] == [(KIND, [1])]


def test_a_breath_right_up_to_the_cue_start_is_not_reported():
    # Quiet lead ending in two quiet hops: the word starts on the cue start.
    raw, regions, levels, first = _recording(LAGGING, lead_db=BREATH_DB, lead_levels=[-30, -31, -45, -44, -46])

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first] == raw[first]
    assert flags == []


def test_lead_ends_in_speech_judges_the_last_hops_before_the_start():
    hops = array("f", [-90.0]) * 300
    hops[100:160] = array("f", [-22.0]) * 60
    levels = vad.SpeechLevels(levels=hops, hop_seconds=0.01, offset_seconds=0.0)
    for lead, ends_in_speech, whole in (
        (CINCO_LEAD, True, False), (UM_LEAD, True, False), ([-45.0] * 21, False, False), ([-24.0] * 21, True, True),
        ([-45.0] * 18 + [-24.0] * 3, False, False), ([-45.0] * 17 + [-24.0] * 4, True, False),
    ):
        hops[100 - len(lead):100] = array("f", [float(level) for level in lead])
        assert levels.lead_ends_in_speech(1.0 - 0.01 * len(lead), 1.0, 1.6) is ends_in_speech
        assert levels.lead_is_speech(1.0 - 0.01 * len(lead), 1.0, 1.6) is whole
    # A lead shorter than the window needs all of its hops.
    hops[97:100] = array("f", [-24.0, -24.0, -45.0])
    assert levels.lead_ends_in_speech(0.97, 1.0, 1.6) is False
    assert levels.lead_ends_in_speech(0.0, 0.0, 1.6) is False


def test_lead_beyond_the_plausible_lag_is_not_moved_onto_but_still_reported():
    # Word repair does not trust a 715 ms lag; a speech-level lead that nobody owns is still the customer's to hear.
    raw, regions, levels, first = _recording(LAGGING, 71.55)

    words, cue, flags = _timed(raw, regions, levels, first)

    assert words[first].start == 71.55
    assert cue.start_ms == PROFILE.snap_floor(71550)
    assert [(flag.kind, flag.cue_ids) for flag in flags] == [(KIND, [1])]
    assert flags[0].start == pytest.approx(BURST.start)


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


# The shape of 2A-scribe cues 50 and 51: Scribe gives 三 21 ms inside the burst 64.375-65.135 and puts the
# 分 of the same cue 6 s late, as a 20 ms token in the lead of the next phrase.
_COLLAPSED_BURSTS = [SpeechRegion(start=64.375, end=65.135), SpeechRegion(start=70.615, end=72.095)]
_COLLAPSED_WORDS = [("三", 64.739, 64.760), ("分", 71.14, 71.16)]
_NEXT_PHRASE_WORDS = [("山", 71.28, 71.32), ("下", 71.36, 71.361), ("森", 71.46, 71.48), ("彦", 71.48, 71.6),
                      ("に", 71.6, 71.82), ("伝", 71.82, 71.9), ("え", 71.9, 72.02), ("ろ", 72.02, 72.08)]


def _collapsed_phrase_recording():
    """A lagging recording whose last two cues are shaped like 2A-scribe cues 50 and 51."""
    words, regions = [], []
    for index, lag in enumerate(LAGGING):
        onset = 1.0 + 2.0 * index
        regions.append(SpeechRegion(start=onset, end=onset + 1.0))
        words.extend(_characters("そうだな", onset + lag, onset + 0.98))
    regions.extend(_COLLAPSED_BURSTS)
    first = len(words)
    words.extend(Word(text=text, start=start, end=end) for text, start, end in (*_COLLAPSED_WORDS, *_NEXT_PHRASE_WORDS))
    hops = array("f", [-90.0]) * 7400
    for region in regions:
        for hop in range(round((region.start - 0.005) * 100), round((region.end - 0.005) * 100)):
            hops[hop] = SPEECH_DB
    return words, regions, vad.SpeechLevels(levels=hops, hop_seconds=0.01, offset_seconds=0.005), first


def test_a_collapsed_token_is_not_stretched_onto_the_onset_and_its_cue_stays_held():
    raw, regions, levels, first = _collapsed_phrase_recording()
    cues = [Cue(index=50, start_ms=64666, end_ms=65466, lines=["三分？"]),
            Cue(index=51, start_ms=70700, end_ms=72133, lines=["山下森彦に伝えろ"])]
    alignment = AlignmentResult(
        cue_word_indices={50: [first, first + 1], 51: list(range(first + 2, first + 10))}, anchor_coverage=1.0,
    )

    words, _ = repair_asr_word_edges(raw, regions, max_region_overrun=0.3, levels=levels)
    rebuilt, flags = rebuild_cues(cues, words, alignment, PROFILE)

    # The recording lags (the first phrase of the sample moves), but a 21 ms token is no start evidence.
    assert words[0].start == regions[0].start
    assert words[first:first + 2] == raw[first:first + 2]
    # 分 stays a 20 ms stray at 71.14: it does not cover the lead of the next phrase, whose starts are untouched.
    assert (words[first + 1].start, words[first + 1].end) == (71.14, 71.16)
    assert [word.start for word in words[first + 2:]] == [word.start for word in raw[first + 2:]]
    # The cue keeps its source timing and its hold, which routes it to whole-utterance hearing.
    assert (rebuilt[0].start_ms, rebuilt[0].end_ms) == (64666, 65466)
    assert [(flag.kind, flag.cue_ids) for flag in flags if flag.kind == "timing_evidence_held"] == [
        ("timing_evidence_held", [50]),
    ]
    assert rebuilt[1].start_ms == PROFILE.snap_floor(71280)


@pytest.mark.parametrize("duration_ms", [1, 20, 21, 40])
def test_a_token_without_a_duration_of_its_own_never_moves_in_a_lagging_recording(duration_ms):
    raw, regions, levels, first = _recording(LAGGING, 71.5)
    raw[first] = raw[first].model_copy(update={"end": round(71.5 + duration_ms / 1000, 3)})

    words, _ = repair_asr_word_edges(raw, regions, max_region_overrun=0.3, levels=levels)

    assert words[first] == raw[first]
    # A sibling with a duration of its own still moves under the same evidence.
    raw[first] = raw[first].model_copy(update={"end": 71.56})
    moved, _ = repair_asr_word_edges(raw, regions, max_region_overrun=0.3, levels=levels)
    assert (moved[first].start, moved[first].end) == (BURST.start, 71.56)


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


@pytest.mark.parametrize(("lead_amplitude", "last_lag", "reported"), [
    (6000, 0.41, True), (1000, 0.41, False),
    # No upper bound for the review item: 900 ms of speech-level sound that nobody owns is still reported.
    (6000, 0.9, True),
])
def test_sync_keeps_a_late_start_when_the_recording_does_not_lag(tmp_path, lead_amplitude, last_lag, reported):
    source, audio, providers, regions = _episode(tmp_path, ON_TIME, last_lag=last_lag, lead_amplitude=lead_amplitude)

    result = pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, fps=30.0)

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert delivered[-1].start_ms == PROFILE.snap_floor(49000 + round(last_lag * 1000))
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

