"""A cue that starts after its own speech began is reported for review; its timing is not moved."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline, vad
from dubsync.asr_timing import repair_asr_word_edges
from dubsync.models import AlignmentResult, Cue, QCFlag, SpeechRegion, Word
from dubsync.qc_review import build_review
from dubsync.recue import cue_spoken_spans, rebuild_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, SpeechEvidence, refine_cues_to_speech_activity

KIND = "cue_starts_after_speech_onset"
PROFILE = StyleProfile(fps=30.0)
# The shape of 2B-scribe cue 51: the burst starts at 70.835 s, Scribe packs the phrase into its end.
BURST = SpeechRegion(start=70.835, end=72.285)
PHRASE = "山下森彦に伝えろ"


def _characters(text: str, start: float, end: float) -> list[Word]:
    step = (end - start) / len(text)
    return [
        Word(text=character, start=round(start + index * step, 3), end=round(start + (index + 1) * step, 3))
        for index, character in enumerate(text)
    ]


def _timed(cues: list[Cue], raw_words: list[Word], regions: list[SpeechRegion], ownership: dict[int, list[int]]):
    """Word repair, rebuild and refinement as the pipeline runs them."""
    words, _ = repair_asr_word_edges(raw_words, regions, max_region_overrun=0.3)
    alignment = AlignmentResult(cue_word_indices=ownership, anchor_coverage=1.0)
    rebuilt, _ = rebuild_cues(cues, words, alignment, PROFILE)
    refined, _ = refine_cues_to_speech_activity(
        rebuilt, regions, PROFILE, BoundaryRefinementConfig(), words=words, alignment=alignment,
    )
    return refined, words, alignment


def _late_start_flags(cues, words, alignment, regions, **options) -> list[QCFlag]:
    return vad.late_start_flags_for_cues(
        cues, regions, words, alignment.cue_word_indices, cue_spoken_spans(cues, words, alignment),
        max_onset_lead_ms=200, frame_ms=PROFILE.frame_ms, end_pad_ms=40, **options,
    )


def _single_phrase(first_word_start: float):
    raw = _characters(PHRASE, first_word_start, 72.26)
    cue = Cue(index=1, start_ms=70700, end_ms=72133, lines=[PHRASE])
    return _timed([cue], raw, [BURST], {1: list(range(len(raw)))})


@pytest.mark.parametrize("first_word_start", [71.036, 71.5])
def test_first_word_past_the_snap_window_is_flagged_and_the_cue_is_not_moved(first_word_start):
    cues, words, alignment = _single_phrase(first_word_start)

    flags = _late_start_flags(cues, words, alignment, [BURST])

    # Detector only: the cue still starts on its late first word.
    assert words[0].start == first_word_start
    assert cues[0].start_ms == PROFILE.snap_floor(first_word_start * 1000)
    assert [(flag.kind, flag.cue_ids, flag.severity) for flag in flags] == [(KIND, [1], "warning")]
    assert flags[0].start == pytest.approx(BURST.start)
    assert flags[0].end == pytest.approx(cues[0].start_ms / 1000)
    assert f"{cues[0].start_ms - 70835} ms" in flags[0].message
    assert flags[0].old_text == PHRASE


@pytest.mark.parametrize("first_word_start", [70.835, 71.03, 71.035])
def test_first_word_inside_the_snap_window_snaps_and_is_not_flagged(first_word_start):
    cues, words, alignment = _single_phrase(first_word_start)

    assert words[0].start == BURST.start
    assert _late_start_flags(cues, words, alignment, [BURST]) == []


def test_lead_spoken_by_the_previous_cue_in_one_burst_is_not_flagged():
    burst = SpeechRegion(start=10.0, end=12.0)
    raw = [*_characters("今から", 10.0, 10.6), *_characters("行くぞ", 10.9, 11.9)]
    cues = [Cue(index=1, start_ms=10000, end_ms=10600, lines=["今から"]),
            Cue(index=2, start_ms=10900, end_ms=12000, lines=["行くぞ"])]
    timed, words, alignment = _timed(cues, raw, [burst], {1: [0, 1, 2], 2: [3, 4, 5]})

    assert timed[1].start_ms - 10000 > 200
    assert _late_start_flags(timed, words, alignment, [burst]) == []


def test_lead_holding_a_word_no_cue_owns_is_not_flagged():
    burst = SpeechRegion(start=10.0, end=12.0)
    raw = [Word(text="え", start=10.05, end=10.3), *_characters("行くぞ", 10.5, 11.9)]
    cue = Cue(index=1, start_ms=10500, end_ms=12000, lines=["行くぞ"])
    timed, words, alignment = _timed([cue], raw, [burst], {1: [1, 2, 3]})

    assert timed[0].start_ms == 10500
    assert _late_start_flags(timed, words, alignment, [burst]) == []


def test_lead_under_another_displayed_cue_is_not_flagged():
    burst = SpeechRegion(start=10.0, end=12.0)
    raw = _characters("行くぞ", 10.5, 11.9)
    # A laugh without ASR words stays at script timing and is on screen while the burst begins.
    laugh = Cue(index=1, start_ms=9000, end_ms=10200, lines=["ははは"])
    cue = Cue(index=2, start_ms=10500, end_ms=12000, lines=["行くぞ"])
    timed, words, alignment = _timed([laugh, cue], raw, [burst], {2: [0, 1, 2]})

    assert (timed[0].end_ms, timed[1].start_ms) == (10200, 10500)
    assert _late_start_flags(timed, words, alignment, [burst]) == []


def test_display_padding_of_the_previous_cue_does_not_hide_a_late_start():
    regions = [SpeechRegion(start=9.0, end=9.98), SpeechRegion(start=10.0, end=12.0)]
    raw = [*_characters("今から", 9.0, 9.98), *_characters("行くぞ", 10.5, 11.9)]
    cues = [Cue(index=1, start_ms=9000, end_ms=9980, lines=["今から"]),
            Cue(index=2, start_ms=10500, end_ms=12000, lines=["行くぞ"])]
    timed, words, alignment = _timed(cues, raw, regions, {1: [0, 1, 2], 2: [3, 4, 5]})

    # Only the first cue's tail padding reaches past the onset of the second burst.
    assert 10000 < timed[0].end_ms <= 10000 + 40 + PROFILE.frame_ms
    flags = _late_start_flags(timed, words, alignment, regions)
    assert [(flag.kind, flag.cue_ids) for flag in flags] == [(KIND, [2])]


def test_held_cue_is_not_flagged():
    cues, words, alignment = _single_phrase(71.5)

    assert _late_start_flags(cues, words, alignment, [BURST], excluded_cue_ids={1}) == []


def test_without_speech_regions_nothing_is_flagged():
    cues, words, alignment = _single_phrase(71.5)

    assert _late_start_flags(cues, words, alignment, []) == []


def test_cue_already_starting_at_the_onset_is_not_flagged():
    cues, words, alignment = _single_phrase(71.5)
    moved = [cues[0].with_timing(PROFILE.snap_floor(70835), cues[0].end_ms)]

    assert _late_start_flags(moved, words, alignment, [BURST]) == []


def test_only_requested_cues_are_checked_against_all_delivered_cues():
    cues, words, alignment = _single_phrase(71.5)

    assert _late_start_flags(cues, words, alignment, [BURST], cue_ids={2}) == []
    assert len(_late_start_flags(cues, words, alignment, [BURST], cue_ids={1})) == 1


def test_late_start_is_a_review_warning_in_the_customer_report():
    cue = Cue(index=1, start_ms=71500, end_ms=72334, lines=[PHRASE])
    flag = QCFlag(kind=KIND, cue_ids=[1], message="Cue starts 665 ms after the detected speech onset.",
                  old_text=PHRASE, start=70.835, end=71.5)

    review = build_review([flag], [], [cue], source_cues=[cue.with_timing(70700, 72133)])

    item, = review.review
    assert (item.kind, item.severity, item.srt_numbers, item.raw_flags) == (KIND, "warning", [1], [0])
    assert item.title == "Cue starts after the speech begins"
    assert item.action
    assert review.verdict == "check"
    assert review.diagnostics == [] and review.notes == []


# --- Pipeline: emitted with the other speech-evidence findings, recomputed on resume ---------------


def test_a_saved_late_start_finding_is_dropped_before_verification_recomputes_it():
    stale = QCFlag(kind=KIND, cue_ids=[2], message="Cue starts 500 ms after the detected speech onset.")
    kept = QCFlag(kind="text_changed", cue_ids=[2], message="Wording changed.")

    assert pipeline._without_stale_verify_flags([stale, kept]) == [kept]


def _sync_assets(tmp_path, *, late_ms: int, timing: dict | None = None, with_vad: bool = True):
    srt, audio, wordstream, vad_fixture, providers = (
        tmp_path / name for name in ("episode.srt", "episode.wav", "words.json", "vad.json", "providers.yaml")
    )
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n"
        "2\n00:00:05,000 --> 00:00:06,400\ntell him now\n\n",
        encoding="utf-8",
    )
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 128000)
    first = 5.0 + late_ms / 1000
    wordstream.write_text(json.dumps({"words": [
        {"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.98, "speaker_id": "A"},
        {"text": "there", "start": 0.25, "end": 0.55, "confidence": 0.97, "speaker_id": "A"},
        {"text": "tell", "start": first, "end": first + 0.2, "confidence": 0.98, "speaker_id": "A"},
        {"text": "him", "start": first + 0.25, "end": first + 0.45, "confidence": 0.98, "speaker_id": "A"},
        {"text": "now", "start": first + 0.5, "end": first + 0.8, "confidence": 0.97, "speaker_id": "A"},
    ]}), encoding="utf-8")
    vad_fixture.write_text(json.dumps({"regions": [
        {"start": 0.0, "end": 0.6, "confidence": 0.9}, {"start": 5.0, "end": first + 0.8, "confidence": 0.9},
    ]}), encoding="utf-8")
    config: dict[str, object] = {"asr": {"fixture_path": str(wordstream)}}
    if with_vad:
        config["vad"] = {"fixture_path": str(vad_fixture)}
    if timing is not None:
        config["timing"] = timing
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    return srt, audio, providers


def _late_flags(report: dict[str, object]) -> list[dict[str, object]]:
    return [flag for flag in report["flags"] if flag["kind"] == KIND]


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
def test_sync_reports_a_late_start_once_and_recomputes_it_on_resume(tmp_path, mode):
    srt, audio, providers = _sync_assets(tmp_path, late_ms=500)

    def run(**kwargs):
        return pipeline.sync_episode(srt, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=providers, no_llm=True, fps=30.0, **kwargs)

    result = run()
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert result.output_srt.read_bytes() == first

    flags = _late_flags(result.report)
    assert [(flag["cue_ids"], flag["severity"]) for flag in flags] == [([2], "warning")]
    assert (flags[0]["start"], flags[0]["end"]) == (5.0, 5.5)
    # Timing is reported, not moved: the cue still starts on its first ASR word.
    delivered = parse_srt_text(first.decode("utf-8"))
    assert [cue.start_ms for cue in delivered] == [0, 5500]
    items = [item for item in result.report["review"] if item["kind"] == KIND]
    assert [(item["severity"], item["srt_numbers"], item["title"]) for item in items] == [
        ("warning", [2], "Cue starts after the speech begins"),
    ]
    saved = json.loads((result.episode_workdir / "qc_report.json").read_text(encoding="utf-8"))
    assert len(_late_flags(saved)) == 1
    assert "Cue starts after the speech begins" in (result.episode_workdir / "qc_report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(("late_ms", "timing", "with_vad", "start_ms"), [
    (200, None, True, 5000),                              # inside the snap window: moved onto the onset
    (500, None, False, 5500),                             # no speech regions: no acoustic evidence
    (150, {"phrase_edge_snap": False}, True, 5134),       # snap disabled: the default window stays the floor
])
def test_sync_stays_silent_without_a_sustained_exclusive_lead(tmp_path, late_ms, timing, with_vad, start_ms):
    srt, audio, providers = _sync_assets(tmp_path, late_ms=late_ms, timing=timing, with_vad=with_vad)

    result = pipeline.sync_episode(srt, audio, tmp_path / "output.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, fps=30.0)

    assert _late_flags(result.report) == []
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert delivered[1].start_ms == start_ms


def test_sync_reports_a_late_start_when_the_snap_is_disabled(tmp_path):
    srt, audio, providers = _sync_assets(tmp_path, late_ms=500, timing={"phrase_edge_snap": False})

    result = pipeline.sync_episode(srt, audio, tmp_path / "output.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, fps=30.0)

    assert [flag["cue_ids"] for flag in _late_flags(result.report)] == [[2]]


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_late_start_of_a_split_cue_is_reported_on_its_first_delivered_child_only(tmp_path, monkeypatch, mode):
    lines = ["We finished the work.", "Now we can go home.", "Please bring the keys."]
    tokens = " ".join(lines).split()
    words = [Word(text=token, start=1 + index * .3, end=1.22 + index * .3, speaker_id="actor")
             for index, token in enumerate(tokens)]
    audio, fixture, config, source = (
        tmp_path / name for name in ("episode.wav", "words.json", "providers.yaml", "episode.srt")
    )
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 128000)
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    source.write_text(write_srt([Cue(index=1, start_ms=1000, end_ms=5000, lines=lines)]), encoding="utf-8")
    profile = StyleProfile(max_chars_per_line=24, max_lines_per_cue=4, min_cue_dur=.1, tail_ms=0)
    # The burst begins half a second before the first ASR word.
    monkeypatch.setattr(pipeline, "speech_evidence_for_words", lambda *_a, **_k: SpeechEvidence(
        words=words, regions=[SpeechRegion(start=.5, end=words[-1].end)], detected=True,
    ))

    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, style_profile=profile, no_llm=True, **kwargs)

    result = run()
    if mode != "fresh":
        result = run(resume=mode)

    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    delivered = sorted(payload["cues"], key=lambda cue: cue["start_ms"])
    assert len(delivered) >= 2
    flags = _late_flags(result.report)
    assert [flag["cue_ids"] for flag in flags] == [[delivered[0]["index"]]]
    assert (flags[0]["start"], flags[0]["end"]) == (.5, 1.0)
