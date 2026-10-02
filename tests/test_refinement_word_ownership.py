"""Boundary refinement judges burst ownership of a cue's own words by the rule word repair uses.

A cue edge may only land inside its own first or last word where the burst owns that word; a burst
that word repair judged too small to own it does not move the edge, so the delivered cue never
starts after its first word or ends before its last.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
import yaml

from dubsync import pipeline
from dubsync.asr_timing import PhraseEdgeSnap, repair_asr_word_edges
from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.recue import cue_spoken_spans, rebuild_cues
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity
from dubsync.transcription import generate_srt_from_audio

PROFILE = StyleProfile(fps=25.0)
SNAP = PhraseEdgeSnap(start_advance=0.2, end_extension=0.3)

# ep02-pt Scribe cue 599 (origin 1695 s removed): "E" starts in silence and overlaps the burst
# after it by 43 ms, under the 50 ms word repair needs to move it.
START_WORDS = (("E", 1.388, 1.568), ("aí,", 1.848, 2.048), ("surpresa?", 2.128, 2.708))
START_REGIONS = ((1.525, 2.035), (2.115, 2.275), (2.355, 2.535), (2.615, 2.715))
# The same words with a short first burst whose offset the phrase snap extends "E" to.
START_SNAPPED_REGIONS = ((1.525, 1.700), (1.900, 2.035), (2.115, 2.275), (2.355, 2.535), (2.615, 2.715))
# ep02-pt MAI cue 662 (origin 1830 s removed): "casaco." overlaps a 70 ms burst by 25 ms; the phrase
# snap moves its start onto the burst onset, which alone made the burst look like its owner.
END_WORDS = (
    ("Obrigada", 3.560, 3.939), ("por", 3.960, 4.060), ("me", 4.120, 4.179),
    ("emprestar", 4.220, 4.560), ("o", 4.600, 4.620), ("casaco.", 4.720, 5.159),
)
END_REGIONS = ((3.475, 4.335), (4.435, 4.595), (4.675, 4.745), (5.705, 6.155))
CASES = {
    "start-ep02-599": ("E aí, surpresa?", 1510, 3000, START_WORDS, START_REGIONS),
    "start-end-snapped": ("E aí, surpresa?", 1510, 3000, START_WORDS, START_SNAPPED_REGIONS),
    "end-ep02-662": ("Obrigada por me emprestar o casaco.", 3240, 4750, END_WORDS, END_REGIONS),
    # The verifier's shapes: a 40 ms overlap 140 ms into the first word; a 30 ms raw overlap of the last.
    "start-overlap-40": (
        "E aí, surpresa?", 4000, 7000,
        (("E", 5.300, 5.480), ("aí,", 5.520, 5.700), ("surpresa?", 5.740, 6.300)), ((5.440, 6.350),),
    ),
    "end-raw-overlap-30": (
        "Deixa o casaco.", 500, 3000,
        (("Deixa", 1.000, 1.300), ("o", 1.320, 1.400), ("casaco.", 1.730, 2.240)), ((0.980, 1.420), (1.690, 1.760)),
    ),
}


def _words(rows) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in rows]


def _regions(rows) -> list[SpeechRegion]:
    return [SpeechRegion(start=start, end=end) for start, end in rows]


def _assert_cue_covers_its_own_words(cue: Cue, words: list[Word]) -> None:
    assert cue.start_ms <= PROFILE.snap_floor(min(word.start for word in words) * 1000)
    assert cue.end_ms >= PROFILE.snap_ceil(max(word.end for word in words) * 1000)


def _timed(raw: list[Word], regions: list[SpeechRegion], cue: Cue, *, source_words: bool = True):
    """Word repair, rebuild, refinement and output finalisation as the pipeline runs them."""
    words, _ = repair_asr_word_edges(raw, regions, max_region_overrun=0.3, snap=SNAP)
    alignment = AlignmentResult(cue_word_indices={cue.index: list(range(len(words)))})
    rebuilt, _ = rebuild_cues([cue], words, alignment, PROFILE)
    refined, flags = refine_cues_to_speech_activity(
        rebuilt, regions, PROFILE, BoundaryRefinementConfig(), words=words, alignment=alignment,
        ambiguous_word_indices=set(), **({"source_words": raw} if source_words else {}),
    )
    final, _ = finalize_cues_for_output(
        refined, PROFILE, preserve_timing=True, spoken_spans=cue_spoken_spans(refined, words, alignment),
    )
    return words, rebuilt[0], final[0], flags


@pytest.mark.parametrize("source_words", [True, False], ids=["provider-words", "words-only"])
def test_a_burst_that_does_not_own_the_first_word_does_not_move_the_cue_start(source_words):
    raw, regions = _words(START_WORDS), _regions(START_REGIONS)
    cue = Cue(index=1, start_ms=1510, end_ms=3000, lines=["E aí, surpresa?"])

    words, rebuilt, delivered, flags = _timed(raw, regions, cue, source_words=source_words)

    # Word repair left "E" where the provider put it: 43 ms is too little of it to move it onto the burst.
    assert words[0] == raw[0]
    _assert_cue_covers_its_own_words(delivered, words)
    assert (delivered.start_ms, delivered.end_ms) == (rebuilt.start_ms, rebuilt.end_ms) == (1360, 2760)
    assert not [flag for flag in flags if flag.kind == "timing_refined"]


def test_an_end_snapped_onto_a_short_burst_does_not_make_it_the_owner_of_the_first_word():
    # "E" still overlaps its burst by 43 ms, but the burst now ends 132 ms after the word: the phrase
    # snap extends the word's end onto it, which gives the repaired word 175 ms inside the burst.
    raw, regions = _words(START_WORDS), _regions(START_SNAPPED_REGIONS)
    cue = Cue(index=1, start_ms=1510, end_ms=3000, lines=["E aí, surpresa?"])

    words, rebuilt, delivered, flags = _timed(raw, regions, cue)

    assert (words[0].start, words[0].end) == (1.388, 1.7)
    _assert_cue_covers_its_own_words(delivered, words)
    assert delivered.start_ms == rebuilt.start_ms == 1360
    assert not [flag for flag in flags if flag.kind == "timing_refined"]


def test_a_burst_that_owns_the_first_word_still_starts_the_cue_on_it():
    # 60 ms of "E" lie in the burst: word repair moves the word onto it and refinement agrees.
    raw = _words((("E", 1.388, 1.585), *START_WORDS[1:]))
    cue = Cue(index=1, start_ms=1510, end_ms=3000, lines=["E aí, surpresa?"])

    words, _, delivered, _ = _timed(raw, _regions(START_REGIONS), cue)

    assert words[0].start == 1.525
    assert delivered.start_ms == PROFILE.snap_floor(1525)
    _assert_cue_covers_its_own_words(delivered, words)


def test_a_burst_reached_only_by_the_phrase_snap_does_not_end_the_cue_inside_the_last_word():
    raw, regions = _words(END_WORDS), _regions(END_REGIONS)
    cue = Cue(index=1, start_ms=3240, end_ms=4750, lines=["Obrigada por me emprestar o casaco."])

    words, rebuilt, delivered, flags = _timed(raw, regions, cue)

    # The snap moved the start of "casaco." onto the 70 ms burst; its provider end stays.
    assert (words[-1].start, words[-1].end) == (4.675, 5.159)
    _assert_cue_covers_its_own_words(delivered, words)
    assert (delivered.start_ms, delivered.end_ms) == (rebuilt.start_ms, rebuilt.end_ms) == (3440, 5200)
    assert not [flag for flag in flags if flag.kind == "timing_refined"]


def test_a_burst_that_owns_the_last_word_ends_both_the_word_and_the_cue():
    # 55 ms of the provider's "casaco." lie in the burst: word repair ends the word with the burst
    # and refinement ends the cue there, padded, without cutting into the word.
    raw = _words((*END_WORDS[:-1], ("casaco.", 4.690, 5.159)))
    cue = Cue(index=1, start_ms=3240, end_ms=4750, lines=["Obrigada por me emprestar o casaco."])

    words, _, delivered, _ = _timed(raw, _regions(END_REGIONS), cue)

    assert (words[-1].start, words[-1].end) == (4.675, 4.745)
    assert delivered.end_ms == PROFILE.snap_ceil(4745 + 40)
    _assert_cue_covers_its_own_words(delivered, words)


def test_a_burst_owning_the_provider_word_still_ends_the_cue_at_its_offset():
    # The provider's word runs 80 ms past the voice (a typical MAI end): the burst owns 420 of its 500 ms.
    profile = StyleProfile(fps=30, min_cue_dur=0.1)

    refined, _ = refine_cues_to_speech_activity(
        [Cue(index=1, start_ms=1000, end_ms=2500, lines=["Oi."])], [SpeechRegion(start=1.0, end=1.42)], profile,
        words=_words((("Oi.", 1.0, 1.5),)), alignment=AlignmentResult(cue_word_indices={1: [0]}),
        source_words=_words((("Oi.", 1.0, 1.5),)),
    )

    assert refined[0].end_ms == profile.snap_ceil(1420 + 40)


def test_provider_words_must_line_up_with_the_repaired_words():
    with pytest.raises(ValueError, match="source_words"):
        refine_cues_to_speech_activity(
            [Cue(index=1, start_ms=1000, end_ms=2500, lines=["Oi."])], [SpeechRegion(start=1.0, end=1.42)], PROFILE,
            words=_words((("Oi.", 1.0, 1.5),)), alignment=AlignmentResult(cue_word_indices={1: [0]}),
            source_words=[],
        )


# --- The real entry points: synchronization and generation with fixture ASR and fixture VAD ---------


def _fixture_run(tmp_path: Path, words, regions) -> tuple[Path, Path]:
    audio, wordstream, vad_fixture, providers = (
        tmp_path / name for name in ("episode.wav", "words.json", "regions.json", "providers.yaml")
    )
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 8)
    wordstream.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "speaker_id": "A"} for text, start, end in words
    ]}), encoding="utf-8")
    vad_fixture.write_text(json.dumps({"regions": [{"start": start, "end": end} for start, end in regions]}),
                           encoding="utf-8")
    # The shipped timing defaults: boundary refinement on, phrase edge snap at its defaults.
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "vad": {"fixture_path": str(vad_fixture), "min_coverage": 0.2, "boundary_refinement": {"enabled": True}},
        "output": {"no_overlaps": True},
    }), encoding="utf-8")
    return audio, providers


def _effective_words(words, regions) -> list[Word]:
    return repair_asr_word_edges(_words(words), _regions(regions), max_region_overrun=0.3, snap=SNAP)[0]


@pytest.mark.parametrize("mode", ["fresh", "verify"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_sync_never_moves_a_cue_edge_inside_its_own_words(tmp_path, case, mode):
    text, start_ms, end_ms, words, regions = CASES[case]
    audio, providers = _fixture_run(tmp_path, words, regions)
    source = tmp_path / "episode.srt"
    source.write_text(f"1\n{_srt_time(start_ms)} --> {_srt_time(end_ms)}\n{text}\n", encoding="utf-8")

    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=providers, no_llm=True, style_profile=PROFILE, **kwargs)

    result = run()
    if mode == "verify":
        # Refinement runs again on the cues it refined: the resumed output is the same.
        first = result.output_srt.read_bytes()
        result = run(resume="verify")
        assert result.output_srt.read_bytes() == first

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in delivered] == [text]
    _assert_cue_covers_its_own_words(delivered[0], _effective_words(words, regions))
    assert not [flag for flag in result.report["flags"] if flag["kind"] == "timing_refined"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_generation_never_moves_a_cue_edge_inside_its_own_words(tmp_path, case):
    text, _, _, words, regions = CASES[case]
    audio, providers = _fixture_run(tmp_path, words, regions)

    result = generate_srt_from_audio(audio, tmp_path / "output.srt", tmp_path / "work", providers_path=providers,
                                     no_llm=True, style_profile=PROFILE)

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in delivered] == [text]
    _assert_cue_covers_its_own_words(delivered[0], _effective_words(words, regions))


def _srt_time(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    return f"00:00:{seconds:02d},{millis:03d}"
