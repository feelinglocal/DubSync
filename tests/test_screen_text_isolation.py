from __future__ import annotations

import pytest

import json
import wave
from array import array

from dubsync.aligner import MISSING_AUDIO_GUARD_VERSION, align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.cue_segmentation import split_overlong_existing_cues
from dubsync.forced_alignment import apply_forced_alignment
from dubsync.llm_providers import _speaker_mapping_prompt
from dubsync.models import (
    AdjudicationDecision,
    AlignmentResult,
    Cue,
    DivergenceSpan,
    ForcedAlignmentCue,
    SpeechRegion,
    Word,
)
from dubsync.output_order import finalize_cues_for_output
from dubsync.pipeline import (
    _spoken_source_cue_count,
    _validate_alignment_screen_text_provenance,
)
from dubsync.punctuation import apply_punctuation_pass
from dubsync.silence import silence_flags_for_cues
from dubsync.style_profile import StyleProfile, derive_style_profile
from dubsync.subtitle_annotations import (
    cue_has_bracketed_screen_text,
    is_bracketed_screen_text_cue,
    speech_text_for_alignment,
)
from dubsync.timing_refinement import refine_cues_to_speech_activity
from dubsync.verify import cps_sanity_flags, score_cues


def _profile() -> StyleProfile:
    return StyleProfile(
        fps=30.0,
        min_cue_dur=0.5,
        max_chars_per_line=26,
        max_lines_per_cue=2,
    )


def test_representative_screen_text_shapes_keep_only_spoken_residue() -> None:
    cases = [
        (Cue(index=1, start_ms=0, end_ms=1_000, lines=['[Adapted from "Morning Story"]']), "", True),
        (Cue(index=2, start_ms=0, end_ms=1_000, lines=["[Episode 1]", '["On-screen quote"]']), "", True),
        (Cue(index=3, start_ms=0, end_ms=1_000, lines=["[Group A", "Director Office]"]), "", True),
        (
            Cue(index=4, start_ms=0, end_ms=1_000, lines=["[Company notice]", "Spoken dialogue."]),
            "Spoken dialogue.",
            False,
        ),
        (
            Cue(index=5, start_ms=0, end_ms=1_000, lines=['[A: "Label"]', "Another spoken sentence."]),
            "Another spoken sentence.",
            False,
        ),
    ]

    for cue, expected_speech, expected_visual_only in cases:
        assert speech_text_for_alignment(cue) == expected_speech
        assert is_bracketed_screen_text_cue(cue) is expected_visual_only


def test_malformed_brackets_fail_open_as_spoken_text() -> None:
    cue = Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Document label", "hello"])

    assert speech_text_for_alignment(cue) == "[Document label hello"
    assert cue_has_bracketed_screen_text(cue) is False


def test_vad_refinement_preserves_visual_timing_and_does_not_cap_dialogue() -> None:
    spoken = Cue(index=1, start_ms=1_000, end_ms=1_400, lines=["hello"])
    visual = Cue(index=2, start_ms=1_600, end_ms=2_400, lines=["[Document title]"])
    following = Cue(index=3, start_ms=3_000, end_ms=3_500, lines=["later"])
    words = [
        Word(text="hello", start=1.10, end=1.85),
        Word(text="later", start=3.10, end=3.30),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0], 3: [1]})
    regions = [
        SpeechRegion(start=1.00, end=2.00),
        SpeechRegion(start=3.00, end=3.40),
    ]

    with_visual, visual_flags = refine_cues_to_speech_activity(
        [spoken, visual, following],
        regions,
        _profile(),
        words=words,
        alignment=alignment,
    )
    control, _ = refine_cues_to_speech_activity(
        [spoken, following],
        regions,
        _profile(),
        words=words,
        alignment=alignment,
    )

    assert [(cue.start_ms, cue.end_ms) for cue in with_visual if cue.index != 2] == [
        (cue.start_ms, cue.end_ms) for cue in control
    ]
    preserved = next(cue for cue in with_visual if cue.index == 2)
    assert preserved.model_dump() == visual.model_dump()
    assert not any(2 in flag.cue_ids for flag in visual_flags)


def test_final_output_preserves_visual_timing_and_matches_bracket_stripped_control() -> None:
    spoken = Cue(index=1, start_ms=0, end_ms=1_200, lines=["alpha"])
    visual = Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Location]"])
    following = Cue(index=3, start_ms=1_900, end_ms=2_500, lines=["beta"])

    with_visual, flags = finalize_cues_for_output(
        [spoken, visual, following],
        _profile(),
        no_overlaps=True,
    )
    control, _ = finalize_cues_for_output(
        [spoken, following],
        _profile(),
        no_overlaps=True,
    )

    assert [(cue.index, cue.start_ms, cue.end_ms) for cue in with_visual if cue.index != 2] == [
        (cue.index, cue.start_ms, cue.end_ms) for cue in control
    ]
    preserved = next(cue for cue in with_visual if cue.index == 2)
    assert preserved.model_dump() == visual.model_dump()
    assert not any(2 in flag.cue_ids for flag in flags)


class _RecordingPunctuationAdapter:
    def __init__(self) -> None:
        self.seen_ids: list[int] = []

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        self.seen_ids.extend(cue.index for cue in cues)
        return {cue.index: f"{cue.plain_text}!" for cue in cues}


def test_punctuation_never_receives_or_changes_annotated_cues() -> None:
    visual = Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Episode 1]"])
    mixed = Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello there"])
    spoken = Cue(index=3, start_ms=2_000, end_ms=3_000, lines=["we go now"])
    adapter = _RecordingPunctuationAdapter()

    updated, _ = apply_punctuation_pass(
        [visual, mixed, spoken],
        adapter,
        source_cues=[visual, mixed, spoken],
    )

    assert adapter.seen_ids == [3]
    assert updated[0].model_dump() == visual.model_dump()
    assert updated[1].model_dump() == mixed.model_dump()
    assert updated[2].plain_text == "we go now!"


def test_forced_alignment_cannot_retime_visual_only_cue_from_stale_row() -> None:
    visual = Cue(index=1, start_ms=1_000, end_ms=2_000, lines=["[Episode 1]"])

    updated, flags = apply_forced_alignment(
        [visual],
        [ForcedAlignmentCue(cue_id=1, start=10.0, end=11.0, score=0.99)],
        _profile(),
    )

    assert updated[0].model_dump() == visual.model_dump()
    assert flags == []


def test_mixed_cue_adjudication_edits_spoken_text_without_touching_label() -> None:
    source = Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Document label]", "hello wrong"])
    words = [
        Word(text="hello", start=2.0, end=2.2),
        Word(text="right", start=2.3, end=2.5),
    ]
    alignment = align_cues_to_words([source], words)
    span = alignment.divergence_spans[0]
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="right",
        confidence=0.99,
        reason="fixture audio",
    )

    updated, _ = apply_adjudication_decisions(
        [source],
        alignment.divergence_spans,
        [decision],
        _profile(),
    )

    assert updated[0].lines == ["[Document label]", "hello right"]


def test_mixed_cue_adjudication_holds_edit_that_crosses_screen_text() -> None:
    source = Cue(index=1, start_ms=0, end_ms=1_000, lines=["hello [Document label] wrong"])
    span = DivergenceSpan(
        case_id="crosses-label",
        cue_ids=[1],
        srt_text="hello wrong",
        asr_text="fixed",
        srt_token_indices=[0, 1],
        asr_word_indices=[0],
    )
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="fixed",
        confidence=0.99,
        reason="fixture audio",
    )

    updated, flags = apply_adjudication_decisions([source], [span], [decision], _profile())

    assert updated[0].model_dump() == source.model_dump()
    assert [flag.kind for flag in flags] == ["screen_text_adjudication_held"]


def test_mixed_cue_spoken_deletion_preserves_exact_label_line() -> None:
    source = Cue(index=1, start_ms=0, end_ms=1_000, lines=["[  Document label  ]", "wrong"])
    alignment = align_cues_to_words([source], [])
    span = alignment.divergence_spans[0]
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="",
        confidence=0.99,
        reason="fixture audio",
    )

    updated, flags = apply_adjudication_decisions(
        [source],
        alignment.divergence_spans,
        [decision],
        _profile(),
    )

    assert updated[0].lines == ["[  Document label  ]"]
    assert [flag.kind for flag in flags] == ["text_changed"]


def test_mixed_cue_adjudication_maps_duplicate_label_token_to_spoken_line() -> None:
    source = Cue(index=1, start_ms=0, end_ms=1_000, lines=["[hello label]", "hello wrong"])
    words = [
        Word(text="hello", start=2.0, end=2.2),
        Word(text="right", start=2.3, end=2.5),
    ]
    alignment = align_cues_to_words([source], words)
    span = alignment.divergence_spans[0]
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="right",
        confidence=0.99,
        reason="fixture audio",
    )

    updated, _ = apply_adjudication_decisions(
        [source],
        alignment.divergence_spans,
        [decision],
        _profile(),
    )

    assert updated[0].lines == ["[hello label]", "hello right"]


def test_line_limit_splitter_leaves_annotated_source_structure_untouched() -> None:
    source = Cue(index=1, start_ms=1_000, end_ms=2_000, lines=["[Document title]", "hello there"])
    words = [
        Word(text="hello", start=1.1, end=1.3),
        Word(text="there", start=1.4, end=1.7),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0, 1]})
    profile = _profile().model_copy(update={"max_lines_per_cue": 1, "max_chars_per_line": 12})

    updated, updated_alignment, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        source_cue_ids={1},
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert updated[0].model_dump() == source.model_dump()
    assert updated_alignment == alignment
    assert flags == []
    assert expansions == {}


def test_speaker_mapping_prompt_uses_only_spoken_residue() -> None:
    mixed = Cue(
        index=7,
        start_ms=0,
        end_ms=1_000,
        lines=['[A: "Label"]', "Another spoken sentence."],
        speaker_id="speaker_1",
    )
    visual = Cue(index=8, start_ms=1_000, end_ms=2_000, lines=["[Character name]"], speaker_id="speaker_2")

    payload = json.loads(_speaker_mapping_prompt([mixed, visual]))

    assert payload["speakers"] == [
        {
            "speaker_id": "speaker_1",
            "samples": [{"cue_id": 7, "text": "Another spoken sentence."}],
        }
    ]


def test_silence_qc_does_not_flag_visual_only_cues(tmp_path) -> None:
    audio_path = tmp_path / "silence.wav"
    samples = array("h", [0] * 16_000)
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())
    visual = Cue(index=1, start_ms=0, end_ms=500, lines=["[Episode 1]"])
    spoken = Cue(index=2, start_ms=500, end_ms=1_000, lines=["hello"])

    flags = silence_flags_for_cues(audio_path, [visual, spoken])

    assert [flag.cue_ids for flag in flags] == [[2]]


def test_dialogue_cps_and_scores_ignore_visual_only_text() -> None:
    visual = Cue(index=1, start_ms=0, end_ms=100, lines=["[A very long document title]"])
    mixed = Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello"])
    words = [Word(text="hello", start=1.1, end=1.4, confidence=0.98)]
    alignment = AlignmentResult(cue_word_indices={2: [0]})

    flags = cps_sanity_flags([visual, mixed], max_cps=10.0, min_cps=0.0)
    scores = score_cues([visual, mixed], words, alignment)

    assert flags == []
    assert [score.cue_id for score in scores] == [2]
    assert scores[0].cps == 5.0


def test_derived_dialogue_style_matches_bracket_stripped_control() -> None:
    spoken = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["short line"]),
        Cue(index=3, start_ms=2_000, end_ms=2_500, lines=["next line"]),
        Cue(index=4, start_ms=4_000, end_ms=4_500, lines=["last line"]),
    ]
    visual = Cue(
        index=2,
        start_ms=500,
        end_ms=1_900,
        lines=["[An exceptionally long document heading", "that uses three source lines", "and is not dubbed]"],
    )

    with_visual = derive_style_profile([spoken[0], visual, *spoken[1:]])
    control = derive_style_profile(spoken)

    assert with_visual.max_lines_per_cue == control.max_lines_per_cue
    assert with_visual.max_chars_per_line == control.max_chars_per_line
    assert with_visual.min_cue_dur == control.min_cue_dur
    assert with_visual.allow_zero_gap == control.allow_zero_gap


def test_derived_style_ignores_screen_text_inside_mixed_cues() -> None:
    mixed = Cue(
        index=1,
        start_ms=0,
        end_ms=500,
        lines=["[An exceptionally long on-screen document heading]", "hello"],
    )
    spoken_control = Cue(index=1, start_ms=0, end_ms=500, lines=["hello"])
    following = Cue(index=2, start_ms=1_000, end_ms=1_500, lines=["next line"])

    with_screen_text = derive_style_profile([mixed, following])
    control = derive_style_profile([spoken_control, following])

    assert with_screen_text.max_lines_per_cue == control.max_lines_per_cue
    assert with_screen_text.max_chars_per_line == control.max_chars_per_line
    assert with_screen_text.min_cue_dur == control.min_cue_dur
    assert with_screen_text.allow_zero_gap == control.allow_zero_gap


def test_alignment_health_denominator_counts_only_spoken_source_cues() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Episode 1]"]),
        Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello"]),
        Cue(index=3, start_ms=2_000, end_ms=3_000, lines=["later"]),
    ]

    assert _spoken_source_cue_count(cues) == 2


def test_alignment_records_all_annotated_cue_ids_for_resume_provenance() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Episode 1]"]),
        Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello"]),
    ]
    alignment = align_cues_to_words(
        cues,
        [Word(text="hello", start=3.0, end=3.4, confidence=0.99)],
    )

    assert alignment.diagnostics.excluded_screen_text_cue_ids == [1, 2]


def test_resume_rejects_alignment_without_matching_screen_text_provenance() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Episode 1]"]),
        Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello"]),
    ]
    stale_alignment = AlignmentResult()

    try:
        _validate_alignment_screen_text_provenance(stale_alignment, cues)
    except RuntimeError as exc:
        assert "resume from align" in str(exc).lower()
    else:
        raise AssertionError("stale alignment provenance was accepted")


def test_resume_accepts_matching_screen_text_provenance() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["[Episode 1]"]),
        Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["[Station]", "hello"]),
    ]
    alignment = AlignmentResult(
        diagnostics={
            "excluded_screen_text_cue_ids": [1, 2],
            "missing_audio_guard_version": MISSING_AUDIO_GUARD_VERSION,
        },
    )

    _validate_alignment_screen_text_provenance(alignment, cues)


def test_resume_rejects_alignment_that_predates_missing_audio_guard() -> None:
    cues = [Cue(index=1, start_ms=0, end_ms=1_000, lines=["hello"])]
    stale = AlignmentResult(
        diagnostics={
            "excluded_screen_text_cue_ids": [],
            "missing_audio_guard_version": 0,
        },
    )

    with pytest.raises(RuntimeError, match="predates the missing-audio timing guard"):
        _validate_alignment_screen_text_provenance(stale, cues)
