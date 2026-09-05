from __future__ import annotations

import json
import wave

import pytest

from dubsync.evaluation import evaluate_against_golden
from dubsync.models import AlignmentResult, Cue, ForcedAlignmentCue, QCFlag, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import GenerationConstraints, StyleProfile
from dubsync.transcription import _build_cues_with_word_ownership, build_cues_from_words, generate_srt_from_audio
from dubsync.verify import cps_sanity_flags, lint_cues, score_cues


@pytest.mark.parametrize(
    ("left_text", "right_text", "left_start", "right_start", "left_speaker", "right_speaker"),
    [
        ("I can hear you.", "I can't hear you.", 0, 0, "A", "A"),
        ("Stop!", "Stop!", 0, 0, "A", "B"),
        ("No!", "No!", 0, 100, "A", "A"),
    ],
)
def test_export_preserves_distinct_overlapping_utterances(
    left_text, right_text, left_start, right_start, left_speaker, right_speaker
):
    cues = [
        Cue(index=1, start_ms=left_start, end_ms=600, lines=[left_text], speaker_id=left_speaker),
        Cue(index=2, start_ms=right_start, end_ms=1000, lines=[right_text], speaker_id=right_speaker),
    ]

    actual, flags = finalize_cues_for_output(cues, StyleProfile(), no_overlaps=False)

    assert [cue.plain_text for cue in actual] == [left_text, right_text]
    assert not any(flag.kind == "duplicate_cue_merged" for flag in flags)


def test_export_does_not_delay_acoustically_timed_overlapping_speech():
    cues = [
        Cue(index=1, start_ms=0, end_ms=600, lines=["First speaker."], speaker_id="A"),
        Cue(index=2, start_ms=400, end_ms=700, lines=["Reply."], speaker_id="B"),
    ]

    actual, flags = finalize_cues_for_output(
        cues, StyleProfile(), preserve_timing=True, max_cps=5, no_overlaps=True
    )

    assert actual == cues
    assert any(flag.kind == "output_overlap_unresolved" and flag.severity == "error" for flag in flags)
    assert not any(flag.kind in {"cps_cue_merged", "cps_duration_extended"} for flag in flags)


def test_export_does_not_insert_speaker_gap_after_acoustic_refinement():
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["First."], speaker_id="A"),
        Cue(index=2, start_ms=500, end_ms=700, lines=["Yes."], speaker_id="B"),
    ]

    actual, flags = finalize_cues_for_output(cues, StyleProfile(), preserve_timing=True)

    assert actual == cues
    assert not flags


def test_export_caps_unprotected_tail_at_media_end_and_preserves_source_holds():
    cues = [
        Cue(index=1, start_ms=700, end_ms=1400, lines=["Finish."]),
        Cue(index=2, start_ms=1500, end_ms=2000, lines=["Held source."]),
        Cue(index=3, start_ms=2100, end_ms=2500, lines=["[Screen text]"]),
    ]

    actual, flags = finalize_cues_for_output(
        cues, StyleProfile(), media_duration_ms=950, protected_cue_ids={2}
    )

    assert actual[0].end_ms <= 950
    assert actual[0].end_ms > actual[0].start_ms
    assert actual[1:] == cues[1:]
    assert any(flag.kind == "media_boundary_clamped" and flag.cue_ids == [1] for flag in flags)
    assert {cue_id for flag in flags if flag.kind == "cue_outside_media" for cue_id in flag.cue_ids} == {2, 3}


def test_generation_never_creates_zero_length_cue_for_simultaneous_speaker_starts():
    words = [
        Word(text="Hello!", start=0.10, end=0.20, speaker_id="A"),
        Word(text="Wait!", start=0.11, end=0.30, speaker_id="B"),
    ]

    cues = build_cues_from_words(words, StyleProfile())

    assert [cue.plain_text for cue in cues] == ["Hello!", "Wait!"]
    assert all(cue.duration_ms > 0 for cue in cues)


def test_generation_preserves_earlier_overlapping_word_endpoint():
    words = [
        Word(text="Drawn", start=0.1, end=0.9),
        Word(text="out.", start=0.2, end=0.3),
    ]

    cues = build_cues_from_words(words, StyleProfile(tail_ms=0), preserve_timing=True)

    assert len(cues) == 1
    assert cues[0].end_ms >= 900


def test_generation_confidence_mapping_excludes_touching_and_other_speaker_words():
    words = [
        Word(text="Before.", start=0.0, end=1.0, speaker_id="A"),
        Word(text="Mine.", start=1.0, end=1.4, speaker_id="A"),
        Word(text="Other.", start=1.0, end=1.4, speaker_id="B"),
        Word(text="After.", start=1.4, end=2.0, speaker_id="A"),
    ]
    _, alignment = _build_cues_with_word_ownership(
        words,
        StyleProfile(min_cue_dur=0.1),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
        preserve_timing=True,
    )

    assert alignment.cue_word_indices == {1: [0], 2: [1], 3: [2], 4: [3]}


def test_export_media_cap_does_not_cut_final_speech_to_the_previous_frame():
    cue = Cue(index=1, start_ms=800, end_ms=1000, lines=["Finish."])

    actual, _ = finalize_cues_for_output(
        [cue], StyleProfile(), media_duration_ms=950, preserve_timing=True
    )

    assert actual[0].end_ms == 950


def test_zero_duration_dialogue_is_a_qc_error_even_when_min_duration_is_disabled():
    cue = Cue(index=1, start_ms=1000, end_ms=1000, lines=["Invisible speech."])

    issues = lint_cues([cue], StyleProfile(min_cue_dur=0))
    flags = cps_sanity_flags([cue])

    assert any(issue.kind == "zero_duration" and issue.severity == "error" for issue in issues)
    assert any(flag.kind == "invalid_cue_duration" and flag.severity == "error" for flag in flags)


def _generation_fixture(tmp_path, words, *, duration_seconds=1.0, vad_regions=None):
    audio_path = tmp_path / "dialogue.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * round(duration_seconds * 16000))
    words_path = tmp_path / "words.json"
    words_path.write_text(json.dumps({"words": words}), encoding="utf-8")
    config = f"asr:\n  fixture_path: '{words_path.as_posix()}'\n"
    if vad_regions is not None:
        vad_path = tmp_path / "vad.json"
        vad_path.write_text(json.dumps({"regions": vad_regions}), encoding="utf-8")
        config += f"vad:\n  fixture_path: '{vad_path.as_posix()}'\n  boundary_refinement: true\n"
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(config, encoding="utf-8")
    return generate_srt_from_audio(
        audio_path,
        tmp_path / "result.srt",
        tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
        style_profile=StyleProfile(min_cue_dur=0.5),
        generation_constraints=GenerationConstraints(max_cps=5, min_cps=0),
    )


def test_generation_does_not_extend_last_spoken_word_past_media_end(tmp_path):
    result = _generation_fixture(
        tmp_path, [{"text": "Finishing.", "start": 0.8, "end": 0.95}], duration_seconds=1.0
    )
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))

    assert cues[0].end_ms <= 1000
    assert cues[0].start_ms == 800
    assert any(flag["kind"] == "impossible_cps_fast" for flag in result.report["flags"])


def test_generation_uses_configured_speech_evidence_before_export(tmp_path):
    result = _generation_fixture(
        tmp_path,
        [{"text": "Done.", "start": 0.1, "end": 3.0}],
        duration_seconds=4.0,
        vad_regions=[{"start": 0.1, "end": 0.4}],
    )
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))

    assert cues[0].end_ms <= 467
    assert (result.episode_workdir / "vad.json").exists()
    assert any(flag["kind"] == "asr_word_clamped" for flag in result.report["flags"])


def test_evaluation_rejects_correct_onsets_with_incorrect_offsets():
    golden = [Cue(index=1, start_ms=0, end_ms=500, lines=["Hello."])]
    predicted = [golden[0].with_timing(0, 3500)]

    metrics = evaluate_against_golden(predicted, golden, fps=30)

    assert metrics["meets_timing_target"] is False
    assert metrics["end_mae_ms"] == 3000
    assert metrics["ends_within_1_frame_ratio"] == 0.0


def test_evaluation_cannot_pass_timing_with_missing_dialogue():
    golden = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["Hello."]),
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["Entire missing sentence."]),
    ]

    metrics = evaluate_against_golden(golden[:1], golden, fps=30)

    assert metrics["meets_timing_target"] is False
    assert metrics["golden_match_coverage"] == 0.5
    assert metrics["predicted_match_coverage"] == 1.0


def test_evaluation_review_burden_counts_only_current_cue_ids():
    cues = [Cue(index=1, start_ms=0, end_ms=500, lines=["Hello."])]
    flags = [QCFlag(kind="text_changed", cue_ids=[1, 2, 3], message="Old merged cue IDs.")]

    metrics = evaluate_against_golden(cues, cues, fps=30, flags=flags)

    assert metrics["review_burden_ratio"] == 1.0


def test_generation_refinement_does_not_borrow_next_sentence_from_display_padding(tmp_path):
    result = _generation_fixture(
        tmp_path,
        [
            {"text": "First.", "start": 0.1, "end": 0.7, "speaker_id": "A", "confidence": 0.4},
            {"text": "Next.", "start": 0.71, "end": 1.3, "speaker_id": "A", "confidence": 0.99},
        ],
        duration_seconds=2.0,
        vad_regions=[{"start": 0.1, "end": 1.3}],
    )
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))

    assert [cue.plain_text for cue in cues] == ["First.", "Next."]
    assert all(cue.duration_ms > 0 for cue in cues)
    assert 700 <= cues[0].end_ms <= 767
    assert cues[1].start_ms == 700
    assert result.report["cue_scores"][0]["score"] == 0.4
    generated = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    assert generated["cue_word_indices"] == {"1": [0], "2": [1]}


def test_generation_keeps_repeated_words_with_the_same_snapped_onset(tmp_path):
    result = _generation_fixture(
        tmp_path,
        [
            {"text": "No.", "start": 0.1, "end": 0.7, "speaker_id": "A"},
            {"text": "No.", "start": 0.11, "end": 0.8, "speaker_id": "A"},
        ],
        duration_seconds=1.0,
    )
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))

    assert [cue.plain_text for cue in cues] == ["No.", "No."]
    assert all(cue.duration_ms > 0 for cue in cues)
    assert not any(flag["kind"] == "duplicate_cue_merged" for flag in result.report["flags"])


def test_preserved_source_hold_participates_in_final_overlap_qc():
    held = Cue(index=1, start_ms=1000, end_ms=2000, lines=["Held source."])
    spoken = Cue(index=2, start_ms=1500, end_ms=2500, lines=["Audible line."])

    actual, flags = finalize_cues_for_output(
        [held, spoken], StyleProfile(), protected_cue_ids={1}, preserve_timing=True
    )

    assert actual == [held, spoken]
    overlap_flags = [flag for flag in flags if flag.kind == "output_overlap_unresolved"]
    assert len(overlap_flags) == 1
    assert overlap_flags[0].cue_ids == [1, 2]


@pytest.mark.parametrize("invalid_kind", ["reversed", "duplicate", "protected"])
def test_cue_confidence_does_not_trust_rejected_forced_alignment(invalid_kind):
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=["Spoken."])
    words = [Word(text="Spoken.", start=1.0, end=2.0, confidence=0.4)]
    alignment = AlignmentResult(cue_word_indices={1: [0]})
    forced = [ForcedAlignmentCue(cue_id=1, start=1.0, end=2.0, score=1.0)]
    if invalid_kind == "reversed":
        forced[0] = forced[0].model_copy(update={"start": 3.0, "end": 2.0})
    elif invalid_kind == "duplicate":
        forced.append(forced[0].model_copy())
    else:
        alignment.diagnostics.missing_audio_cue_ids = [1]

    scores = score_cues([cue], words, alignment, forced)

    assert scores[0].source != "forced_alignment"
    assert scores[0].score != 1.0
