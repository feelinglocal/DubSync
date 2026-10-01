from __future__ import annotations

import json

import pytest
import yaml

from dubsync.aligner import align_cues_to_words
from dubsync.models import AlignmentResult, Cue, SpeechRegion, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.pipeline import sync_episode
from dubsync.recue import rebuild_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import BoundaryRefinementConfig, refine_cues_to_speech_activity


def _source_cues():
    return [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["今日は"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["晴れです"]),
    ]


def _timings(cues):
    return [(cue.index, cue.start_ms, cue.end_ms) for cue in cues]


def test_shared_japanese_asr_phrase_keeps_source_boundaries_with_advisory():
    cues = _source_cues()
    words = [Word(text="今日は晴れです", start=0.5, end=1.8)]
    alignment = align_cues_to_words(cues, words)
    assert alignment.cue_word_indices == {1: [0], 2: [0]}

    rebuilt, flags = rebuild_cues(cues, words, alignment, StyleProfile())

    assert _timings(rebuilt) == _timings(cues)
    assert [flag.kind for flag in flags] == ["shared_word_timing_preserved"] * 2
    assert all(flag.severity == "warning" for flag in flags)


def test_shared_word_fallback_is_not_shifted_by_previous_cue():
    cues = [Cue(index=0, start_ms=0, end_ms=300, lines=["前"]), *_source_cues()]
    words = [Word(text="前", start=0.0, end=0.7), Word(text="今日は晴れです", start=0.5, end=1.8)]
    alignment = AlignmentResult(cue_word_indices={0: [0], 1: [1], 2: [1]})

    rebuilt, _ = rebuild_cues(cues, words, alignment, StyleProfile())

    assert _timings(rebuilt[1:]) == _timings(cues[1:])


def test_shared_word_source_fallback_does_not_drag_next_reliable_cue():
    cues = [
        Cue(index=1, start_ms=10000, end_ms=11000, lines=["今日は"]),
        Cue(index=2, start_ms=11000, end_ms=12000, lines=["晴れです"]),
        Cue(index=3, start_ms=13000, end_ms=14000, lines=["終わり"]),
    ]
    words = [Word(text="今日は晴れです", start=0.5, end=1.8), Word(text="終わり", start=2.5, end=3.0)]
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [0], 3: [1]})

    rebuilt, _ = rebuild_cues(cues, words, alignment, StyleProfile(tail_ms=0))

    assert _timings(rebuilt) == [(1, 10000, 11000), (2, 11000, 12000), (3, 2500, 3000)]


def test_output_fallback_does_not_shift_overlapping_reliable_cue():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=3000, lines=["今日は"]),
        Cue(index=2, start_ms=2000, end_ms=2500, lines=["終わり"]),
    ]

    finalized, flags = finalize_cues_for_output(cues, StyleProfile(), protected_cue_ids={1})

    assert _timings(finalized) == _timings(cues)
    assert any(flag.kind == "output_overlap_preserved" for flag in flags)


def test_output_finalization_does_not_extend_merge_or_shift_protected_cues():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1500, lines=["今日は"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["今日は"]),
    ]

    finalized, flags = finalize_cues_for_output(
        cues, StyleProfile(), max_cps=1, protected_cue_ids={1, 2},
    )

    assert finalized == cues
    assert not any(flag.kind in {"cps_duration_extended", "cps_cue_merged", "duplicate_cue_merged"} for flag in flags)


def test_preserved_overlap_does_not_block_output_when_earlier_cue_moves_past_it():
    cues = [
        Cue(index=1, start_ms=0, end_ms=3000, lines=["first"]),
        Cue(index=2, start_ms=500, end_ms=1500, lines=["another"]),
        Cue(index=3, start_ms=1000, end_ms=2000, lines=["今日は"]),
    ]

    finalized, flags = finalize_cues_for_output(cues, StyleProfile(), protected_cue_ids={3})

    assert [(cue.start_ms, cue.end_ms) for cue in finalized if cue.index == 3] == [(1000, 2000)]
    assert [cue.start_ms for cue in finalized] == sorted(cue.start_ms for cue in finalized)
    assert any(flag.kind == "output_overlap_preserved" for flag in flags)


def _sync_fixture(tmp_path, *, forced_rows=None, vad=True, max_cps=1):
    srt = tmp_path / "japanese.srt"
    audio = tmp_path / "japanese.wav"
    words = tmp_path / "asr.json"
    providers = tmp_path / "providers.yaml"
    srt.write_text(write_srt(_source_cues()), encoding="utf-8")
    audio.write_bytes(b"fixture audio; providers use recorded timestamps")
    words.write_text(json.dumps({"words": [{"text": "今日は晴れです", "start": 0.5, "end": 1.8}]}), encoding="utf-8")
    config = {"asr": {"fixture_path": str(words)}, "timing": {"max_cps": max_cps, "min_cps": 0.1}}
    if vad:
        regions = tmp_path / "vad.json"
        regions.write_text(json.dumps({"regions": [{"start": 0.5, "end": 1.8}]}), encoding="utf-8")
        config["vad"] = {"fixture_path": str(regions), "boundary_refinement": True}
    if forced_rows is not None:
        forced = tmp_path / "forced.json"
        forced.write_text(json.dumps({"cues": forced_rows}), encoding="utf-8")
        config["forced_alignment"] = {"fixture_path": str(forced)}
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {
        "srt_path": srt, "audio_path": audio, "output_path": tmp_path / "output.srt",
        "workdir": tmp_path / "work", "providers_path": providers, "no_llm": True, "fps": 30,
    }


@pytest.mark.parametrize("resume", [False, True])
def test_fixture_sync_and_verify_resume_preserve_shared_word_source_timing(tmp_path, resume):
    args = _sync_fixture(tmp_path)
    result = sync_episode(**args)
    if resume:
        # Verification must repair damaged cue timing while retaining current policy metadata.
        artifact = result.episode_workdir / "rebuild.json"
        damaged = [cue.with_timing(500, 1866 + 500 * index) for index, cue in enumerate(_source_cues())]
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["cues"] = [cue.model_dump() for cue in damaged]
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        result = sync_episode(**args, resume="verify")

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert _timings(output) == _timings(_source_cues())
    flags = result.report["flags"]
    assert sum(flag["kind"] == "shared_word_timing_preserved" for flag in flags) == 2
    assert not any(flag["kind"] in {"timing_refined", "cps_cue_merged", "cps_duration_extended"} for flag in flags)


def test_real_per_cue_forced_alignment_resolves_shared_word_timing(tmp_path):
    args = _sync_fixture(tmp_path, max_cps=30, forced_rows=[
        {"cue_id": 1, "start": 0.2, "end": 0.9},
        {"cue_id": 2, "start": 1.1, "end": 1.7},
    ])
    result = sync_episode(**args)

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert _timings(output) == [(1, 200, 900), (2, 1100, 1700)]
    assert not any(flag["kind"] == "shared_word_timing_preserved" for flag in result.report["flags"])


def test_empty_forced_alignment_keeps_advisory_source_timing(tmp_path):
    result = sync_episode(**_sync_fixture(tmp_path, forced_rows=[]))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert _timings(output) == _timings(_source_cues())
    assert sum(flag["kind"] == "shared_word_timing_preserved" for flag in result.report["flags"]) == 2
    assert any(flag["kind"] == "forced_alignment_unavailable" for flag in result.report["flags"])


@pytest.mark.parametrize("invalid_start,invalid_end", [
    (1.2, 1.2), (1.8, 1.2), (-0.1, 1.2), (float("nan"), 1.8), (1.2, float("inf")),
])
def test_partial_forced_alignment_only_releases_resolved_cue(tmp_path, invalid_start, invalid_end):
    result = sync_episode(**_sync_fixture(tmp_path, max_cps=30, forced_rows=[
        {"cue_id": 1, "start": 0.2, "end": 0.9},
        {"cue_id": 2, "start": invalid_start, "end": invalid_end},
    ]))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert _timings(output) == [(1, 200, 900), (2, 1000, 2000)]
    assert [flag["cue_ids"] for flag in result.report["flags"] if flag["kind"] == "shared_word_timing_preserved"] == [[2]]


def test_successful_forced_alignment_resume_removes_stale_warning(tmp_path):
    args = _sync_fixture(tmp_path, max_cps=30, forced_rows=[])
    sync_episode(**args)
    (tmp_path / "forced.json").write_text(json.dumps({"cues": [
        {"cue_id": 1, "start": 0.2, "end": 0.9},
        {"cue_id": 2, "start": 1.1, "end": 1.7},
    ]}), encoding="utf-8")

    result = sync_episode(**args, resume="verify")

    assert _timings(parse_srt_text(result.output_srt.read_text(encoding="utf-8"))) == [(1, 200, 900), (2, 1100, 1700)]
    assert not any(flag["kind"] in {"forced_alignment_unavailable", "shared_word_timing_preserved"} for flag in result.report["flags"])


def test_fixture_vad_does_not_cap_reliable_cue_at_shared_source_boundary(tmp_path):
    args = _sync_fixture(tmp_path, max_cps=30)
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["前"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["今日は"]),
        Cue(index=3, start_ms=2000, end_ms=3000, lines=["晴れです"]),
    ]
    args["srt_path"].write_text(write_srt(cues), encoding="utf-8")
    (tmp_path / "asr.json").write_text(json.dumps({"words": [
        {"text": "前", "start": 2.5, "end": 3.0},
        {"text": "今日は晴れです", "start": 3.5, "end": 4.8},
    ]}), encoding="utf-8")
    (tmp_path / "vad.json").write_text(json.dumps({"regions": [
        {"start": 2.5, "end": 3.0}, {"start": 3.5, "end": 4.8},
    ]}), encoding="utf-8")

    result = sync_episode(**args)

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    by_text = {cue.plain_text: (cue.start_ms, cue.end_ms) for cue in output}
    assert by_text == {"前": (2500, 3067), "今日は": (1000, 2000), "晴れです": (2000, 3000)}


def test_vad_preserves_acoustic_endpoint_across_protected_source_cues():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["前"]),
        Cue(index=2, start_ms=10000, end_ms=11000, lines=["今日は"]),
        Cue(index=3, start_ms=11000, end_ms=12000, lines=["晴れです"]),
        Cue(index=4, start_ms=1500, end_ms=2500, lines=["終わり"]),
    ]
    words = [Word(text="前", start=0, end=2), Word(text="終わり", start=1.5, end=2.5)]
    alignment = AlignmentResult(cue_word_indices={1: [0], 4: [1]})

    refined, _ = refine_cues_to_speech_activity(
        cues, [SpeechRegion(start=0, end=2.5)], StyleProfile(),
        BoundaryRefinementConfig(max_end_extension_ms=3000),
        words=words, alignment=alignment, protected_cue_ids={2, 3},
    )

    # The next cue can cap padding, but cannot cut a word ending at 2s.
    assert refined[0].end_ms == 2000
    assert refined[1:3] == cues[1:3]


def test_forced_aligned_shared_cue_remains_valid_vad_cap():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["前"]),
        Cue(index=2, start_ms=1500, end_ms=2100, lines=["今日は"]),
        Cue(index=3, start_ms=2200, end_ms=3000, lines=["晴れです"]),
    ]

    refined, _ = refine_cues_to_speech_activity(
        cues, [SpeechRegion(start=0, end=3)], StyleProfile(),
        BoundaryRefinementConfig(max_end_extension_ms=3000),
        words=[Word(text="前", start=0, end=2.9)],
        alignment=AlignmentResult(cue_word_indices={1: [0]}),
        fixed_cue_ids={2, 3},
    )

    assert refined[0].end_ms == 1500
    assert refined[1:] == cues[1:]


def test_duplicate_forced_alignment_rows_do_not_release_shared_source_timing(tmp_path):
    result = sync_episode(**_sync_fixture(tmp_path, max_cps=30, forced_rows=[
        {"cue_id": 1, "start": 0.2, "end": 0.9},
        {"cue_id": 1, "start": 0.3, "end": 0.8},
        {"cue_id": 2, "start": 1.1, "end": 1.7},
    ]))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert _timings(output) == [(1, 0, 1000), (2, 1100, 1700)]
    assert [flag["cue_ids"] for flag in result.report["flags"] if flag["kind"] == "shared_word_timing_preserved"] == [[1]]
