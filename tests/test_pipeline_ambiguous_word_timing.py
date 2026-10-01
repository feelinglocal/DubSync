from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import Cue
from dubsync.output_order import finalize_cues_for_output
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.transcription import generate_srt_from_audio


def _inputs(tmp_path, words, regions, *, boundary=True, rewrite=False):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 12)
    word_path = tmp_path / "words.json"
    word_path.write_text(json.dumps({"words": words}), encoding="utf-8")
    region_path = tmp_path / "regions.json"
    region_path.write_text(json.dumps({"regions": regions}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(word_path)},
        "vad": {"fixture_path": str(region_path), "boundary_refinement": {"enabled": boundary}},
        "timing": {"phrase_edge_snap": False},
        "llm": {"provider": "fixture", "responses": {"case-1": {
            "case_id": "case-1", "verdict": "use_audio" if rewrite else "keep_srt",
            "final_text": "Luke.", "confidence": 0.99,
            "evidence": "heard_clearly", "heard_text": "Luke.",
            "reason": "Fixture recognizes the name but supplies no burst ownership.",
        }}},
    }), encoding="utf-8")
    return audio, providers


@pytest.mark.parametrize("resume", [None, "rebuild", "verify"])
def test_full_sync_preserves_a_cue_before_ambiguous_word_trimming_and_refinement(tmp_path, resume):
    words = [
        {"text": "Hello", "start": 1.05, "end": 1.2, "speaker_id": "A"},
        {"text": "Luke.", "start": 1.24, "end": 7.45, "speaker_id": "A"},
    ]
    audio, providers = _inputs(tmp_path, words, [{"start": 1.0, "end": 1.8}, {"start": 7.0, "end": 7.5}])
    source = tmp_path / "source.srt"
    source_cue = Cue(index=1, start_ms=1000, end_ms=1800, lines=["Hello Luke."])
    source.write_text(write_srt([source_cue]), encoding="utf-8")
    options = dict(srt_path=source, audio_path=audio, output_path=tmp_path / "out.srt", workdir=tmp_path / "work",
                   providers_path=providers, no_llm=True, style_profile=StyleProfile(fps=25, min_cue_dur=0.1, tail_ms=0))

    result = pipeline.sync_episode(**options)
    if resume:
        result = pipeline.sync_episode(**options, resume=resume)

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert len(output) == 1
    assert (output[0].start_ms, output[0].end_ms, output[0].text) == (1000, 1800, "Hello Luke.")
    assert any(flag["kind"] == "timing_evidence_held" and flag["cue_ids"] == [1] for flag in result.report["flags"])
    assert not any(flag["kind"] == "timing_outlier_trimmed" for flag in result.report["flags"])
    raw = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))
    assert raw["words"][1]["end"] == 7.45


def test_full_sync_does_not_retime_a_short_ambiguous_word_to_the_first_burst(tmp_path):
    audio, providers = _inputs(tmp_path, [{"text": "Luke.", "start": 1.05, "end": 2.4}],
                               [{"start": 1.0, "end": 1.4}, {"start": 2.0, "end": 2.5}])
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:02,000 --> 00:00:02,500\nLuke.\n", encoding="utf-8")

    result = pipeline.sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, style_profile=StyleProfile(fps=25, tail_ms=0))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [(cue.start_ms, cue.end_ms) for cue in output] == [(2000, 2500)]
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])


def test_ambiguous_source_word_is_protected_before_adjudicated_text_and_ownership(tmp_path):
    audio, providers = _inputs(tmp_path, [
        {"text": "Hello", "start": 1.05, "end": 1.2, "speaker_id": "A"},
        {"text": "Luke.", "start": 1.24, "end": 7.45, "speaker_id": "B"},
    ], [{"start": 1.0, "end": 1.8}, {"start": 7.0, "end": 7.5}], rewrite=True)
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:01,800\nHello Luan.\n", encoding="utf-8")

    result = pipeline.sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work",
                                   providers_path=providers, style_profile=StyleProfile(fps=25, tail_ms=0))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert len(output) == 1
    assert (output[0].start_ms, output[0].end_ms, output[0].text) == (1000, 1800, "Hello Luan.")
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])


@pytest.mark.parametrize("boundary", [False, True])
def test_generation_preserves_raw_ambiguous_bounds_with_actionable_review(tmp_path, boundary):
    audio, providers = _inputs(tmp_path, [{"text": "Luke.", "start": 1.05, "end": 7.45}],
                               [{"start": 1.0, "end": 1.4}, {"start": 7.0, "end": 7.5}], boundary=boundary)

    result = generate_srt_from_audio(audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers,
                                    no_llm=True, style_profile=StyleProfile(fps=25, tail_ms=0))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in output] == [(1040, 7480, "Luke.")]
    assert any(flag["kind"] == "timing_evidence_held" and flag["cue_ids"] == [1] for flag in result.report["flags"])


def test_generation_readability_extension_cannot_change_ambiguous_raw_bounds(tmp_path):
    audio, providers = _inputs(tmp_path, [{"text": "Luke.", "start": 1.05, "end": 2.4}],
                               [{"start": 1.0, "end": 1.4}, {"start": 2.0, "end": 2.5}], boundary=False)

    result = generate_srt_from_audio(audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers,
                                    no_llm=True, style_profile=StyleProfile(fps=25, tail_ms=0, min_cue_dur=3.0))

    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [(cue.start_ms, cue.end_ms) for cue in output] == [(1040, 2400)]
    assert not any(flag["kind"] == "cps_duration_extended" for flag in result.report["flags"])


def test_final_overlap_resolution_does_not_clip_a_fixed_ambiguous_interval():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=3000, lines=["Luke."]),
        Cue(index=2, start_ms=2800, end_ms=3500, lines=["Hello."]),
    ]
    options = dict(protected_cue_ids={1}, preserve_timing=True, spoken_spans={2: (2800, 3500)})
    ordinary_hold, _ = finalize_cues_for_output(cues, StyleProfile(fps=25), **options)
    assert ordinary_hold[0].end_ms == 2800  # Existing source-hold clipping behavior.

    fixed, flags = finalize_cues_for_output(cues, StyleProfile(fps=25), fixed_cue_ids={1}, **options)

    assert fixed == cues
    overlap = next(flag for flag in flags if flag.kind == "output_overlap_unresolved")
    assert "uncertain" in overlap.message.lower()
    assert "simultaneous" not in overlap.message.lower()
