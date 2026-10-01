from __future__ import annotations

import json

import pytest
import yaml

from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.transcription import generate_srt_from_audio


def _generate_with_timing(tmp_path, *, asr_model, timing, boundary=None, word_window=(1.12, 1.52)):
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"fixture audio")
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"words": [
        {"text": "Hello.", "start": word_window[0], "end": word_window[1]},
    ]}), encoding="utf-8")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({"regions": [{"start": 1.0, "end": 1.68}]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(words), **asr_model},
        "vad": {
            "fixture_path": str(regions),
            # Isolate pre-segmentation word repair from later cue refinement.
            "boundary_refinement": {"enabled": False, **(boundary or {})},
        },
        "timing": timing,
    }), encoding="utf-8")
    output = tmp_path / "output.srt"

    result = generate_srt_from_audio(
        audio, output, tmp_path / "work", providers_path=config, no_llm=True,
        style_profile=StyleProfile(fps=25, min_cue_dur=0.1, tail_ms=0),
    )

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert len(cues) == 1
    assert cues[0].plain_text == "Hello."
    return cues[0], result


@pytest.mark.parametrize("snap", [False, {"start_advance_ms": 80, "end_extension_ms": 80}])
def test_generation_respects_shared_phrase_edge_snap_policy(tmp_path, snap):
    cue, _ = _generate_with_timing(
        tmp_path, asr_model={"model_id": "scribe_v2"}, timing={"phrase_edge_snap": snap},
    )

    assert (cue.start_ms, cue.end_ms) == (1120, 1520)


@pytest.mark.parametrize(
    ("asr_model", "expected_start", "expected_end"),
    [
        ({"model_id": "scribe_v2"}, 1120, 1520),
        ({"model": "microsoft/mai-transcribe-2"}, 1000, 1520),
        ({"model": "other-model"}, 1000, 1680),
    ],
)
def test_generation_uses_selected_models_phrase_edge_limits(tmp_path, asr_model, expected_start, expected_end):
    cue, result = _generate_with_timing(
        tmp_path, asr_model=asr_model, timing={"phrase_edge_snap": {
            "start_advance_ms": 200,
            "end_extension_ms": 300,
            "models": {
                "scribe_v2": {"start_advance_ms": 80, "end_extension_ms": 80},
                "microsoft/mai-transcribe-2": {"end_extension_ms": 80},
            },
        }},
    )

    assert (cue.start_ms, cue.end_ms) == (expected_start, expected_end)
    artifact = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    assert artifact["asr"]["model"] == next(iter(asr_model.values()))


def test_generation_inherits_boundary_end_window_for_word_repair(tmp_path):
    cue, _ = _generate_with_timing(
        tmp_path, asr_model={"model_id": "scribe_v2"}, timing={},
        boundary={"max_end_extension_ms": 80},
    )

    assert (cue.start_ms, cue.end_ms) == (1000, 1520)


def test_disabling_phrase_snap_still_repairs_edges_in_silence(tmp_path):
    cue, _ = _generate_with_timing(
        tmp_path, asr_model={"model_id": "scribe_v2"}, timing={"phrase_edge_snap": False},
        word_window=(0.5, 2.0),
    )

    assert (cue.start_ms, cue.end_ms) == (1000, 1680)
