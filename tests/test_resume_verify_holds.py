from __future__ import annotations

import json
import math
import random
import wave
from array import array

import pytest
import yaml

from dubsync import pipeline
from dubsync.srt_io import parse_srt_text

# Cue 2 shares only "Hause." with the spoken words, so its rebuild keeps it at
# source timing. Without a model its timing hold is folded into the
# unconfirmed-wording finding and lives only in the rebuild's memory.
_SOURCE = (
    "1\n00:00:00,500 --> 00:00:01,600\nGuten Morgen zusammen.\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\nWir gehen jetzt nach Hause.\n\n"
    "3\n00:00:05,000 --> 00:00:06,500\nBis morgen dann.\n"
)
_WORDS = [
    ("Guten", 0.70, 0.95), ("Morgen", 0.97, 1.25), ("zusammen.", 1.27, 1.60),
    ("Ich", 2.50, 2.62), ("laufe", 2.64, 2.90), ("heute", 2.92, 3.10),
    ("weit", 3.12, 3.26), ("Hause.", 3.28, 3.60),
    ("Bis", 5.30, 5.45), ("morgen", 5.47, 5.80), ("dann.", 5.82, 6.10),
]
_LOW_CONFIDENCE_KEEP = {"case-1": {
    "case_id": "case-1", "verdict": "keep_srt", "final_text": "Wir gehen jetzt nach",
    "confidence": 0.5, "reason": "unsure which words were spoken",
}}


def _episode(tmp_path, *, llm: bool, timing: dict | None = None, refinement: dict | None = None):
    rate = 16000
    rng = random.Random(1234)
    samples = array("h", (int(rng.uniform(-20, 20)) for _ in range(8 * rate)))
    for _, start, end in _WORDS:
        for i in range(int(start * rate), int(end * rate)):
            samples[i] = int(9000 * math.sin(2 * math.pi * 220 * i / rate) + rng.uniform(-3000, 3000))
    audio = tmp_path / "episode.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    source = tmp_path / "episode.srt"
    source.write_text(_SOURCE, encoding="utf-8")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end} for text, start, end in _WORDS
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    # The shipped energy VAD with boundary refinement, as in provider.yaml.
    config = {
        "asr": {"fixture_path": str(fixture)},
        "vad": {"provider": "energy", "min_coverage": 0.2,
                "boundary_refinement": {"enabled": True, **(refinement or {})}},
        **({"timing": timing} if timing else {}),
        **({"llm": {"provider": "fixture", "responses": _LOW_CONFIDENCE_KEEP}} if llm else {}),
    }
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    return dict(srt_path=source, audio_path=audio, workdir=tmp_path / "work",
                providers_path=providers, no_llm=not llm)


def _flags(result) -> list[str]:
    return sorted(json.dumps(flag, sort_keys=True, ensure_ascii=False) for flag in result.report["flags"])


@pytest.mark.parametrize("llm", [False, True], ids=["no-llm", "low-confidence-keep"])
def test_verify_resume_keeps_a_folded_source_timing_hold(tmp_path, llm):
    options = _episode(tmp_path, llm=llm)
    fresh = pipeline.sync_episode(**options, output_path=tmp_path / "fresh.srt")
    held = [cue for cue in parse_srt_text(fresh.output_srt.read_text(encoding="utf-8")) if cue.index == 2]
    assert [(cue.start_ms, cue.end_ms) for cue in held] == [(2000, 4000)]
    assert not any(flag["kind"] == "timing_evidence_held" for flag in fresh.report["flags"])
    checkpoint_path = fresh.episode_workdir / "rebuild.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    resumed = pipeline.sync_episode(**options, output_path=tmp_path / "resumed.srt", resume="verify")

    assert resumed.output_srt.read_bytes() == fresh.output_srt.read_bytes()
    assert _flags(resumed) == _flags(fresh)
    assert 2 in checkpoint["verify_input"]["source_timing_held_cue_ids"]
    # A second resume starts from the same checkpoint again.
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == checkpoint


def _artifacts(result) -> dict:
    paths = [result.output_srt, *result.episode_workdir.glob("*.json")]
    return {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("change", [
    {"fps": 24.0},
    {"timing": {"min_duration_policy": "acoustic"}},
    {"refinement": {"enabled": False}},
    {"refinement": {"start_pad_ms": 0, "max_leading_silence_ms": 0}},
], ids=["fps", "min-duration-policy", "refinement-off", "refinement-pads"])
def test_verify_resume_refuses_a_checkpoint_timed_under_other_settings(tmp_path, change):
    options = _episode(tmp_path, llm=False)
    first = pipeline.sync_episode(**options, output_path=tmp_path / "out.srt")
    before = _artifacts(first)
    changed = {**_episode(tmp_path, llm=False, timing=change.get("timing"), refinement=change.get("refinement")),
               "output_path": tmp_path / "out.srt", "fps": change.get("fps")}

    # The checkpoint's cues were finished on the old frame grid and under the
    # old minimum-duration and refinement settings, which verify cannot redo.
    with pytest.raises(ValueError, match="other frame-rate.*resume from rebuild"):
        pipeline.sync_episode(**changed, resume="verify")

    assert _artifacts(first) == before
    rebuilt = pipeline.sync_episode(**changed, resume="rebuild")
    output = rebuilt.output_srt.read_bytes()
    # The rebuilt checkpoint records the new settings; verify then resumes from it.
    assert pipeline.sync_episode(**changed, resume="verify").output_srt.read_bytes() == output


def test_verify_resume_accepts_the_same_settings_given_explicitly(tmp_path):
    options = _episode(tmp_path, llm=False)
    first = pipeline.sync_episode(**options, output_path=tmp_path / "out.srt")
    output = first.output_srt.read_bytes()
    fps = json.loads((first.episode_workdir / "style_profile.json").read_text(encoding="utf-8"))["fps"]

    resumed = pipeline.sync_episode(**options, output_path=tmp_path / "out.srt", resume="verify", fps=fps)

    assert resumed.output_srt.read_bytes() == output


def test_verify_resume_refuses_a_checkpoint_without_its_verification_input(tmp_path):
    options = _episode(tmp_path, llm=False)
    first = pipeline.sync_episode(**options, output_path=tmp_path / "out.srt")
    path = first.episode_workdir / "rebuild.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("verify_input")
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = _artifacts(first)

    # Its finished cues alone would lose the folded hold of cue 2.
    with pytest.raises(ValueError, match="verification input.*resume from rebuild"):
        pipeline.sync_episode(**options, output_path=tmp_path / "out.srt", resume="verify")

    assert _artifacts(first) == before
