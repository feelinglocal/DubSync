"""The delivered SRT has at most two lines in sync and generation routes."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline, transcription
from dubsync.models import Cue, SpeechRegion, Word
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import GenerationConstraints, StyleProfile
from dubsync.timing_refinement import SpeechEvidence
from dubsync.transcription import generate_srt_from_audio


def _assets(tmp_path, words):
    audio, fixture, config = (tmp_path / name for name in ("episode.wav", "words.json", "providers.yaml"))
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 128000)
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    return audio, config


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
@pytest.mark.parametrize("forced", [False, True])
def test_sync_splits_authored_three_line_cue_at_spoken_sentence_boundaries(tmp_path, monkeypatch, mode, forced):
    lines = ["We finished the work.", "Now we can go home.", "Please bring the keys."]
    tokens = " ".join(lines).split()
    words = [Word(text=token, start=1 + index * .3, end=1.22 + index * .3, speaker_id="actor")
             for index, token in enumerate(tokens)]
    audio, config = _assets(tmp_path, words)
    if forced:
        forced_fixture = tmp_path / "forced.json"
        forced_fixture.write_text(json.dumps({"cues": [{
            "cue_id": 1, "start": 1, "end": words[-1].end, "score": .99,
        }]}), encoding="utf-8")
        settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        settings["forced_alignment"] = {"fixture_path": str(forced_fixture)}
        config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    source = tmp_path / "episode.srt"
    source.write_text(write_srt([Cue(index=1, start_ms=1000, end_ms=5000, lines=lines)]), encoding="utf-8")
    profile = StyleProfile(max_chars_per_line=24, max_lines_per_cue=4, min_cue_dur=.1, tail_ms=0)
    monkeypatch.setattr(pipeline, "speech_evidence_for_words", lambda *_a, **_k: SpeechEvidence(
        words=words, regions=[SpeechRegion(start=1, end=words[-1].end)], detected=True,
    ))
    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, style_profile=profile, no_llm=True, **kwargs)
    result = run()
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        result = run(**({} if mode == "cache" else {"resume": mode}))
        assert result.output_srt.read_bytes() == first
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert len(delivered) >= 2
    assert all(len(cue.lines) <= 2 for cue in delivered)
    assert " ".join(cue.plain_text for cue in delivered) == " ".join(lines)
    assert all(left.end_ms <= right.start_ms for left, right in zip(delivered, delivered[1:]))
    assert all(cue.plain_text.endswith(".") for cue in delivered)
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    owned = payload["alignment"]["cue_word_indices"]
    assert sorted(index for indices in owned.values() for index in indices) == list(range(len(words)))
    for raw in payload["cues"]:
        indices = owned[str(raw["index"])]
        assert raw["start_ms"] <= words[indices[0]].start * 1000 + 1
        assert raw["end_ms"] >= words[indices[-1]].end * 1000 - 1
    # A whole-parent forced score is not evidence for any individual child.
    assert all(score["source"] != "forced_alignment" for score in result.report["cue_scores"])


def test_generation_enforces_two_lines_when_one_asr_word_cannot_be_split(tmp_path):
    text = "We brought every single document for the meeting today."
    words = [Word(text=text, start=1, end=3, speaker_id="actor")]
    audio, config = _assets(tmp_path, words)
    result = generate_srt_from_audio(
        audio, tmp_path / "output.srt", tmp_path / "work", providers_path=config, no_llm=True,
        style_profile=StyleProfile(max_chars_per_line=20, max_lines_per_cue=4, min_cue_dur=.1, tail_ms=0),
    )
    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert len(cues) == 1
    assert len(cues[0].lines) <= 2
    assert cues[0].plain_text == text
    assert (cues[0].start_ms, cues[0].end_ms) == (1000, 3000)
    assert any(issue["kind"] == "line_length" for issue in result.report["style_issues"])
    payload = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    assert payload["cue_word_indices"] == {"1": [0]}


def test_generation_checks_speech_coverage_on_delivered_children(tmp_path, monkeypatch):
    tokens = "Keep your bag close, bring the blue coat.".split()
    words = [Word(text=token, start=1 + index * .2 + (.6 if index >= 4 else 0),
                  end=1.18 + index * .2 + (.6 if index >= 4 else 0), speaker_id="actor")
             for index, token in enumerate(tokens)]
    audio, config = _assets(tmp_path, words)
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    settings["vad"] = {"min_coverage": .85}
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")

    class Activity:
        def detect(self, _audio):
            return [SpeechRegion(start=1, end=1.78), SpeechRegion(start=2.4, end=3.18)]

    monkeypatch.setattr(transcription, "speech_activity_adapter_from_config", lambda _config: Activity())
    result = generate_srt_from_audio(
        audio, tmp_path / "output.srt", tmp_path / "work", providers_path=config, no_llm=True,
        style_profile=StyleProfile(max_chars_per_line=12, max_lines_per_cue=4, min_cue_dur=.1, tail_ms=0),
        generation_constraints=GenerationConstraints(max_gap_seconds=1, max_cue_duration_seconds=6),
    )
    payload = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    assert payload["output_segmentation"]["expansions"]
    assert all(len(cue["lines"]) <= 2 for cue in payload["cues"])
    assert not any(flag["kind"] == "cue_without_speech_activity" for flag in result.report["flags"])
