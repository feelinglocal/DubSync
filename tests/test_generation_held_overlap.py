from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync.models import Word
from dubsync.transcription import generate_srt_from_audio
from dubsync.web.generation_styles import GenerationStyleRequest, resolve_generation_style


# Actual Scribe11 windows, translated by whole seconds so the frame phase stays
# unchanged. Ei, andar, and Esta each intersect several plausible VAD bursts.
CASES = {
    "ei": (
        [("Ei,", .292, 5.432, "speaker_3"), ("o", 6.012, 6.112, "speaker_3"),
         ("chefe", 6.132, 6.392, "speaker_3"), ("me", 6.412, 6.492, "speaker_3"),
         ("procurou", 6.512, 6.932, "speaker_3"), ("para", 6.952, 7.052, "speaker_3"),
         ("alugar", 7.062, 7.292, "speaker_3"), ("uma", 7.332, 7.492, "speaker_3"),
         ("casa", 7.512, 7.712, "speaker_3"), ("com", 7.772, 7.872, "speaker_3"),
         ("quintal", 7.892, 8.172, "speaker_3"), ("perto", 8.252, 8.492, "speaker_3"),
         ("da", 8.512, 8.612, "speaker_3"), ("minha", 8.632, 8.792, "speaker_3"),
         ("avó.", 8.802, 8.992, "speaker_3")],
        [(.005, .665), (5.245, 5.405), (6.005, 9.015), (9.335, 10.745)], 0,
    ),
    "andar": (
        [("Eu", 1.048, 1.158, "speaker_7"), ("te", 1.178, 1.238, "speaker_7"),
         ("ajudo", 1.258, 1.598, "speaker_7"), ("a", 1.608, 1.609, "speaker_7"),
         ("andar.", 1.678, 2.398, "speaker_7"), ("Não", 2.498, 2.598, "speaker_6"),
         ("precisa.", 2.638, 3.058, "speaker_6")],
        [(.935, 1.895), (2.145, 2.195), (2.295, 3.095), (3.195, 3.495), (3.575, 3.655)], 4,
    ),
    "frio": (
        [("Está", .318, 1.138, "speaker_7"), ("fazendo", 1.198, 1.478, "speaker_7"),
         ("um", 1.498, 1.578, "speaker_7"), ("friozinho.", 1.618, 2.138, "speaker_7"),
         ("Que", 2.258, 2.338, "speaker_7"), ("tal", 2.398, 2.528, "speaker_7"),
         ("a", 2.558, 2.578, "speaker_7"), ("gente", 2.658, 2.878, "speaker_7"),
         ("descer", 2.918, 3.218, "speaker_7"), ("e", 3.298, 3.308, "speaker_7"),
         ("procurar", 3.318, 3.638, "speaker_7"), ("alguma", 3.658, 3.878, "speaker_7"),
         ("coisa", 3.898, 4.098, "speaker_7"), ("pra", 4.118, 4.178, "speaker_7"),
         ("comer?", 4.238, 4.498, "speaker_7")],
        [(.255, .625), (.925, 7.305)], 0,
    ),
}


def _generate(tmp_path, tuples, regions, preset):
    words = [Word(text=text, start=start, end=end, speaker_id=speaker)
             for text, start, end, speaker in tuples]
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000 * 12)
    fixture, vad = tmp_path / "words.json", tmp_path / "vad.json"
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    vad.write_text(json.dumps({"regions": [dict(start=start, end=end) for start, end in regions]}), encoding="utf-8")
    config = {
        "asr": {"fixture_path": str(fixture), "model_id": "scribe_v2"},
        "timing": {"max_word_duration": 2, "min_duration_policy": "extend_into_silence"},
        "vad": {"fixture_path": str(vad), "boundary_refinement": {
            "enabled": True, "start_pad_ms": 40, "end_pad_ms": 40, "max_end_extension_ms": 300,
            "max_leading_silence_ms": 150, "max_trailing_silence_ms": 300,
        }},
        "output": {"no_overlaps": True},
    }
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    style = resolve_generation_style(GenerationStyleRequest(source="preset", preset=preset), fps=30)
    result = generate_srt_from_audio(audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers,
                                     no_llm=True, style_profile=style.profile, generation_constraints=style.constraints)
    artifact = json.loads((result.episode_workdir / "generate.json").read_text(encoding="utf-8"))
    return words, artifact, result, style


@pytest.mark.parametrize("case,preset", [
    ("ei", "broadcast"), ("ei", "streaming"), ("andar", "broadcast"),
    ("andar", "streaming"), ("frio", "streaming"),
])
def test_actual_held_generation_windows_do_not_overlap_from_display_padding(tmp_path, case, preset):
    tuples, regions, ambiguous_index = CASES[case]
    words, artifact, result, style = _generate(tmp_path, tuples, regions, preset)
    cues, mapping = artifact["cues"], artifact["cue_word_indices"]

    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(cues, cues[1:]))
    assert [index for cue in cues for index in mapping[str(cue["index"])]] == list(range(len(words)))
    assert all(len(cue["lines"]) <= style.profile.max_lines_per_cue for cue in cues)
    assert all(cue["end_ms"] - cue["start_ms"] <= style.constraints.max_cue_duration_seconds * 1000 for cue in cues)
    owner = next(cue for cue in cues if ambiguous_index in mapping[str(cue["index"])])
    assert owner["start_ms"] <= words[ambiguous_index].start * 1000
    assert owner["end_ms"] >= words[ambiguous_index].end * 1000
    assert len([flag for flag in result.report["flags"] if flag["kind"] == "timing_evidence_held"
                and owner["index"] in flag["cue_ids"]]) == 1
    assert any(item["kind"] == "timing_evidence_held" for item in result.report["review"])
    assert not any(flag["kind"] == "output_overlap_unresolved" for flag in result.report["flags"])
    for cue in cues:
        assert len({words[index].speaker_id for index in mapping[str(cue["index"])]}) == 1


def test_real_overlapping_speaker_speech_stays_visible_with_its_hold(tmp_path):
    words, artifact, result, _ = _generate(tmp_path, [
        ("Held.", 1.05, 2.45, "speaker_1"), ("Reply.", 2.2, 2.7, "speaker_2"),
    ], [(1.0, 1.3), (2.0, 2.8)], "broadcast")
    cues, mapping = artifact["cues"], artifact["cue_word_indices"]
    assert len(cues) == 2
    assert [index for cue in cues for index in mapping[str(cue["index"])]] == [0, 1]
    assert cues[0]["end_ms"] >= words[0].end * 1000
    assert cues[1]["start_ms"] <= words[1].start * 1000
    assert cues[0]["end_ms"] > cues[1]["start_ms"]
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])
    assert any(flag["kind"] == "output_overlap_unresolved" for flag in result.report["flags"])


@pytest.mark.parametrize("speakers", [(None, None), ("speaker_1", "speaker_2")])
def test_tiny_frame_gap_never_borrows_an_unknown_or_other_speakers_words(tmp_path, speakers):
    words, artifact, result, _ = _generate(tmp_path, [
        ("Held.", 1.05, 2.45, speakers[0]), ("Reply.", 2.46, 2.8, speakers[1]),
    ], [(1.0, 1.3), (2.0, 2.9)], "broadcast")
    assert artifact["cue_word_indices"] == {"1": [0], "2": [1]}
    assert artifact["cues"][0]["end_ms"] >= words[0].end * 1000
    assert artifact["cues"][1]["start_ms"] <= words[1].start * 1000
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])
    assert any(flag["kind"] == "output_overlap_unresolved" for flag in result.report["flags"])


def test_tiny_frame_gap_can_merge_bounded_same_speaker_generated_groups(tmp_path):
    words, artifact, result, style = _generate(tmp_path, [
        ("Held.", 1.05, 2.45, "speaker_1"), ("Reply.", 2.46, 2.8, "speaker_1"),
    ], [(1.0, 1.3), (2.0, 2.9)], "broadcast")
    assert artifact["cue_word_indices"] == {"1": [0, 1]}
    assert len(artifact["cues"]) == 1
    assert artifact["cues"][0]["end_ms"] >= words[-1].end * 1000
    assert artifact["cues"][0]["end_ms"] - artifact["cues"][0]["start_ms"] <= style.constraints.max_cue_duration_seconds * 1000
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])
    assert not any(flag["kind"] == "output_overlap_unresolved" for flag in result.report["flags"])
