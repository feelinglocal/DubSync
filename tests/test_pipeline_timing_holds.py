import json
import wave
import pytest
from dubsync import pipeline
from dubsync.cost import CostMeter
from dubsync.models import AlignmentResult, Cue, QCFlag, Word
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
import yaml


@pytest.mark.parametrize("kind", ["timing_evidence_held", "adjudication_word_mapping_held"])
def test_timing_hold_survives_verification_without_undoing_approved_words(tmp_path, kind):
    source = Cue(index=1, start_ms=1000, end_ms=3000, lines=["Source wording."])
    approved = Cue(index=1, start_ms=2033, end_ms=2166, lines=["Não consegui pegar ele."])
    words = [Word(text=t, start=a, end=b, speaker_id="A") for t,a,b in [
        ("Não",2.058,2.118), ("consegui",2.138,2.148),
        ("pegar",2.148,2.149), ("ele.",2.148,2.149)]]
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1,2,16000,0,"NONE","not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 4)
    vad = tmp_path / "vad-fixture.json"
    vad.write_text(json.dumps({"regions":[{"start":2.05,"end":2.25}]}),encoding="utf-8")
    output = tmp_path / "output.srt"
    result = pipeline._run_verify_stage(
        episode_workdir=tmp_path, output_path=output, audio_path=audio, audio_for_asr=audio,
        provider_config={"vad":{"fixture_path":str(vad),"boundary_refinement":True}},
        profile=StyleProfile(fps=30), source_cues=[source], rebuilt=[approved], words=words,
        alignment=AlignmentResult(cue_word_indices={1:list(range(4))}),
        flags=[QCFlag(kind=kind,cue_ids=[1],severity="error",message="unusable local acoustic evidence")],
        cost_meter=CostMeter(), include_dropped_line_flags=False,
    )
    restored = parse_srt_text(output.read_text(encoding="utf-8"))[0]
    assert (restored.start_ms,restored.end_ms)==(1000,3000)
    assert restored.text==approved.text
    assert any(f["kind"]==kind for f in result.report["flags"])


def test_full_sync_preserves_parent_when_one_actor_has_collapsed_word_times(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:03,000\n- Hurry fast.\n- Come here.\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 4)
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": "Hurry", "start": 1.3, "end": 1.5, "speaker_id": "A"},
        {"text": "fast.", "start": 1.6, "end": 2.0, "speaker_id": "A"},
        {"text": "Come", "start": 2.01, "end": 2.011, "speaker_id": "B"},
        {"text": "here.", "start": 2.01, "end": 2.011, "speaker_id": "B"},
    ]}), encoding="utf-8")
    config = tmp_path / "provider.yaml"
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    output = tmp_path / "output.srt"

    result = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=config,
                                   no_llm=True, style_profile=StyleProfile(min_cue_dur=.1, fps=30))

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert len(cues) == 1
    assert (cues[0].start_ms, cues[0].end_ms) == (1000, 3000)
    assert cues[0].lines == ["- Hurry fast.", "- Come here."]
    assert any(flag["kind"] == "timing_evidence_held" for flag in result.report["flags"])
