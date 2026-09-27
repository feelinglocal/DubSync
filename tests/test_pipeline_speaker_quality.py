import json
import wave

import yaml

from dubsync.pipeline import sync_episode
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


def test_new_speaker_children_also_receive_the_requested_line_capacity(tmp_path):
    text = "Yes. one two three four five six seven eight nine ten eleven twelve."
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:06,000\nYes.\none two three four five six seven eight nine ten eleven twelve.\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 7)
    words = [{"text": word, "start": 1 + i*.3, "end": 1.2+i*.3, "speaker_id": "A" if i==0 else "B"} for i, word in enumerate(text.split())]
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}), encoding="utf-8")
    config = tmp_path / "provider.yaml"
    config.write_text(yaml.safe_dump({"asr":{"fixture_path":str(fixture)}}), encoding="utf-8")
    output = tmp_path / "result.srt"
    sync_episode(source, audio, output, tmp_path/"work", providers_path=config, no_llm=True,
                 style_profile=StyleProfile(max_chars_per_line=12, max_lines_per_cue=2, min_cue_dur=.1, fps=30))
    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert len(cues) >= 4
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in cues)) == alphanumeric_signature(text)
    assert all(cue.duration_ms > 0 for cue in cues)
