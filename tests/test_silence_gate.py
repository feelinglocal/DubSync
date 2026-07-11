from __future__ import annotations

import wave

from dubsync.models import Cue
from dubsync.silence import silence_flags_for_cues


def test_silence_gate_flags_cue_on_quiet_wav(tmp_path):
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    flags = silence_flags_for_cues(
        audio,
        [Cue(index=1, start_ms=100, end_ms=600, lines=["silent line"])],
        threshold_dbfs=-45.0,
    )

    assert flags[0].kind == "cue_on_silence"
    assert flags[0].cue_ids == [1]
