"""Two cues spoken in swapped order: the reconciled cue follows its own words."""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AlignmentDiagnostics, AlignmentResult, QCFlag
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile

_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\nWir müssen jetzt sofort gehen.\n\n"
    "2\n00:00:03,500 --> 00:00:04,500\nWarte kurz.\n\n"
    "3\n00:00:05,000 --> 00:00:07,000\nIch hole meine Jacke.\n\n"
    "4\n00:00:08,000 --> 00:00:10,000\nDann komm endlich mit.\n"
)
# The actors say cue 3 first (3.5 s) and cue 2 afterwards (5.2 s).
_WORDS = [
    ("Wir", 1.0, 1.2), ("müssen", 1.25, 1.6), ("jetzt", 1.65, 1.9), ("sofort", 1.95, 2.3), ("gehen.", 2.35, 2.8),
    ("Ich", 3.5, 3.65), ("hole", 3.7, 4.0), ("meine", 4.05, 4.35), ("Jacke.", 4.4, 4.8),
    ("Warte", 5.2, 5.5), ("kurz.", 5.55, 5.9),
    ("Dann", 8.0, 8.2), ("komm", 8.25, 8.5), ("endlich", 8.55, 9.0), ("mit.", 9.05, 9.5),
]


@pytest.mark.parametrize("confidence", [1.0, None])
def test_reconciled_swapped_cue_is_timed_from_its_spoken_words(tmp_path, confidence):
    source = tmp_path / "episode.srt"
    source.write_text(_SRT, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "confidence": confidence} for text, start, end in _WORDS
    ]}, ensure_ascii=False), encoding="utf-8")
    responses = {
        # The source cue is not spoken at its place (deterministically held) ...
        "case-1": {"case_id": "case-1", "verdict": "use_audio", "final_text": "", "confidence": 0.97, "reason": "not spoken here"},
        # ... its words are heard after the following cue.
        "case-2": {"case_id": "case-2", "verdict": "use_audio", "final_text": "Warte kurz.", "confidence": 0.97, "reason": "spoken"},
    }
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": responses},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"

    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5),
    )

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == [
        "Wir müssen jetzt sofort gehen.", "Ich hole meine Jacke.", "Warte kurz.", "Dann komm endlich mit.",
    ]
    jacke, warte = cues[1], cues[2]
    assert abs(jacke.start_ms - 3500) <= 34 and abs(jacke.end_ms - 4840) <= 34
    assert abs(warte.start_ms - 5200) <= 34 and abs(warte.end_ms - 5940) <= 34
    assert jacke.end_ms <= warte.start_ms
    flags = result.report["flags"]
    kinds = [flag["kind"] for flag in flags]
    assert kinds.count("adlib_reconciled") == 1
    # The cue has its audio: it is no missing-audio hold, overlap or order error.
    assert not any(kind.startswith("missing_audio") for kind in kinds)
    assert not any("overlap" in kind for kind in kinds)
    assert "output_order_inversion" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_released_cue_leaves_the_protected_and_unmatched_sets():
    alignment = AlignmentResult(
        unmatched_cue_ids=[2, 7],
        diagnostics=AlignmentDiagnostics(missing_audio_cue_ids=[2, 7]),
    )
    flags = [
        QCFlag(kind="missing_audio_timing_held", cue_ids=[2], severity="error", message="locked"),
        QCFlag(kind="missing_audio_timing_held", cue_ids=[7], severity="error", message="locked"),
        QCFlag(kind="missing_audio_source_cue_held", cue_ids=[2], severity="error", message="held"),
        QCFlag(kind="missing_audio_source_cue_held", cue_ids=[2, 7], severity="error", message="held"),
        QCFlag(kind="adlib_reconciled", cue_ids=[2], message="reused"),
    ]

    released, kept = pipeline._release_reconciled_cues(alignment, flags)

    assert released.diagnostics.missing_audio_cue_ids == [7]
    assert released.unmatched_cue_ids == [7]
    assert [(flag.kind, flag.cue_ids) for flag in kept] == [
        ("missing_audio_timing_held", [7]),
        ("missing_audio_source_cue_held", [7]),
        ("adlib_reconciled", [2]),
    ]
    # Idempotent: verify re-derives the same state from persisted flags.
    assert pipeline._release_reconciled_cues(released, kept) == (released, kept)
