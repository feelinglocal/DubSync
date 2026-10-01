"""A generated ad-lib whose ASR words have collapsed timing still exports a usable cue.

It is merged into the adjacent cue it belongs to, padded inside the free gap,
or removed with one flag. It never exports as a 1 ms cue and never fails the
job after ASR and adjudication were paid.
"""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile

_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\nWir müssen jetzt sofort gehen.\n\n"
    "2\n00:00:08,000 --> 00:00:10,000\nDann komm endlich mit.\n"
)
_BASE = [("Wir", 1.0, 1.2), ("müssen", 1.25, 1.6), ("jetzt", 1.65, 1.9), ("sofort", 1.95, 2.3), ("gehen.", 2.35, 2.8)]
_TAIL = [("Dann", 8.0, 8.2), ("komm", 8.25, 8.5), ("endlich", 8.55, 9.0), ("mit.", 9.05, 9.5)]


def _sync(tmp_path, adlib: list[tuple], *, srt: str = _SRT, base=_BASE, tail=_TAIL, final_text: str | None = None):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    words = [
        {"text": item[0], "start": item[1], "end": item[2], "speaker_id": item[3] if len(item) > 3 else None}
        for item in [*base, *adlib, *tail]
    ]
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    spoken = final_text or " ".join(item[0] for item in adlib)
    responses = {"case-1": {"case_id": "case-1", "verdict": "use_audio", "final_text": spoken,
                            "confidence": 0.97, "reason": "spoken"}}
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": responses},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


def _kinds(flags) -> list[str]:
    return [flag["kind"] for flag in flags]


@pytest.mark.parametrize("adlib", [
    [("Hä?", 4.003, 4.004)],                        # int() truncation made this 4003 --> 4003 and failed the export
    [("Hä?", 5.0, 5.001)],                          # exported as 00:00:05,000 --> 00:00:05,001
    [("Oh", 5.0, 5.001), ("nein!", 5.001, 5.002)],  # two collapsed words
])
def test_isolated_collapsed_adlib_is_padded_inside_the_free_gap(tmp_path, adlib):
    cues, flags = _sync(tmp_path, adlib)

    assert len(cues) == 3
    inserted = cues[1]
    assert inserted.plain_text == " ".join(item[0] for item in adlib)
    assert abs(inserted.start_ms - round(adlib[0][1] * 1000)) <= 34
    assert inserted.end_ms - inserted.start_ms == 500
    assert cues[0].end_ms <= inserted.start_ms and inserted.end_ms <= cues[2].start_ms
    kinds = _kinds(flags)
    assert kinds.count("adlib_inserted") == 1
    assert kinds.count("adlib_timing_estimated") == 1
    assert "timing_evidence_held" not in kinds
    assert "impossible_cps_fast" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_adlib_collapsed_just_before_the_next_cue_is_merged_into_it(tmp_path):
    # Scribe ep11 "Ah," 1421.968-1421.969 belongs to the sentence that starts
    # right after it. (A gap of at most 0.2 s is already attached inline.)
    cues, flags = _sync(tmp_path, [("Ah,", 7.75, 7.751)])

    assert [cue.plain_text for cue in cues] == ["Wir müssen jetzt sofort gehen.", "Ah, Dann komm endlich mit."]
    assert abs(cues[1].start_ms - 8000) <= 34 and cues[1].end_ms >= 9500
    kinds = _kinds(flags)
    assert "adlib_inserted" not in kinds and "timing_evidence_held" not in kinds
    merged = [flag for flag in flags if flag["kind"] == "text_changed" and flag["cue_ids"] == [2]]
    assert len(merged) == 1
    assert (merged[0]["old_text"], merged[0]["new_text"]) == ("Dann komm endlich mit.", "Ah, Dann komm endlich mit.")
    assert not [flag for flag in flags if flag["severity"] == "error"]


_TOUCHING_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\nWir müssen jetzt sofort gehen.\n\n"
    "2\n00:00:03,000 --> 00:00:05,000\nDann komm endlich mit.\n"
)


def _touching(next_speaker: str):
    base = [(text, start, end, "A") for text, start, end in _BASE]
    tail = [("Dann", 2.9, 3.0, next_speaker), ("komm", 3.05, 3.3, next_speaker),
            ("endlich", 3.35, 3.8, next_speaker), ("mit.", 3.85, 4.3, next_speaker)]
    return base, tail


def test_collapsed_adlib_without_room_joins_the_same_speakers_previous_cue(tmp_path):
    base, tail = _touching("B")
    cues, flags = _sync(tmp_path, [("Ja?", 2.86, 2.861, "A")], srt=_TOUCHING_SRT, base=base, tail=tail)

    assert [cue.plain_text for cue in cues] == ["Wir müssen jetzt sofort gehen. Ja?", "Dann komm endlich mit."]
    assert "adlib_inserted" not in _kinds(flags) and "timing_evidence_held" not in _kinds(flags)
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_collapsed_adlib_without_room_or_owner_is_removed_with_one_flag(tmp_path):
    # Another actor's reaction with a placeholder timestamp between two touching cues.
    base, tail = _touching("A")
    cues, flags = _sync(tmp_path, [("Hä?", 2.86, 2.861, "B")], srt=_TOUCHING_SRT, base=base, tail=tail)

    assert [cue.plain_text for cue in cues] == ["Wir müssen jetzt sofort gehen.", "Dann komm endlich mit."]
    kinds = _kinds(flags)
    assert kinds.count("adlib_removed_collapsed_timing") == 1
    assert "adlib_inserted" not in kinds and "timing_evidence_held" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_generated_adlib_cue_always_has_a_positive_duration():
    cues = [Cue(index=1, start_ms=1000, end_ms=3000, lines=["Wir gehen."])]
    span = DivergenceSpan(case_id="case-1", cue_ids=[], srt_text="", asr_text="Hä?", start=4.003, end=4.004,
                          asr_word_indices=[2])
    decision = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text="Hä?", confidence=0.97, reason="spoken")

    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile(), {"case-1": 2})

    adlib = next(cue for cue in changed if cue.index == 2)
    assert adlib.start_ms == 4003
    assert adlib.end_ms > adlib.start_ms
