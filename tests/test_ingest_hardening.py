"""Broken source numbering or cue timing must not corrupt or fail a paid job."""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.srt_io import SRTParseError, parse_srt_text
from dubsync.style_profile import StyleProfile

_WORDS = [
    ("Hallo", 1.2, 1.5), ("Welt.", 1.55, 1.9),
    ("Wie", 3.1, 3.3), ("geht", 3.35, 3.6), ("es", 3.65, 3.8), ("dir?", 3.85, 4.2),
    ("Gut,", 5.3, 5.6), ("danke.", 5.65, 6.1),
]


def _sync(tmp_path, srt: str, words=_WORDS, *, resume: str | None = None):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end} for text, start, end in words
    ]}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers, no_llm=True,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5), resume=resume,
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result


def _kinds(result) -> list[str]:
    return [flag["kind"] for flag in result.report["flags"]]


def test_restarted_cue_numbering_gets_internal_sequential_ids(tmp_path):
    # Two SRT parts were concatenated: the numbering restarts at 1.
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n\n"
        "1\n00:00:05,000 --> 00:00:06,000\nGut, danke.\n"
    )

    cues, result = _sync(tmp_path, srt)

    assert [cue.plain_text for cue in cues] == ["Hallo Welt.", "Wie geht es dir?", "Gut, danke."]
    # "Gut, danke." is spoken at 5.3-6.1 s; it was exported at 2.0 s before the second cue.
    assert abs(cues[2].start_ms - 5300) <= 34 and abs(cues[2].end_ms - 6140) <= 34
    kinds = _kinds(result)
    assert "output_order_inversion" not in kinds and "alignment_outlier" not in kinds
    renumbered = [flag for flag in result.report["flags"] if flag["kind"] == "source_cue_numbers_reassigned"]
    assert len(renumbered) == 1
    assert renumbered[0]["severity"] == "info"
    assert renumbered[0]["cue_ids"] == [3]
    ingest = json.loads((result.episode_workdir / "ingest.json").read_text(encoding="utf-8"))
    assert [cue["index"] for cue in ingest["cues"]] == [1, 2, 3]
    assert ingest["source_cue_numbers"] == {"1": 1, "2": 2, "3": 1}


def test_reassigned_numbering_survives_a_resumed_run(tmp_path):
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n\n"
        "1\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n\n"
        "0\n00:00:05,000 --> 00:00:06,000\nGut, danke.\n"
    )
    _sync(tmp_path, srt)

    cues, result = _sync(tmp_path, srt, resume="rebuild")

    assert [cue.plain_text for cue in cues] == ["Hallo Welt.", "Wie geht es dir?", "Gut, danke."]
    assert _kinds(result).count("source_cue_numbers_reassigned") == 1


def test_unique_cue_numbers_are_kept_even_with_gaps(tmp_path):
    srt = (
        "10\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n\n"
        "20\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n\n"
        "35\n00:00:05,000 --> 00:00:06,000\nGut, danke.\n"
    )

    _, result = _sync(tmp_path, srt)

    ingest = json.loads((result.episode_workdir / "ingest.json").read_text(encoding="utf-8"))
    assert [cue["index"] for cue in ingest["cues"]] == [10, 20, 35]
    assert "source_cue_numbers" not in ingest
    assert "source_cue_numbers_reassigned" not in _kinds(result)


@pytest.mark.parametrize("timing, repaired_end", [
    ("00:00:02,300 --> 00:00:02,300", 2800),   # zero duration
    ("00:00:02,300 --> 00:00:02,100", 2800),   # inverted
    ("00:00:02,800 --> 00:00:02,800", 3000),   # no room before the next cue starts at 3.0 s
])
def test_unusable_source_cue_duration_is_repaired_before_any_paid_stage(tmp_path, timing, repaired_end):
    # The middle cue is not spoken, so it keeps its source timing until export,
    # where a non-positive duration used to fail the finished job.
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n\n"
        f"2\n{timing}\nKomm schon mit!\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n"
    )

    cues, result = _sync(tmp_path, srt, words=_WORDS[:6])

    held = next(cue for cue in cues if cue.plain_text == "Komm schon mit!")
    assert held.start_ms == int(timing[6:8]) * 1000 + int(timing[9:12])
    assert held.end_ms == repaired_end
    repaired = [flag for flag in result.report["flags"] if flag["kind"] == "source_cue_timing_repaired"]
    assert [(flag["cue_ids"], flag["severity"]) for flag in repaired] == [([2], "warning")]
    ingest = json.loads((result.episode_workdir / "ingest.json").read_text(encoding="utf-8"))
    assert all(cue["end_ms"] > cue["start_ms"] for cue in ingest["cues"])


def test_cue_timestamp_without_number_or_separator_fails_before_any_paid_stage(tmp_path):
    # Kept as text, the timestamp line would merge two speakers' cues and be
    # delivered as dialogue; the customer is told which line to fix instead.
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n"
        "00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nGut, danke.\n"
    )

    with pytest.raises(SRTParseError, match="subtitle line 4 is a cue timestamp"):
        _sync(tmp_path, srt)

    assert not (tmp_path / "work" / "episode" / "ingest.json").exists()
