from __future__ import annotations

import json
import asyncio
from io import BytesIO
import wave

import pytest
import yaml

from dubsync.models import Cue, Word
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import GenerationConstraints, StyleProfile
from dubsync.text_metrics import wrap_visual_width
from dubsync.transcription import build_cues_from_words, generate_srt_from_audio
from dubsync.web.generation_styles import GenerationStyleRequest, resolve_generation_style


def _generate(tmp_path, words, *, regions=None, profile=None, constraints=None, boundary_enabled=True):
    audio = tmp_path / "episode.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000 * 5)
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}), encoding="utf-8")
    config = {"asr": {"fixture_path": str(fixture)}, "timing": {"phrase_edge_snap": False}}
    if regions is not None:
        regions_path = tmp_path / "regions.json"
        regions_path.write_text(json.dumps({"regions": regions}), encoding="utf-8")
        config["vad"] = {"fixture_path": str(regions_path), "boundary_refinement": {"enabled": boundary_enabled}}
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = generate_srt_from_audio(
        audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers, no_llm=True,
        style_profile=profile or StyleProfile(), generation_constraints=constraints,
    )
    return parse_srt_text(result.output_srt.read_text(encoding="utf-8")), result


def test_generation_resolves_padding_overlap_without_vad(tmp_path):
    cues, result = _generate(tmp_path, [
        {"text": "First.", "start": 1.0, "end": 1.61, "speaker_id": "A"},
        {"text": "Second.", "start": 1.65, "end": 2.3, "speaker_id": "B"},
    ])
    assert cues[0].end_ms <= cues[1].start_ms
    assert cues[0].end_ms >= 1610
    assert cues[1].start_ms <= 1650
    assert not any(flag["kind"] == "output_overlap_unresolved" for flag in result.report["flags"])


def test_generation_reading_speed_extends_only_end_into_verified_silence(tmp_path):
    cues, _ = _generate(tmp_path, [
        {"text": "Read this line.", "start": 1.0, "end": 1.4, "speaker_id": "A"},
        {"text": "Next.", "start": 3.0, "end": 3.5, "speaker_id": "B"},
    ], regions=[{"start": 1.0, "end": 1.4}, {"start": 3.0, "end": 3.5}],
        profile=StyleProfile(fps=25, min_cue_dur=0.1, tail_ms=0),
        constraints=GenerationConstraints(max_cps=10))
    assert cues[0].start_ms == 1000
    assert cues[0].end_ms >= 2500
    assert cues[0].end_ms <= cues[1].start_ms


def test_generation_minimum_duration_uses_silence_even_without_boundary_refinement(tmp_path):
    cues, _ = _generate(tmp_path, [{"text": "Yes.", "start": 1.0, "end": 1.2}],
        regions=[{"start": 1.0, "end": 1.2}], boundary_enabled=False,
        profile=StyleProfile(fps=25, min_cue_dur=1.0, tail_ms=0))
    assert cues[0].start_ms == 1000
    assert cues[0].duration_ms >= 1000


@pytest.mark.parametrize("regions", [None, [{"start": 1.0, "end": 4.0}]])
def test_generation_never_invents_verified_silence(tmp_path, regions):
    cues, _ = _generate(tmp_path, [{"text": "Read this line.", "start": 1.0, "end": 1.4}],
        regions=regions, boundary_enabled=False,
        profile=StyleProfile(fps=25, min_cue_dur=0.1, tail_ms=0),
        constraints=GenerationConstraints(max_cps=10))
    assert cues[0].start_ms == 1000
    assert cues[0].end_ms == 1400


def test_build_generation_never_clips_genuine_simultaneous_speech():
    cues = build_cues_from_words([
        Word(text="First.", start=1.0, end=2.0, speaker_id="A"),
        Word(text="Reply.", start=1.5, end=2.2, speaker_id="B"),
    ], StyleProfile())
    assert cues[0].end_ms >= 2000
    assert cues[1].start_ms <= 1500


def test_generation_keeps_overlapping_same_speaker_sentence_parts_in_one_cue():
    words = [
        Word(text="Já está melhor?", start=1.0, end=1.735),
        Word(text="Melhor.", start=1.52, end=1.735),
    ]
    cues = build_cues_from_words(words, StyleProfile(), preserve_timing=True)
    assert [cue.plain_text for cue in cues] == ["Já está melhor? Melhor."]
    assert cues[0].start_ms <= 1000
    assert cues[0].end_ms >= 1735


@pytest.mark.parametrize("profile,max_duration", [
    (StyleProfile(), 1.5),
    (StyleProfile(max_lines_per_cue=1, max_chars_per_line=10), 5.0),
])
def test_overlapping_generated_grouping_retains_line_and_duration_limits(profile, max_duration):
    cues = build_cues_from_words([
        Word(text="First.", start=1.0, end=2.0),
        Word(text="Reply.", start=1.9, end=3.0),
    ], profile, preserve_timing=True, max_cue_duration_seconds=max_duration)
    assert [cue.plain_text for cue in cues] == ["First.", "Reply."]


def test_generation_moves_only_the_boundary_when_overlapping_phrases_exceed_two_lines():
    texts = ["E", "E", "eu", "eu", "e", "ia", "ela.", "lá", "saber", "que", "ele", "tinha", "namorado?"]
    starts = [0.405, 0.405, 0.56, 0.62, 0.72, 0.76, 0.82, 0.92, 1.12, 1.32, 1.42, 1.56, 1.76]
    ends = [0.48, 0.48, 0.699, 0.699, 0.779, 0.88, 1.0, 1.04, 1.279, 1.399, 1.519, 1.699, 2.155]
    words = [Word(text=text, start=start, end=end) for text, start, end in zip(texts, starts, ends)]
    cues = build_cues_from_words(words, StyleProfile(), preserve_timing=False)
    assert " ".join(cue.plain_text for cue in cues) == " ".join(texts)
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues, cues[1:]))
    assert all(len(cue.lines) <= 2 and all(len(line) <= 26 for line in cue.lines) for cue in cues)


def test_generation_keeps_readability_extension_out_of_acoustic_qc(tmp_path):
    _, result = _generate(tmp_path, [{"text": "Yes.", "start": 1.0, "end": 1.2}],
        regions=[{"start": 1.0, "end": 1.2}], profile=StyleProfile(min_cue_dur=1.0))
    assert not any(flag["kind"] in {"cue_without_speech_activity", "cue_with_excessive_trailing_silence"}
                   for flag in result.report["flags"])


def test_balanced_wrapping_avoids_orphan_last_word():
    assert wrap_visual_width("Du Idiot, das ist verdammt Mist.", 26) == [
        "Du Idiot, das ist", "verdammt Mist.",
    ]


@pytest.mark.parametrize("words", [
    ["Wir", "sind", "am", "3.", "Oktober", "fertig."],
    ["Das", "sagt", "Dr.", "Müller", "heute."],
    ["Ich", "dachte…", "du", "wärst", "hier."],
    ["Ich", "dachte...", "du", "wärst", "hier."],
])
def test_generation_does_not_split_nonterminal_punctuation(words):
    cues = build_cues_from_words([
        Word(text=text, start=i * 0.3, end=i * 0.3 + 0.25) for i, text in enumerate(words)
    ], StyleProfile(max_chars_per_line=50))
    assert len(cues) == 1


def test_srt_period_timestamps_and_empty_cues_are_ingestible():
    cues = parse_srt_text("1\n00:00:01.000 --> 00:00:02.000\n\n2\n00:00:02.000 --> 00:00:03.000\nText.\n")
    assert len(cues) == 2
    assert not cues[0].plain_text
    assert cues[1].start_ms == 2000


def test_srt_missing_blank_separator_does_not_merge_dialogue():
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nOne.\n2\n00:00:02,000 --> 00:00:03,000\nTwo.\n")
    assert [cue.plain_text for cue in cues] == ["One.", "Two."]


@pytest.mark.parametrize("source, line_number", [
    ("1\n00:00:01,000 --> 00:00:02,000\nHallo Welt.\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n", 4),
    ("1\r\n00:00:01,000 --> 00:00:02,000\r\nHallo\r\nWelt.\r\n00:00:03.000 --> 00:00:04.000\r\nWie geht es dir?\r\n", 5),
    ("1\n00:00:01,000 --> 00:00:02,000\n00:00:03,000 --> 00:00:04,000\nWie geht es dir?\n", 3),
])
def test_srt_cue_timestamp_without_number_or_separator_is_rejected_not_merged(source, line_number):
    # Kept as text, the timestamp would merge two cues and be delivered as dialogue.
    from dubsync.srt_io import SRTParseError
    with pytest.raises(SRTParseError, match=f"subtitle line {line_number} is a cue timestamp"):
        parse_srt_text(source + "\n3\n00:00:05,000 --> 00:00:06,000\nGut, danke.\n")


def test_srt_writer_removes_interior_empty_lines_and_embedded_newlines():
    cues = [Cue(index=1, start_ms=1000, end_ms=2000, lines=["First", "", "Second\n\nThird"])]
    text = write_srt(cues)
    assert len(parse_srt_text(text)) == 1
    assert parse_srt_text(text)[0].lines == ["First", "Second", "Third"]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\x0b", "\x0c"])
def test_srt_writer_breaks_lines_only_at_line_feeds(separator):
    # Only CR/LF end an SRT line when it is read, so any other separator inside
    # an authored line is kept instead of becoming an extra subtitle line.
    cues = [Cue(index=1, start_ms=1000, end_ms=2000, lines=[f"A.{separator}B.", "C."])]
    assert parse_srt_text(write_srt(cues)) == cues


def test_sample_style_uses_robust_percentiles_and_detected_fps():
    cues = [Cue(index=i + 1, start_ms=i * 1240 + 40, end_ms=i * 1240 + 840, lines=["A normal subtitle line."])
            for i in range(40)]
    cues[-1] = cues[-1].model_copy(update={"end_ms": cues[-1].start_ms + 1, "lines": ["x" * 200]})
    result = resolve_generation_style(GenerationStyleRequest(source="sample"), fps=30.0, sample_cues=cues)
    assert result.profile.fps == 25.0
    assert result.profile.max_chars_per_line < 40
    assert result.profile.min_cue_dur == 0.8
    assert result.constraints.max_cps < 40


def test_srt_bytes_accept_legacy_encoding_with_visible_notice():
    from dubsync.srt_io import parse_srt_bytes
    with pytest.warns(UserWarning, match="Windows-1252"):
        cues = parse_srt_bytes("1\n00:00:01,000 --> 00:00:02,000\nGrüße.\n".encode("cp1252"))
    assert cues[0].plain_text == "Grüße."


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16", "utf-32"])
def test_legacy_srt_upload_preserves_authored_bytes_and_decodes_text(encoding):
    from starlette.datastructures import UploadFile
    from dubsync.srt_io import SRTParseLimits
    from dubsync.web.srt_uploads import read_validated_srt_upload
    data = "1\r\n00:00:01,000 --> 00:00:02,000\r\nGrüße.\r\n".encode(encoding)
    upload = UploadFile(BytesIO(data), filename="episode.srt")
    result = asyncio.run(read_validated_srt_upload(
        upload, max_bytes=2000, max_line_bytes=200,
        parse_limits=SRTParseLimits(max_lines=5, max_cues=1, max_line_chars=100), label="SRT",
    ))
    assert result.data == data
    assert result.cues[0].plain_text == "Grüße."
    # A BOM identifies UTF-16/32 losslessly; only a legacy fallback needs checking.
    assert bool(result.encoding_notice) is (encoding == "cp1252")


def test_srt_encoding_does_not_guess_windows_for_japanese():
    from dubsync.srt_io import SRTParseError, parse_srt_bytes
    data = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n".encode("shift_jis")
    with pytest.raises(SRTParseError, match="cannot be identified safely"):
        parse_srt_bytes(data)
    with pytest.warns(UserWarning, match="shift_jis"):
        cues = parse_srt_bytes(data, encoding="shift_jis")
    assert cues[0].plain_text == "こんにちは。"
