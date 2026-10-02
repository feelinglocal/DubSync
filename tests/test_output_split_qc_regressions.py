"""Display children keep QC attached to the cues the customer actually receives."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue, QCFlag, SpeechRegion, Word
from dubsync.qc_review import build_review
from dubsync.semantic_output import split_crowded_output_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.timing_refinement import SpeechEvidence


def _assets(tmp_path, words: list[Word], *, frames: int = 16000 * 8):
    audio, fixture, config = (tmp_path / name for name in ("episode.wav", "words.json", "providers.yaml"))
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * frames)
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    return audio, config


def _new_children(expansions) -> set[int]:
    return {child for children in expansions.values() for child in children[1:]}


@pytest.mark.parametrize("mode", ["fresh", "verify"])
@pytest.mark.parametrize("labels", [("speaker_0", "speaker_1"), ("chunk_0:1", "chunk_0:2")], ids=["scribe", "mai"])
def test_genuine_overlap_stays_in_review_when_its_earlier_cue_is_split(tmp_path, monkeypatch, mode, labels):
    # Speaker A's three authored lines need the two-line split; A's last word
    # "keys." (4.60-4.95 s) is still spoken when speaker B starts at 4.70 s.
    a_lines = ["We finished the work.", "Now we can go home.", "Please bring the keys."]
    tokens = " ".join(a_lines).split()
    last = len(tokens) - 1
    words = [Word(text=token, start=round(4.6 - (last - i) * .3, 3), end=round(4.82 - (last - i) * .3, 3),
                  speaker_id=labels[0]) for i, token in enumerate(tokens)]
    words[-1] = words[-1].model_copy(update={"end": 4.95})
    words += [Word(text=text, start=start, end=end, speaker_id=labels[1])
              for text, start, end in [("Wait", 4.70, 4.90), ("for", 5.00, 5.20), ("me!", 5.25, 5.60)]]
    words.sort(key=lambda word: (word.start, word.end))
    audio, config = _assets(tmp_path, words)
    source = tmp_path / "episode.srt"
    source.write_text(write_srt([
        Cue(index=1, start_ms=round(words[0].start * 1000), end_ms=4900, lines=a_lines),
        Cue(index=2, start_ms=4700, end_ms=5700, lines=["Wait for me!"]),
    ]), encoding="utf-8")
    profile = StyleProfile(max_chars_per_line=24, max_lines_per_cue=4, min_cue_dur=.1, tail_ms=0)
    monkeypatch.setattr(pipeline, "speech_evidence_for_words", lambda *_a, **_k: SpeechEvidence(
        words=words, regions=[SpeechRegion(start=words[0].start, end=5.6)], detected=True,
    ))

    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, style_profile=profile, no_llm=True, **kwargs)

    result = run()
    if mode != "fresh":
        first = result.output_srt.read_bytes()
        result = run(resume=mode)
        assert result.output_srt.read_bytes() == first
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    expansions = payload["output_segmentation"]["expansions"]
    assert list(expansions) == ["1"] and len(expansions["1"]) == 2
    tail_child = expansions["1"][-1]
    by_id = {cue["index"]: cue for cue in payload["cues"]}
    # The split child that keeps "keys." is still on screen with B's first word.
    assert by_id[tail_child]["end_ms"] > by_id[2]["start_ms"]
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert any(left.end_ms > right.start_ms for left, right in zip(delivered, delivered[1:]))

    report = result.report
    overlaps = [item for item in report["review"] if item["kind"] == "output_overlap_unresolved"]
    assert [item["cue_ids"] for item in overlaps] == [[tail_child, 2]]
    assert overlaps[0]["srt_numbers"] == [2, 3]
    assert "stale_overlap" not in {item["kind"] for item in report["diagnostics"]}
    assert report["summary"]["verdict"] != "clean"


def test_split_overlap_flag_reviews_every_pair_still_on_screen_and_drops_separated_siblings():
    # expand_output_flags listed [earlier, child, later] after the earlier cue was split.
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2134, lines=["We finished the work."]),
        Cue(index=3, start_ms=2200, end_ms=4967, lines=["Now we can go home.", "Please bring the keys."]),
        Cue(index=2, start_ms=4700, end_ms=5600, lines=["Wait for me!"]),
    ]
    flags = [QCFlag(kind="output_overlap_unresolved", cue_ids=[1, 3, 2], severity="error",
                    message="Acoustically timed cues overlap.", start=4.7, end=4.9)]

    review = build_review(flags, [], cues, source_cues=[])

    assert [(item.kind, item.cue_ids, item.srt_numbers) for item in review.review] == [
        ("output_overlap_unresolved", [3, 2], [2, 3])]
    assert "267 ms" in review.review[0].detail
    assert not any(item.kind == "stale_overlap" for item in review.diagnostics)
    assert review.verdict == "check"
    assert review.review[0].raw_flags == [0]

    # Control: once the tail child no longer meets the later cue the flag is stale.
    separated = [*cues[:1], cues[1].with_timing(2200, 4700), cues[2]]
    stale = build_review(flags, [], separated, source_cues=[])
    assert stale.review == [] and stale.verdict == "clean"
    assert [item.kind for item in stale.diagnostics] == ["stale_overlap"]
    assert stale.diagnostics[0].raw_flags == [0]


_LONG = ["We waited at the station for hours and hours,", "but nobody came to meet us,",
         "so we walked all the way home."]
_LONG_WORDS = [
    ("We", 0.50, 0.62), ("waited", 0.66, 0.98), ("at", 1.02, 1.10), ("the", 1.12, 1.20),
    ("station", 1.24, 1.66), ("for", 1.70, 1.82), ("hours", 1.86, 2.10), ("and", 2.14, 2.24),
    ("hours,", 2.28, 2.70),
    ("but", 3.10, 3.22), ("nobody", 3.26, 3.60), ("came", 3.64, 3.86), ("to", 3.90, 3.98),
    ("meet", 4.02, 4.20), ("us,", 4.24, 4.50),
    ("so", 4.90, 5.02), ("we", 5.06, 5.16), ("walked", 5.20, 5.50), ("all", 5.54, 5.66),
    ("the", 5.70, 5.78), ("way", 5.82, 5.98), ("home.", 6.02, 6.40),
    ("Thank", 8.00, 8.20), ("you", 8.24, 8.40), ("so", 8.44, 8.56), ("much.", 8.60, 9.00),
]


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_removed_last_source_cue_keeps_its_removal_when_an_output_split_adds_a_child(tmp_path, mode):
    # Default sync style (derived from the source): only the output splitter
    # divides the three-line cue, after the unspoken trailing duplicate (the
    # highest source id) was removed.
    source_cues = [
        Cue(index=1, start_ms=400, end_ms=6600, lines=_LONG),
        Cue(index=2, start_ms=7900, end_ms=9200, lines=["Thank you so much."]),
        Cue(index=3, start_ms=8500, end_ms=9000, lines=["Thank you."]),
    ]
    source = tmp_path / "episode.srt"
    source.write_text(write_srt(source_cues), encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture, config = tmp_path / "words.json", tmp_path / "providers.yaml"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "confidence": None} for text, start, end in _LONG_WORDS
    ]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")

    def run(**kwargs):
        return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                     providers_path=config, no_llm=True, fps=30, **kwargs)

    result = run()
    if mode != "fresh":
        first = result.output_srt.read_bytes()
        result = run(resume=mode)
        assert result.output_srt.read_bytes() == first
    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in delivered] == [
        "We waited at the station for hours and hours,",
        "but nobody came to meet us, so we walked all the way home.",
        "Thank you so much.",
    ]
    payload = json.loads((result.episode_workdir / "rebuild.json").read_text(encoding="utf-8"))
    expansions = payload["output_segmentation"]["expansions"]
    assert list(expansions) == ["1"] and len(expansions["1"]) == 2
    children = _new_children(expansions)
    assert children and not children & {cue.index for cue in source_cues}
    assert 3 not in {cue["index"] for cue in payload["cues"]}

    changes = result.report["changes"]
    removed = [item for item in changes if item["change"] == "removed"]
    assert [(item["cue_id"], item["kind"], item["old_text"]) for item in removed] == [
        (3, "duplicate_cue_merged", "Thank you.")]
    assert not any(item["cue_id"] == 3 and item["change"] != "removed" for item in changes)
    log = (result.episode_workdir / "changes.diff.srt").read_text(encoding="utf-8")
    assert "# removed after SRT #3 (cue 3)" in log
    assert "edited (cue 3)" not in log


def test_output_child_never_takes_the_id_of_an_omitted_source_cue():
    # Customer numbering is not in time order: omitted cue 3 holds the highest id.
    profile = StyleProfile(fps=30, min_cue_dur=0.5)
    timed = [("We", .5, .62), ("waited", .66, .98), ("at", 1.02, 1.1), ("the", 1.12, 1.2),
             ("station", 1.24, 1.66), ("for", 1.7, 1.82), ("hours,", 1.86, 2.3), ("but", 3.1, 3.22),
             ("nobody", 3.26, 3.6), ("came", 3.64, 3.86), ("to", 3.9, 3.98), ("meet", 4.02, 4.2),
             ("us.", 4.24, 4.5), ("Goodbye.", 9.0, 9.5)]
    words = [Word(text=text, start=start, end=end) for text, start, end in timed]
    source = [
        Cue(index=1, start_ms=400, end_ms=4600,
            lines=["We waited at the station for hours,", "but nobody came to meet us."]),
        Cue(index=3, start_ms=5600, end_ms=6400, lines=["Are you still there?"]),
        Cue(index=2, start_ms=8900, end_ms=9600, lines=["Goodbye."]),
    ]
    flags = [
        QCFlag(kind="low_confidence_adjudication", cue_ids=[3], severity="warning",
               message="Adjudication audio evidence is unclear or inaudible; source wording retained.",
               old_text="Are you still there?", new_text="Are you still there?", start=5.6, end=6.4),
        QCFlag(kind="missing_dialogue_audio_reconciled", cue_ids=[3], severity="info", confidence=1,
               old_text="Are you still there?", new_text="",
               message="Complete anchored audio confirmed this source cue was omitted; only this cue was removed.",
               start=4.6, end=8.9),
    ]
    pre_output = [source[0].with_timing(500, 4600), source[2].with_timing(9000, 9600)]
    reserved = {cue.index for cue in source} | {cue_id for flag in flags for cue_id in flag.cue_ids}

    segmented = split_crowded_output_cues(pre_output, words, {1: list(range(13)), 2: [13]}, profile,
                                          reserved_cue_ids=reserved)

    children = _new_children(segmented.expansions)
    assert children and not children & reserved
    review = build_review([*flags, *segmented.flags], [], segmented.cues, source_cues=source)
    assert review.review == []
    assert "resolved_omission_adjudication" in {item.kind for item in review.diagnostics}
    assert any(item.change == "removed" and item.cue_id == 3 for item in review.changes)
    assert not any(item.change == "edited" and item.cue_id == 3 for item in review.changes)


def test_composition_children_and_caption_slots_never_take_a_reserved_id():
    profile = StyleProfile(max_chars_per_line=30, min_cue_dur=.1, tail_ms=0)
    # Source cue 4 was removed earlier in the run; a finding still names cue 7.
    reserved = {1, 2, 3, 4, 7}
    # A two-page caption inside one spoken cue divides the speech at word boundaries.
    speech = Cue(index=1, start_ms=1000, end_ms=4000, lines=["Vamos embora daqui. Agora mesmo, por favor."])
    caption = Cue(index=2, start_ms=1500, end_ms=3500, lines=["[Hospital Central da Cidade de São Paulo]"])
    tokens = speech.plain_text.split()
    words = [Word(text=token, start=round(1.05 + position * .4, 3), end=round(1.3 + position * .4, 3))
             for position, token in enumerate(tokens)]

    split = compose_bracketed_annotations([speech, caption], {1: list(range(len(tokens)))}, words=words,
                                          profile=profile, reserved_cue_ids=reserved)

    children = _new_children(split.expansions)
    assert children and not children & reserved
    assert {cue.index for cue in split.cues} - {1, 2} == children

    # A caption across two spoken cues needs new display cues for its later gaps.
    spoken = [Cue(index=1, start_ms=1000, end_ms=2500, lines=["Vamos embora daqui."]),
              Cue(index=3, start_ms=3500, end_ms=5000, lines=["Agora mesmo."])]
    long_caption = Cue(index=2, start_ms=0, end_ms=6000, lines=["[Hospital Central]"])
    for composed in (
        compose_bracketed_annotations([long_caption, *spoken], {1: [], 3: []}, profile=profile,
                                      reserved_cue_ids=reserved),
        compose_bracketed_annotations([long_caption, *spoken], {1: [], 3: []}, reserved_cue_ids=reserved),
    ):
        created = {cue.index for cue in composed.cues} - {1, 2, 3}
        assert len(created) == 2 and not created & reserved
