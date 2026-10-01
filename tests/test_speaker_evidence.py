"""Speaker labels are evidence of a speaker change only when they are comparable.

MAI labels are scoped to one transcription chunk ("chunk_3:1"); the same actor
gets an unrelated label in the next chunk. Missing labels prove nothing either.
"""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.changes import apply_adjudication_decisions, indexed_multi_cue_replacements
from dubsync.cue_segmentation import group_word_indices_for_cues, split_speaker_turn_cues
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.speaker_evidence import has_known_different_speakers, speakers_known_different
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


@pytest.mark.parametrize("left, right, different", [
    ("speaker_1", "speaker_2", True),          # Scribe: episode-wide labels
    ("speaker_1", "speaker_1", False),
    ("chunk_1:0", "chunk_1:1", True),          # MAI: two actors inside one chunk
    ("chunk_1:0", "chunk_2:1", False),         # MAI: unrelated label scopes
    ("chunk_1:0", "chunk_2:0", False),
    ("chunk_1:0", "speaker_1", False),         # mixed scopes cannot be compared
    (None, "speaker_1", False),
    ("", "speaker_1", False),
])
def test_speaker_relation(left, right, different):
    assert speakers_known_different(left, right) is different
    assert speakers_known_different(right, left) is different


def test_known_different_speakers_in_a_group_of_labels():
    assert has_known_different_speakers(["chunk_1:0", "chunk_2:1", None]) is False
    assert has_known_different_speakers(["chunk_1:0", "chunk_2:1", "chunk_2:0"]) is True
    assert has_known_different_speakers([]) is False


def _boundary_words(first: str, second: str) -> list[Word]:
    return [Word(text=text, start=start, end=end, confidence=None, speaker_id=speaker) for text, start, end, speaker in [
        ("Ich", 299.5, 299.65, first), ("gehe", 299.7, 299.95, first),
        ("jetzt", 300.02, 300.3, second), ("nach", 300.35, 300.55, second), ("Hause.", 300.6, 301.1, second),
    ]]


def test_sentence_across_a_mai_chunk_boundary_is_not_split_into_two_speakers():
    # pipeline.md P-2: "Ich gehe" / "jetzt nach Hause." at the 300 s chunk boundary.
    cue = Cue(index=2, start_ms=299_500, end_ms=301_500, lines=["Ich gehe jetzt nach Hause."])
    words = _boundary_words("chunk_1:0", "chunk_2:1")
    alignment = AlignmentResult(cue_word_indices={2: [0, 1, 2, 3, 4]})

    cues, _, flags, expansions = split_speaker_turn_cues([cue], words, alignment, StyleProfile(fps=30))

    assert cues == [cue]
    assert flags == [] and expansions == {}


def test_two_actors_inside_one_chunk_are_still_split():
    cue = Cue(index=2, start_ms=299_500, end_ms=301_500, lines=["Ich gehe jetzt nach Hause."])
    words = _boundary_words("chunk_1:0", "chunk_1:1")
    alignment = AlignmentResult(cue_word_indices={2: [0, 1, 2, 3, 4]})

    cues, _, flags, _ = split_speaker_turn_cues([cue], words, alignment, StyleProfile(fps=30))

    assert [item.plain_text for item in cues] == ["Ich gehe", "jetzt nach Hause."]
    assert [flag.kind for flag in flags] == ["speaker_turn_split"]


def test_generated_cue_grouping_ignores_an_unrelated_chunk_label():
    words = _boundary_words("chunk_1:0", "chunk_2:1")

    groups = group_word_indices_for_cues(
        words, list(range(len(words))), StyleProfile(fps=30), max_gap_seconds=0.8, max_cue_duration_seconds=5.0,
    )

    assert groups == [[0, 1, 2, 3, 4]]


def test_full_sync_keeps_a_chunk_boundary_sentence_in_one_cue(tmp_path):
    srt = (
        "1\n00:04:58,000 --> 00:04:59,400\nGuten Morgen zusammen.\n\n"
        "2\n00:04:59,500 --> 00:05:01,500\nIch gehe jetzt nach Hause.\n\n"
        "3\n00:05:02,000 --> 00:05:03,000\nBis morgen dann.\n"
    )
    words = [{"text": text, "start": start, "end": end, "speaker_id": speaker, "confidence": None}
             for text, start, end, speaker in [
        ("Guten", 298.0, 298.3, "chunk_1:0"), ("Morgen", 298.35, 298.7, "chunk_1:0"), ("zusammen.", 298.75, 299.3, "chunk_1:0"),
        ("Ich", 299.5, 299.65, "chunk_1:0"), ("gehe", 299.7, 299.95, "chunk_1:0"),
        ("jetzt", 300.02, 300.3, "chunk_2:1"), ("nach", 300.35, 300.55, "chunk_2:1"), ("Hause.", 300.6, 301.1, "chunk_2:1"),
        ("Bis", 302.0, 302.2, "chunk_2:1"), ("morgen", 302.25, 302.6, "chunk_2:1"), ("dann.", 302.65, 302.95, "chunk_2:1"),
    ]]
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"

    result = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=providers, no_llm=True,
                                   style_profile=StyleProfile(fps=30, min_cue_dur=0.5))

    cues = parse_srt_text(output.read_text(encoding="utf-8"))
    assert [cue.plain_text for cue in cues] == ["Guten Morgen zusammen.", "Ich gehe jetzt nach Hause.", "Bis morgen dann."]
    assert not [flag for flag in result.report["flags"] if flag["kind"].startswith("speaker_turn")]


# --- rules that used to require a diarized speaker ----------------------------------------------


def _case174(speakers: list[str], *, right_anchor_start: float | None = 1322.75):
    # MAI ep11 case-174: 'Vamos.' / 'Vamos rápido.' / 'Yuanzhu vai pagar.', spoken "E o Yuanzhu vai pagar".
    cues = [
        Cue(index=452, start_ms=1321160, end_ms=1321680, lines=["Vamos."]),
        Cue(index=453, start_ms=1321750, end_ms=1322310, lines=["Vamos rápido."]),
        Cue(index=454, start_ms=1322310, end_ms=1323030, lines=["Yuanzhu vai pagar."]),
    ]
    span = DivergenceSpan(
        case_id="case-174", cue_ids=[452, 453, 454],
        srt_text="Vamos Vamos rápido Yuanzhu", asr_text="E o Antzu",
        srt_token_indices=[0, 1, 2, 3], asr_word_indices=[0, 1, 2],
        start=1322.24, end=1322.74, speaker_ids=speakers,
        left_anchor_cue_id=451, right_anchor_cue_id=454, right_anchor_start=right_anchor_start,
    )
    return cues, span


@pytest.mark.parametrize("speakers", [
    [],                                # MAI without diarization
    ["chunk_4:0", "chunk_5:2"],        # labels of unrelated chunk scopes
    ["speaker_4"],                     # one diarized actor (unchanged)
])
def test_unfinished_phrase_joins_its_spoken_continuation_without_speaker_labels(speakers):
    cues, span = _case174(speakers)

    assert indexed_multi_cue_replacements(cues, span, "E o Yuanzhu") == {
        452: (0, 1, ""), 453: (0, 2, ""), 454: (0, 1, "E o Yuanzhu"),
    }
    decision = AdjudicationDecision(case_id=span.case_id, verdict="hybrid", final_text="E o Yuanzhu",
                                    confidence=0.95, reason="confirmed unfinished phrase")
    changed, _ = apply_adjudication_decisions(cues, [span], [decision], StyleProfile())
    # No one-letter orphan cues "E." / "o." in the source slots.
    assert [(cue.index, cue.plain_text) for cue in changed] == [(454, "E o Yuanzhu vai pagar.")]


@pytest.mark.parametrize("speakers, right_anchor_start", [
    (["speaker_4", "speaker_5"], 1322.75),   # two different known actors
    (["chunk_4:0", "chunk_4:1"], 1322.75),
    ([], 1323.60),                           # no labels and a pause before the retained continuation
    ([], None),                              # no labels and no acoustic anchor at all
])
def test_continuation_without_evidence_is_not_assumed(speakers, right_anchor_start):
    cues, span = _case174(speakers, right_anchor_start=right_anchor_start)

    edits = indexed_multi_cue_replacements(cues, span, "E o Yuanzhu")

    assert edits is not None
    assert any(edits[cue_id][2] for cue_id in (452, 453))
