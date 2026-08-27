from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan, Word
from dubsync.pipeline import _adlib_cue_ids_by_case
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import normalize_token


def test_profanity_compound_suffix_remains_visible_to_alignment() -> None:
    assert normalize_token("Sch*i\u00dfe") == normalize_token("Schei\u00dfe") == "scheisse"
    assert normalize_token("Schei\u00dfsystem") == "scheisssystem"
    assert normalize_token("Schei\u00dfsystem") != normalize_token("Schei\u00dfe")

    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=1800,
            lines=["Was f\u00fcr ein Sch*i\u00dfe System"],
        )
    ]
    words = [
        Word(text="Was", start=0.00, end=0.20, speaker_id="A"),
        Word(text="f\u00fcr", start=0.22, end=0.38, speaker_id="A"),
        Word(text="ein", start=0.40, end=0.55, speaker_id="A"),
        Word(text="Schei\u00dfsystem", start=0.57, end=1.20, speaker_id="A"),
    ]

    alignment = align_cues_to_words(cues, words)

    assert len(alignment.divergence_spans) == 1
    span = alignment.divergence_spans[0]
    assert span.srt_text == "Sch*i\u00dfe System"
    assert span.asr_text == "Schei\u00dfsystem"
    assert span.srt_token_indices == [3, 4]
    assert span.asr_word_indices == [3]


def test_masked_productive_profanity_compound_matches_uncensored_asr() -> None:
    assert normalize_token("Sch*iss-Wetter") == "scheiss-wetter"
    assert normalize_token("Sch*iss-Wetter") == normalize_token("Scheiss-Wetter")

    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=1600,
            lines=["Das ist Sch*iss-Wetter."],
        )
    ]
    words = [
        Word(text="Das", start=0.0, end=0.2, speaker_id="A"),
        Word(text="ist", start=0.22, end=0.4, speaker_id="A"),
        Word(text="Scheiss-Wetter", start=0.42, end=1.1, speaker_id="A"),
    ]

    alignment = align_cues_to_words(cues, words)

    assert alignment.divergence_spans == []
    assert alignment.anchor_coverage == 1.0


def test_percent_symbol_aligns_with_spoken_german_percent_word() -> None:
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=2200,
            lines=["Die Chance liegt bei 10%."],
        )
    ]
    words = [
        Word(text="Die", start=0.00, end=0.20, speaker_id="A"),
        Word(text="Chance", start=0.22, end=0.55, speaker_id="A"),
        Word(text="liegt", start=0.57, end=0.77, speaker_id="A"),
        Word(text="bei", start=0.79, end=0.94, speaker_id="A"),
        Word(text="zehn", start=0.96, end=1.18, speaker_id="A"),
        Word(text="Prozent", start=1.20, end=1.58, speaker_id="A"),
    ]

    alignment = align_cues_to_words(cues, words)

    assert alignment.divergence_spans == []
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4, 5]}
    assert alignment.anchor_coverage == 1.0


def test_cross_cue_partial_compound_replacement_preserves_context_and_cues() -> None:
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=1600,
            lines=["Zwei Drittel des Eroberungs-"],
        ),
        Cue(
            index=2,
            start_ms=1600,
            end_ms=3000,
            lines=["fortschritts sind abgeschlossen."],
        ),
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2],
        srt_text="Eroberungs- fortschritts",
        asr_text="Eroberungsfortschritts",
        start=1.20,
        end=1.90,
        confidence=0.99,
        speaker_ids=["A"],
        srt_token_indices=[3, 4],
        asr_word_indices=[3],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="Eroberungsfortschritts",
        confidence=0.99,
        speaker="A",
        reason="word timestamps confirm one spoken compound",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert [cue.index for cue in transformed] == [1, 2]
    assert [cue.plain_text for cue in transformed] == [
        "Zwei Drittel des Eroberungsfortschritts",
        "sind abgeschlossen.",
    ]


def test_cross_cue_full_drop_does_not_leave_punctuation_only_cues() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["Für eine Suppe,"]),
        Cue(index=2, start_ms=500, end_ms=1000, lines=["Idiot!"]),
        Cue(index=3, start_ms=1000, end_ms=1500, lines=["Gib dir die Schuld,"]),
        Cue(index=4, start_ms=1500, end_ms=2000, lines=["du Dummkopf."]),
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[1, 2, 3, 4],
        srt_text="Für eine Suppe Idiot Gib dir die Schuld du Dummkopf",
        asr_text="Ah",
        start=2.1,
        end=2.4,
        confidence=0.99,
        speaker_ids=["A"],
        srt_token_indices=list(range(11)),
        asr_word_indices=[11],
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="Ah!",
        confidence=0.99,
        speaker="A",
        reason="the source block is absent and the actor vocalizes instead",
    )

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
    )

    assert [cue.index for cue in transformed] == [1]
    assert transformed[0].plain_text == "Ah!"


def test_nearby_same_speaker_insertion_moves_terminal_punctuation() -> None:
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=1000,
            lines=["Jungs, schnappt ihn!"],
            speaker_id="A",
        )
    ]
    words = [
        Word(text="Jungs", start=0.10, end=0.30, speaker_id="A"),
        Word(text="schnappt", start=0.32, end=0.68, speaker_id="A"),
        Word(text="ihn", start=0.70, end=1.00, speaker_id="A"),
        Word(text="euch", start=1.04, end=1.16, speaker_id="A"),
    ]
    alignment = align_cues_to_words(cues, words)
    assert len(alignment.divergence_spans) == 1
    span = alignment.divergence_spans[0]
    assert span.cue_ids == []
    assert span.asr_text == "euch"
    assert span.left_anchor_cue_id == 1
    assert span.left_anchor_end == 1.00
    assert span.left_anchor_speaker_id == "A"

    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="euch",
        confidence=0.99,
        speaker="A",
        reason="same-speaker word begins 40 ms after the left anchor",
    )
    adlib_ids, _ = _adlib_cue_ids_by_case(
        cues,
        [span],
        [decision],
        alignment.unmatched_cue_ids,
    )

    assert adlib_ids == {span.case_id: 1}

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
        adlib_cue_ids_by_case=adlib_ids,
    )

    assert len(transformed) == 1
    assert transformed[0].plain_text == "Jungs, schnappt ihn euch!"


@pytest.mark.parametrize(
    ("gap_seconds", "speaker_ids"),
    [
        (0.10, ["A"]),
        (0.04, []),
    ],
)
def test_terminal_punctuation_insertion_requires_narrow_confirmed_continuation(
    gap_seconds: float,
    speaker_ids: list[str],
) -> None:
    cues = [
        Cue(
            index=1,
            start_ms=0,
            end_ms=1000,
            lines=["Hallo!"],
            speaker_id="A",
        )
    ]
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[],
        srt_text="",
        asr_text="Ja",
        start=1.0 + gap_seconds,
        end=1.2 + gap_seconds,
        confidence=0.99,
        speaker_ids=speaker_ids,
        srt_token_indices=[],
        asr_word_indices=[1],
        left_anchor_cue_id=1,
        left_anchor_end=1.0,
        left_anchor_speaker_id="A",
    )
    decision = AdjudicationDecision(
        case_id="case-1",
        verdict="use_audio",
        final_text="Ja",
        confidence=0.99,
        speaker="A",
        reason="fixture",
    )

    adlib_ids, _ = _adlib_cue_ids_by_case(
        cues,
        [span],
        [decision],
        [],
    )

    assert adlib_ids == {"case-1": 2}

    transformed, _ = apply_adjudication_decisions(
        cues,
        [span],
        [decision],
        StyleProfile(),
        adlib_cue_ids_by_case=adlib_ids,
    )

    assert [cue.plain_text for cue in transformed] == ["Hallo!", "Ja"]


def test_generated_adlib_far_beyond_source_span_is_rejected_with_qc_flag() -> None:
    cues = [
        Cue(index=1, start_ms=0, end_ms=1800, lines=["Wir gehen jetzt."]),
        Cue(index=2, start_ms=3200, end_ms=5000, lines=["Kommst du mit?"]),
    ]
    span = DivergenceSpan(
        case_id="far-tail-adlib",
        cue_ids=[],
        srt_text="",
        asr_text="Warte auf mich",
        start=20.0,
        end=21.2,
        confidence=0.98,
        speaker_ids=["A"],
        asr_word_indices=[8, 9, 10],
    )
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="Warte auf mich!",
        confidence=0.98,
        speaker="A",
        reason="ASR-only tail content",
    )

    adlib_ids, flags = _adlib_cue_ids_by_case(cues, [span], [decision], [])

    assert adlib_ids == {}
    assert [flag.kind for flag in flags] == ["adlib_rejected_outside_source_span"]
    assert flags[0].new_text == "Warte auf mich!"
    assert flags[0].start == 20.0
    assert flags[0].end == 21.2


def test_highly_repetitive_song_like_adlib_is_rejected_with_qc_flag() -> None:
    cues = [Cue(index=1, start_ms=0, end_ms=120_000, lines=["Episode dialogue"])]
    lyrics = "Go, let it all go, let it all go. Go, let it all go, let it all go."
    span = DivergenceSpan(
        case_id="repetitive-adlib",
        cue_ids=[],
        srt_text="",
        asr_text=lyrics,
        start=50.0,
        end=54.0,
        confidence=0.99,
        speaker_ids=["A"],
        asr_word_indices=list(range(16)),
    )
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text=lyrics,
        confidence=0.99,
        speaker="A",
        reason="ASR-only repeated content",
    )

    adlib_ids, flags = _adlib_cue_ids_by_case(cues, [span], [decision], [])

    assert adlib_ids == {}
    assert [flag.kind for flag in flags] == ["adlib_rejected_repetitive_content"]
    assert flags[0].new_text == lyrics
    assert flags[0].start == 50.0
    assert flags[0].end == 54.0


def test_normalize_token_uses_bounded_cache_for_hot_alignment_paths():
    normalize_token.cache_clear()

    assert normalize_token("gehen") == "gehen"
    assert normalize_token("gehen") == "gehen"

    info = normalize_token.cache_info()
    assert info.hits >= 1
    assert info.maxsize is not None
