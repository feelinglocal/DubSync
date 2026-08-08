from dubsync.cue_segmentation import (
    group_word_indices_for_cues,
    segment_generated_adlib_cues,
)
from dubsync.models import AlignmentResult, Cue, Word
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


def _profile() -> StyleProfile:
    return StyleProfile(
        fps=30.0,
        max_lines_per_cue=2,
        max_chars_per_line=26,
        min_cue_dur=0.5,
        lead_in_ms=0,
        tail_ms=40,
    )


def test_generated_adlib_uses_only_the_asr_window_retained_by_adjudication():
    words = [
        Word(text="omitted", start=1.00, end=1.20, speaker_id="A"),
        Word(text="earlier", start=1.25, end=1.45, speaker_id="A"),
        Word(text="narration.", start=1.50, end=1.80, speaker_id="A"),
        Word(text="„VERRATEN", start=10.00, end=10.30, speaker_id="A"),
        Word(text="von", start=10.35, end=10.50, speaker_id="A"),
        Word(text='Feinden."', start=10.55, end=11.10, speaker_id="A"),
        Word(text="Unser", start=12.20, end=12.45, speaker_id="A"),
        Word(text="Ziel", start=12.50, end=12.70, speaker_id="A"),
        Word(text="bleibt.", start=12.75, end=13.20, speaker_id="A"),
        Word(text="omitted", start=20.00, end=20.20, speaker_id="B"),
        Word(text="dialogue.", start=20.25, end=20.60, speaker_id="B"),
    ]
    retained_text = "Verraten von Feinden. Unser Ziel bleibt."
    source = Cue(
        index=7,
        start_ms=1_000,
        end_ms=20_600,
        lines=[retained_text],
    )
    alignment = AlignmentResult(cue_word_indices={7: list(range(len(words)))})

    cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {7},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) == 2
    assert alphanumeric_signature(" ".join(cue.plain_text for cue in cues)) == alphanumeric_signature(
        retained_text
    )
    assert cues[0].start_ms == 10_000
    assert cues[-1].end_ms == 13_266
    assert [index for cue_id in expansions[7] for index in updated.cue_word_indices[cue_id]] == list(
        range(3, 9)
    )
    assert [flag.kind for flag in flags] == [
        "generated_adlib_word_window_refined",
        "generated_adlib_segmented",
    ]


def test_word_grouping_filters_invalid_indices_and_splits_known_speaker_changes():
    words = [
        Word(text="first", start=0.00, end=0.20, speaker_id="A"),
        Word(text="second", start=0.25, end=0.45, speaker_id="B"),
        Word(text="line.", start=0.50, end=0.80, speaker_id="B"),
    ]

    groups = group_word_indices_for_cues(
        words,
        [99, 2, 1, 0, 1, -1],
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert groups == [[0], [1, 2]]


def test_generated_adlib_keeps_full_word_mapping_when_retained_text_has_no_exact_window():
    words = [
        Word(text="spoken", start=1.00, end=1.20, speaker_id="A"),
        Word(text="audio", start=1.25, end=1.45, speaker_id="A"),
        Word(text="that", start=1.50, end=1.70, speaker_id="A"),
        Word(text="does", start=1.75, end=1.95, speaker_id="A"),
        Word(text="not", start=2.00, end=2.20, speaker_id="A"),
        Word(text="match.", start=2.25, end=2.55, speaker_id="A"),
    ]
    source = Cue(index=4, start_ms=1_000, end_ms=2_550, lines=["corrected editorial text"])
    alignment = AlignmentResult(cue_word_indices={4: list(range(len(words)))})

    cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {4},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated.cue_word_indices[4] == list(range(len(words)))
    assert [flag.kind for flag in flags] == ["generated_adlib_word_mapping_unavailable"]
    assert expansions == {}


def test_generated_adlib_does_not_guess_between_repeated_exact_word_windows():
    words = [
        Word(text="same", start=1.00, end=1.20, speaker_id="A"),
        Word(text="phrase.", start=1.25, end=1.55, speaker_id="A"),
        Word(text="same", start=3.00, end=3.20, speaker_id="A"),
        Word(text="phrase.", start=3.25, end=3.55, speaker_id="A"),
    ]
    source = Cue(index=4, start_ms=1_000, end_ms=3_550, lines=["same phrase."])
    alignment = AlignmentResult(cue_word_indices={4: list(range(len(words)))})

    cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {4},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert [index for cue_id in expansions[4] for index in updated.cue_word_indices[cue_id]] == [
        0,
        1,
        2,
        3,
    ]
    assert [flag.kind for flag in flags] == [
        "generated_adlib_word_mapping_unavailable",
        "generated_adlib_segmented",
    ]
    assert len(cues) == 2


def test_generated_adlib_accepts_full_span_timing_for_equal_length_word_corrections():
    words = [
        Word(text="wrong", start=1.00, end=1.20, speaker_id="A"),
        Word(text="audio", start=1.25, end=1.45, speaker_id="A"),
        Word(text="wording.", start=1.50, end=1.80, speaker_id="A"),
    ]
    source = Cue(index=4, start_ms=1_000, end_ms=1_800, lines=["correct editorial wording."])
    alignment = AlignmentResult(cue_word_indices={4: [0, 1, 2]})

    cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {4},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated.cue_word_indices[4] == [0, 1, 2]
    assert flags == []
    assert expansions == {}


def test_generated_adlib_flags_text_expansion_beyond_available_asr_words():
    words = [
        Word(text="kurzer", start=1.00, end=1.20, speaker_id="A"),
        Word(text="Satz.", start=1.25, end=1.55, speaker_id="A"),
    ]
    source = Cue(
        index=4,
        start_ms=1_000,
        end_ms=1_550,
        lines=["Dieser erheblich längere Text hat keine akustische Wortabdeckung."],
    )
    alignment = AlignmentResult(cue_word_indices={4: [0, 1]})

    _cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {4},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert [
        word_index
        for cue_id in expansions[4]
        for word_index in updated.cue_word_indices[cue_id]
    ] == [0, 1]
    assert flags[0].kind == "generated_adlib_word_mapping_unavailable"


def test_generated_adlib_flags_when_candidate_indices_have_no_valid_word_timing():
    words = [Word(text="invalid", start=-1.0, end=-0.5, speaker_id="A")]
    source = Cue(index=4, start_ms=1_000, end_ms=1_500, lines=["generated dialogue"])
    alignment = AlignmentResult(cue_word_indices={4: [0, -1, 99]})

    cues, updated, flags, expansions = segment_generated_adlib_cues(
        [source],
        words,
        alignment,
        {4},
        _profile(),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated.cue_word_indices[4] == []
    assert [flag.kind for flag in flags] == ["generated_adlib_word_mapping_unavailable"]
    assert expansions == {}
