import re

import pytest

from dubsync.cue_segmentation import (
    group_word_indices_for_cues,
    split_overlong_existing_cues,
    segment_generated_adlib_cues,
)
from dubsync.models import AlignmentResult, Cue, Word
from dubsync.output_order import finalize_cues_for_output
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


def test_existing_sync_cue_splits_by_line_limit_with_word_timing():
    words = [
        Word(text="Team", start=1.00, end=1.18, speaker_id="A"),
        Word(text="Falcon", start=1.20, end=1.48, speaker_id="A"),
        Word(text="hat", start=1.50, end=1.62, speaker_id="A"),
        Word(text="eigenmächtig", start=1.64, end=2.05, speaker_id="A"),
        Word(text="die", start=2.07, end=2.20, speaker_id="A"),
        Word(text="Position", start=2.22, end=2.55, speaker_id="A"),
        Word(text="verraten.", start=2.57, end=2.95, speaker_id="A"),
    ]
    source = Cue(
        index=12,
        start_ms=1_000,
        end_ms=2_950,
        lines=["Team Falcon hat eigenmächtig die Position verraten."],
    )
    alignment = AlignmentResult(cue_word_indices={12: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 18, "max_lines_per_cue": 2})

    cues, updated_alignment, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) == 2
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert " ".join(cue.plain_text for cue in cues) == source.plain_text
    assert cues[0].start_ms == 1_000
    assert cues[-1].end_ms == 3_000
    assert [index for cue_id in expansions[12] for index in updated_alignment.cue_word_indices[cue_id]] == list(
        range(len(words))
    )
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_split"]


def test_existing_sync_cue_does_not_split_when_word_mapping_is_unavailable():
    source = Cue(
        index=12,
        start_ms=1_000,
        end_ms=3_000,
        lines=["Team Falcon hat eigenmächtig die Position verraten."],
    )
    alignment = AlignmentResult(cue_word_indices={12: []})
    profile = _profile().model_copy(update={"max_chars_per_line": 18, "max_lines_per_cue": 2})

    cues, updated_alignment, flags, expansions = split_overlong_existing_cues(
        [source],
        [],
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated_alignment == alignment
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_timing_unavailable"]


def test_existing_sync_cue_does_not_split_inline_markup_tags():
    words = [
        Word(text="Team", start=1.00, end=1.18),
        Word(text="Falcon", start=1.20, end=1.48),
        Word(text="hat", start=1.50, end=1.62),
        Word(text="eigenmächtig", start=1.64, end=2.05),
        Word(text="die", start=2.07, end=2.20),
        Word(text="Position", start=2.22, end=2.55),
        Word(text="verraten.", start=2.57, end=2.95),
    ]
    source = Cue(
        index=12,
        start_ms=1_000,
        end_ms=2_950,
        lines=["<i>Team Falcon hat eigenmächtig die Position verraten.</i>"],
    )
    alignment = AlignmentResult(cue_word_indices={12: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 18, "max_lines_per_cue": 2})

    cues, updated_alignment, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated_alignment == alignment
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_markup_unsupported"]


def test_existing_sync_cue_conserves_case_and_punctuation_at_acoustic_boundary():
    editorial_text = '„NICHT blind trennen“, sagte Weiß-Bär — JETZT!'
    words = [
        Word(text="NICHT", start=1.00, end=1.18, speaker_id="A"),
        Word(text="blind", start=1.20, end=1.38, speaker_id="A"),
        Word(text="trennen", start=1.40, end=1.65, speaker_id="A"),
        Word(text="sagte", start=1.67, end=1.82, speaker_id="A"),
        Word(text="Weiß-Bär", start=1.84, end=2.10, speaker_id="A"),
        Word(text="JETZT!", start=2.12, end=2.42, speaker_id="A"),
    ]
    source = Cue(
        index=7,
        start_ms=1_000,
        end_ms=2_420,
        lines=[editorial_text],
        speaker_id="A",
        character="Kapitänin Adler",
    )
    alignment = AlignmentResult(cue_word_indices={7: list(range(len(words)))})
    profile = _profile().model_copy(
        update={"max_chars_per_line": 12, "max_lines_per_cue": 2, "tail_ms": 0}
    )

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) > 1
    assert " ".join(cue.plain_text for cue in cues) == editorial_text
    assert [cue.character for cue in cues] == ["Kapitänin Adler"] * len(cues)
    assert all(cue.speaker_id == "A" for cue in cues)
    assert [
        word_index
        for cue_id in expansions[7]
        for word_index in updated.cue_word_indices[cue_id]
    ] == list(range(len(words)))
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_split"]


@pytest.mark.parametrize("line_count", [3, 4, 5])
def test_existing_sync_genuine_multiline_source_becomes_at_most_two_lines(line_count):
    source_lines = [f"Abschnitt {position}." for position in range(1, line_count + 1)]
    words: list[Word] = []
    for position in range(line_count):
        base = 1.0 + position * 0.8
        words.extend(
            [
                Word(text="Abschnitt", start=base, end=base + 0.25, speaker_id="A"),
                Word(
                    text=f"{position + 1}.",
                    start=base + 0.27,
                    end=base + 0.52,
                    speaker_id="A",
                ),
            ]
        )
    source = Cue(
        index=18,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=source_lines,
        speaker_id="A",
    )
    alignment = AlignmentResult(cue_word_indices={18: list(range(len(words)))})
    # Deliberately wide: the configured line-count limit, not character wrapping,
    # must detect a genuine 3-5-line source cue.
    profile = _profile().model_copy(
        update={"max_chars_per_line": 120, "max_lines_per_cue": 2, "tail_ms": 0}
    )

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) > 1
    assert all(1 <= len(cue.lines) <= 2 for cue in cues)
    assert [line for cue in cues for line in cue.lines] == source_lines
    assert [
        word_index
        for cue_id in expansions[18]
        for word_index in updated.cue_word_indices[cue_id]
    ] == list(range(len(words)))
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_split"]


def test_existing_sync_split_preserves_balanced_source_italics_and_line_breaks():
    source_lines = [
        "<i>„NICHT blind trennen“,</i>",
        "<i>sagte Weiß-Bär.</i>",
        "<i>JETZT bleibt alles</i>",
        "<i>GENAU so!</i>",
    ]
    spoken_tokens = [
        "NICHT",
        "blind",
        "trennen",
        "sagte",
        "Weiß-Bär.",
        "JETZT",
        "bleibt",
        "alles",
        "GENAU",
        "so!",
    ]
    words = [
        Word(
            text=token,
            start=1.0 + position * 0.25,
            end=1.18 + position * 0.25,
            speaker_id="A",
        )
        for position, token in enumerate(spoken_tokens)
    ]
    source = Cue(
        index=22,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=source_lines,
        speaker_id="A",
        character="Weiß-Bär",
    )
    alignment = AlignmentResult(cue_word_indices={22: list(range(len(words)))})
    profile = _profile().model_copy(
        update={"max_chars_per_line": 80, "max_lines_per_cue": 2, "tail_ms": 0}
    )

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) > 1
    assert [line for cue in cues for line in cue.lines] == source_lines
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert all(re.fullmatch(r"(?:<i>.*</i>)(?:\n<i>.*</i>)?", cue.text) for cue in cues)
    assert [cue.character for cue in cues] == ["Weiß-Bär"] * len(cues)
    assert [
        word_index
        for cue_id in expansions[22]
        for word_index in updated.cue_word_indices[cue_id]
    ] == list(range(len(words)))
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_split"]


def test_existing_sync_cue_does_not_split_markup_spanning_source_lines():
    source = Cue(
        index=23,
        start_ms=1_000,
        end_ms=2_500,
        lines=["<i>Alpha beta", "gamma delta", "epsilon zeta.</i>"],
    )
    words = [
        Word(text=token, start=1.0 + position * 0.2, end=1.16 + position * 0.2)
        for position, token in enumerate("Alpha beta gamma delta epsilon zeta.".split())
    ]
    alignment = AlignmentResult(cue_word_indices={23: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 80, "max_lines_per_cue": 2})

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated == alignment
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_markup_unsupported"]


@pytest.mark.parametrize("boundary_kind", ["sentence", "speaker", "pause"])
def test_existing_sync_split_respects_sentence_speaker_and_pause_boundaries(boundary_kind):
    first_tail = "dort." if boundary_kind == "sentence" else "dort"
    words = [
        Word(text="Wir", start=1.00, end=1.18, speaker_id="A"),
        Word(text="bleiben", start=1.20, end=1.42, speaker_id="A"),
        Word(text=first_tail, start=1.44, end=1.70, speaker_id="A"),
        Word(
            text="Danach",
            start=2.80 if boundary_kind == "pause" else 1.72,
            end=3.00 if boundary_kind == "pause" else 1.92,
            speaker_id="B" if boundary_kind == "speaker" else "A",
        ),
        Word(
            text="gehen",
            start=3.02 if boundary_kind == "pause" else 1.94,
            end=3.22 if boundary_kind == "pause" else 2.14,
            speaker_id="B" if boundary_kind == "speaker" else "A",
        ),
        Word(
            text="wir.",
            start=3.24 if boundary_kind == "pause" else 2.16,
            end=3.50 if boundary_kind == "pause" else 2.42,
            speaker_id="B" if boundary_kind == "speaker" else "A",
        ),
    ]
    source = Cue(
        index=31,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=["Wir bleiben", first_tail, "Danach gehen", "wir."],
    )
    alignment = AlignmentResult(cue_word_indices={31: list(range(len(words)))})
    profile = _profile().model_copy(
        update={"max_chars_per_line": 100, "max_lines_per_cue": 2, "tail_ms": 0}
    )

    cues, updated, _flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert len(cues) == 2
    first_id, second_id = expansions[31]
    assert updated.cue_word_indices[first_id] == [0, 1, 2]
    assert updated.cue_word_indices[second_id] == [3, 4, 5]
    assert cues[0].end_ms == profile.snap_ceil(words[2].end * 1000)
    assert cues[1].start_ms == profile.snap_floor(words[3].start * 1000)
    assert cues[0].plain_text == f"Wir bleiben {first_tail}"
    assert cues[1].plain_text == "Danach gehen wir."


def test_existing_sync_split_does_not_pack_different_speakers_into_one_cue():
    words = [
        Word(text="Nur", start=1.00, end=1.18, speaker_id="A"),
        Word(text="ich", start=1.20, end=1.42, speaker_id="A"),
        Word(text="Dann", start=1.44, end=1.66, speaker_id="B"),
        Word(text="reden", start=1.68, end=1.90, speaker_id="B"),
        Word(text="wir", start=1.92, end=2.12, speaker_id="B"),
        Word(text="gemeinsam", start=2.14, end=2.42, speaker_id="B"),
        Word(text="weiter", start=2.44, end=2.66, speaker_id="B"),
        Word(text="hier.", start=2.68, end=2.92, speaker_id="B"),
    ]
    source = Cue(
        index=32,
        start_ms=1_000,
        end_ms=2_920,
        lines=["Nur ich", "Dann reden", "wir gemeinsam", "weiter hier."],
    )
    alignment = AlignmentResult(cue_word_indices={32: list(range(len(words)))})
    profile = _profile().model_copy(
        update={"max_chars_per_line": 100, "max_lines_per_cue": 2, "tail_ms": 0}
    )

    cues, updated, _flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues[0].lines == ["Nur ich"]
    assert updated.cue_word_indices[expansions[32][0]] == [0, 1]
    for cue_id in expansions[32]:
        speakers = {words[index].speaker_id for index in updated.cue_word_indices[cue_id]}
        assert len(speakers) == 1


@pytest.mark.parametrize("mapping_kind", ["ambiguous", "partial"])
def test_existing_sync_cue_fails_closed_for_ambiguous_or_partial_asr_mapping(mapping_kind):
    source = Cue(
        index=40,
        start_ms=1_000,
        end_ms=4_000,
        lines=["Alpha beta gamma delta."],
        speaker_id="A",
        character="Editorial Name",
    )
    tokens = ["Alpha", "beta", "gamma", "delta."]
    if mapping_kind == "ambiguous":
        tokens = [*tokens, *tokens]
    else:
        tokens = tokens[:-1]
    words = [
        Word(
            text=token,
            start=1.0 + position * 0.3,
            end=1.2 + position * 0.3,
            speaker_id="A",
        )
        for position, token in enumerate(tokens)
    ]
    alignment = AlignmentResult(cue_word_indices={40: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 8, "max_lines_per_cue": 2})

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert cues[0].model_dump_json() == source.model_dump_json()
    assert updated.cue_word_indices[40] == list(range(len(words)))
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_timing_unavailable"]


def test_existing_sync_cue_fails_closed_for_equal_length_asr_substitution():
    source = Cue(
        index=41,
        start_ms=1_000,
        end_ms=3_000,
        lines=["Alpha beta", "gamma delta", "epsilon zeta."],
    )
    # Same token count as the source, but one substituted word means the source
    # line boundaries no longer have exact per-word timing evidence.
    words = [
        Word(text=token, start=1.0 + position * 0.25, end=1.16 + position * 0.25)
        for position, token in enumerate("Alpha beta gamma theta epsilon zeta.".split())
    ]
    alignment = AlignmentResult(cue_word_indices={41: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 80, "max_lines_per_cue": 2})

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source]
    assert updated.cue_word_indices[41] == list(range(len(words)))
    assert expansions == {}
    assert [flag.kind for flag in flags] == ["sync_cue_line_limit_timing_unavailable"]


def test_existing_sync_splitter_only_processes_explicit_source_cue_ids():
    source = Cue(index=1, start_ms=0, end_ms=500, lines=["Kurz und gut."])
    generated = Cue(
        index=99,
        start_ms=1_000,
        end_ms=2_900,
        lines=["Generated dialogue that would otherwise exceed two lines."],
    )
    words = [
        Word(text=token, start=1.0 + i * 0.25, end=1.18 + i * 0.25, speaker_id="A")
        for i, token in enumerate(generated.plain_text.split())
    ]
    alignment = AlignmentResult(cue_word_indices={1: [], 99: list(range(len(words)))})
    profile = _profile().model_copy(update={"max_chars_per_line": 12, "max_lines_per_cue": 2})

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source, generated],
        words,
        alignment,
        profile,
        source_cue_ids={1},
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues == [source, generated]
    assert cues[1].model_dump_json() == generated.model_dump_json()
    assert updated == alignment
    assert flags == []
    assert expansions == {}


def test_existing_sync_normal_two_line_cue_is_bit_for_bit_unchanged():
    source = Cue(
        index=51,
        start_ms=12_345,
        end_ms=14_678,
        lines=["„Genau so“,", "sagte Weiß-Bär."],
        speaker_id="speaker_7",
        character="Weiß-Bär",
    )
    alignment = AlignmentResult(cue_word_indices={51: [4, 5, 6]})
    profile = _profile().model_copy(update={"max_chars_per_line": 26, "max_lines_per_cue": 2})

    cues, updated, flags, expansions = split_overlong_existing_cues(
        [source],
        [],
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert cues[0] is source
    assert cues[0].model_dump_json() == source.model_dump_json()
    assert updated == alignment
    assert flags == []
    assert expansions == {}


def test_split_outputs_finalize_without_more_than_two_simultaneous_visible_lines():
    words = [
        Word(text=token, start=1.0 + i * 0.22, end=1.20 + i * 0.22, speaker_id="A")
        for i, token in enumerate("Eins zwei drei vier fünf sechs sieben acht neun zehn.".split())
    ]
    source = Cue(
        index=60,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=["Eins zwei drei vier fünf sechs sieben acht neun zehn."],
    )
    alignment = AlignmentResult(cue_word_indices={60: list(range(len(words)))})
    profile = _profile().model_copy(
        update={"max_chars_per_line": 12, "max_lines_per_cue": 2, "tail_ms": 40}
    )

    split, _updated, _flags, _expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )
    finalized, _output_flags = finalize_cues_for_output(split, profile, no_overlaps=True)

    assert all(len(cue.lines) <= 2 for cue in finalized)
    assert all(left.end_ms <= right.start_ms for left, right in zip(finalized, finalized[1:]))
    event_times = sorted({cue.start_ms for cue in finalized} | {cue.end_ms for cue in finalized})
    for timestamp in event_times:
        visible_lines = sum(
            len(cue.lines) for cue in finalized if cue.start_ms <= timestamp < cue.end_ms
        )
        assert visible_lines <= 2


def test_existing_sync_split_preserves_cue_ids_and_unrelated_alignment_entries():
    words = [
        Word(text=token, start=1.0 + i * 0.3, end=1.2 + i * 0.3, speaker_id="A")
        for i, token in enumerate("Alpha beta gamma delta epsilon zeta.".split())
    ]
    split_source = Cue(
        index=5,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=["Alpha beta gamma delta epsilon zeta."],
    )
    untouched = Cue(index=90, start_ms=5_000, end_ms=5_800, lines=["Untouched."])
    alignment = AlignmentResult(
        cue_word_indices={5: list(range(len(words))), 90: [77]},
        unmatched_cue_ids=[90],
    )
    profile = _profile().model_copy(update={"max_chars_per_line": 10, "max_lines_per_cue": 2})

    cues, updated, _flags, expansions = split_overlong_existing_cues(
        [split_source, untouched],
        words,
        alignment,
        profile,
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert expansions[5][0] == 5
    assert expansions[5][1:] == list(range(91, 91 + len(expansions[5]) - 1))
    assert len({cue.index for cue in cues}) == len(cues)
    assert next(cue for cue in cues if cue.index == 90) is untouched
    assert updated.cue_word_indices[90] == [77]
    assert updated.unmatched_cue_ids == [90]
    assert [
        word_index
        for cue_id in expansions[5]
        for word_index in updated.cue_word_indices[cue_id]
    ] == list(range(len(words)))


def test_existing_sync_line_limit_is_exactly_the_selected_maximum():
    words = [
        Word(text=token, start=1.0 + i * 0.3, end=1.2 + i * 0.3, speaker_id="A")
        for i, token in enumerate("Eine Zeile. Zweite Zeile. Dritte Zeile.".split())
    ]
    source = Cue(
        index=72,
        start_ms=1_000,
        end_ms=int(words[-1].end * 1000),
        lines=["Eine Zeile.", "Zweite Zeile.", "Dritte Zeile."],
    )
    alignment = AlignmentResult(cue_word_indices={72: list(range(len(words)))})
    base = _profile().model_copy(update={"max_chars_per_line": 100, "tail_ms": 0})

    allowed, allowed_alignment, allowed_flags, allowed_expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        base.model_copy(update={"max_lines_per_cue": 3}),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )
    limited, _limited_alignment, limited_flags, limited_expansions = split_overlong_existing_cues(
        [source],
        words,
        alignment,
        base.model_copy(update={"max_lines_per_cue": 2}),
        max_gap_seconds=0.8,
        max_cue_duration_seconds=5.0,
    )

    assert allowed == [source]
    assert allowed_alignment == alignment
    assert allowed_flags == []
    assert allowed_expansions == {}
    assert len(limited) > 1
    assert all(len(cue.lines) <= 2 for cue in limited)
    assert [flag.kind for flag in limited_flags] == ["sync_cue_line_limit_split"]
    assert limited_expansions[72][0] == 72
