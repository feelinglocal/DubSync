from __future__ import annotations

import pytest

from dubsync import aligner
from dubsync.models import Cue, Word
from dubsync.pipeline import _hold_incomplete_source_insertions


def _supported_omission(*, gap: float = 0.0, confidence: float | None = .99):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=3000, lines=["Hoje eu realmente preciso ir embora."]),
        Cue(index=2, start_ms=3500, end_ms=4500, lines=["Tudo certo."]),
    ]
    times = [(1.05, 1.30), (1.30, 1.50), (1.50 + gap, 1.80), (1.80, 2.00),
             (2.00, 2.35), (2.35, 2.70), (3.60, 3.90), (3.90, 4.20)]
    words = [Word(text=text, start=start, end=end, confidence=confidence, speaker_id="A")
             for text, (start, end) in zip(
                 "Hoje eu preciso ir embora agora Tudo certo".split(), times, strict=True)]
    return cues, words


@pytest.mark.parametrize("confidence", [.99, None], ids=["scored-asr", "mai-unscored"])
def test_touching_word_boundaries_do_not_hide_well_supported_internal_omission(confidence):
    cues, words = _supported_omission(confidence=confidence)
    original_cues = [cue.model_dump() for cue in cues]
    original_words = [word.model_dump() for word in words]

    alignment = aligner.align_cues_to_words(cues, words)
    provider, held, flags = _hold_incomplete_source_insertions(
        alignment.divergence_spans, {}, source_cue_count=len(cues),
        alignment_unresolved=alignment.diagnostics.unresolved,
        missing_audio_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
    )

    assert alignment.diagnostics.missing_audio_cue_ids == []
    assert alignment.diagnostics.unresolved is False
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4], 2: [6, 7]}
    assert [(span.srt_text, span.asr_text) for span in provider] == [
        ("realmente", ""), ("", "agora"),
    ]
    omission = provider[0]
    assert omission.start == words[0].start
    assert omission.end == words[3].end
    assert omission.confidence == 0.0
    assert omission.asr_word_indices == []
    assert held == flags == []
    assert [cue.model_dump() for cue in cues] == original_cues
    assert [word.model_dump() for word in words] == original_words


def test_positive_omission_window_retains_its_existing_word_boundaries():
    cues, words = _supported_omission(gap=.001)

    alignment = aligner.align_cues_to_words(cues, words)

    assert alignment.diagnostics.missing_audio_cue_ids == []
    omission = alignment.divergence_spans[0]
    assert (omission.start, omission.end) == (1.5, 1.501)


@pytest.mark.parametrize("defect", [
    "low-confidence", "overlong-word", "overlap",
    "speaker-change", "remote-flank", "outer-collapsed-1-ms", "outer-collapsed-20-ms",
])
def test_unreliable_flanking_words_do_not_reclassify_missing_audio(defect):
    cues, words = _supported_omission()
    if defect.startswith("outer-collapsed-"):
        # The outer anchor 'Hoje' collapses onto the start of 'eu' (F32).
        words[0] = words[0].model_copy(update={"start": 1.299 if defect.endswith("1-ms") else 1.28})
    elif defect == "low-confidence":
        words[1] = words[1].model_copy(update={"confidence": .4})
    elif defect == "overlong-word":
        words[0] = words[0].model_copy(update={"start": -1.0})
    elif defect == "overlap":
        words[1] = words[1].model_copy(update={"end": 1.6})
    elif defect == "speaker-change":
        words[2] = words[2].model_copy(update={"speaker_id": "B"})
    else:
        words[0] = words[0].model_copy(update={"start": .05, "end": .20})

    alignment = aligner.align_cues_to_words(cues, words)

    assert alignment.diagnostics.missing_audio_cue_ids == [1]
    assert alignment.divergence_spans[0].end <= alignment.divergence_spans[0].start


@pytest.mark.parametrize("internal_start", [1.499, 1.48], ids=["1-ms-internal-word", "20-ms-internal-word"])
def test_short_internal_word_uses_reliable_outer_anchors_for_the_whole_clause_question(internal_start):
    # The short word cannot establish a local omission window. Guard 8 may
    # still ask about the whole mostly matched phrase using its reliable outer
    # words; opening that hearing question does not repair any word timestamp.
    cues, words = _supported_omission()
    words[1] = words[1].model_copy(update={"start": internal_start})
    before = [word.model_dump() for word in words]
    alignment = aligner.align_cues_to_words(cues, words)
    assert alignment.diagnostics.missing_audio_cue_ids == []
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4], 2: [6, 7]}
    omission = next(span for span in alignment.divergence_spans if span.srt_text == "realmente")
    assert (omission.start, omission.end) == (1.05, 2.35)
    assert omission.asr_word_indices == []
    assert [word.model_dump() for word in words] == before


@pytest.mark.parametrize("flank,update,window", [
    ("left", {"end": 1.301}, (1.301, 1.5)),
    ("left", {"end": 1.32}, (1.32, 1.5)),
    ("right", {"start": 1.799}, (1.5, 1.799)),
    ("right", {"start": 1.78}, (1.5, 1.78)),
], ids=["left-1-ms", "left-20-ms", "right-1-ms", "right-20-ms"])
def test_collapsed_flank_that_leaves_a_gap_bounds_only_the_real_gap(flank, update, window):
    # F32: a 1-20 ms flank collapsed away from the omitted word leaves a real gap.
    # That gap is an ordinary positive omission window (as for any other flank
    # defect); the collapse never widens it into the outer anchors, never lends
    # the window ASR words or confidence and never rewrites a timestamp. Only a
    # flank collapsed onto the omission point (above) cannot bound a window.
    cues, words = _supported_omission()
    index = 1 if flank == "left" else 2
    words[index] = words[index].model_copy(update=update)
    before = [word.model_dump() for word in words]

    alignment = aligner.align_cues_to_words(cues, words)

    assert alignment.diagnostics.missing_audio_cue_ids == []
    assert alignment.cue_word_indices == {1: [0, 1, 2, 3, 4], 2: [6, 7]}
    omission = next(span for span in alignment.divergence_spans if span.srt_text == "realmente")
    assert (omission.start, omission.end) == window
    assert omission.asr_word_indices == [] and omission.confidence == 0.0
    assert [word.model_dump() for word in words] == before


@pytest.mark.parametrize("text", [
    "really Hoje eu preciso ir embora",
    "Hoje eu preciso ir embora really",
])
def test_missing_source_prefix_or_tail_of_a_well_matched_cue_is_reviewed_in_its_speech(text):
    # Five of six source words were heard as compact, trustworthy speech. The
    # unspoken edge word becomes a reviewable omission inside that speech
    # instead of locking the whole cue to its source timing.
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=[text])
    _, words = _supported_omission()

    alignment = aligner.align_cues_to_words([cue], words[:5])

    assert alignment.diagnostics.missing_audio_cue_ids == []
    omission, = alignment.divergence_spans
    assert (omission.srt_text, omission.asr_text) == ("really", "")
    assert (omission.start, omission.end) == (words[0].start, words[4].end)


@pytest.mark.parametrize("text", [
    "really truly Hoje eu preciso",
    "Hoje eu preciso really truly",
])
def test_missing_source_prefix_or_tail_of_a_sparse_cue_keeps_full_cue_protection(text):
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=[text])
    _, words = _supported_omission()

    alignment = aligner.align_cues_to_words([cue], words[:3])

    assert alignment.diagnostics.missing_audio_cue_ids == [1]


def test_sparse_partial_cue_stays_locked_despite_touching_internal_anchors():
    cues = [Cue(index=1, start_ms=0, end_ms=2000, lines=["before missing words after"])]
    words = [Word(text="before", start=.7, end=1.0), Word(text="after", start=1.0, end=1.3)]

    alignment = aligner.align_cues_to_words(cues, words)

    assert alignment.diagnostics.missing_audio_cue_ids == [1]


def test_mixed_screen_text_cue_does_not_gain_internal_omission_exception():
    cues, words = _supported_omission()
    cues[0] = cues[0].with_lines(["[Apartment]", *cues[0].lines])

    alignment = aligner.align_cues_to_words(cues, words)

    assert alignment.diagnostics.excluded_screen_text_cue_ids == [1]
    assert alignment.diagnostics.missing_audio_cue_ids == [1]


def test_global_unresolved_alignment_still_holds_every_source_span(monkeypatch):
    cues, words = _supported_omission()
    monkeypatch.setattr(aligner, "ALIGNMENT_CELL_BUDGET", 1)

    alignment = aligner.align_cues_to_words(cues, words)
    provider, held, flags = _hold_incomplete_source_insertions(
        alignment.divergence_spans, {}, alignment_unresolved=alignment.diagnostics.unresolved,
        missing_audio_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
    )

    assert alignment.diagnostics.unresolved is True
    assert provider == []
    assert [decision.verdict for decision in held] == ["keep_srt"]
    assert [flag.kind for flag in flags] == ["unresolved_alignment_adjudication_held"]
