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
    "low-confidence", "collapsed-word", "20-ms-word", "overlong-word", "overlap",
    "speaker-change", "remote-flank",
])
def test_unreliable_flanking_words_do_not_reclassify_missing_audio(defect):
    cues, words = _supported_omission()
    if defect == "low-confidence":
        words[1] = words[1].model_copy(update={"confidence": .4})
    elif defect == "collapsed-word":
        words[1] = words[1].model_copy(update={"start": 1.499})
    elif defect == "20-ms-word":
        words[1] = words[1].model_copy(update={"start": 1.48})
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


@pytest.mark.parametrize("text", [
    "really Hoje eu preciso ir embora",
    "Hoje eu preciso ir embora really",
])
def test_missing_source_prefix_or_tail_keeps_full_cue_protection(text):
    cue = Cue(index=1, start_ms=1000, end_ms=3000, lines=[text])
    _, words = _supported_omission()

    alignment = aligner.align_cues_to_words([cue], words[:5])

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
