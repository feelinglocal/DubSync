from __future__ import annotations

import pytest

from dubsync.models import Cue, QCFlag
from dubsync.qc_review import build_review
from dubsync.style_profile import StyleProfile
from dubsync.verify import lint_cues


def _cue(index: int, start_ms: int, end_ms: int, text: str = "Hallo.") -> Cue:
    return Cue(index=index, start_ms=start_ms, end_ms=end_ms, lines=[text])


def _timing_hold(cue: Cue) -> QCFlag:
    return QCFlag(
        kind="missing_audio_timing_held", cue_ids=[cue.index], severity="error",
        message="No trustworthy local speech evidence was available for this source cue.",
        old_text=cue.text, start=cue.start_ms / 1000, end=cue.end_ms / 1000,
    )


@pytest.mark.parametrize(
    "fps,duration_ms", [(24, 459), (25, 460), (29.97, 467), (30, 467), (120, 492)],
)
def test_duration_lint_accepts_one_frame_of_minimum_duration_slack(fps, duration_ms):
    profile = StyleProfile(fps=fps, min_cue_dur=0.5)
    cue = _cue(1, 1000, 1000 + duration_ms)

    assert not any(issue.kind == "min_duration" for issue in lint_cues([cue], profile))
    assert cue.start_ms == 1000
    assert cue.end_ms == 1000 + duration_ms


@pytest.mark.parametrize(
    "fps,duration_ms", [(24, 458), (25, 459), (29.97, 466), (30, 466), (120, 491)],
)
def test_duration_lint_preserves_real_shortfalls_beyond_one_frame(fps, duration_ms):
    profile = StyleProfile(fps=fps, min_cue_dur=0.5)
    cue = _cue(1, 1000, 1000 + duration_ms)

    issue, = [issue for issue in lint_cues([cue], profile) if issue.kind == "min_duration"]
    assert issue.cue_id == 1
    assert issue.severity == "warning"


@pytest.mark.parametrize("end_ms,kind", [(1000, "zero_duration"), (999, "negative_duration")])
def test_duration_lint_keeps_invalid_intervals_as_errors(end_ms, kind):
    issues = lint_cues([_cue(1, 1000, end_ms)], StyleProfile())

    assert any(issue.kind == kind and issue.severity == "error" for issue in issues)


@pytest.mark.parametrize("first_text", ["Hallo.", "♪Hallo♪"])
def test_duplicate_source_hold_is_attached_to_each_separate_affected_block(first_text):
    cues = [_cue(10, 1000, 2000, first_text), _cue(20, 3000, 4000), _cue(30, 5000, 6000)]
    flags = [
        _timing_hold(cues[0]), _timing_hold(cues[2]),
        QCFlag(kind="missing_audio_source_cue_held", cue_ids=[10, 30], severity="error",
               message="Source-backed text was not sent to adjudication."),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    last, = [item for item in review.review if item.cue_ids == [30]]
    assert last.raw_flags == [1, 2]
    assert "missing_audio_source_cue_held" in last.reasons
    assert last.severity == "error"
    assert last.srt_numbers == [3]
    if first_text.startswith("♪"):
        song, = [note for note in review.notes if note.kind == "song_lyrics_without_voice"]
        assert song.raw_flags == [0, 2]
        assert len(review.review) == 1
    else:
        first, = [item for item in review.review if item.cue_ids == [10]]
        assert first.raw_flags == [0, 2]
        assert len(review.review) == 2
    assert [(flag.kind, flag.severity) for flag in flags] == [
        ("missing_audio_timing_held", "error"), ("missing_audio_timing_held", "error"),
        ("missing_audio_source_cue_held", "error"),
    ]


def test_mixed_source_hold_keeps_unconfirmed_cue_reviewable_and_folds_the_duplicate():
    cues = [_cue(10, 1000, 2000), _cue(20, 3000, 4000), _cue(30, 5000, 6000)]
    flags = [
        _timing_hold(cues[0]),
        QCFlag(kind="missing_audio_source_cue_held", cue_ids=[10, 30], severity="error",
               message="Source-backed text was not sent to adjudication."),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    assert len(review.review) == 2
    first, = [item for item in review.review if item.cue_ids == [10]]
    unconfirmed, = [item for item in review.review if item.cue_ids == [30]]
    assert first.kind == "missing_audio_timing_held"
    assert first.raw_flags == [0, 1]
    assert "missing_audio_source_cue_held" in first.reasons
    assert unconfirmed.kind == "missing_audio_source_cue_held"
    assert unconfirmed.raw_flags == [1]
    assert not review.diagnostics


def test_word_clamp_diagnostic_summarizes_all_corrections_and_retains_raw_evidence():
    flags = [
        QCFlag(kind="asr_word_clamped", message="ASR word endpoint was clamped.",
               old_text="Hallo 1.000 --> 4.000", new_text="Hallo 1.050 --> 2.150",
               start=1.05, end=2.15),
        QCFlag(kind="asr_word_clamped", message="ASR word endpoint was clamped.",
               old_text="Welt 4.050 --> 8.000", new_text="Welt 4.100 --> 4.800",
               start=4.1, end=4.8, severity="error"),
    ]
    original_flags = [flag.model_dump() for flag in flags]

    review = build_review(flags, [], [_cue(1, 1000, 5000)])

    diagnostic, = review.diagnostics
    assert diagnostic.kind == "asr_word_clamped"
    assert diagnostic.count == 2
    assert diagnostic.raw_flags == [0, 1]
    assert diagnostic.severity == "error"
    assert "2 ASR word timing corrections" in diagnostic.message
    assert "00:00:01,050" in diagnostic.message
    assert "00:00:04,800" in diagnostic.message
    assert "50 ms" in diagnostic.message
    assert "3200 ms" in diagnostic.message
    assert review.review == []
    assert [flag.model_dump() for flag in flags] == original_flags
