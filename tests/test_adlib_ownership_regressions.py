from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.overlap import apply_overlap_policy
from dubsync.pipeline import _adlib_cue_ids_by_case, _alignment_with_decision_words
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile


@pytest.mark.parametrize("speaker", ["actor", None])
@pytest.mark.parametrize("min_duration", [0.5, 1.0])
def test_episode_11_eu_insertion_keeps_earlier_audio_position(speaker, min_duration):
    # Frozen source/QC windows from test fix/11.srt and dubsync.qc.json.
    # The phrase-level word below uses the emitted Colho cue window; this
    # reproduces rebuilding order without pretending to recover its ASR tokens.
    source = [Cue(index=536, start_ms=1490070, end_ms=1491880, lines=["Colho, sim."])]
    words = [
        Word(text="Eu", start=1490.19, end=1490.55, speaker_id=speaker),
        Word(text="Colho, sim.", start=1490.866, end=1491.8, speaker_id=speaker),
    ]
    span = DivergenceSpan(
        case_id="episode-11-eu",
        cue_ids=[],
        srt_text="",
        asr_text="Eu",
        start=1490.19,
        end=1490.55,
        asr_word_indices=[0],
    )
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="Eu",
        confidence=0.95,
        speaker=speaker,
        reason="The speaker says 'Eu colho, sim', uttering 'Eu' before 'colho'.",
    )
    adlib_ids = {span.case_id: 951}
    profile = StyleProfile(tail_ms=0, min_cue_dur=min_duration)
    changed, _ = apply_adjudication_decisions(source, [span], [decision], profile, adlib_ids)
    # Source onset precedes the ad-lib, even though its audio onset follows it.
    assert [cue.index for cue in changed] == [536, 951]
    alignment = _alignment_with_decision_words(
        AlignmentResult(cue_word_indices={536: [1]}), [decision], [span], adlib_ids
    )

    rebuilt, _ = rebuild_cues(changed, words, alignment, profile)
    output, _ = finalize_cues_for_output(rebuilt, profile)

    assert [cue.index for cue in output] == [951, 536]
    assert [cue.plain_text for cue in output] == ["Eu", "Colho, sim."]
    assert output[0].start_ms == profile.snap_floor(1490190)
    assert output[0].end_ms <= output[1].start_ms
    assert output[1].start_ms == profile.snap_floor(1490866)
    assert output[1].end_ms == max(
        profile.snap_ceil(1491800),
        profile.snap_ceil(output[1].start_ms + min_duration * 1000),
    )
    assert alignment.cue_word_indices == {536: [1], 951: [0]}


@pytest.mark.parametrize("anchor_speaker", ["A", None])
def test_another_actor_between_same_cue_anchors_keeps_separate_word_ownership(anchor_speaker):
    source = [Cue(index=1, start_ms=1000, end_ms=2400, lines=["Alpha omega."], speaker_id="A")]
    words = [
        Word(text="Alpha", start=1.0, end=1.3, speaker_id=anchor_speaker),
        Word(text="Yes", start=1.35, end=1.55, speaker_id="B"),
        Word(text="omega", start=1.6, end=2.2, speaker_id=anchor_speaker),
    ]
    alignment = align_cues_to_words(source, words)
    span = alignment.divergence_spans[0]
    assert span.cue_ids == []
    assert span.left_anchor_cue_id == span.right_anchor_cue_id == 1
    assert span.speaker_ids == ["B"]
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="Yes",
        confidence=0.99,
        speaker="B",
        reason="A second actor interjects between the first actor's words.",
    )
    profile = StyleProfile(tail_ms=0, min_cue_dur=0.1)

    adlib_ids, _ = _adlib_cue_ids_by_case(source, [span], [decision], [])
    changed, _ = apply_adjudication_decisions(source, [span], [decision], profile, adlib_ids)
    updated = _alignment_with_decision_words(alignment, [decision], [span], adlib_ids)
    rebuilt, _ = rebuild_cues(changed, words, updated, profile)

    assert adlib_ids == {span.case_id: 2}
    assert [cue.plain_text for cue in rebuilt] == ["Alpha omega.", "Yes"]
    assert updated.cue_word_indices == {1: [0, 2], 2: [1]}
    assert rebuilt[0].speaker_id == anchor_speaker
    assert rebuilt[1].speaker_id == "B"
    assert rebuilt[1].start_ms < rebuilt[0].end_ms


def test_mixed_actor_insertion_is_left_for_speaker_segmentation():
    source = [Cue(index=1, start_ms=1000, end_ms=2400, lines=["Alpha omega."], speaker_id="A")]
    words = [
        Word(text="Alpha", start=1.0, end=1.3, speaker_id="A"),
        Word(text="Yes", start=1.35, end=1.55, speaker_id="A"),
        Word(text="No", start=1.56, end=1.8, speaker_id="B"),
        Word(text="omega", start=1.85, end=2.2, speaker_id="A"),
    ]
    alignment = align_cues_to_words(source, words)
    span = alignment.divergence_spans[0]
    assert span.speaker_ids == ["A", "B"]
    decision = AdjudicationDecision(
        case_id=span.case_id,
        verdict="use_audio",
        final_text="Yes No",
        confidence=0.99,
        reason="Two actors contribute distinct words inside the anchor interval.",
    )

    adlib_ids, _ = _adlib_cue_ids_by_case(source, [span], [decision], [])
    updated = _alignment_with_decision_words(alignment, [decision], [span], adlib_ids)

    assert adlib_ids == {span.case_id: 2}
    assert updated.cue_word_indices == {1: [0, 3], 2: [1, 2]}


@pytest.mark.parametrize("min_duration,tail_ms", [(0.5, 40), (0.1, 500)])
@pytest.mark.parametrize("allow_zero_gap", [True, False])
def test_sequential_actors_are_not_dash_merged_by_display_padding(
    min_duration, tail_ms, allow_zero_gap
):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1166, lines=["Hã?"], speaker_id="A"),
        Cue(index=2, start_ms=1200, end_ms=1666, lines=["Uau!"], speaker_id="B"),
    ]
    words = [
        Word(text="Hã?", start=1.0, end=1.166, speaker_id="A"),
        Word(text="Uau!", start=1.2, end=1.666, speaker_id="B"),
    ]
    profile = StyleProfile(
        min_cue_dur=min_duration,
        tail_ms=tail_ms,
        allow_zero_gap=allow_zero_gap,
        overlap_policy="dash",
    )

    rebuilt, _ = rebuild_cues(
        cues, words, AlignmentResult(cue_word_indices={1: [0], 2: [1]}), profile
    )
    output, flags = apply_overlap_policy(rebuilt, policy="dash")

    assert [cue.plain_text for cue in output] == ["Hã?", "Uau!"]
    assert rebuilt[0].end_ms >= profile.snap_ceil(words[0].end * 1000)
    assert rebuilt[0].end_ms <= rebuilt[1].start_ms
    if not allow_zero_gap:
        assert rebuilt[0].end_ms < rebuilt[1].start_ms
    assert rebuilt[1].start_ms == 1200
    assert not any(flag.kind == "overlap_dash_merge" for flag in flags)


def test_real_actor_overlap_survives_display_padding_cap_and_keeps_two_turns():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1300, lines=["Hã?"], speaker_id="A"),
        Cue(index=2, start_ms=1200, end_ms=1666, lines=["Uau!"], speaker_id="B"),
    ]
    words = [
        Word(text="Hã?", start=1.0, end=1.3, speaker_id="A"),
        Word(text="Uau!", start=1.2, end=1.666, speaker_id="B"),
    ]
    profile = StyleProfile(min_cue_dur=1.0, tail_ms=200, overlap_policy="dash")

    rebuilt, _ = rebuild_cues(
        cues, words, AlignmentResult(cue_word_indices={1: [0], 2: [1]}), profile
    )
    output, flags = apply_overlap_policy(rebuilt, policy="dash")

    assert rebuilt[0].end_ms == profile.snap_ceil(words[0].end * 1000)
    assert rebuilt[0].end_ms > rebuilt[1].start_ms
    assert len(output) == 1
    assert output[0].lines == ["- Hã?", "- Uau!"]
    assert any(flag.kind == "overlap_dash_merge" for flag in flags)


def test_same_actor_monotonic_adjustment_cannot_pad_into_the_next_actor():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1300, lines=["First"], speaker_id="A"),
        Cue(index=2, start_ms=1200, end_ms=1400, lines=["Second"], speaker_id="A"),
        Cue(index=3, start_ms=1600, end_ms=2200, lines=["Third"], speaker_id="B"),
    ]
    words = [
        Word(text="First", start=1.0, end=1.3, speaker_id="A"),
        Word(text="Second", start=1.2, end=1.4, speaker_id="A"),
        Word(text="Third", start=1.6, end=2.2, speaker_id="B"),
    ]
    profile = StyleProfile(min_cue_dur=0.5, tail_ms=0, overlap_policy="dash")

    rebuilt, _ = rebuild_cues(
        cues, words, AlignmentResult(cue_word_indices={1: [0], 2: [1], 3: [2]}), profile
    )
    output, flags = apply_overlap_policy(rebuilt, policy="dash")

    assert rebuilt[1].start_ms == rebuilt[0].end_ms
    assert rebuilt[1].end_ms >= profile.snap_ceil(words[1].end * 1000)
    assert rebuilt[1].end_ms <= rebuilt[2].start_ms
    assert [cue.plain_text for cue in output] == ["First", "Second", "Third"]
    assert not any(flag.kind == "overlap_dash_merge" for flag in flags)


def test_held_overlap_keeps_prior_same_actor_boundary_for_following_cues():
    cues = [
        Cue(index=1, start_ms=1000, end_ms=2000, lines=["First"], speaker_id="A"),
        Cue(index=2, start_ms=1200, end_ms=1300, lines=["Second"], speaker_id="A"),
        Cue(index=3, start_ms=1333, end_ms=1500, lines=["Third"], speaker_id="B"),
        Cue(index=4, start_ms=1500, end_ms=2000, lines=["Fourth"], speaker_id="A"),
    ]
    words = [
        Word(text="First", start=1.0, end=2.0, speaker_id="A"),
        Word(text="Second", start=1.2, end=1.3, speaker_id="A"),
        Word(text="Third", start=1.333, end=1.5, speaker_id="B"),
        Word(text="Fourth", start=1.5, end=2.0, speaker_id="A"),
    ]
    profile = StyleProfile(min_cue_dur=0.5, tail_ms=0)

    rebuilt, _ = rebuild_cues(
        cues, words,
        AlignmentResult(cue_word_indices={1: [0], 2: [1], 3: [2], 4: [3]}),
        profile,
    )

    # The second cue cannot be shifted past the first without crossing B.
    assert rebuilt[1].start_ms == 1200
    assert rebuilt[1].end_ms <= rebuilt[2].start_ms
    assert rebuilt[3].start_ms == rebuilt[0].end_ms
    assert all(cue.end_ms > cue.start_ms for cue in rebuilt)
