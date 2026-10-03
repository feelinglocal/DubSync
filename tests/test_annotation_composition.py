"""Screen-caption composition preserves timed speech and continuous coverage."""

import json

import pytest

from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue
from dubsync.qc_review import KIND_REGISTRY, build_review
from dubsync.reports import write_change_log
from dubsync.style_profile import StyleProfile


def cue(index, start, end, lines, **metadata):
    return Cue(index=index, start_ms=start, end_ms=end, lines=lines, **metadata)


def pairs(cues):
    return {
        frozenset((left.index, right.index))
        for i, left in enumerate(cues)
        for right in cues[i + 1:]
        if left.start_ms < right.end_ms and right.start_ms < left.end_ms
    }


def test_nonoverlap_and_touching_intervals_are_identity():
    original = [cue(4, 0, 1000, ["[Luan Nian]"]), cue(5, 1000, 2000, ["Vem cá."])]
    words = {5: [0, 1]}
    result = compose_bracketed_annotations(original, words)

    assert result.cues == original
    assert all(a is b for a, b in zip(result.cues, original))
    assert result.cue_word_indices == words
    assert result.tracks == result.cue_annotations == {}
    result.cue_word_indices[5].append(9)
    assert words == {5: [0, 1]}


def test_fresh_luan_preserves_speech_identity_and_complete_caption_coverage():
    annotation = cue(170, 538660, 539320, ["[Luan Nian]"])
    spoken = cue(171, 539234, 540234, ["Você já foi embora?"], speaker_id="actor", character="Luan", prompt_scene_id=2)
    original = [annotation, spoken]
    before = [c.model_dump() for c in original]
    words = {170: [], 171: [90, 91, 92, 93]}
    result = compose_bracketed_annotations(original, words)

    assert [(c.index, c.start_ms, c.end_ms, c.lines) for c in result.cues] == [
        (170, 538660, 539234, ["[Luan Nian]"]),
        (171, 539234, 540234, ["Você já foi embora?", "[Luan Nian]"]),
    ]
    assert result.cues[1].model_dump(exclude={"lines"}) == spoken.model_dump(exclude={"lines"})
    assert result.cues[1].prompt_scene_id == 2
    assert result.cue_word_indices[171] == [90, 91, 92, 93]
    assert result.cue_annotations == {170: [170], 171: [170]}
    assert result.tracks[170]["original_start_ms"] == 538660
    assert result.tracks[170]["original_end_ms"] == 539320
    assert result.tracks[170]["lines"] == ["[Luan Nian]"]
    assert result.tracks[170]["display_cue_ids"] == [170, 171]
    assert result.tracks[170]["display_intervals"] == [[538660, 540234]]
    assert result.tracks[170]["early_extension_ms"] == 0
    assert result.tracks[170]["late_extension_ms"] == 914
    assert [c.model_dump() for c in original] == before
    assert words == {170: [], 171: [90, 91, 92, 93]}
    assert not pairs(result.cues)


def test_annotation_prefix_gap_and_suffix_with_two_spoken_cues():
    original = [cue(10, 0, 5000, ["[Email]", "[Send to me]"]),
                cue(11, 1000, 2000, ["First authored line.", "Second authored line."]),
                cue(12, 3000, 4000, ["Other actor."])]
    words = {11: [4, 5], 12: [6], 101: []}
    result = compose_bracketed_annotations(original, words)

    assert [(c.start_ms, c.end_ms) for c in result.cues] == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
    by_id = {c.index: c for c in result.cues}
    assert by_id[11].lines == original[1].lines + original[0].lines
    assert by_id[12].lines == original[2].lines + original[0].lines
    assert result.cue_word_indices[11] == [4, 5]
    assert result.cue_word_indices[12] == [6]
    assert set(by_id) == {10, 11, 12, 102, 103}
    assert result.tracks[10]["display_intervals"] == [[0, 5000]]
    assert len(result.tracks[10]["display_cue_ids"]) == 5
    assert not pairs(result.cues)


def test_multiple_annotations_keep_all_authored_lines_in_order():
    original = [cue(1, 0, 40, ["[First]", "[Authored break]"]),
                cue(2, 10, 50, ["[Second]"]), cue(3, 20, 30, ["Speech."])]
    result = compose_bracketed_annotations(original, {3: [8]})

    assert [(c.start_ms, c.end_ms, c.lines) for c in result.cues] == [
        (0, 10, ["[First]", "[Authored break]"]),
        (10, 20, ["[First]", "[Authored break]", "[Second]"]),
        (20, 30, ["Speech.", "[First]", "[Authored break]", "[Second]"]),
        (30, 40, ["[First]", "[Authored break]", "[Second]"]),
        (40, 50, ["[Second]"]),
    ]
    assert result.tracks[1]["display_intervals"] == [[0, 40]]
    assert result.tracks[2]["display_intervals"] == [[10, 50]]
    assert len({c.index for c in result.cues}) == len(result.cues)
    assert not pairs(result.cues)


def test_connected_annotation_neighbor_composes_without_touching_isolated_caption():
    isolated = cue(8, 100, 200, ["[Isolated]"])
    original = [cue(1, 0, 40, ["[First]"]), cue(2, 10, 50, ["[Second]"]), cue(3, 0, 5, ["Speech."]), isolated]
    result = compose_bracketed_annotations(original)

    assert result.cues[-1] is isolated
    assert set(result.tracks) == {1, 2}
    assert not pairs(result.cues)


def test_simultaneous_speech_overlap_and_ownership_remain_exact():
    original = [cue(1, 0, 5000, ["[Title]"]),
                cue(2, 1000, 3000, ["Actor one."], speaker_id="one"),
                cue(3, 2000, 4000, ["Actor two."], speaker_id="two")]
    result = compose_bracketed_annotations(original, {2: [0], 3: [1]})

    assert pairs(result.cues) == {frozenset((2, 3))}
    for spoken in original[1:]:
        updated = next(c for c in result.cues if c.index == spoken.index)
        assert updated.model_dump(exclude={"lines"}) == spoken.model_dump(exclude={"lines"})
        assert updated.lines == spoken.lines + ["[Title]"]
    assert result.cue_word_indices[2] == [0]
    assert result.cue_word_indices[3] == [1]
    assert result.tracks[1]["display_intervals"] == [[0, 5000]]


@pytest.mark.parametrize("other_lines", [["♪Leve-me para fugir♪"], ["♫Music♫"], ["[♪Song♪]"]])
def test_lyrics_are_never_pure_annotation_tracks(other_lines):
    original = [cue(1, 0, 1000, other_lines), cue(2, 300, 800, ["Uhum."])]
    result = compose_bracketed_annotations(original, {1: [0], 2: [1]})

    assert result.cues == original
    assert result.tracks == {}
    assert result.cue_word_indices == {1: [0], 2: [1]}
    assert pairs(result.cues) == pairs(original)


def test_mixed_caption_and_dialogue_remains_spoken_and_is_not_duplicated():
    original = [cue(1, 1930550, 1931510, ["[Então procure outro emprego rápido]"]),
                cue(2, 1929120, 1930800, ["E aconselhou ela a pedir demissão?"]),
                cue(3, 1931440, 1932480, ["[Esta empresa não é para você]", "A Lingmei não é para você."])]
    result = compose_bracketed_annotations(original, {2: [0, 1], 3: [2, 3]})

    by_id = {c.index: c for c in result.cues}
    assert by_id[2].lines == original[1].lines + original[0].lines
    assert by_id[3].lines == original[2].lines + original[0].lines
    assert set(result.tracks) == {1}
    assert [(c.start_ms, c.end_ms) for c in result.cues] == [(1929120, 1930800), (1930800, 1931440), (1931440, 1932480)]
    assert result.cue_word_indices[3] == [2, 3]
    assert sum("A Lingmei não é para você." in c.lines for c in result.cues) == 1
    assert not pairs(result.cues)


def test_nonempty_caption_ownership_is_preserved_without_guessing_reassignment():
    original = [cue(1, 0, 1000, ["[Unexpected owner]"]), cue(2, 300, 800, ["Speech."])]
    result = compose_bracketed_annotations(original, {1: [8], 2: [9]})

    assert result.cues == original
    assert result.cue_word_indices == {1: [8], 2: [9]}
    assert result.tracks == {}


def test_composition_is_deterministic_and_cues_and_ownership_are_idempotent():
    original = [cue(3, 1000, 2000, ["Speech."]), cue(1, 0, 3000, ["[Title]"])]
    first = compose_bracketed_annotations(original, {3: [5]})
    again = compose_bracketed_annotations(list(reversed(original)), {3: [5]})
    second = compose_bracketed_annotations(first.cues, first.cue_word_indices)

    assert first == again
    assert second.cues == first.cues
    assert second.cue_word_indices == first.cue_word_indices
    assert len({c.index for c in first.cues}) == len(first.cues)


def test_empty_input_is_identity():
    result = compose_bracketed_annotations([])
    assert result.cues == []
    assert result.cue_word_indices == result.tracks == result.cue_annotations == {}


def test_continuous_short_caption_fragments_have_json_ready_track_provenance():
    original = [cue(478, 1377580, 1378900, ["[Announcement]"]),
                cue(479, 1378200, 1378700, ["Hum,"])]
    result = compose_bracketed_annotations(original, {479: [4]})

    assert [(c.start_ms, c.end_ms) for c in result.cues] == [
        (1377580, 1378200), (1378200, 1378700), (1378700, 1378900),
    ]
    assert result.tracks[478]["display_intervals"] == [[1377580, 1378900]]
    assert result.tracks[478]["early_extension_ms"] == result.tracks[478]["late_extension_ms"] == 0
    artifact = result.artifact()
    assert json.loads(json.dumps(artifact))["tracks"]["478"]["display_intervals"] == [[1377580, 1378900]]
    artifact["tracks"][478]["lines"].append("Changed")
    assert result.tracks[478]["lines"] == ["[Announcement]"]


def test_caption_contained_inside_spoken_cue_needs_no_annotation_only_piece():
    original = [cue(1, 400, 600, ["[Title]"]), cue(2, 0, 1000, ["A whole sentence."], speaker_id="actor")]
    result = compose_bracketed_annotations(original, {1: [], 2: [0, 1]})

    assert len(result.cues) == 1
    assert result.cues[0].index == 2
    assert result.cues[0].lines == ["A whole sentence.", "[Title]"]
    assert result.cue_word_indices == {2: [0, 1]}
    assert result.tracks[1]["display_cue_ids"] == [2]
    assert result.tracks[1]["display_intervals"] == [[0, 1000]]
    assert result.tracks[1]["early_extension_ms"] == result.tracks[1]["late_extension_ms"] == 400


def test_caption_can_append_to_a_lyric_without_repeating_or_retiming_the_lyric():
    title = cue(1, 0, 1000, ["[Title]"])
    lyric = cue(2, 200, 800, ["♪Song♪"], speaker_id="singer")
    result = compose_bracketed_annotations([title, lyric], {2: [7]})

    updated = next(c for c in result.cues if c.index == lyric.index)
    assert updated.model_dump(exclude={"lines"}) == lyric.model_dump(exclude={"lines"})
    assert updated.lines == ["♪Song♪", "[Title]"]
    assert result.cue_word_indices[2] == [7]
    assert set(result.tracks) == {1}
    assert sum("♪Song♪" in c.lines for c in result.cues) == 1
    assert not pairs(result.cues)


# W4C-4: dash turns that fill the two-line display for a contained caption's
# whole time. The caption is never a third line and never silently lost.
_TURNS = ["- Você vem com a gente amanhã?", "- Não, fico aqui em casa."]
_SIGN = ["[PLACA: SAÍDA DE EMERGÊNCIA]"]


def _full_speech_case(*neighbors, media_end_ms=None, pre_annotation=True):
    speech = cue(1, 10000, 14000, _TURNS)
    caption = cue(2, 11000, 13000, _SIGN)
    source = sorted([speech, caption, *neighbors], key=lambda item: (item.start_ms, item.index))
    profile = StyleProfile(fps=25.0, max_chars_per_line=42, min_cue_dur=.5)
    composed = compose_bracketed_annotations(source, {item.index: [] for item in source}, words=[], profile=profile,
                                             **({} if media_end_ms is None else {"media_end_ms": media_end_ms}))
    # Never more than two lines, never an overlap, speech and neighbours exactly as they were.
    assert all(len(item.lines) <= 2 for item in composed.cues), [item.lines for item in composed.cues]
    assert all(left.end_ms <= right.start_ms for left, right in zip(composed.cues, composed.cues[1:]))
    assert [item for item in composed.cues if item.index != 2] == [item for item in source if item.index != 2]
    if media_end_ms is not None:
        assert all(item.end_ms <= media_end_ms for item in composed.cues)
    review = build_review(composed.flags, [], composed.cues, source_cues=source,
                          pre_annotation_cues=source if pre_annotation else None)
    return caption, composed, review


def _assert_moved(caption, composed, review, interval, side):
    moved, = [item for item in composed.cues if item.lines == _SIGN]
    assert (moved.index, moved.start_ms, moved.end_ms) == (2, *interval)
    # A moved caption is a display change, not a review item: QC says where it went.
    assert not [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
    assert not [item for item in review.review if "annotation_display_full" in item.reasons]
    flag, = [flag for flag in composed.flags if flag.kind == "annotation_line_limit_pagination"]
    assert (flag.severity, flag.cue_ids, flag.old_text, flag.new_text) == ("info", [2], caption.text, caption.text)
    assert f"just {side} that speech" in flag.message
    assert (flag.start, flag.end) == (interval[0] / 1000, interval[1] / 1000)
    change, = [item for item in review.changes if item.kind == "annotation_line_limit_pagination"]
    assert (change.change, change.cue_id) == ("timing", 2)
    assert change.old_timing == "00:00:11,000 --> 00:00:13,000"
    assert change.new_timing == f"00:00:{interval[0] // 1000:02d},{interval[0] % 1000:03d} --> "                                 f"00:00:{interval[1] // 1000:02d},{interval[1] % 1000:03d}"
    assert change.srt_number == [item.index for item in composed.cues].index(2) + 1
    assert f"just {side} that speech" in change.reason
    track = composed.tracks[2]
    assert track["display_cue_ids"] == [2] and track["display_intervals"] == [list(interval)]
    assert track["coverage_gaps_ms"] == [[11000, 13000]]


def test_caption_beside_full_speech_moves_to_the_free_interval_after_it_as_a_logged_display_change():
    caption, composed, review = _full_speech_case(cue(5, 9800, 9900, ["Antes."]), cue(6, 15000, 16000, ["Depois."]))

    _assert_moved(caption, composed, review, (14000, 15000), "after")
    assert composed.tracks[2]["late_extension_ms"] == 2000
    assert review.verdict == "clean"


def test_caption_beside_full_speech_moves_before_it_when_only_the_earlier_interval_is_free():
    caption, composed, review = _full_speech_case(cue(5, 5000, 6000, ["Antes."]), cue(6, 14000, 15000, ["Depois."]))

    _assert_moved(caption, composed, review, (8000, 10000), "before")
    assert composed.tracks[2]["early_extension_ms"] == 3000


@pytest.mark.parametrize("media_end_ms,interval", [(15500, (14000, 15500)), (20000, (14000, 16000))])
def test_caption_beside_the_last_full_speech_may_use_the_media_after_it(media_end_ms, interval):
    caption, composed, review = _full_speech_case(cue(5, 9800, 9900, ["Antes."]), media_end_ms=media_end_ms)

    _assert_moved(caption, composed, review, interval, "after")


@pytest.mark.parametrize("pre_annotation", [True, False])
@pytest.mark.parametrize("neighbors,media_end_ms", [
    # Displays touch the speech on both sides.
    ((cue(5, 8000, 10000, ["Antes."]), cue(6, 14000, 16000, ["Depois."])), None),
    # Less than the minimum display before it and before the end of the media.
    ((cue(5, 9800, 9900, ["Antes."]),), 14400),
    # Nothing follows and the media end is unknown: never past known timing.
    ((cue(5, 9800, 9900, ["Antes."]),), None),
])
def test_caption_with_no_free_interval_beside_full_speech_is_an_error_and_a_removed_change(
        tmp_path, neighbors, media_end_ms, pre_annotation):
    caption, composed, review = _full_speech_case(*neighbors, media_end_ms=media_end_ms,
                                                  pre_annotation=pre_annotation)

    assert not [item for item in composed.cues if any(line.startswith("[PLACA") for line in item.lines)]
    assert not [flag for flag in composed.flags if flag.kind == "annotation_line_limit_pagination"]
    flag, = [flag for flag in composed.flags if flag.kind == "annotation_display_full"]
    assert (flag.severity, flag.cue_ids, flag.old_text, flag.new_text) == ("error", [2], caption.text, None)
    assert caption.text in flag.message and "could not be displayed" in flag.message
    assert (flag.start, flag.end) == (11.0, 13.0)
    assert composed.tracks[2]["display_intervals"] == [] and composed.tracks[2]["display_cue_ids"] == []
    assert composed.tracks[2]["coverage_gaps_ms"] == [[11000, 13000]]
    spec = KIND_REGISTRY["annotation_display_full"]
    assert (spec.category, spec.severity) == ("review", "error")
    # Its own review item at the caption's time, right after the speech that fills the display.
    item, = [item for item in review.review if "annotation_display_full" in item.reasons]
    speech_number = [cue.index for cue in composed.cues].index(1) + 1
    assert (item.kind, item.severity, item.srt_numbers, item.after_srt_number) == (
        "annotation_display_full", "error", [], speech_number)
    assert (item.timecode, item.old_text, item.cue_ids) == ("00:00:11,000", caption.text, [2])
    assert caption.text in item.detail and "could not be displayed" in item.detail
    assert review.verdict == "attention" and review.counts["review_error_count"] == 1
    # The change log records the caption as a removed cue, once, with the reason.
    removed, = [change for change in review.changes if change.change == "removed"]
    assert (removed.kind, removed.cue_id, removed.old_text, removed.new_text) == (
        "annotation_display_full", 2, caption.text, None)
    assert (removed.srt_number, removed.after_srt_number, removed.start, removed.end) == (
        None, speech_number, 11.0, 13.0)
    assert "could not be displayed" in removed.reason
    assert composed.flags.index(flag) in removed.raw_flags
    write_change_log(tmp_path / "changes.diff.srt", [change.model_dump() for change in review.changes])
    diff = (tmp_path / "changes.diff.srt").read_text(encoding="utf-8")
    assert f"# removed after SRT #{speech_number} (cue 2)" in diff
    assert f"- {_SIGN[0]}" in diff
