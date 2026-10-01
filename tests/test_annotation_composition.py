"""Screen-caption composition preserves timed speech and continuous coverage."""

import json

import pytest

from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue


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
