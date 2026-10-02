from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dubsync.models import Cue, QCFlag, StyleIssue
from dubsync.qc_review import KIND_REGISTRY, STYLE_REGISTRY, build_review


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "dubsync"


def _cue(index: int, start_ms: int, end_ms: int, text: str) -> Cue:
    return Cue(index=index, start_ms=start_ms, end_ms=end_ms, lines=text.split("\n"))


def _covered(review) -> tuple[set[int], set[int]]:
    flags: set[int] = set()
    style: set[int] = set()
    for item in [*review.review, *review.changes, *review.notes, *review.diagnostics]:
        flags.update(item.raw_flags)
        style.update(item.raw_style)
    return flags, style


def _assert_every_raw_finding_is_mapped(review, flags, style_issues) -> None:
    covered_flags, covered_style = _covered(review)
    assert covered_flags == set(range(len(flags)))
    assert covered_style == set(range(len(style_issues)))


def _hold_cascade(cue: Cue) -> list[QCFlag]:
    window = {"start": cue.start_ms / 1000, "end": cue.end_ms / 1000, "old_text": cue.text}
    return [
        QCFlag(kind="missing_audio_timing_held", cue_ids=[cue.index], severity="error",
               message="No trustworthy local speech evidence was available for this source cue.", **window),
        QCFlag(kind="dropped_line_candidate", cue_ids=[cue.index], confidence=0.0,
               message="Unmatched source cue overlaps speech activity for only 0%; actor may have dropped this line.",
               **window),
        QCFlag(kind="cue_without_speech_activity", cue_ids=[cue.index], confidence=0.0,
               message="Cue overlaps speech activity for only 0% of its duration.", **window),
        QCFlag(kind="cue_on_silence", cue_ids=[cue.index], message="Cue audio is silent.", **window),
        QCFlag(kind="unmatched_cue", cue_ids=[cue.index], message="No ASR word timestamps matched this cue.",
               **window),
    ]


def test_song_lyric_holds_become_one_episode_note_without_review_items():
    lyrics = [
        _cue(1, 21_380, 23_940, "♪Seu riso, sua bondade♪"),
        _cue(2, 24_020, 25_820, "♪Sua crença♪"),
        _cue(9, 90_000, 92_000, "♪Só restam lágrimas♪"),
    ]
    dialogue = _cue(3, 30_000, 31_000, "Bom dia.")
    cues = [lyrics[0], lyrics[1], dialogue, lyrics[2]]
    flags = [flag for cue in lyrics for flag in _hold_cascade(cue)]
    flags.append(QCFlag(
        kind="missing_audio_source_cue_held", cue_ids=[1, 2], severity="error",
        message="Source-backed text was not sent to adjudication.",
    ))
    flags.append(QCFlag(
        kind="low_confidence_adjudication", cue_ids=[1, 2], confidence=0.0,
        message=(
            "Adjudication confidence is below the configured gate; source SRT was preserved. Proposed verdict: "
            "keep_srt. Reason: The source cue had no trustworthy local speech evidence."
        ),
    ))
    style = [StyleIssue(kind="frame_grid", cue_id=1, message="Cue timestamp is off the frame grid.")]

    review = build_review(flags, style, cues, source_cues=cues)

    assert review.review == []
    assert review.verdict == "clean"
    lyric_notes = [note for note in review.notes if note.kind == "song_lyrics_without_voice"]
    assert len(lyric_notes) == 1
    assert lyric_notes[0].count == 3
    assert lyric_notes[0].srt_numbers == [1, 2, 4]
    assert "2 passages" in lyric_notes[0].detail
    _assert_every_raw_finding_is_mapped(review, flags, style)


def test_dialogue_hold_run_is_one_error_item_with_delivered_numbers():
    cues = [
        _cue(10, 1_000, 2_000, "Olá."),
        _cue(11, 1_199_000, 1_201_000, "Sejam bem-vindos"),
        _cue(12, 1_201_500, 1_203_000, "à Lime China."),
        _cue(13, 1_204_000, 1_205_000, "Obrigado."),
        _cue(14, 1_300_000, 1_301_000, "Tchau."),
    ]
    flags = [flag for cue in cues[1:4] for flag in _hold_cascade(cue)]
    style = [StyleIssue(kind="frame_grid", cue_id=12, message="Cue timestamp is off the frame grid.")]

    review = build_review(flags, style, cues, source_cues=cues)

    assert len(review.review) == 1
    item = review.review[0]
    assert item.severity == "error"
    assert item.kind == "missing_audio_timing_held"
    assert item.srt_numbers == [2, 3, 4]
    assert item.cue_ids == [11, 12, 13]
    assert item.srt_label == "#2–#4"
    assert item.timecode == "00:19:59,000"
    assert "dropped_line_candidate" in item.reasons
    assert review.verdict == "attention"
    assert review.counts["review_cue_count"] == 3
    _assert_every_raw_finding_is_mapped(review, flags, style)


def test_one_overlap_reported_four_ways_is_one_warning_item():
    cues = [
        _cue(227, 683_000, 683_900, "Cuidado,"),
        _cue(228, 683_710, 685_000, "Vem cá."),
    ]
    window = {"start": 683.71, "end": 683.9}
    flags = [
        QCFlag(kind="overlap_stacked", cue_ids=[227, 228], message="Overlapping speaker cues require QC review.",
               **window),
        QCFlag(kind="output_overlap_unresolved", cue_ids=[227, 228], severity="error",
               message="Acoustically timed cues overlap.", **window),
        QCFlag(kind="output_overlap_preserved", cue_ids=[227, 228], message="Uncertain source timing.", **window),
    ]
    style = [StyleIssue(kind="overlap", cue_id=228, message="Cue overlaps the previous cue.")]

    review = build_review(flags, style, cues, source_cues=cues)

    assert len(review.review) == 1
    item = review.review[0]
    assert item.kind == "output_overlap_unresolved"
    assert item.severity == "warning"
    assert item.srt_numbers == [1, 2]
    assert review.verdict == "check"
    _assert_every_raw_finding_is_mapped(review, flags, style)


def test_one_frame_overlap_needs_review_but_a_stale_overlap_does_not():
    cues = [
        _cue(1, 0, 1_033, "Primeira."),
        _cue(2, 1_000, 2_000, "Segunda."),
        _cue(3, 3_000, 4_000, "Terceira."),
        _cue(4, 4_000, 5_000, "Quarta."),
    ]
    flags = [
        # The human-corrected episodes contain no overlaps at all, even of one frame.
        QCFlag(kind="output_overlap_unresolved", cue_ids=[1, 2], severity="error", message="overlap",
               start=1.0, end=1.033),
        # Emitted at rebuild time, before verify moved cue 3 apart from cue 4.
        QCFlag(kind="overlap_stacked", cue_ids=[3, 4], message="overlap", start=3.9, end=4.2),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    assert [(item.kind, item.srt_numbers) for item in review.review] == [("output_overlap_unresolved", [1, 2])]
    assert any(item.kind == "stale_overlap" for item in review.diagnostics)
    _assert_every_raw_finding_is_mapped(review, flags, [])


def test_style_lint_on_untouched_customer_cues_is_diagnostic_only():
    source = [_cue(1, 1_001, 2_002, "A long customer line that is wider than the profile allows")]
    cues = list(source)
    style = [
        StyleIssue(kind="frame_grid", cue_id=1, message="Cue timestamp is off the frame grid."),
        StyleIssue(kind="line_length", cue_id=1, message="Cue line exceeds profile length."),
    ]

    review = build_review([], style, cues, source_cues=source,
                          summary_metadata={"fps_detection_confident": True})

    assert review.review == []
    assert {item.kind for item in review.diagnostics} == {"style:frame_grid", "style:line_length"}
    _assert_every_raw_finding_is_mapped(review, [], style)


def test_style_lint_on_tool_timed_cue_with_confident_fps_needs_review():
    source = [_cue(1, 1_000, 2_000, "Hello.")]
    cues = [_cue(1, 1_210, 2_005, "Hello.")]
    style = [StyleIssue(kind="frame_grid", cue_id=1, message="Cue timestamp is off the frame grid.")]

    confident = build_review([], style, cues, source_cues=source, summary_metadata={"fps_detection_confident": True})
    fallback = build_review([], style, cues, source_cues=source, summary_metadata={"fps_detection_confident": False})

    assert [item.kind for item in confident.review] == ["style:frame_grid"]
    assert fallback.review == []


def test_text_changes_use_delivered_numbering_and_hide_route_tags():
    source = [
        _cue(30, 176_680, 177_360, "Sim,"),
        _cue(31, 180_000, 181_000, "Bom dia, Luke."),
    ]
    adlib = _cue(940, 178_000, 178_400, "Ei.")
    cues = [_cue(30, 176_680, 177_360, "Sim, já comecei."), adlib, source[1]]
    flags = [
        QCFlag(kind="text_changed", cue_ids=[30], confidence=1.0,
               message="Adjudication verdict use_audio: [hybrid:primary] The actor clearly says já comecei.",
               old_text="Sim,", new_text="Sim, já comecei.", start=176.9, end=177.2),
        QCFlag(kind="adlib_inserted", cue_ids=[940], confidence=1.0,
               message="Adjudication verdict use_audio: [hybrid:fallback] The actor says Ei.",
               new_text="Ei.", start=178.0, end=178.4),
        QCFlag(kind="unsourced_word_substitution", cue_ids=[30], old_text="conhece", new_text="comecei",
               message="Output spelling 'comecei' is absent from the source."),
    ]

    review = build_review(flags, [], cues, source_cues=source)

    assert review.review == []
    assert [(change.change, change.srt_number, change.cue_id) for change in review.changes] == [
        ("edited", 1, 30),
        ("added", 2, 940),
    ]
    edited = review.changes[0]
    assert (edited.old_text, edited.new_text) == ("Sim,", "Sim, já comecei.")
    assert edited.reason == "The actor clearly says já comecei."
    assert edited.route == "primary"
    assert edited.timecode == "00:02:56,680"
    assert all("[hybrid" not in (change.reason or "") for change in review.changes)
    assert review.counts["text_change_count"] == 2
    _assert_every_raw_finding_is_mapped(review, flags, [])


def test_unsourced_word_without_a_logged_change_stays_reviewable():
    cues = [_cue(1, 0, 1_000, "Tomem cuidado.")]
    flags = [QCFlag(kind="name_spelling_inconsistency", cue_ids=[1], old_text="tem", new_text="Tomem",
                    message="Output spelling 'Tomem' is absent from the source.")]

    review = build_review(flags, [], cues, source_cues=[_cue(1, 0, 1_000, "Tomem cuidado.")])

    assert [item.kind for item in review.review] == ["name_spelling_inconsistency"]


def test_inserted_adlib_shifts_review_numbering_to_the_delivered_srt():
    source = [_cue(33, 10_000, 11_000, "Primeiro."), _cue(34, 12_000, 13_000, "Segundo.")]
    cues = [source[0], _cue(940, 11_200, 11_600, "Ei."), _cue(34, 12_000, 13_000, "Segundo.")]
    flags = [QCFlag(kind="timing_evidence_held", cue_ids=[34], severity="error",
                    message="Sparse lexical timing evidence.", start=12.0, end=13.0)]

    review = build_review(flags, [], cues, source_cues=source)

    item, = review.review
    assert item.srt_numbers == [3]
    assert item.cue_ids == [34]
    assert item.timecode == "00:00:12,000"
    assert item.text == "Segundo."


def test_unknown_kinds_fail_closed_by_severity():
    cues = [_cue(1, 0, 1_000, "Hello.")]
    flags = [
        QCFlag(kind="brand_new_problem", cue_ids=[1], message="Something new.", severity="warning"),
        QCFlag(kind="brand_new_note", cue_ids=[], message="Informational.", severity="info"),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    assert [item.kind for item in review.review] == ["brand_new_problem"]
    assert [note.kind for note in review.notes] == ["brand_new_note"]


def test_synthetic_low_confidence_hold_is_absorbed_but_real_low_confidence_is_reviewed():
    cues = [_cue(1, 0, 1_000, "Olá."), _cue(2, 2_000, 3_000, "Até logo.")]
    flags = [
        *_hold_cascade(cues[0]),
        QCFlag(kind="low_confidence_adjudication", cue_ids=[1], confidence=0.0,
               message="Proposed verdict: keep_srt. Reason: The source cue had no trustworthy local speech evidence."),
        QCFlag(kind="low_confidence_adjudication", cue_ids=[2], confidence=0.55,
               message="Adjudication confidence is below the configured gate; source SRT was preserved. "
                       "Proposed verdict: use_audio."),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    assert [(item.kind, item.srt_numbers) for item in review.review] == [
        ("missing_audio_timing_held", [1]),
        ("low_confidence_adjudication", [2]),
    ]
    assert "low_confidence_adjudication" in review.review[0].reasons
    _assert_every_raw_finding_is_mapped(review, flags, [])


def test_minor_timing_refinements_are_one_note_and_large_ones_are_changes():
    source = [_cue(1, 1_000, 2_000, "Um."), _cue(2, 3_000, 4_000, "Dois.")]
    cues = [_cue(1, 1_000, 2_066, "Um."), _cue(2, 3_000, 4_900, "Dois.")]
    flags = [
        QCFlag(kind="timing_refined", cue_ids=[1], message="Cue boundary adjusted.",
               old_text="1.000 --> 2.000", new_text="1.000 --> 2.066", start=1.0, end=2.066),
        QCFlag(kind="timing_refined", cue_ids=[2], message="Cue boundary adjusted.",
               old_text="3.000 --> 4.000", new_text="3.000 --> 4.900", start=3.0, end=4.9),
    ]

    review = build_review(flags, [], cues, source_cues=source)

    assert review.review == []
    assert [(change.change, change.srt_number) for change in review.changes] == [("timing", 2)]
    assert review.changes[0].old_timing == "00:00:03,000 --> 00:00:04,000"
    assert [note.kind for note in review.notes] == ["minor_timing_adjustments"]
    _assert_every_raw_finding_is_mapped(review, flags, [])


@pytest.mark.parametrize("resolved_kind", [
    "missing_dialogue_audio_reconciled", "accepted_anchor_omission_reconciled",
])
@pytest.mark.parametrize("scenario", [
    "removed", "retained", "mixed", "insertion", "partial", "uncertain", "unresolved", "retimed",
])
def test_failed_hearing_is_diagnostic_only_after_complete_confirmed_omission(resolved_kind, scenario):
    source = [_cue(2, 26_500, 27_365, "いいかしら？"), _cue(3, 32_300, 32_630, "あなたは…")]
    cues = source if scenario == "retained" else source[:1]
    old = QCFlag(
        kind="low_confidence_adjudication", cue_ids=[3], confidence=0.0,
        message="Adjudication audio evidence is unclear or inaudible; source SRT was preserved.",
        old_text="あなたは", new_text="あなたは", start=27.440, end=27.479,
    )
    resolved = QCFlag(
        kind=resolved_kind, cue_ids=[3], confidence=1.0, severity="info",
        message="Complete anchored audio confirmed this source cue was omitted.",
        old_text="あなたは…", new_text="", start=27.365, end=32.155,
    )
    if scenario == "mixed":
        old.cue_ids = [2, 3]
    elif scenario == "insertion":
        old.cue_ids = []
    elif scenario == "partial":
        resolved.new_text = "あなた"
    elif scenario == "uncertain":
        resolved.confidence = 0.9
    elif scenario == "retimed":
        old.kind = "collapsed_singleton_timing_held"
    flags = [old] if scenario == "unresolved" else [old, resolved]
    before = [flag.model_dump() for flag in flags]

    review = build_review(flags, [], cues, source_cues=source)

    if scenario == "removed":
        assert review.review == []
        diagnostic, = [item for item in review.diagnostics if 0 in item.raw_flags]
        assert diagnostic.kind == "resolved_omission_adjudication"
        assert diagnostic.cue_ids == [3]
    else:
        assert any(0 in item.raw_flags for item in review.review)
    assert [flag.model_dump() for flag in flags] == before
    _assert_every_raw_finding_is_mapped(review, flags, [])


def test_operator_diagnostics_are_aggregated_by_kind():
    cues = [_cue(1, 0, 1_000, "Hello.")]
    flags = [
        QCFlag(kind="asr_word_clamped", message="ASR word endpoint clamped.", old_text="a", new_text="b"),
        QCFlag(kind="asr_word_clamped", message="ASR word endpoint clamped.", old_text="c", new_text="d"),
        QCFlag(kind="cost_unmetered", message="LLM usage not metered."),
    ]

    review = build_review(flags, [], cues, source_cues=cues)

    assert review.review == []
    assert {(item.kind, item.count) for item in review.diagnostics} == {("asr_word_clamped", 2), ("cost_unmetered", 1)}
    assert review.verdict == "clean"


def test_removed_adlib_is_located_after_the_previous_delivered_cue():
    cues = [_cue(1, 250_000, 251_000, "Olá."), _cue(2, 256_000, 257_000, "Tudo bem?")]
    flags = [QCFlag(kind="adlib_removed_without_speech_activity", cue_ids=[942], old_text="Você...", new_text="",
                    message="Generated ad-lib cue was removed.", start=254.166, end=254.266)]

    review = build_review(flags, [], cues, source_cues=cues)

    item, = review.review
    assert item.srt_numbers == []
    assert item.after_srt_number == 1
    assert item.timecode == "00:04:14,166"
    assert item.old_text == "Você..."


def test_every_emitted_flag_kind_is_registered():
    """A new emitter must choose its customer bucket instead of silently vanishing."""

    flag_kinds = _emitted_kind_literals("QCFlag")
    style_kinds = _emitted_kind_literals("StyleIssue")
    # Kinds built from a variable, an f-string or model_copy(update=...).
    dynamic_kinds = {
        "overlap_stacked", "overlap_flag_only", "output_overlap_preserved",
        "name_spelling_inconsistency", "unsourced_word_substitution",
        "missing_audio_source_cue_restored", "low_confidence_source_cue_restored",
        "timing_evidence_source_cue_restored", "protected_region_source_cue_restored",
    }
    flag_kinds = {kind for kind in flag_kinds if not kind.startswith("_")} | dynamic_kinds

    missing = sorted(flag_kinds - KIND_REGISTRY.keys())
    assert missing == [], f"add these QC flag kinds to qc_review.KIND_REGISTRY: {missing}"
    assert sorted(style_kinds - STYLE_REGISTRY.keys()) == []


def _emitted_kind_literals(constructor: str) -> set[str]:
    kinds: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name != constructor:
                continue
            for keyword in node.keywords:
                if keyword.arg != "kind":
                    continue
                for constant in _emitted_value_nodes(keyword.value):
                    if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                        kinds.add(constant.value)
    return kinds


def _emitted_value_nodes(value):
    if isinstance(value, ast.IfExp):
        yield from _emitted_value_nodes(value.body)
        yield from _emitted_value_nodes(value.orelse)
    else:
        yield from ast.walk(value)


def test_kind_registry_scan_keeps_both_branches_without_treating_conditions_as_kinds():
    expression = ast.parse("'unknown_branch' if voice == 'different' else 'registered_branch'", mode="eval").body
    values = {node.value for node in _emitted_value_nodes(expression)
              if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert values == {"unknown_branch", "registered_branch"}


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], "clean"),
        ([QCFlag(kind="adjudication_word_mapping_held", cue_ids=[1], severity="warning", message="x")], "check"),
        ([QCFlag(kind="timing_evidence_held", cue_ids=[1], severity="error", message="x")], "attention"),
        ([QCFlag(kind="text_changed", cue_ids=[1], message="x", old_text="a", new_text="b")], "clean"),
    ],
)
def test_verdict_is_computed_from_review_items_only(flags, expected):
    cues = [_cue(1, 0, 1_000, "b")] + [_cue(index, index * 2_000, index * 2_000 + 500, "x") for index in range(2, 20)]
    source = [_cue(1, 0, 1_000, "a"), *cues[1:]]

    assert build_review(flags, [], cues, source_cues=source).verdict == expected
