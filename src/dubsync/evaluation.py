from __future__ import annotations

from .models import Cue, QCFlag
from .tokenize import alphanumeric_signature


TEXT_CHANGE_FLAG_KINDS = {"text_changed", "adlib_inserted"}


def evaluate_against_golden(
    predicted: list[Cue],
    golden: list[Cue],
    fps: float,
    flags: list[QCFlag] | None = None,
    style_violations: int = 0,
    source: list[Cue] | None = None,
) -> dict[str, object]:
    by_predicted = {cue.index: cue for cue in predicted}
    by_golden = {cue.index: cue for cue in golden}
    matched_ids = sorted(set(by_predicted) & set(by_golden))
    frame_ms = 1000.0 / fps if fps > 0 else 0.0
    deltas = [abs(by_predicted[cue_id].start_ms - by_golden[cue_id].start_ms) for cue_id in matched_ids]
    matched_count = len(deltas)
    start_mae_ms = round(sum(deltas) / matched_count, 3) if matched_count else None
    within_1 = _ratio(sum(1 for delta in deltas if delta <= frame_ms), matched_count)
    within_3 = _ratio(sum(1 for delta in deltas if delta <= frame_ms * 3), matched_count)
    review_burden = _review_burden_ratio(predicted, flags or [])
    improv_metrics = _improv_detection_metrics(predicted, golden, flags or [], source=source)

    return {
        "cue_count_predicted": len(predicted),
        "cue_count_golden": len(golden),
        "matched_cues": matched_count,
        "start_mae_ms": start_mae_ms,
        "starts_within_1_frame_ratio": within_1,
        "starts_within_3_frames_ratio": within_3,
        "review_burden_ratio": review_burden,
        "style_violations": style_violations,
        **improv_metrics,
        "meets_timing_target": bool(
            matched_count
            and within_1 >= 0.9
            and within_3 >= 0.98
            and start_mae_ms is not None
            and start_mae_ms < 50.0
        ),
        "meets_structure_target": len(predicted) == len(golden) and style_violations == 0,
        "meets_review_burden_target": review_burden <= 0.1,
        "meets_improv_target": (
            improv_metrics["improv_precision"] is not None
            and improv_metrics["improv_recall"] is not None
            and improv_metrics["improv_precision"] >= 0.9
            and improv_metrics["improv_recall"] >= 0.85
        ),
    }


def _review_burden_ratio(cues: list[Cue], flags: list[QCFlag]) -> float:
    if not cues:
        return 0.0
    flagged_cues = {cue_id for flag in flags for cue_id in flag.cue_ids}
    return len(flagged_cues) / len(cues)


def _improv_detection_metrics(
    predicted: list[Cue],
    golden: list[Cue],
    flags: list[QCFlag],
    source: list[Cue] | None = None,
) -> dict[str, object]:
    by_predicted = {cue.index: cue for cue in predicted}
    by_golden = {cue.index: cue for cue in golden}
    by_source = {cue.index: cue for cue in source or []}
    matched_ids = set(by_predicted) & set(by_golden)
    flagged_change_ids = {
        cue_id
        for flag in flags
        if flag.kind in TEXT_CHANGE_FLAG_KINDS
        for cue_id in flag.cue_ids
        if cue_id in matched_ids
    }
    if by_source:
        actual_changed_ids = _source_changed_ids(by_golden, by_source, matched_ids)
        true_positive_ids = {
            cue_id
            for cue_id in flagged_change_ids
            if cue_id in actual_changed_ids
            and _text_signature(by_predicted[cue_id]) == _text_signature(by_golden[cue_id])
        }
        true_positives = len(true_positive_ids)
        false_positives = len(flagged_change_ids - true_positive_ids)
        false_negatives = len(actual_changed_ids - true_positive_ids)
    else:
        golden_mismatch_ids = {
            cue_id
            for cue_id in matched_ids
            if _text_signature(by_predicted[cue_id]) != _text_signature(by_golden[cue_id])
        }
        true_positives = sum(
            1
            for cue_id in flagged_change_ids
            if _text_signature(by_predicted[cue_id]) == _text_signature(by_golden[cue_id])
        )
        false_positives = len(flagged_change_ids) - true_positives
        false_negatives = len(golden_mismatch_ids - flagged_change_ids)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)

    return {
        "improv_true_positives": true_positives,
        "improv_false_positives": false_positives,
        "improv_false_negatives": false_negatives,
        "improv_precision": precision,
        "improv_recall": recall,
    }


def _source_changed_ids(
    by_golden: dict[int, Cue],
    by_source: dict[int, Cue],
    matched_ids: set[int],
) -> set[int]:
    source_matched_ids = matched_ids & set(by_source)
    return {
        cue_id
        for cue_id in source_matched_ids
        if _text_signature(by_source[cue_id]) != _text_signature(by_golden[cue_id])
    }


def _text_signature(cue: Cue) -> tuple[str, ...]:
    return tuple(alphanumeric_signature(cue.plain_text))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
