from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import Cue, QCFlag
from .tokenize import alphanumeric_signature


TEXT_CHANGE_FLAG_KINDS = {"text_changed", "adlib_inserted"}
CUE_MATCH_THRESHOLD = 0.85


@dataclass(frozen=True)
class _CueMatch:
    predicted: Cue
    golden: Cue
    similarity: float


def evaluate_against_golden(
    predicted: list[Cue],
    golden: list[Cue],
    fps: float,
    flags: list[QCFlag] | None = None,
    style_violations: int = 0,
    source: list[Cue] | None = None,
) -> dict[str, object]:
    cue_alignment = _match_cues_by_content(predicted, golden)
    matched_pairs = [
        match for match in cue_alignment if match.similarity >= CUE_MATCH_THRESHOLD
    ]
    frame_ms = 1000.0 / fps if fps > 0 else 0.0
    deltas = [
        abs(match.predicted.start_ms - match.golden.start_ms) for match in matched_pairs
    ]
    matched_count = len(deltas)
    start_mae_ms = round(sum(deltas) / matched_count, 3) if matched_count else None
    within_1 = _ratio(sum(1 for delta in deltas if delta <= frame_ms), matched_count)
    within_3 = _ratio(sum(1 for delta in deltas if delta <= frame_ms * 3), matched_count)
    review_burden = _review_burden_ratio(predicted, flags or [])
    improv_metrics = _improv_detection_metrics(
        predicted,
        golden,
        flags or [],
        cue_alignment,
        source=source,
    )

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


def _match_cues_by_content(predicted: list[Cue], golden: list[Cue]) -> list[_CueMatch]:
    """Globally align cue sequences before comparing their timestamps.

    Cue numbers are export-local and can shift after an insertion or deletion. This
    text-only alignment keeps timing evaluation monotonic without using timestamps
    themselves to choose favorable pairs.
    """

    predicted_count = len(predicted)
    golden_count = len(golden)
    if not predicted_count or not golden_count:
        return []

    gap_score = -0.75
    scores = [[0.0] * (golden_count + 1) for _ in range(predicted_count + 1)]
    back = [[""] * (golden_count + 1) for _ in range(predicted_count + 1)]
    for predicted_index in range(1, predicted_count + 1):
        scores[predicted_index][0] = scores[predicted_index - 1][0] + gap_score
        back[predicted_index][0] = "delete"
    for golden_index in range(1, golden_count + 1):
        scores[0][golden_index] = scores[0][golden_index - 1] + gap_score
        back[0][golden_index] = "insert"

    for predicted_index in range(1, predicted_count + 1):
        for golden_index in range(1, golden_count + 1):
            match = scores[predicted_index - 1][golden_index - 1] + _cue_match_score(
                predicted[predicted_index - 1],
                golden[golden_index - 1],
            )
            delete = scores[predicted_index - 1][golden_index] + gap_score
            insert = scores[predicted_index][golden_index - 1] + gap_score
            best_score = match
            best_op = "match"
            if delete > best_score:
                best_score = delete
                best_op = "delete"
            if insert > best_score:
                best_score = insert
                best_op = "insert"
            scores[predicted_index][golden_index] = best_score
            back[predicted_index][golden_index] = best_op

    pairs: list[_CueMatch] = []
    predicted_index = predicted_count
    golden_index = golden_count
    while predicted_index > 0 or golden_index > 0:
        op = back[predicted_index][golden_index]
        if op == "match":
            predicted_cue = predicted[predicted_index - 1]
            golden_cue = golden[golden_index - 1]
            pairs.append(
                _CueMatch(
                    predicted=predicted_cue,
                    golden=golden_cue,
                    similarity=_cue_match_similarity(predicted_cue, golden_cue),
                )
            )
            predicted_index -= 1
            golden_index -= 1
        elif op == "delete":
            predicted_index -= 1
        elif op == "insert":
            golden_index -= 1
        else:
            raise RuntimeError("cue evaluation alignment reached an empty operation")
    pairs.reverse()
    return pairs


def _cue_match_score(predicted: Cue, golden: Cue) -> float:
    similarity = _cue_match_similarity(predicted, golden)
    return 2.0 * similarity if similarity >= CUE_MATCH_THRESHOLD else -0.6


def _cue_match_similarity(predicted: Cue, golden: Cue) -> float:
    predicted_signature = _text_signature(predicted)
    golden_signature = _text_signature(golden)
    if predicted_signature and predicted_signature == golden_signature:
        return 1.0
    if not predicted_signature or not golden_signature:
        return 0.0
    return fuzz.ratio(" ".join(predicted_signature), " ".join(golden_signature)) / 100.0


def _review_burden_ratio(cues: list[Cue], flags: list[QCFlag]) -> float:
    if not cues:
        return 0.0
    flagged_cues = {cue_id for flag in flags for cue_id in flag.cue_ids}
    return len(flagged_cues) / len(cues)


def _improv_detection_metrics(
    predicted: list[Cue],
    golden: list[Cue],
    flags: list[QCFlag],
    cue_alignment: list[_CueMatch],
    source: list[Cue] | None = None,
) -> dict[str, object]:
    by_predicted = {cue.index: cue for cue in predicted}
    aligned_by_predicted_id = {match.predicted.index: match for match in cue_alignment}
    flagged_change_ids = {
        cue_id
        for flag in flags
        if flag.kind in TEXT_CHANGE_FLAG_KINDS
        for cue_id in flag.cue_ids
        if cue_id in by_predicted
    }
    if source:
        source_alignment = _match_cues_by_content(source or [], golden)
        matched_golden_ids = {match.golden.index for match in source_alignment}
        changed_aligned_ids = {
            match.golden.index
            for match in source_alignment
            if _text_signature(match.predicted) != _text_signature(match.golden)
        }
        inserted_golden_ids = {cue.index for cue in golden if cue.index not in matched_golden_ids}
        actual_changed_ids = changed_aligned_ids | inserted_golden_ids
        true_positive_ids = {
            predicted_id
            for predicted_id in flagged_change_ids
            if (match := aligned_by_predicted_id.get(predicted_id)) is not None
            and match.golden.index in actual_changed_ids
            and _text_signature(match.predicted) == _text_signature(match.golden)
        }
        true_positives = len(true_positive_ids)
        false_positives = len(flagged_change_ids - true_positive_ids)
        detected_golden_ids = {
            aligned_by_predicted_id[predicted_id].golden.index
            for predicted_id in true_positive_ids
        }
        false_negatives = len(actual_changed_ids - detected_golden_ids)
    else:
        aligned_mismatch_ids = {
            match.predicted.index
            for match in cue_alignment
            if _text_signature(match.predicted) != _text_signature(match.golden)
        }
        true_positive_ids = {
            predicted_id
            for predicted_id in flagged_change_ids
            if (match := aligned_by_predicted_id.get(predicted_id)) is not None
            and _text_signature(match.predicted) == _text_signature(match.golden)
        }
        true_positives = len(true_positive_ids)
        false_positives = len(flagged_change_ids - true_positive_ids)
        false_negatives = len(aligned_mismatch_ids - flagged_change_ids)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)

    return {
        "improv_true_positives": true_positives,
        "improv_false_positives": false_positives,
        "improv_false_negatives": false_negatives,
        "improv_precision": precision,
        "improv_recall": recall,
    }


def _text_signature(cue: Cue) -> tuple[str, ...]:
    return tuple(alphanumeric_signature(cue.plain_text))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
