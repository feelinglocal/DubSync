from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import Cue, QCFlag
from .tokenize import alphanumeric_signature


TEXT_CHANGE_FLAG_KINDS = {"text_changed", "adlib_inserted"}
CUE_MATCH_THRESHOLD = 0.85
EVALUATION_BAND_MARGIN = 32


@dataclass(frozen=True)
class _CueMatch:
    predicted: Cue
    golden: Cue
    similarity: float
    is_exact: bool


def evaluate_against_golden(
    predicted: list[Cue],
    golden: list[Cue],
    fps: float,
    flags: list[QCFlag] | None = None,
    style_violations: int = 0,
    source: list[Cue] | None = None,
) -> dict[str, object]:
    predicted_signatures = [_text_signature(cue) for cue in predicted]
    golden_signatures = [_text_signature(cue) for cue in golden]
    cue_alignment = _match_cues_by_content(
        predicted,
        golden,
        predicted_signatures=predicted_signatures,
        golden_signatures=golden_signatures,
    )
    matched_pairs = [
        match for match in cue_alignment if match.similarity >= CUE_MATCH_THRESHOLD
    ]
    frame_ms = 1000.0 / fps if fps > 0 else 0.0
    deltas = [
        abs(match.predicted.start_ms - match.golden.start_ms) for match in matched_pairs
    ]
    end_deltas = [
        abs(match.predicted.end_ms - match.golden.end_ms) for match in matched_pairs
    ]
    matched_count = len(deltas)
    start_mae_ms = round(sum(deltas) / matched_count, 3) if matched_count else None
    within_1 = _ratio(sum(1 for delta in deltas if delta <= frame_ms), matched_count)
    within_3 = _ratio(sum(1 for delta in deltas if delta <= frame_ms * 3), matched_count)
    end_mae_ms = round(sum(end_deltas) / matched_count, 3) if matched_count else None
    ends_within_1 = _ratio(sum(1 for delta in end_deltas if delta <= frame_ms), matched_count)
    ends_within_3 = _ratio(sum(1 for delta in end_deltas if delta <= frame_ms * 3), matched_count)
    review_burden = _review_burden_ratio(predicted, flags or [])
    improv_metrics = _improv_detection_metrics(
        predicted,
        golden,
        flags or [],
        cue_alignment,
        golden_signatures,
        source=source,
    )

    return {
        "cue_count_predicted": len(predicted),
        "cue_count_golden": len(golden),
        "matched_cues": matched_count,
        "golden_match_coverage": _ratio(matched_count, len(golden)),
        "predicted_match_coverage": _ratio(matched_count, len(predicted)),
        "start_mae_ms": start_mae_ms,
        "end_mae_ms": end_mae_ms,
        "starts_within_1_frame_ratio": within_1,
        "starts_within_3_frames_ratio": within_3,
        "ends_within_1_frame_ratio": ends_within_1,
        "ends_within_3_frames_ratio": ends_within_3,
        "review_burden_ratio": review_burden,
        "style_violations": style_violations,
        **improv_metrics,
        "meets_timing_target": bool(
            matched_count
            and matched_count == len(golden) == len(predicted)
            and within_1 >= 0.9
            and within_3 >= 0.98
            and start_mae_ms is not None
            and start_mae_ms < 50.0
            and ends_within_1 >= 0.9
            and ends_within_3 >= 0.98
            and end_mae_ms is not None
            and end_mae_ms < 50.0
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


def _match_cues_by_content(
    predicted: list[Cue],
    golden: list[Cue],
    *,
    predicted_signatures: list[tuple[str, ...]] | None = None,
    golden_signatures: list[tuple[str, ...]] | None = None,
) -> list[_CueMatch]:
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
    if predicted_signatures is None:
        predicted_signatures = [_text_signature(cue) for cue in predicted]
    if golden_signatures is None:
        golden_signatures = [_text_signature(cue) for cue in golden]
    if len(predicted_signatures) != predicted_count or len(golden_signatures) != golden_count:
        raise ValueError("cue signature counts must match cue counts")
    reachability = _evaluation_reachability_centers(
        predicted_signatures,
        golden_signatures,
    )
    similarities: dict[tuple[int, int], float] = {}
    scores: list[dict[int, float]] = [{0: 0.0}]
    back: list[dict[int, str]] = [{0: ""}]
    _, first_end = _evaluation_band_window(
        0,
        predicted_count,
        golden_count,
        reachability.get(0, ()),
    )
    for golden_index in range(1, first_end + 1):
        scores[0][golden_index] = scores[0][golden_index - 1] + gap_score
        back[0][golden_index] = "insert"

    for predicted_index in range(1, predicted_count + 1):
        previous = scores[predicted_index - 1]
        row: dict[int, float] = {}
        row_back: dict[int, str] = {}
        start, end = _evaluation_band_window(
            predicted_index,
            predicted_count,
            golden_count,
            reachability.get(predicted_index, ()),
        )
        for golden_index in range(start, end + 1):
            best_score = float("-inf")
            best_op = ""
            if golden_index in previous:
                best_score = previous[golden_index] + gap_score
                best_op = "delete"
            if golden_index > 0 and (golden_index - 1) in row:
                insert = row[golden_index - 1] + gap_score
                if insert > best_score:
                    best_score = insert
                    best_op = "insert"
            if golden_index > 0 and (golden_index - 1) in previous:
                similarity = _signature_similarity(
                    predicted_signatures[predicted_index - 1],
                    golden_signatures[golden_index - 1],
                )
                similarities[(predicted_index - 1, golden_index - 1)] = similarity
                match = previous[golden_index - 1] + _cue_match_score_from_similarity(similarity)
                if match > best_score:
                    best_score = match
                    best_op = "match"
            if best_op:
                row[golden_index] = best_score
                row_back[golden_index] = best_op
        scores.append(row)
        back.append(row_back)

    if golden_count not in scores[predicted_count]:
        return _match_cues_by_content_unbanded(
            predicted,
            golden,
            predicted_signatures,
            golden_signatures,
        )

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
                    similarity=similarities.get(
                        (predicted_index - 1, golden_index - 1),
                        _signature_similarity(
                            predicted_signatures[predicted_index - 1],
                            golden_signatures[golden_index - 1],
                        ),
                    ),
                    is_exact=(
                        predicted_signatures[predicted_index - 1]
                        == golden_signatures[golden_index - 1]
                    ),
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
    if _misses_unique_evaluation_pair(
        pairs,
        predicted,
        golden,
        predicted_signatures,
        golden_signatures,
    ):
        return _match_cues_by_content_unbanded(
            predicted,
            golden,
            predicted_signatures,
            golden_signatures,
        )
    return pairs


def _match_cues_by_content_unbanded(
    predicted: list[Cue],
    golden: list[Cue],
    predicted_signatures: list[tuple[str, ...]],
    golden_signatures: list[tuple[str, ...]],
) -> list[_CueMatch]:
    gap_score = -0.75
    predicted_count = len(predicted)
    golden_count = len(golden)
    scores = [[0.0] * (golden_count + 1) for _ in range(predicted_count + 1)]
    back = [[""] * (golden_count + 1) for _ in range(predicted_count + 1)]
    similarities: dict[tuple[int, int], float] = {}
    for predicted_index in range(1, predicted_count + 1):
        scores[predicted_index][0] = scores[predicted_index - 1][0] + gap_score
        back[predicted_index][0] = "delete"
    for golden_index in range(1, golden_count + 1):
        scores[0][golden_index] = scores[0][golden_index - 1] + gap_score
        back[0][golden_index] = "insert"
    for predicted_index in range(1, predicted_count + 1):
        for golden_index in range(1, golden_count + 1):
            similarity = _signature_similarity(
                predicted_signatures[predicted_index - 1],
                golden_signatures[golden_index - 1],
            )
            similarities[(predicted_index - 1, golden_index - 1)] = similarity
            match = scores[predicted_index - 1][golden_index - 1] + _cue_match_score_from_similarity(similarity)
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
            pairs.append(
                _CueMatch(
                    predicted=predicted[predicted_index - 1],
                    golden=golden[golden_index - 1],
                    similarity=similarities[(predicted_index - 1, golden_index - 1)],
                    is_exact=(
                        predicted_signatures[predicted_index - 1]
                        == golden_signatures[golden_index - 1]
                    ),
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


def _evaluation_band_window(
    row: int,
    predicted_count: int,
    golden_count: int,
    prior_centers: tuple[int, ...] = (),
) -> tuple[int, int]:
    if predicted_count <= 0:
        return 0, golden_count
    center = round(row * golden_count / predicted_count)
    start = max(0, center - EVALUATION_BAND_MARGIN)
    end = min(golden_count, center + EVALUATION_BAND_MARGIN)
    if row == 0:
        start = 0
    if row == predicted_count:
        end = golden_count
    for prior_center in prior_centers:
        start = min(start, max(0, prior_center - EVALUATION_BAND_MARGIN))
        end = max(end, min(golden_count, prior_center + EVALUATION_BAND_MARGIN))
    return start, end


def _evaluation_unique_exact_pairs(
    predicted_signatures: list[tuple[str, ...]],
    golden_signatures: list[tuple[str, ...]],
) -> list[tuple[int, int]]:
    predicted_positions: dict[tuple[str, ...], list[int]] = {}
    golden_positions: dict[tuple[str, ...], list[int]] = {}
    for index, signature in enumerate(predicted_signatures):
        if signature:
            predicted_positions.setdefault(signature, []).append(index)
    for index, signature in enumerate(golden_signatures):
        if signature:
            golden_positions.setdefault(signature, []).append(index)
    pairs = sorted(
        (predicted_indices[0], golden_positions[signature][0])
        for signature, predicted_indices in predicted_positions.items()
        if len(predicted_indices) == 1
        and signature in golden_positions
        and len(golden_positions[signature]) == 1
    )
    if any(left_golden >= right_golden for (_, left_golden), (_, right_golden) in zip(pairs, pairs[1:])):
        return []
    return pairs


def _evaluation_reachability_centers(
    predicted_signatures: list[tuple[str, ...]],
    golden_signatures: list[tuple[str, ...]],
) -> dict[int, tuple[int, ...]]:
    centers: dict[int, set[int]] = {}
    for predicted_index, golden_index in _evaluation_unique_exact_pairs(
        predicted_signatures,
        golden_signatures,
    ):
        centers.setdefault(predicted_index, set()).add(golden_index)
        centers.setdefault(predicted_index + 1, set()).add(golden_index + 1)
    return {row: tuple(sorted(values)) for row, values in centers.items()}


def _misses_unique_evaluation_pair(
    matches: list[_CueMatch],
    predicted: list[Cue],
    golden: list[Cue],
    predicted_signatures: list[tuple[str, ...]],
    golden_signatures: list[tuple[str, ...]],
) -> bool:
    expected = {
        (predicted[predicted_index].index, golden[golden_index].index)
        for predicted_index, golden_index in _evaluation_unique_exact_pairs(
            predicted_signatures,
            golden_signatures,
        )
    }
    matched = {(match.predicted.index, match.golden.index) for match in matches if match.is_exact}
    return not expected.issubset(matched)


def _cue_match_score(predicted: Cue, golden: Cue) -> float:
    similarity = _cue_match_similarity(predicted, golden)
    return _cue_match_score_from_similarity(similarity)


def _cue_match_score_from_similarity(similarity: float) -> float:
    return 2.0 * similarity if similarity >= CUE_MATCH_THRESHOLD else -0.6


def _cue_match_similarity(predicted: Cue, golden: Cue) -> float:
    return _signature_similarity(_text_signature(predicted), _text_signature(golden))


def _signature_similarity(predicted_signature: tuple[str, ...], golden_signature: tuple[str, ...]) -> float:
    if predicted_signature and predicted_signature == golden_signature:
        return 1.0
    if not predicted_signature or not golden_signature:
        return 0.0
    return fuzz.ratio(" ".join(predicted_signature), " ".join(golden_signature)) / 100.0


def _review_burden_ratio(cues: list[Cue], flags: list[QCFlag]) -> float:
    if not cues:
        return 0.0
    existing_cue_ids = {cue.index for cue in cues}
    flagged_cues = {cue_id for flag in flags for cue_id in flag.cue_ids if cue_id in existing_cue_ids}
    return len(flagged_cues) / len(cues)


def _improv_detection_metrics(
    predicted: list[Cue],
    golden: list[Cue],
    flags: list[QCFlag],
    cue_alignment: list[_CueMatch],
    golden_signatures: list[tuple[str, ...]],
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
        source_signatures = [_text_signature(cue) for cue in source]
        source_alignment = _match_cues_by_content(
            source,
            golden,
            predicted_signatures=source_signatures,
            golden_signatures=golden_signatures,
        )
        matched_golden_ids = {match.golden.index for match in source_alignment}
        changed_aligned_ids = {
            match.golden.index
            for match in source_alignment
            if not match.is_exact
        }
        inserted_golden_ids = {cue.index for cue in golden if cue.index not in matched_golden_ids}
        actual_changed_ids = changed_aligned_ids | inserted_golden_ids
        true_positive_ids = {
            predicted_id
            for predicted_id in flagged_change_ids
            if (match := aligned_by_predicted_id.get(predicted_id)) is not None
            and match.golden.index in actual_changed_ids
            and match.is_exact
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
            if not match.is_exact
        }
        true_positive_ids = {
            predicted_id
            for predicted_id in flagged_change_ids
            if (match := aligned_by_predicted_id.get(predicted_id)) is not None
            and match.is_exact
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
