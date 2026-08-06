from __future__ import annotations

import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag

_SENTENCE_BOUNDARY_BEFORE_WORD = re.compile(r"[.!?][\"'»“”„’\)\]]*\s*$")


def span_coverage_flags(
    source_cues: list[Cue],
    rebuilt_cues: list[Cue],
    spans: list[DivergenceSpan],
    decisions: list[AdjudicationDecision],
    min_ratio: float = 0.5,
) -> list[QCFlag]:
    decisions_by_case = {decision.case_id: decision for decision in decisions}
    flags: list[QCFlag] = []
    for span in spans:
        decision = decisions_by_case.get(span.case_id)
        if decision is None or decision.verdict == "keep_srt":
            continue
        source_window = _cue_window(source_cues, span.cue_ids)
        if source_window is None:
            continue
        rebuilt_window = _cue_window(rebuilt_cues, span.cue_ids)
        source_duration = max(1, source_window[1] - source_window[0])
        rebuilt_duration = 0 if rebuilt_window is None else max(0, rebuilt_window[1] - rebuilt_window[0])
        ratio = rebuilt_duration / source_duration
        if ratio >= min_ratio:
            continue
        flags.append(
            QCFlag(
                kind="span_coverage_low",
                cue_ids=span.cue_ids,
                message=(
                    f"Adjudicated replacement covers only {ratio:.0%} of the source span; "
                    "hold for editor review before accepting compressed dialogue."
                ),
                severity="error",
                old_text=span.srt_text,
                new_text=decision.final_text,
                start=source_window[0] / 1000.0,
                end=source_window[1] / 1000.0,
            )
        )
    return flags


def name_spelling_inconsistency_flags(
    source_cues: list[Cue],
    output_cues: list[Cue],
    min_source_count: int = 2,
    min_source_consistency: float = 0.8,
    similarity_threshold: float = 0.7,
) -> list[QCFlag]:
    source_occurrences = _case_preserving_source_occurrences(source_cues)
    source_tokens = [occurrence[0] for occurrence in source_occurrences]
    source_counts = Counter(token.casefold() for token in source_tokens)
    source_forms: dict[str, Counter[str]] = defaultdict(Counter)
    source_name_like: dict[str, bool] = defaultdict(bool)
    for token, sentence_initial in source_occurrences:
        source_forms[token.casefold()][token] += 1
        if _is_capitalized_token(token) and not sentence_initial:
            source_name_like[token.casefold()] = True

    output_occurrences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cue in output_cues:
        for token in _case_preserving_tokens(cue.plain_text):
            output_occurrences[token.casefold()].append((token, cue.index))

    flags: list[QCFlag] = []
    for output_key, occurrences in sorted(output_occurrences.items()):
        if output_key in source_forms or len(output_key) < 4:
            continue
        output_spelling = occurrences[0][0]
        candidates: list[tuple[float, str]] = []
        for source_key, source_spellings in source_forms.items():
            if source_counts.get(source_key, 0) < min_source_count:
                continue
            if abs(len(output_key) - len(source_key)) > 3:
                continue
            source_spelling = source_spellings.most_common(1)[0][0]
            similarity = fuzz.ratio(output_key, source_key) / 100.0
            if similarity < similarity_threshold:
                continue
            candidates.append((similarity, source_key))
        if not candidates:
            continue
        _, source_key = max(candidates, key=lambda candidate: (source_counts[candidate[1]], candidate[0]))
        similar_source_count = sum(source_counts[candidate_key] for _, candidate_key in candidates)
        if source_counts[source_key] / similar_source_count < min_source_consistency:
            continue
        source_spelling = source_forms[source_key].most_common(1)[0][0]
        is_name_like = source_name_like[source_key]
        kind = "name_spelling_inconsistency" if is_name_like else "unsourced_word_substitution"
        label = "a possible name drift" if is_name_like else "an unsourced word substitution"
        flags.append(
            QCFlag(
                kind=kind,
                cue_ids=sorted({cue_id for _, cue_id in occurrences}),
                message=(
                    f"Output spelling '{output_spelling}' is absent from the source, while near spelling "
                    f"'{source_spelling}' appears {source_counts[source_key]} times; review as {label}."
                ),
                severity="warning",
                old_text=source_spelling,
                new_text=output_spelling,
            )
        )
    return flags


def _cue_window(cues: list[Cue], cue_ids: list[int]) -> tuple[int, int] | None:
    selected_ids = set(cue_ids)
    selected = [cue for cue in cues if cue.index in selected_ids]
    if not selected:
        return None
    return min(cue.start_ms for cue in selected), max(cue.end_ms for cue in selected)


def _case_preserving_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text, re.UNICODE)


def _case_preserving_source_occurrences(cues: list[Cue]) -> list[tuple[str, bool]]:
    occurrences: list[tuple[str, bool]] = []
    sentence_initial = True
    for cue in cues:
        cue_occurrences, sentence_initial = _cue_token_occurrences(
            cue.plain_text,
            sentence_initial=sentence_initial,
        )
        occurrences.extend(cue_occurrences)
    return occurrences


def _cue_token_occurrences(
    text: str,
    *,
    sentence_initial: bool,
) -> tuple[list[tuple[str, bool]], bool]:
    occurrences: list[tuple[str, bool]] = []
    previous_end = 0
    for match in re.finditer(r"[\w]+", text, re.UNICODE):
        separator = text[previous_end : match.start()]
        if _SENTENCE_BOUNDARY_BEFORE_WORD.search(separator):
            sentence_initial = True
        token = match.group(0)
        occurrences.append((token, sentence_initial))
        sentence_initial = False
        previous_end = match.end()
    if _SENTENCE_BOUNDARY_BEFORE_WORD.search(text[previous_end:]):
        sentence_initial = True
    return occurrences, sentence_initial


def _is_capitalized_token(token: str) -> bool:
    return bool(token) and token[0].isupper()
