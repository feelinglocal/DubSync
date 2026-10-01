from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from rapidfuzz import fuzz

from .models import AlignmentResult, Cue, QCFlag, Word
from .style_profile import StyleProfile
from .subtitle_annotations import is_bracketed_screen_text_cue, speech_text_for_alignment
from .tokenize import alphanumeric_signature, number_alias, spoken_number_values


# A pause inside a cue may be this many ``max_intra_cue_gap`` long before the
# words beyond it stop timing the cue, provided the cue's text contains them.
LEXICAL_BRIDGE_GAP_FACTOR = 2.0


@dataclass(frozen=True)
class _CueTiming:
    start_ms: int
    spoken_end_ms: int
    spoken_end_raw_ms: float
    end_ms: int
    min_end_ms: int
    speaker_id: str | None


def shared_word_cue_ids(alignment: AlignmentResult) -> set[int]:
    """Find cues whose boundary falls inside one indivisible ASR timestamp."""
    owners: dict[int, set[int]] = {}
    for cue_id, word_indices in alignment.cue_word_indices.items():
        for word_index in word_indices:
            owners.setdefault(word_index, set()).add(cue_id)
    return {cue_id for cue_ids in owners.values() if len(cue_ids) > 1 for cue_id in cue_ids}


def ambiguous_word_cue_ids(alignment: AlignmentResult, word_indices: set[int] | None) -> set[int]:
    """Find every cue owning a word whose speech burst could not be identified."""
    if not word_indices:
        return set()
    return {
        cue_id for cue_id, owned in alignment.cue_word_indices.items()
        if any(index in word_indices for index in owned)
    }


def ambiguous_word_timing_flags(cues: list[Cue], cue_ids: set[int]) -> list[QCFlag]:
    return [
        QCFlag(
            kind="timing_evidence_held", cue_ids=[cue.index], severity="error",
            message=(
                "An ASR word overlaps multiple plausible speech bursts, so its position is uncertain. "
                "The complete cue and its input timing were preserved for review."
            ),
            old_text=cue.text, start=cue.start_ms / 1000.0, end=cue.end_ms / 1000.0,
        )
        for cue in cues if cue.index in cue_ids and not is_bracketed_screen_text_cue(cue)
    ]


def preserve_source_timings(cues: list[Cue], source_cues: list[Cue], cue_ids: set[int]) -> list[Cue]:
    sources = {cue.index: cue for cue in source_cues if cue.index in cue_ids}
    return [
        cue.with_timing(sources[cue.index].start_ms, sources[cue.index].end_ms)
        if cue.index in sources else cue
        for cue in cues
    ]


def shared_word_timing_flags(cues: list[Cue], cue_ids: set[int]) -> list[QCFlag]:
    return [
        QCFlag(
            kind="shared_word_timing_preserved",
            cue_ids=[cue.index],
            message=(
                "One ASR timestamp spans multiple cues, so their internal speech boundary is uncertain. "
                "Source timing was preserved for review."
            ),
            start=cue.start_ms / 1000.0,
            end=cue.end_ms / 1000.0,
        )
        for cue in cues if cue.index in cue_ids
    ]


def rebuild_cues(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    max_word_duration: float = 2.0,
    max_intra_cue_gap: float = 1.5,
    protected_cue_ids: set[int] | None = None,
    min_duration_policy: str = "extend_into_silence",
    ambiguous_word_indices: set[int] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    rebuilt: list[Cue] = []
    flags: list[QCFlag] = []
    shared_cue_ids = shared_word_cue_ids(alignment)
    protected = set(protected_cue_ids or ()) | shared_cue_ids
    ambiguous = ambiguous_word_cue_ids(alignment, ambiguous_word_indices)
    flags.extend(ambiguous_word_timing_flags(cues, ambiguous - protected))
    protected |= ambiguous
    timings, timing_flags = _cue_timings(
        cues,
        words,
        alignment,
        profile,
        max_word_duration=max_word_duration,
        max_intra_cue_gap=max_intra_cue_gap,
        protected_cue_ids=protected,
        # The "acoustic" policy ends a short cue with its speech; refinement
        # applies the same policy with the audio evidence rebuild lacks.
        extend_short_cues=min_duration_policy != "acoustic",
    )
    flags.extend(timing_flags)
    held_cue_ids = protected | {
        cue_id for flag in timing_flags if flag.kind == "timing_evidence_held"
        for cue_id in flag.cue_ids
    }
    preserved_cue_ids = held_cue_ids | {
        cue.index for cue in cues
        if cue.index not in timings and profile.drop_policy != "remove"
        and not is_bracketed_screen_text_cue(cue)
    }
    next_start_by_cue = _next_acoustic_start_by_cue(cues, timings, preserved_cue_ids)

    for cue in cues:
        if cue.index in held_cue_ids:
            rebuilt.append(cue)
            continue
        timing = timings.get(cue.index)
        if timing is None:
            should_remove = profile.drop_policy == "remove"
            if is_bracketed_screen_text_cue(cue):
                rebuilt.append(cue)
                continue
            if not should_remove:
                rebuilt.append(cue)
            flags.append(
                QCFlag(
                    kind="dropped_unmatched_cue" if should_remove else "unmatched_cue",
                    cue_ids=[cue.index],
                    message=(
                        "No ASR word timestamps matched this cue; removed by drop_policy."
                        if should_remove
                        else "No ASR word timestamps matched this cue."
                    ),
                    old_text=cue.text,
                    start=cue.start_ms / 1000.0,
                    end=cue.end_ms / 1000.0,
                )
            )
            continue

        end_ms = _extend_into_available_gap(timing, next_start_by_cue.get(cue.index), profile)
        rebuilt.append(cue.with_timing(timing.start_ms, end_ms).model_copy(update={"speaker_id": timing.speaker_id}))

    # A cue always starts with its own first word. Display padding and frame
    # snapping were capped at the following start above; what still overlaps
    # is simultaneous speech, which stays visible for the overlap policy.
    flags.extend(shared_word_timing_flags(cues, shared_cue_ids))
    return rebuilt, flags


def cue_spoken_spans(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    *,
    max_word_duration: float = 2.0,
    max_intra_cue_gap: float = 1.5,
    ambiguous_word_indices: set[int] | None = None,
) -> dict[int, tuple[int, int]]:
    """First-word onset and last-word offset (ms) of every cue that owns timed words.

    Final overlap resolution uses these to tell display padding, which may be
    trimmed, from a cue's own speech, which may not.
    """
    spans: dict[int, tuple[int, int]] = {}
    ambiguous = ambiguous_word_cue_ids(alignment, ambiguous_word_indices)
    for cue in cues:
        if cue.index in ambiguous:
            continue
        owned = [
            words[index]
            for index in alignment.cue_word_indices.get(cue.index, [])
            if 0 <= index < len(words)
        ]
        if not owned:
            continue
        selected, _ = select_cue_word_window(
            cue, owned, max_word_duration=max_word_duration, max_intra_cue_gap=max_intra_cue_gap,
        )
        spans[cue.index] = (
            round(min(word.start for word in selected) * 1000),
            round(max(word.end for word in selected) * 1000),
        )
    return spans


def _cue_timings(
    cues: list[Cue],
    words: list[Word],
    alignment: AlignmentResult,
    profile: StyleProfile,
    *,
    max_word_duration: float,
    max_intra_cue_gap: float,
    protected_cue_ids: set[int],
    extend_short_cues: bool = True,
) -> tuple[dict[int, _CueTiming], list[QCFlag]]:
    timings: dict[int, _CueTiming] = {}
    flags: list[QCFlag] = []
    for cue in cues:
        if cue.index in protected_cue_ids or is_bracketed_screen_text_cue(cue):
            continue
        word_indices = alignment.cue_word_indices.get(cue.index, [])
        if not word_indices:
            continue
        matched_words = [words[index] for index in word_indices]
        selected_words, trimmed = select_cue_word_window(
            cue,
            matched_words,
            max_word_duration=max_word_duration,
            max_intra_cue_gap=max_intra_cue_gap,
        )
        if trimmed:
            flags.append(
                QCFlag(
                    kind="timing_outlier_trimmed",
                    cue_ids=[cue.index],
                    message="Cue timing ignored an impossible ASR word duration or intra-cue gap.",
                    old_text=_word_span_label(matched_words),
                    new_text=_word_span_label(selected_words),
                    start=selected_words[0].start if selected_words else None,
                    end=selected_words[-1].end if selected_words else None,
                )
            )
        matched_words = selected_words
        issue = timing_evidence_issue(cue, matched_words)
        if issue is not None:
            flags.append(QCFlag(
                kind="timing_evidence_held", cue_ids=[cue.index], severity="error",
                message=(
                    f"{issue} Approved dialogue and its input timing were preserved for review; "
                    "no neighboring speech boundaries were borrowed."
                ),
                old_text=cue.text, start=cue.start_ms / 1000, end=cue.end_ms / 1000,
            ))
            continue
        start_ms = max(0, profile.snap_floor(min(word.start for word in matched_words) * 1000 - profile.lead_in_ms))
        spoken_end = max(word.end for word in matched_words) * 1000
        end_ms = profile.snap_ceil(spoken_end + profile.tail_ms)
        min_end_ms = profile.snap_ceil(start_ms + profile.min_cue_dur * 1000) if extend_short_cues else end_ms
        timings[cue.index] = _CueTiming(
            start_ms=start_ms,
            spoken_end_ms=profile.snap_ceil(spoken_end),
            spoken_end_raw_ms=spoken_end,
            end_ms=end_ms,
            min_end_ms=min_end_ms,
            speaker_id=_dominant_speaker(matched_words),
        )
    return timings, flags


def timing_evidence_issue(cue: Cue, words: list[Word]) -> str | None:
    """Identify clearly unusable anchors without estimating missing speech.

    These conservative bounds detect placeholder timestamps and sparse lexical
    matches, not ordinary reading-speed problems. Display padding cannot turn
    those anchors into reliable timing for a complete source phrase.
    """
    source_tokens = alphanumeric_signature(speech_text_for_alignment(cue))
    if not source_tokens or not words:
        return None
    if any(
        not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start
        for word in words
    ):
        return "Matched ASR words contain invalid timestamps."
    evidence_tokens = alphanumeric_signature(" ".join(word.text for word in words))
    span = max(word.end for word in words) - min(word.start for word in words)
    short_word_count = sum(word.end - word.start <= 0.020 + 1e-9 for word in words)
    if span <= 0.005 + 1e-9 or (
        len(evidence_tokens) >= 2 and (
            span < 0.080 - 1e-9
            or short_word_count * 2 >= len(words) and span < 0.060 * len(evidence_tokens) - 1e-9
        )
    ):
        return (
            f"Matched ASR timing is collapsed: {len(evidence_tokens)} lexical tokens occupy "
            f"{span * 1000:.1f} ms, including {short_word_count} words of at most 20 ms."
        )
    if len(source_tokens) >= 3:
        supported = _ordered_lexical_support(source_tokens, evidence_tokens)
        if supported * 2 < len(source_tokens):
            return (
                f"Sparse lexical timing evidence covers only {supported}/{len(source_tokens)} "
                "spoken source tokens."
            )
    return None


# The aligner pairs the same speech written another way; such words are owned
# evidence. Limits mirror its compound and spoken-number groups.
_COMPOUND_MAX_PARTS = 3
_COMPOUND_MIN_CHARACTERS = 4
_NUMBER_GROUP_MAX_WORDS = 6


def _ordered_lexical_support(source: list[str], evidence: list[str]) -> int:
    """How many source tokens the timestamped tokens support, in order.

    Each timestamped token is used once. Normalization handles numeric and
    accent aliases; the aligner's 85% spelling tolerance retains minor name
    variants. A compound written open on one side and closed on the other
    ("Drachen Evolutionssystem" / "Drachen-Evolutionssystem") and a number
    written in digits on one side and spoken in words on the other
    ("vinte e seis" / "26") support every source token of the group.
    """
    def same_word(token: str, candidate: str) -> bool:
        return (
            token == candidate or fuzz.ratio(token, candidate, score_cutoff=85) >= 85
            or (candidate.isdigit() and number_alias(token) == candidate)
            or (token.isdigit() and number_alias(candidate) == token)
        )

    def same_group(tokens: list[str], candidates: list[str]) -> bool:
        if len(tokens) + len(candidates) <= 2:
            return False
        for digits, spoken in ((tokens, candidates), (candidates, tokens)):
            if len(digits) == 1 and digits[0].isdigit() and int(digits[0]) in spoken_number_values(spoken):
                return True
        spelled = "".join(tokens)
        return (
            max(len(tokens), len(candidates)) <= _COMPOUND_MAX_PARTS
            and len(spelled) >= _COMPOUND_MIN_CHARACTERS and not spelled.isdigit()
            and spelled == "".join(candidates)
        )

    support = [[0] * (len(evidence) + 1) for _ in range(len(source) + 1)]
    for row in range(1, len(source) + 1):
        for column in range(1, len(evidence) + 1):
            best = max(
                support[row - 1][column], support[row][column - 1],
                support[row - 1][column - 1] + int(same_word(source[row - 1], evidence[column - 1])),
            )
            for token_count in range(1, min(row, _NUMBER_GROUP_MAX_WORDS) + 1):
                for word_count in range(1, min(column, _NUMBER_GROUP_MAX_WORDS) + 1):
                    if same_group(source[row - token_count:row], evidence[column - word_count:column]):
                        best = max(best, support[row - token_count][column - word_count] + token_count)
            support[row][column] = best
    return support[-1][-1]


def select_cue_word_window(
    cue: Cue,
    words: list[Word],
    *,
    max_word_duration: float,
    max_intra_cue_gap: float,
) -> tuple[list[Word], bool]:
    """Choose the owned words that time a cue; shared by rebuild and refinement.

    Words are grouped at gaps above ``max_intra_cue_gap`` and around impossible
    word durations. The group with the most usable evidence anchors the cue. A
    neighbouring group is kept as well when the cue's own text contains its
    words and the pause is at most ``LEXICAL_BRIDGE_GAP_FACTOR`` gaps long, so
    a cue with a mid-sentence pause still starts on its first spoken word.
    Returns the selected words in time order and whether any word was left out.
    """
    if len(words) <= 1:
        return words, False
    sorted_words = sorted(words, key=lambda word: (word.start, word.end))
    clusters: list[list[Word]] = []
    current: list[Word] = []
    previous: Word | None = None
    for word in sorted_words:
        word_is_outlier = _word_duration(word) > max_word_duration
        starts_new_cluster = False
        if previous is not None:
            starts_new_cluster = word.start - previous.end > max_intra_cue_gap or _word_duration(previous) > max_word_duration
        if word_is_outlier and current:
            clusters.append(current)
            current = []
        if starts_new_cluster and current:
            clusters.append(current)
            current = []
        current.append(word)
        if word_is_outlier:
            clusters.append(current)
            current = []
        previous = word
    if current:
        clusters.append(current)
    if len(clusters) <= 1:
        return sorted_words, False

    supported = _lexically_supported_words(cue, sorted_words)
    # Equal evidence must not pick an arbitrary half: the earliest group wins,
    # because a cue is read from its first words.
    anchor = max(
        range(len(clusters)),
        key=lambda index: (*_cluster_score(clusters[index], supported, max_word_duration), -index),
    )
    def belongs(cluster: list[Word], gap: float) -> bool:
        return gap <= max_intra_cue_gap * LEXICAL_BRIDGE_GAP_FACTOR and all(
            _word_duration(word) <= max_word_duration and id(word) in supported for word in cluster
        )

    first = last = anchor
    while first > 0 and belongs(clusters[first - 1], clusters[first][0].start - clusters[first - 1][-1].end):
        first -= 1
    while last + 1 < len(clusters) and belongs(clusters[last + 1], clusters[last + 1][0].start - clusters[last][-1].end):
        last += 1
    selected = [word for cluster in clusters[first:last + 1] for word in cluster]
    return selected, len(selected) != len(sorted_words)


def _cluster_score(words: list[Word], supported: set[int], max_word_duration: float) -> tuple[int, int, int]:
    normal = [word for word in words if _word_duration(word) <= max_word_duration]
    return sum(1 for word in normal if id(word) in supported), len(normal), len(words)


def _lexically_supported_words(cue: Cue, words: list[Word]) -> set[int]:
    """Identities of the words whose text appears, in order, in the cue's own dialogue."""
    source = alphanumeric_signature(speech_text_for_alignment(cue))
    evidence: list[tuple[str, int]] = [
        (token, id(word)) for word in words for token in alphanumeric_signature(word.text)
    ]
    if not source or not evidence:
        return set()
    # Longest ordered match with the aligner's spelling tolerance, then walk it
    # back to learn which timestamped words took part.
    table = [[0] * (len(evidence) + 1) for _ in range(len(source) + 1)]
    for row, token in enumerate(source, start=1):
        for column, (candidate, _) in enumerate(evidence, start=1):
            matches = token == candidate or fuzz.ratio(token, candidate, score_cutoff=85) >= 85
            table[row][column] = max(
                table[row - 1][column], table[row][column - 1], table[row - 1][column - 1] + int(matches),
            )
    supported: set[int] = set()
    row, column = len(source), len(evidence)
    while row > 0 and column > 0:
        if table[row][column] == table[row - 1][column]:
            row -= 1
        elif table[row][column] == table[row][column - 1]:
            column -= 1
        else:
            supported.add(evidence[column - 1][1])
            row -= 1
            column -= 1
    return supported


def _word_duration(word: Word) -> float:
    return word.end - word.start


def _word_span_label(words: list[Word]) -> str:
    if not words:
        return ""
    return " ".join(f"{word.text}({word.start:.3f}-{word.end:.3f})" for word in words)


def _next_acoustic_start_by_cue(
    cues: list[Cue], timings: dict[int, _CueTiming], preserved_cue_ids: set[int],
) -> dict[int, int]:
    # Insertions are initially placed using source SRT times. Once the source
    # cues are retimed, that list can differ from their actual spoken order.
    # A source timing hold is still a display boundary. Omitting it lets the
    # preceding cue's optional padding extend over the preserved dialogue.
    starts = {
        cue.index: timings[cue.index].start_ms if cue.index in timings else cue.start_ms
        for cue in cues
        if cue.index in timings or (
            cue.index in preserved_cue_ids and not is_bracketed_screen_text_cue(cue)
        )
    }
    timed_cues = sorted(
        (cue for cue in cues if cue.index in starts),
        key=lambda cue: (starts[cue.index], cue.index in preserved_cue_ids),
    )
    return {
        cue.index: starts[following.index]
        for cue, following in zip(timed_cues, timed_cues[1:])
    }


def _extend_into_available_gap(timing: _CueTiming, next_start_ms: int | None, profile: StyleProfile) -> int:
    desired_end_ms = max(timing.end_ms, timing.min_end_ms)
    if next_start_ms is None:
        return desired_end_ms
    cap_ms = next_start_ms if profile.allow_zero_gap else profile.snap_floor(max(0, next_start_ms - 1))
    if cap_ms >= timing.spoken_end_ms:
        # A following actor limits optional display padding too.
        return min(desired_end_ms, cap_ms)
    # The next cue begins before this one's frame-ceiled last word. Words less
    # than a frame apart only collide because the start is floored and the end
    # is ceiled: the shared boundary is the next start. A true word overlap
    # remains intact; readability must never manufacture one.
    snap_slack_ms = profile.frame_ms * (1 if profile.allow_zero_gap else 2)
    if cap_ms > timing.start_ms and cap_ms >= timing.spoken_end_raw_ms - snap_slack_ms:
        return cap_ms
    return timing.spoken_end_ms


def _dominant_speaker(words: list[Word]) -> str | None:
    speakers = [word.speaker_id for word in words if word.speaker_id]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]
