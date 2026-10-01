"""Customer-facing presentation of QC findings.

Emitters keep their kinds and severities: pipeline stages read them as an
internal message bus and many tests pin them. This module runs at report time
only. It sorts every raw finding into one customer bucket:

* ``review``      a human must look (counted, decides the verdict);
* ``changes``     a log of normal successful operations (wording, ad-libs, retimes);
* ``notes``       episode-level information (song captions kept, fps fallback);
* ``diagnostics`` operator-only bookkeeping (word clamps, cost, routing).

Duplicate reports of one root cause collapse into a single item, and every item
names the cue number and timecode of the DELIVERED SRT (the output is
renumbered in time order, so internal cue ids are only a secondary key).
Nothing is dropped: each item lists the raw flag and style-issue indices it
covers, and an unknown kind fails closed into review (or a note when info).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .models import Cue, QCFlag, StyleIssue
from .srt_io import format_timestamp
from .subtitle_annotations import is_bracketed_screen_text_cue
from .tokenize import alphanumeric_signature


Category = Literal["review", "change", "note", "diagnostic"]
Severity = Literal["error", "warning"]
Verdict = Literal["clean", "check", "attention"]

# A job is "attention" (red) when a review item is an error or when more than
# this share of the delivered cues needs a look; "check" (amber) otherwise. The
# floor keeps one overlap in a 13-cue clip from turning the job red.
REVIEW_CUE_RATIO_ATTENTION = 0.10
REVIEW_CUE_COUNT_ATTENTION_FLOOR = 10
# Boundary moves below these are folded into one statistic instead of the change log.
LARGE_START_SHIFT_SECONDS = 0.25
LARGE_END_SHIFT_SECONDS = 0.5
# Short or fast cues are reading-speed information unless they are extreme.
SHORT_CUE_REVIEW_SECONDS = 0.25
FAST_CPS_REVIEW = 35.0
LONG_TRAILING_SILENCE_SECONDS = 0.5
# Text changes that move words across this many delivered cues are also reviewed:
# redistributed wording is where word order breaks (58% human precision on ep11).
REDISTRIBUTED_CHANGE_MIN_CUES = 2

_LYRIC_MARKS = ("♪", "♫")
_ROUTE_TAG = re.compile(r"\[hybrid:([a-z_]+)\]\s*")
_VERDICT_PREFIX = re.compile(r"^Adjudication verdict [a-z_]+:\s*")
_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class KindSpec:
    category: Category
    title: str
    action: str = ""
    priority: int = 50
    severity: Severity | None = None
    episode: bool = False


def _review(title: str, action: str, priority: int, severity: Severity | None = None, *, episode: bool = False) -> KindSpec:
    return KindSpec("review", title, action, priority, severity, episode)


def _change(title: str) -> KindSpec:
    return KindSpec("change", title, priority=90)


def _note(title: str) -> KindSpec:
    return KindSpec("note", title, priority=95)


def _diagnostic(title: str) -> KindSpec:
    return KindSpec("diagnostic", title, priority=99)


_LISTEN = "Listen to this passage and correct the subtitle if needed."

# Default bucket per raw kind. Rules in ``_FindingSorter`` refine the kinds whose
# meaning depends on the cue (song caption, held cue, untouched customer cue).
KIND_REGISTRY: dict[str, KindSpec] = {
    # Problems a human must look at.
    "missing_audio_timing_held": _review(
        "No matching speech in the audio",
        "Check whether the dub omits or replaces these lines; fix or delete the subtitles.", 10, "error"),
    "invalid_cue_duration": _review("Cue has no readable duration", "Retime the cue.", 5, "error"),
    "cue_outside_media": _review("Cue lies outside the audio", "Retime or delete the cue.", 5, "error"),
    "output_order_inversion": _review(
        "Cue order differs from the script order", "Check the word order of these cues against the audio.", 8, "error"),
    "adjudication_span_edit_held": _review(
        "Approved wording could not be applied", "Compare the cue with the audio and type the spoken words.", 12, "error"),
    "editorial_guard_rejected": _review(
        "Wording edit was rejected to protect quotes or styling", _LISTEN, 12, "error"),
    "screen_text_adjudication_held": _review(
        "Spoken change touches on-screen text", "Edit the spoken part without changing the bracketed text.", 12, "error"),
    "timing_evidence_held": _review(
        "Timing could not be confirmed from the audio", "Check the cue timing against the audio; script timing was kept.",
        15, "error"),
    "timing_refinement_held": _review(
        "Timing refinement was not possible", "Check the cue timing against the audio.", 15, "error"),
    "forced_alignment_unresolved": _review(
        "Forced alignment found no unique timing", "Check the cue timing against the audio.", 15, "error"),
    "generated_adlib_rejected_incomplete_source": _review(
        "Speech in the audio is missing from the script", "Listen and add subtitles for the missing speech.", 18, "error"),
    "span_coverage_low": _review(
        "Much of the scripted line was replaced or removed", "Check that no spoken words are missing.", 20),
    "implausible_matched_word_duration": _review(
        "Implausible word timing", "Check the cue timing against the audio.", 25),
    "alignment_outlier": _review(
        "Cue timing disagrees with the rest of the episode", "Check the cue timing against the audio.", 25),
    "adjudication_word_mapping_held": _review(
        "New wording kept at script timing", "Check the cue timing: the approved words could not be located exactly.",
        30, "warning"),
    "adjudication_replacement_ownership_held": _review(
        "Spoken words could not be placed in a cue", "Listen and add the missing words.", 30, "warning"),
    "adlib_removed_without_speech_activity": _review(
        "Added line removed: no speech detected", "Listen at this time and restore the line if it is spoken.",
        32, "warning"),
    "text_redistributed": _review(
        "Wording moved between cues", "Check word order and line breaks across these cues.", 33, "warning"),
    "adjudication_audio_unavailable": _review(
        "AI review had no audio for this passage", _LISTEN, 35, "warning"),
    "low_confidence_adjudication": _review(
        "AI review was not confident; script wording kept", _LISTEN, 35, "warning"),
    "generated_adlib_word_window_refined": _review(
        "Some spoken words were left out of an added line", "Listen and add the missing words.", 35, "warning"),
    "dropped_line_candidate": _review(
        "Line may not be spoken", "Listen; delete the subtitle if the actor dropped the line.", 36, "warning"),
    "missing_audio_source_cue_held": _review(
        "Script wording kept without AI review", _LISTEN, 37, "warning"),
    "unmatched_cue": _review(
        "No spoken words matched this cue", "Check that the line is spoken and timed correctly.", 38, "warning"),
    "speaker_turn_split_held": _review(
        "Speaker change inside a cue", "Check whether the cue should be split between speakers.", 40, "warning"),
    "shared_word_timing_preserved": _review(
        "Cue boundary falls inside one recognised word", "Check the boundary between these cues.", 40, "warning"),
    "generated_adlib_word_mapping_unavailable": _review(
        "Added line timing is approximate", "Check the timing of the added line.", 40, "warning"),
    "cue_without_speech_activity": _review(
        "No speech detected under this cue", _LISTEN, 42, "warning"),
    "cue_on_silence": _review("Cue is on silence", _LISTEN, 42, "warning"),
    "output_overlap_unresolved": _review(
        "Cues overlap", "Trim the earlier cue or merge the two lines.", 45, "warning"),
    "output_overlap_preserved": _review("Cues overlap", "Trim the earlier cue or merge the two lines.", 45, "warning"),
    "overlap_stacked": _review("Cues overlap", "Trim the earlier cue or merge the two lines.", 45, "warning"),
    "overlap_flag_only": _review("Cues overlap", "Trim the earlier cue or merge the two lines.", 45, "warning"),
    "overlap_detected": _review("Overlapping speech detected", "Check both speakers are subtitled.", 45, "warning"),
    "min_duration_unattainable": _review(
        "Very short cue", "Check the cue is readable; merge it with a neighbour if needed.", 50, "warning"),
    "sync_cue_line_limit_markup_unsupported": _review(
        "Cue exceeds the line limit (styled text)", "Split the cue by hand.", 50, "warning"),
    "sync_cue_line_limit_timing_unavailable": _review(
        "Cue exceeds the line limit (no safe split point)", "Split the cue by hand.", 50, "warning"),
    "impossible_cps_fast": _review(
        "Cue reads extremely fast", "Check the timing; merge or shorten if needed.", 55, "warning"),
    "impossible_cps_slow": _review(
        "Cue stays on screen far longer than its text", "Check the cue timing.", 55, "warning"),
    "cue_with_excessive_trailing_silence": _review(
        "Cue stays on screen long after the speech ends", "Check the cue end.", 55, "warning"),
    "name_spelling_inconsistency": _review(
        "Possible name spelling drift", "Check the spelling against the script.", 60, "warning"),
    "unsourced_word_substitution": _review(
        "Word not found in the script", "Check the word against the audio.", 60, "warning"),
    # Episode-level problems: one item per kind.
    "alignment_unresolved": _review(
        "The script could not be aligned to the audio", "Check that the SRT belongs to this audio.", 1, "error",
        episode=True),
    "unresolved_alignment_adjudication_held": _review(
        "Wording differences were not reviewed because alignment failed", _LISTEN, 2, "error", episode=True),
    "alignment_model_unavailable": _review(
        "No stable timing model for this episode", "Spot-check timing across the episode.", 3, "warning",
        episode=True),
    "alignment_anchor_coverage_low": _review(
        "Little of the script matched the audio", "Check that the SRT belongs to this audio.", 3, episode=True),
    "oversized_adjudication_span_held": _review(
        "A long passage was not AI-reviewed", _LISTEN, 4, "error", episode=True),
    "editorial_signature_unexplained": _review(
        "Output has quotes, styling or masks not in the script", "Search the SRT for the added marks.", 6, "error",
        episode=True),
    "divergence_unresolved": _review(
        "Script and audio differences were not AI-reviewed", _LISTEN, 7, "warning", episode=True),
    "llm_provider_unavailable": _review(
        "AI review failed for some passages", _LISTEN, 7, "warning", episode=True),
    "invalid_llm_response": _review(
        "AI review failed for some passages", _LISTEN, 7, "warning", episode=True),
    # Normal successful operations.
    "text_changed": _change("Wording changed to match the audio"),
    "adlib_inserted": _change("Added line spoken in the audio"),
    "adlib_reconciled": _change("Added line matched to a script cue"),
    "dropped_adjudicated_cue": _change("Line removed"),
    "dropped_unmatched_cue": _change("Line removed"),
    "speaker_turn_split": _change("Cue split at a speaker change"),
    "generated_adlib_segmented": _change("Added speech split into cues"),
    "sync_cue_line_limit_split": _change("Cue split to the line limit"),
    "duplicate_cue_merged": _change("Duplicate cue merged"),
    "overlap_dash_merge": _change("Overlapping lines merged into a dash cue"),
    "german_profanity_censored": _change("Profanity masked"),
    "cps_cue_merged": _change("Cues merged for reading speed"),
    "timing_refined": _change("Cue boundary moved to the speech"),
    "forced_alignment_refined": _change("Cue retimed by forced alignment"),
    "media_boundary_clamped": _change("Cue end capped at the end of the audio"),
    "cps_duration_extended": _change("Cue extended for reading speed"),
    "output_overlap_resolved": _change("Overlap resolved"),
    "speaker_transition_gap_inserted": _change("Gap inserted at a speaker change"),
    # Episode-level notes.
    "fps_detection_low_confidence": _note("Frame rate guessed"),
    "fps_override_mismatch": _note("Selected frame rate differs from the script"),
    "punctuation_skipped_for_long_audio": _note("AI punctuation skipped for long audio"),
    "punctuation_provider_unavailable": _note("AI punctuation unavailable; script punctuation kept"),
    "source_out_of_order": _note("Script cues were re-sorted by time"),
    "adlib_rejected_repetitive_content": _note("Repetitive speech outside the script was not subtitled"),
    "adlib_rejected_outside_source_span": _note("Speech outside the script range was not subtitled"),
    # Operator-only bookkeeping.
    "asr_word_clamped": _diagnostic("ASR word end clamped"),
    "asr_timestamp_rounding_clamped": _diagnostic("ASR timestamp rounding clamped"),
    "word_stream_repaired": _diagnostic("ASR word stream repaired"),
    "asr_audio_provenance_unverified": _diagnostic("ASR checkpoint provenance unverified"),
    "hybrid_adjudication_summary": _diagnostic("AI review routing summary"),
    "cost_estimate_uncertain": _diagnostic("Cost estimate uncertain"),
    "cost_unmetered": _diagnostic("Cost not metered"),
    "gemini_audio_context_warning": _diagnostic("Audio context warning"),
    "audio_snippet_unavailable": _diagnostic("Optional audio clip unavailable"),
    "vad_provider_fallback": _diagnostic("Speech detector fallback"),
    "forced_alignment_unavailable": _diagnostic("Forced alignment unavailable"),
    "alignment_band_limited": _diagnostic("Alignment search was bounded"),
    "protected_source_region_held": _diagnostic("Song captions protected"),
    "adlib_speaker_ownership_held": _diagnostic("Added line kept as its own cue"),
    "speaker_turn_word_window_refined": _diagnostic("Speaker-turn word window refined"),
    # 1 of 6 such items was a human fix on ep11 that no other item already covered.
    "timing_outlier_trimmed": _diagnostic("Outlier word ignored for timing"),
    "source_error": _diagnostic("Repeated phrase in adjacent script cues"),
    "invalid_punctuation_change": _diagnostic("Invalid punctuation change reverted"),
    "punctuation_source_structure_preserved": _diagnostic("Punctuation kept script line structure"),
    "speaker_character_mapped": _diagnostic("Speaker mapped to character"),
    "missing_audio_source_cue_restored": _diagnostic("Held cue restored to script"),
    "low_confidence_source_cue_restored": _diagnostic("Held cue restored to script"),
    "timing_evidence_source_cue_restored": _diagnostic("Held cue restored to script"),
    "protected_region_source_cue_restored": _diagnostic("Held cue restored to script"),
    "interpolated_timing": _diagnostic("Interpolated timing (legacy)"),
}

STYLE_REGISTRY: dict[str, KindSpec] = {
    "negative_duration": _review("Cue ends before it starts", "Retime the cue.", 5, "error"),
    "zero_duration": _review("Cue has no display duration", "Retime the cue.", 5, "error"),
    "min_duration": _review(
        "Very short cue", "Check the cue is readable; merge it with a neighbour if needed.", 50, "warning"),
    "frame_grid": _review("Timecode is off the frame grid", "Snap the cue to frames.", 70, "warning"),
    "line_count": _review("Too many lines", "Re-break the cue.", 65, "warning"),
    "line_length": _review("Line too long", "Re-break the cue.", 65, "warning"),
    "overlap": _review("Cues overlap", "Trim the earlier cue or merge the two lines.", 45, "warning"),
}

_OVERLAP_KINDS = frozenset({"output_overlap_unresolved", "output_overlap_preserved", "overlap_stacked", "overlap_flag_only"})
# Follow-on findings that only restate "this held cue has no matching speech".
_HOLD_ECHO_KINDS = frozenset({
    "missing_audio_timing_held", "missing_audio_source_cue_held", "cue_without_speech_activity", "cue_on_silence",
    "dropped_line_candidate", "unmatched_cue", "dropped_unmatched_cue", "low_confidence_adjudication",
    "adjudication_audio_unavailable", "timing_evidence_held", "impossible_cps_fast", "impossible_cps_slow",
    "cue_with_excessive_trailing_silence", "timing_outlier_trimmed", "source_error", "alignment_outlier",
    "implausible_matched_word_duration", "min_duration_unattainable",
    "missing_audio_source_cue_restored", "low_confidence_source_cue_restored",
    "timing_evidence_source_cue_restored", "protected_region_source_cue_restored",
})
_HOLD_ECHO_STYLE = frozenset({"frame_grid", "min_duration", "line_length", "line_count"})
# Speech-absence findings on a song caption are summarised with the song note.
_LYRIC_ABSENCE_KINDS = frozenset({
    "dropped_line_candidate", "cue_without_speech_activity", "cue_on_silence", "unmatched_cue",
})
# Change kinds whose effect is visible as different delivered text.
_TEXT_CHANGE_KINDS = frozenset({
    "text_changed", "adlib_inserted", "adlib_reconciled", "dropped_adjudicated_cue", "dropped_unmatched_cue",
    "speaker_turn_split", "generated_adlib_segmented", "sync_cue_line_limit_split", "duplicate_cue_merged",
    "overlap_dash_merge", "german_profanity_censored", "cps_cue_merged",
})
_TIMING_CHANGE_KINDS = frozenset({
    "timing_refined", "forced_alignment_refined", "media_boundary_clamped", "cps_duration_extended",
    "output_overlap_resolved", "speaker_transition_gap_inserted",
})
_MOVE_THRESHOLD_KINDS = frozenset({"timing_refined", "forced_alignment_refined"})
# Diagnostic buckets that do not correspond to one raw kind.
_DIAGNOSTIC_TITLES = {
    "stale_overlap": "Overlap reported before a later stage separated the cues",
    "style:frame_grid": "Timecode off the frame grid (your timing, or the frame rate was guessed)",
    "style:line_length": "Line longer than the style profile (your text)",
    "style:line_count": "More lines than the style profile (your text)",
    "style:min_duration": "Cue shorter than the style minimum (your timing)",
    "style:overlap": "Cue overlaps the previous cue",
    "low_confidence_adjudication": "Script kept by a pipeline hold (no model proposal)",
    "unmatched_cue": "No ASR words matched, but the cue was retimed to detected speech",
    "cue_without_speech_activity": "Energy detector found little speech under a word-timed cue",
    "cue_on_silence": "Quiet audio under a word-timed cue",
    "alignment_anchor_coverage_low": "Anchor coverage below target (includes unspoken song captions)",
    "missing_audio_source_cue_held": "Held cues were not sent to AI review",
    "impossible_cps_slow": "Slow reading speed on your own timing",
    "cue_with_excessive_trailing_silence": "Cue ends up to 0.5 s after the detected speech",
}


class ReviewItem(BaseModel):
    id: str
    severity: Severity
    kind: str
    reasons: list[str] = Field(default_factory=list)
    title: str
    detail: str
    action: str = ""
    srt_numbers: list[int] = Field(default_factory=list)
    srt_label: str = ""
    after_srt_number: int | None = None
    timecode: str | None = None
    start: float | None = None
    end: float | None = None
    cue_ids: list[int] = Field(default_factory=list)
    text: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    raw_flags: list[int] = Field(default_factory=list)
    raw_style: list[int] = Field(default_factory=list)


class ChangeItem(BaseModel):
    id: str
    change: Literal["edited", "added", "removed", "timing"]
    kind: str
    title: str
    srt_number: int | None = None
    srt_label: str = ""
    after_srt_number: int | None = None
    timecode: str | None = None
    start: float | None = None
    end: float | None = None
    cue_id: int | None = None
    old_text: str | None = None
    new_text: str | None = None
    old_timing: str | None = None
    new_timing: str | None = None
    reason: str | None = None
    route: str | None = None
    confidence: float | None = None
    raw_flags: list[int] = Field(default_factory=list)
    raw_style: list[int] = Field(default_factory=list)


class NoteItem(BaseModel):
    id: str
    kind: str
    title: str
    detail: str
    count: int
    srt_numbers: list[int] = Field(default_factory=list)
    raw_flags: list[int] = Field(default_factory=list)
    raw_style: list[int] = Field(default_factory=list)


class DiagnosticItem(BaseModel):
    id: str
    kind: str
    title: str
    count: int
    severity: Literal["info", "warning", "error"]
    message: str
    cue_ids: list[int] = Field(default_factory=list)
    raw_flags: list[int] = Field(default_factory=list)
    raw_style: list[int] = Field(default_factory=list)


class QCReview(BaseModel):
    verdict: Verdict
    counts: dict[str, int | float]
    review: list[ReviewItem] = Field(default_factory=list)
    changes: list[ChangeItem] = Field(default_factory=list)
    notes: list[NoteItem] = Field(default_factory=list)
    diagnostics: list[DiagnosticItem] = Field(default_factory=list)


def build_review(
    flags: Sequence[QCFlag],
    style_issues: Sequence[StyleIssue],
    cues: Sequence[Cue],
    *,
    source_cues: Sequence[Cue] | None = None,
    summary_metadata: Mapping[str, object] | None = None,
) -> QCReview:
    """Classify raw findings for the customer.

    ``cues`` must be the delivered list in SRT order (``write_srt(renumber=True)``),
    so a cue's delivered number is its position + 1. ``source_cues`` are the
    customer's cues; without them (generate mode) every cue counts as tool-made.
    """

    return _FindingSorter(flags, style_issues, cues, source_cues, summary_metadata or {}).build()


def clean_customer_text(text: str | None) -> str | None:
    """Strip internal routing tags and verdict prefixes from a customer-visible message."""

    if text is None:
        return None
    return " ".join(_VERDICT_PREFIX.sub("", strip_route_tags(text)).split())


def strip_route_tags(text: str) -> str:
    """Remove ``[hybrid:primary]``-style adjudication routing tags."""

    return _ROUTE_TAG.sub("", text)


def srt_label(numbers: Sequence[int]) -> str:
    ordered = sorted(set(numbers))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f"#{ordered[0]}"
    parts = [
        f"#{run[0]}" if len(run) == 1 else f"#{run[0]}–#{run[-1]}"
        for run in _runs(ordered)
    ]
    if len(parts) <= 6:
        return ", ".join(parts)
    return f"{', '.join(parts[:6])} and {len(parts) - 6} more"


@dataclass
class _Candidate:
    kind: str
    severity: Severity
    cue_ids: tuple[int, ...]
    message: str
    start: float | None = None
    end: float | None = None
    old_text: str | None = None
    new_text: str | None = None
    episode: bool = False
    raw_flags: list[int] = field(default_factory=list)
    raw_style: list[int] = field(default_factory=list)


@dataclass
class _Bucket:
    kind: str
    raw_flags: list[int] = field(default_factory=list)
    raw_style: list[int] = field(default_factory=list)
    cue_ids: set[int] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    severity: str = "info"

    def add_flag(self, index: int, flag: QCFlag) -> None:
        self.raw_flags.append(index)
        self.cue_ids.update(flag.cue_ids)
        self.messages.append(flag.message)
        self.severity = _max_severity(self.severity, flag.severity)

    def add_style(self, index: int, issue: StyleIssue) -> None:
        self.raw_style.append(index)
        self.cue_ids.add(issue.cue_id)
        self.messages.append(issue.message)
        self.severity = _max_severity(self.severity, issue.severity)


class _FindingSorter:
    def __init__(
        self,
        flags: Sequence[QCFlag],
        style_issues: Sequence[StyleIssue],
        cues: Sequence[Cue],
        source_cues: Sequence[Cue] | None,
        summary_metadata: Mapping[str, object],
    ) -> None:
        self.flags = list(flags)
        self.style_issues = list(style_issues)
        self.cues = list(cues)
        self.position: dict[int, int] = {}
        for position, cue in enumerate(self.cues):
            self.position.setdefault(cue.index, position)
        self.by_id = {cue.index: cue for cue in reversed(self.cues)}
        self.has_source = source_cues is not None
        self.source = {cue.index: cue for cue in source_cues or []}
        self.fps_confident = summary_metadata.get("fps_detection_confident") is True
        self.held = {
            cue_id for flag in self.flags if flag.kind == "missing_audio_timing_held" for cue_id in flag.cue_ids
        }
        self.candidates: list[_Candidate] = []
        self.hold_raw: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        self.lyric_absent: set[int] = set()
        self.notes: dict[str, _Bucket] = {}
        self.diagnostics: dict[str, _Bucket] = {}
        self.change_flags: list[int] = []
        self.deferred_spelling: list[int] = []
        self.overlaps: dict[tuple[int, int], _Candidate] = {}

    # Cue facts -------------------------------------------------------------

    def is_lyric(self, cue_id: int) -> bool:
        return any(
            mark in cue.text
            for cue in (self.by_id.get(cue_id), self.source.get(cue_id))
            if cue is not None
            for mark in _LYRIC_MARKS
        )

    def tool_timed(self, cue_id: int) -> bool:
        cue = self.by_id.get(cue_id)
        source = self.source.get(cue_id)
        if cue is None:
            return False
        if source is None:
            return True
        return (cue.start_ms, cue.end_ms) != (source.start_ms, source.end_ms)

    def tool_text(self, cue_id: int) -> bool:
        cue = self.by_id.get(cue_id)
        source = self.source.get(cue_id)
        if cue is None:
            return False
        return source is None or cue.plain_text != source.plain_text

    def is_screen_text(self, cue_id: int) -> bool:
        cue = self.by_id.get(cue_id)
        return cue is not None and is_bracketed_screen_text_cue(cue)

    def delivered_ids(self, cue_ids: Iterable[int]) -> list[int]:
        return [cue_id for cue_id in cue_ids if cue_id in self.position]

    def all_held(self, cue_ids: Sequence[int]) -> bool:
        ids = self.delivered_ids(cue_ids) or list(cue_ids)
        return bool(ids) and all(cue_id in self.held for cue_id in ids)

    # Classification --------------------------------------------------------

    def build(self) -> QCReview:
        for index, flag in enumerate(self.flags):
            self._sort_flag(index, flag)
        for index, issue in enumerate(self.style_issues):
            self._sort_style(index, issue)
        hold_candidates, lyric_note = self._hold_runs()
        self.candidates.extend(hold_candidates)
        if lyric_note is not None:
            self.notes["song_lyrics_without_voice"] = lyric_note
        self.candidates.extend(self._real_overlaps())
        changes = self._changes()
        self._sort_spelling(changes)
        review = self._merge_candidates()
        notes = self._finish_notes()
        diagnostics = self._finish_diagnostics()
        counts = self._counts(review, changes, notes, diagnostics)
        return QCReview(
            verdict=_verdict(review, counts),
            counts=counts,
            review=review,
            changes=changes,
            notes=notes,
            diagnostics=diagnostics,
        )

    def _sort_flag(self, index: int, flag: QCFlag) -> None:
        kind = flag.kind
        spec = KIND_REGISTRY.get(kind)
        if spec is None:
            if flag.severity == "info":
                self._note(kind, index, flag)
            else:
                self._candidate(index, flag, kind, flag.severity)
            return
        cue_ids = list(flag.cue_ids)

        if kind == "missing_audio_timing_held":
            for cue_id in cue_ids:
                self._absorb(cue_id, flag=index)
            return
        if kind in _OVERLAP_KINDS:
            self._overlap(index, flag)
            return
        if kind in ("name_spelling_inconsistency", "unsourced_word_substitution"):
            if cue_ids and self.all_held(cue_ids):
                self._absorb(self._first_held(cue_ids), flag=index)
            else:
                # Needs the change log to decide; see _sort_spelling.
                self.deferred_spelling.append(index)
            return
        if cue_ids and kind in _HOLD_ECHO_KINDS and self.all_held(cue_ids) and not self._real_proposal(flag):
            self._absorb(self._first_held(cue_ids), flag=index)
            return
        if cue_ids and kind in _LYRIC_ABSENCE_KINDS and all(self.is_lyric(cue_id) for cue_id in cue_ids):
            self.lyric_absent.update(cue_ids)
            self._absorb(cue_ids[0], flag=index)
            return
        if spec.category == "change":
            # Resolved against the delivered cues in _changes().
            self.change_flags.append(index)
            return
        if spec.category == "note":
            self._note(kind, index, flag)
            return
        if spec.category == "diagnostic":
            self._diagnostic(kind, index, flag)
            return
        self._sort_review_flag(index, flag, spec)

    def _sort_review_flag(self, index: int, flag: QCFlag, spec: KindSpec) -> None:
        kind = flag.kind
        cue_ids = flag.cue_ids
        if kind == "alignment_anchor_coverage_low" and flag.severity != "error":
            # Coverage counts unspoken song captions; only a collapse is actionable.
            self._diagnostic(kind, index, flag)
            return
        if kind == "low_confidence_adjudication" and not self._real_proposal(flag):
            self._diagnostic(kind, index, flag)
            return
        if kind in ("unmatched_cue", "cue_without_speech_activity", "cue_on_silence"):
            # A script cue timed on matched words that the energy detector misses
            # is a fact about the detector; a cue still at script timing, or a
            # line the tool added from ASR alone, needs a listen.
            delivered = self.delivered_ids(cue_ids)
            if not delivered or all(cue_id in self.source and self.tool_timed(cue_id) for cue_id in delivered):
                self._diagnostic(kind, index, flag)
                return
        if kind == "missing_audio_source_cue_held":
            unheld = [cue_id for cue_id in self.delivered_ids(cue_ids) if cue_id not in self.held]
            if not unheld:
                self._diagnostic(kind, index, flag)
                return
            self._candidate(index, flag, kind, _customer_severity(spec, flag), cue_ids=unheld)
            return
        if kind == "min_duration_unattainable":
            duration = _flag_duration(flag, self._delivered_window(cue_ids))
            if duration is None or duration >= SHORT_CUE_REVIEW_SECONDS:
                self._note("short_cues", index, flag)
                return
        if kind == "impossible_cps_fast":
            if (
                (flag.confidence or 0.0) <= FAST_CPS_REVIEW
                or not any(self.tool_timed(cue_id) for cue_id in cue_ids)
                or all(self.is_screen_text(cue_id) for cue_id in cue_ids)
            ):
                self._note("fast_reading_speed", index, flag)
                return
        if kind == "impossible_cps_slow" and not any(self.tool_timed(cue_id) for cue_id in cue_ids):
            self._diagnostic(kind, index, flag)
            return
        if kind == "cue_with_excessive_trailing_silence":
            duration = _flag_duration(flag, None)
            if duration is None or duration <= LONG_TRAILING_SILENCE_SECONDS:
                self._diagnostic(kind, index, flag)
                return
        self._candidate(index, flag, kind, _customer_severity(spec, flag), episode=spec.episode)

    def _sort_style(self, index: int, issue: StyleIssue) -> None:
        key = f"style:{issue.kind}"
        spec = STYLE_REGISTRY.get(issue.kind)
        cue_id = issue.cue_id
        if spec is None:
            self._style_candidate(index, issue, key, issue.severity)
            return
        if issue.kind == "overlap":
            previous = self.position.get(cue_id)
            if previous is None or previous == 0:
                self._diagnostic(key, index, issue=issue)
                return
            pair = (self.cues[previous - 1].index, cue_id)
            self._overlap_pair(pair, style_index=index, start=None, end=None, message=issue.message)
            return
        if cue_id in self.held and issue.kind in _HOLD_ECHO_STYLE:
            self._absorb(cue_id, style=index)
            return
        if issue.kind == "frame_grid":
            if self.tool_timed(cue_id) and self.fps_confident:
                self._style_candidate(index, issue, key, "warning")
            else:
                self._diagnostic(key, index, issue=issue)
            return
        if issue.kind in ("line_length", "line_count"):
            if self.tool_text(cue_id):
                self._style_candidate(index, issue, key, "warning")
            else:
                self._diagnostic(key, index, issue=issue)
            return
        if issue.kind == "min_duration":
            if self.tool_timed(cue_id):
                self._note("short_cues", index, issue=issue)
            else:
                self._diagnostic(key, index, issue=issue)
            return
        self._style_candidate(index, issue, key, spec.severity or issue.severity)

    def _real_proposal(self, flag: QCFlag) -> bool:
        """A low-confidence flag that carries a real model proposal, not a pipeline hold."""

        if flag.kind != "low_confidence_adjudication":
            return False
        if not flag.confidence:
            return False
        message = flag.message
        return "no trustworthy local speech evidence" not in message and "Punctuation/casing-only" not in message

    def _first_held(self, cue_ids: Sequence[int]) -> int:
        return next((cue_id for cue_id in cue_ids if cue_id in self.held), cue_ids[0])

    def _absorb(self, cue_id: int, *, flag: int | None = None, style: int | None = None) -> None:
        raw_flags, raw_style = self.hold_raw[cue_id]
        if flag is not None:
            raw_flags.append(flag)
        if style is not None:
            raw_style.append(style)

    def _candidate(
        self,
        index: int,
        flag: QCFlag,
        kind: str,
        severity: str,
        *,
        cue_ids: Sequence[int] | None = None,
        episode: bool = False,
    ) -> None:
        self.candidates.append(_Candidate(
            kind=kind,
            severity="error" if severity == "error" else "warning",
            cue_ids=tuple(flag.cue_ids if cue_ids is None else cue_ids),
            message=flag.message,
            start=flag.start,
            end=flag.end,
            old_text=flag.old_text,
            new_text=flag.new_text,
            episode=episode,
            raw_flags=[index],
        ))

    def _style_candidate(self, index: int, issue: StyleIssue, kind: str, severity: str) -> None:
        self.candidates.append(_Candidate(
            kind=kind,
            severity="error" if severity == "error" else "warning",
            cue_ids=(issue.cue_id,),
            message=issue.message,
            raw_style=[index],
        ))

    def _note(self, key: str, index: int, flag: QCFlag | None = None, *, issue: StyleIssue | None = None) -> None:
        bucket = self.notes.setdefault(key, _Bucket(key))
        if flag is not None:
            bucket.add_flag(index, flag)
        if issue is not None:
            bucket.add_style(index, issue)

    def _diagnostic(self, key: str, index: int, flag: QCFlag | None = None, *, issue: StyleIssue | None = None) -> None:
        bucket = self.diagnostics.setdefault(key, _Bucket(key))
        if flag is not None:
            bucket.add_flag(index, flag)
        if issue is not None:
            bucket.add_style(index, issue)

    # Overlaps ---------------------------------------------------------------

    def _overlap(self, index: int, flag: QCFlag) -> None:
        ids = list(dict.fromkeys(flag.cue_ids))
        if len(ids) < 2:
            self._candidate(index, flag, "output_overlap_unresolved", "warning")
            return
        pair = tuple(sorted(ids[:2], key=lambda cue_id: self.position.get(cue_id, cue_id)))
        self._overlap_pair(pair, flag_index=index, start=flag.start, end=flag.end, message=flag.message, kind=flag.kind)

    def _overlap_pair(
        self,
        pair: tuple[int, int],
        *,
        flag_index: int | None = None,
        style_index: int | None = None,
        start: float | None,
        end: float | None,
        message: str,
        kind: str = "style:overlap",
    ) -> None:
        candidate = self.overlaps.get(pair)
        if candidate is None:
            candidate = _Candidate(
                kind="output_overlap_unresolved", severity="warning", cue_ids=pair, message=message,
                start=start, end=end,
            )
            self.overlaps[pair] = candidate
        if flag_index is not None:
            candidate.raw_flags.append(flag_index)
        if style_index is not None:
            candidate.raw_style.append(style_index)
        if kind == "output_overlap_unresolved":
            candidate.message = message
        if candidate.start is None and start is not None:
            candidate.start, candidate.end = start, end

    def _real_overlaps(self) -> list[_Candidate]:
        real: list[_Candidate] = []
        for pair, candidate in self.overlaps.items():
            first, second = (self.by_id.get(cue_id) for cue_id in pair)
            if first is not None and second is not None:
                overlap = (min(first.end_ms, second.end_ms) - max(first.start_ms, second.start_ms)) / 1000.0
            elif candidate.start is not None and candidate.end is not None:
                overlap = candidate.end - candidate.start
            else:
                overlap = 1.0
            if overlap <= 0:
                # Reported before a later stage separated the cues.
                bucket = self.diagnostics.setdefault("stale_overlap", _Bucket("stale_overlap"))
                bucket.raw_flags.extend(candidate.raw_flags)
                bucket.raw_style.extend(candidate.raw_style)
                bucket.cue_ids.update(pair)
                bucket.messages.append("Overlap was reported before a later stage separated these cues.")
                bucket.severity = _max_severity(bucket.severity, "warning")
                continue
            # Even one-frame overlaps stay reviewable: the human-corrected
            # episodes have zero overlaps and every delivered one was edited.
            candidate.message = f"The two cues are on screen together for {overlap * 1000:.0f} ms."
            real.append(candidate)
        return real

    # Holds ------------------------------------------------------------------

    def _hold_runs(self) -> tuple[list[_Candidate], _Bucket | None]:
        lyric_cues = {cue_id for cue_id in self.held if self.is_lyric(cue_id)} | self.lyric_absent
        dialogue = sorted(
            (cue_id for cue_id in self.held if cue_id not in lyric_cues),
            key=lambda cue_id: (self.position.get(cue_id, 10**9), cue_id),
        )
        runs: list[list[int]] = []
        for cue_id in dialogue:
            position = self.position.get(cue_id)
            if (
                runs
                and position is not None
                and self.position.get(runs[-1][-1]) is not None
                and position == self.position[runs[-1][-1]] + 1
            ):
                runs[-1].append(cue_id)
            else:
                runs.append([cue_id])

        candidates: list[_Candidate] = []
        for run in runs:
            raw_flags = sorted({index for cue_id in run for index in self.hold_raw[cue_id][0]})
            raw_style = sorted({index for cue_id in run for index in self.hold_raw[cue_id][1]})
            first = self.flags[raw_flags[0]] if raw_flags else None
            candidates.append(_Candidate(
                kind="missing_audio_timing_held",
                severity="error",
                cue_ids=tuple(run),
                message=first.message if first is not None else "",
                start=first.start if first is not None else None,
                end=first.end if first is not None else None,
                raw_flags=raw_flags,
                raw_style=raw_style,
            ))

        stray = [cue_id for cue_id in self.hold_raw if cue_id not in self.held and cue_id not in lyric_cues]
        for cue_id in stray:
            # Absorbed onto a cue that is neither held nor a song caption: keep it
            # reviewable instead of letting it disappear.
            raw_flags, raw_style = self.hold_raw[cue_id]
            first = self.flags[raw_flags[0]] if raw_flags else None
            candidates.append(_Candidate(
                kind=first.kind if first is not None else "missing_audio_timing_held",
                severity="warning",
                cue_ids=(cue_id,),
                message=first.message if first is not None else "",
                start=first.start if first is not None else None,
                end=first.end if first is not None else None,
                raw_flags=list(raw_flags),
                raw_style=list(raw_style),
            ))
        if not lyric_cues:
            return candidates, None
        bucket = _Bucket("song_lyrics_without_voice")
        for cue_id in sorted(lyric_cues):
            raw_flags, raw_style = self.hold_raw.get(cue_id, ([], []))
            bucket.raw_flags.extend(raw_flags)
            bucket.raw_style.extend(raw_style)
        bucket.cue_ids = set(lyric_cues)
        return candidates, bucket

    # Changes ----------------------------------------------------------------

    def _changes(self) -> list[ChangeItem]:
        text_flags: dict[int, list[int]] = defaultdict(list)
        timing_items: list[ChangeItem] = []
        minor_timing: list[int] = []
        for index in self.change_flags:
            flag = self.flags[index]
            if flag.kind in _TIMING_CHANGE_KINDS:
                item = self._timing_change(index, flag)
                if item is None:
                    minor_timing.append(index)
                else:
                    timing_items.append(item)
                continue
            for cue_id in flag.cue_ids or [-1]:
                text_flags[cue_id].append(index)

        items = self._text_changes(text_flags) if self.has_source else self._flag_text_changes(text_flags)
        items.extend(timing_items)
        for index in minor_timing:
            self._note("minor_timing_adjustments", index, self.flags[index])
        items.sort(key=lambda item: (
            item.start if item.start is not None else float("inf"),
            item.srt_number or 0,
            item.change != "removed",
        ))
        for number, item in enumerate(items, start=1):
            item.id = f"C{number}"
        return items

    def _text_changes(self, text_flags: dict[int, list[int]]) -> list[ChangeItem]:
        items: list[ChangeItem] = []
        claimed: set[int] = set()
        for position, cue in enumerate(self.cues):
            source = self.source.get(cue.index)
            if source is not None and source.plain_text == cue.plain_text:
                continue
            raw = sorted(set(text_flags.get(cue.index, [])))
            claimed.update(raw)
            items.append(self._text_item(
                "edited" if source is not None else "added",
                cue.index, raw,
                old_text=source.text if source is not None else None,
                new_text=cue.text,
                start_ms=cue.start_ms, end_ms=cue.end_ms, position=position,
                signature_equal=(
                    source is not None
                    and alphanumeric_signature(source.plain_text) == alphanumeric_signature(cue.plain_text)
                ),
            ))
        for cue_id, source in self.source.items():
            if cue_id in self.position:
                continue
            raw = sorted(set(text_flags.get(cue_id, [])))
            claimed.update(raw)
            items.append(self._text_item(
                "removed", cue_id, raw, old_text=source.text, new_text=None,
                start_ms=source.start_ms, end_ms=source.end_ms, position=None,
            ))
        self._redistributions(items, text_flags)
        for index in {index for raw in text_flags.values() for index in raw} - claimed:
            # The edit was undone later (restored hold, guard) and is not in the delivery.
            self._diagnostic(f"{self.flags[index].kind}:not_delivered", index, self.flags[index])
        return items

    def _flag_text_changes(self, text_flags: dict[int, list[int]]) -> list[ChangeItem]:
        """Generate mode: no script to compare with, so log the change flags themselves."""

        items: list[ChangeItem] = []
        seen: set[int] = set()
        for raw in text_flags.values():
            for index in raw:
                if index in seen:
                    continue
                seen.add(index)
                flag = self.flags[index]
                cue_ids = self.delivered_ids(flag.cue_ids)
                window = self._delivered_window(cue_ids)
                start_ms, end_ms = window if window is not None else _flag_window_ms(flag)
                items.append(self._text_item(
                    "added" if flag.old_text is None else "edited",
                    cue_ids[0] if cue_ids else (flag.cue_ids[0] if flag.cue_ids else None),
                    [index],
                    old_text=flag.old_text, new_text=flag.new_text,
                    start_ms=start_ms, end_ms=end_ms,
                    position=self.position.get(cue_ids[0]) if cue_ids else None,
                ))
        return items

    def _text_item(
        self,
        change: Literal["edited", "added", "removed"],
        cue_id: int | None,
        raw: list[int],
        *,
        old_text: str | None,
        new_text: str | None,
        start_ms: int | None,
        end_ms: int | None,
        position: int | None,
        signature_equal: bool = False,
    ) -> ChangeItem:
        flags = [self.flags[index] for index in raw]
        primary = next((flag for flag in flags if flag.kind in ("text_changed", "adlib_inserted")), None)
        primary = primary or (flags[0] if flags else None)
        if primary is not None:
            kind = primary.kind
            title = KIND_REGISTRY[kind].title if kind in KIND_REGISTRY else kind
            reason = clean_customer_text(primary.message)
            route_match = _ROUTE_TAG.search(primary.message)
            route = route_match.group(1) if route_match else None
            confidence = primary.confidence if kind in ("text_changed", "adlib_inserted") else None
        else:
            kind = "text_changed"
            title = "Punctuation or capitalisation changed" if signature_equal else "Wording changed"
            reason, route, confidence = None, None, None
        if change == "removed" and primary is None:
            title = "Line removed"
        return ChangeItem(
            id="",
            change=change,
            kind=kind,
            title=title,
            srt_number=position + 1 if position is not None else None,
            srt_label=f"#{position + 1}" if position is not None else "",
            after_srt_number=None if position is not None else self._after_number(start_ms),
            timecode=format_timestamp(start_ms) if start_ms is not None else None,
            start=start_ms / 1000.0 if start_ms is not None else None,
            end=end_ms / 1000.0 if end_ms is not None else None,
            cue_id=cue_id,
            old_text=old_text,
            new_text=new_text,
            reason=reason,
            route=route,
            confidence=confidence,
            raw_flags=raw,
        )

    def _redistributions(self, items: list[ChangeItem], text_flags: dict[int, list[int]]) -> None:
        """Review text edits whose words were spread over several delivered cues."""

        changed = {item.cue_id for item in items if item.change in ("edited", "added")}
        seen: set[int] = set()
        for raw in text_flags.values():
            for index in raw:
                flag = self.flags[index]
                if index in seen or flag.kind != "text_changed":
                    continue
                seen.add(index)
                cue_ids = [cue_id for cue_id in self.delivered_ids(flag.cue_ids) if cue_id in changed]
                if len(cue_ids) < REDISTRIBUTED_CHANGE_MIN_CUES:
                    continue
                self.candidates.append(_Candidate(
                    kind="text_redistributed",
                    severity="warning",
                    cue_ids=tuple(cue_ids),
                    message=clean_customer_text(flag.message) or "",
                    start=flag.start,
                    end=flag.end,
                    old_text=flag.old_text,
                    new_text=flag.new_text,
                    raw_flags=[index],
                ))

    def _timing_change(self, index: int, flag: QCFlag) -> ChangeItem | None:
        old = _parse_seconds_window(flag.old_text)
        new = _parse_seconds_window(flag.new_text)
        if flag.kind in _MOVE_THRESHOLD_KINDS and old is not None and new is not None:
            if (
                abs(new[0] - old[0]) < LARGE_START_SHIFT_SECONDS
                and abs(new[1] - old[1]) < LARGE_END_SHIFT_SECONDS
            ):
                return None
        cue_ids = self.delivered_ids(flag.cue_ids)
        cue_id = cue_ids[0] if cue_ids else (flag.cue_ids[0] if flag.cue_ids else None)
        position = self.position.get(cue_id) if cue_id is not None else None
        cue = self.cues[position] if position is not None else None
        start_ms = cue.start_ms if cue is not None else _flag_window_ms(flag)[0]
        end_ms = cue.end_ms if cue is not None else _flag_window_ms(flag)[1]
        return ChangeItem(
            id="",
            change="timing",
            kind=flag.kind,
            title=KIND_REGISTRY[flag.kind].title,
            srt_number=position + 1 if position is not None else None,
            srt_label=f"#{position + 1}" if position is not None else "",
            after_srt_number=None if position is not None else self._after_number(start_ms),
            timecode=format_timestamp(start_ms) if start_ms is not None else None,
            start=start_ms / 1000.0 if start_ms is not None else None,
            end=end_ms / 1000.0 if end_ms is not None else None,
            cue_id=cue_id,
            old_timing=_timing_label(old) if old is not None else flag.old_text,
            new_timing=_timing_label(new) if new is not None else flag.new_text,
            reason=clean_customer_text(flag.message),
            raw_flags=[index],
        )

    def _sort_spelling(self, changes: list[ChangeItem]) -> None:
        by_cue: dict[int, ChangeItem] = {}
        for item in changes:
            if item.change in ("edited", "added") and item.cue_id is not None:
                by_cue.setdefault(item.cue_id, item)
        for index in self.deferred_spelling:
            flag = self.flags[index]
            logged = next((by_cue[cue_id] for cue_id in flag.cue_ids if cue_id in by_cue), None)
            if logged is not None:
                # The approved wording is already in the change log with its reason.
                logged.raw_flags.append(index)
            else:
                self._candidate(index, flag, flag.kind, "warning")

    # Review items -------------------------------------------------------------

    def _merge_candidates(self) -> list[ReviewItem]:
        cue_scoped = [candidate for candidate in self.candidates if candidate.cue_ids and not candidate.episode]
        parent = list(range(len(cue_scoped)))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        owner: dict[int, int] = {}
        for node, candidate in enumerate(cue_scoped):
            for cue_id in candidate.cue_ids:
                if cue_id in owner:
                    parent[find(node)] = find(owner[cue_id])
                else:
                    owner[cue_id] = node
        groups: dict[int, list[_Candidate]] = defaultdict(list)
        for node, candidate in enumerate(cue_scoped):
            groups[find(node)].append(candidate)

        merged = self._merge_adjacent_runs(list(groups.values()))
        episode_groups: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in self.candidates:
            if candidate.episode:
                episode_groups[candidate.kind].append(candidate)
            elif not candidate.cue_ids:
                merged.append([candidate])
        merged.extend(episode_groups.values())

        items = [self._review_item(group) for group in merged]
        items.sort(key=lambda item: (
            item.start if item.start is not None else -1.0,
            _SEVERITY_RANK[item.severity],
            item.srt_numbers[0] if item.srt_numbers else 0,
        ))
        for number, item in enumerate(items, start=1):
            item.id = f"R{number}"
        return items

    def _merge_adjacent_runs(self, groups: list[list[_Candidate]]) -> list[list[_Candidate]]:
        """One item for a block of consecutive cues with the same root cause (e.g. a countdown)."""

        def span(group: list[_Candidate]) -> tuple[int, int] | None:
            positions = [self.position[cue_id] for candidate in group for cue_id in candidate.cue_ids
                         if cue_id in self.position]
            return (min(positions), max(positions)) if positions else None

        located = sorted(
            ((span(group), group) for group in groups if span(group) is not None),
            key=lambda entry: entry[0],
        )
        unlocated = [group for group in groups if span(group) is None]
        runs: list[tuple[tuple[int, int], list[_Candidate]]] = []
        for bounds, group in located:
            if runs:
                (first, last), previous = runs[-1]
                if bounds[0] == last + 1 and _primary_kind(previous) == _primary_kind(group):
                    runs[-1] = ((first, bounds[1]), previous + group)
                    continue
            runs.append((bounds, group))
        return [group for _, group in runs] + unlocated

    def _review_item(self, group: list[_Candidate]) -> ReviewItem:
        ranked = _ranked(group)
        primary = ranked[0]
        spec = _spec(primary.kind)
        cue_ids = list(dict.fromkeys(cue_id for candidate in group for cue_id in candidate.cue_ids))
        delivered = sorted(self.delivered_ids(cue_ids), key=lambda cue_id: self.position[cue_id])
        numbers = [self.position[cue_id] + 1 for cue_id in delivered]
        raw_flags = sorted({index for candidate in group for index in candidate.raw_flags})
        raw_style = sorted({index for candidate in group for index in candidate.raw_style})
        reasons = list(dict.fromkeys([
            *(candidate.kind for candidate in ranked),
            *(self.flags[index].kind for index in raw_flags),
            *(f"style:{self.style_issues[index].kind}" for index in raw_style),
        ]))
        if delivered:
            start_ms = min(self.by_id[cue_id].start_ms for cue_id in delivered)
            end_ms = max(self.by_id[cue_id].end_ms for cue_id in delivered)
        else:
            starts = [candidate.start for candidate in group if candidate.start is not None]
            ends = [candidate.end for candidate in group if candidate.end is not None]
            start_ms = round(min(starts) * 1000) if starts else None
            end_ms = round(max(ends) * 1000) if ends else None
        detail = self._detail(primary, ranked, delivered, numbers)
        return ReviewItem(
            id="",
            severity="error" if any(candidate.severity == "error" for candidate in group) else "warning",
            kind=primary.kind,
            reasons=reasons,
            title=spec.title,
            detail=detail,
            action=spec.action,
            srt_numbers=numbers,
            srt_label=srt_label(numbers),
            after_srt_number=None if numbers else self._after_number(start_ms),
            timecode=format_timestamp(max(0, start_ms)) if start_ms is not None else None,
            start=start_ms / 1000.0 if start_ms is not None else None,
            end=end_ms / 1000.0 if end_ms is not None else None,
            cue_ids=sorted(cue_ids, key=lambda cue_id: self.position.get(cue_id, 10**9)),
            text=self._item_text(delivered),
            old_text=primary.old_text,
            new_text=primary.new_text,
            raw_flags=raw_flags,
            raw_style=raw_style,
        )

    def _detail(self, primary: _Candidate, ranked: list[_Candidate], delivered: list[int], numbers: list[int]) -> str:
        if primary.kind == "missing_audio_timing_held" and len(delivered) > 1:
            first, last = self.by_id[delivered[0]], self.by_id[delivered[-1]]
            sentence = (
                f"{len(delivered)} cues ({format_timestamp(first.start_ms)}–{format_timestamp(last.end_ms)}) "
                "have no matching speech in the dub audio; your text and timing were kept."
            )
        elif primary.kind == "missing_audio_timing_held":
            sentence = "No matching speech was found in the dub audio; your text and timing were kept."
        elif primary.episode and len(ranked) > 1:
            sentence = f"{len(ranked)} passages: {clean_customer_text(primary.message)}"
        else:
            sentence = clean_customer_text(primary.message) or _spec(primary.kind).title
            same_kind = [candidate for candidate in ranked if candidate.kind == primary.kind]
            if len(same_kind) > 1 and len(delivered) > 1:
                sentence = f"{len(delivered)} cues. {sentence}"
        others = [
            _spec(kind).title
            for kind in dict.fromkeys(candidate.kind for candidate in ranked)
            if kind != primary.kind and _spec(kind).title != _spec(primary.kind).title
        ]
        if others:
            sentence = f"{sentence} Also: {'; '.join(dict.fromkeys(others))}."
        return sentence

    def _item_text(self, delivered: list[int]) -> str | None:
        if not delivered:
            return None
        texts = [self.by_id[cue_id].plain_text for cue_id in delivered]
        if len(texts) <= 3:
            return " / ".join(texts)
        return f"{texts[0]} / … / {texts[-1]}"

    def _after_number(self, start_ms: int | None) -> int | None:
        if start_ms is None:
            return None
        before = [position for position, cue in enumerate(self.cues) if cue.start_ms <= start_ms]
        return before[-1] + 1 if before else None

    def _delivered_window(self, cue_ids: Sequence[int]) -> tuple[int, int] | None:
        delivered = self.delivered_ids(cue_ids)
        if not delivered:
            return None
        return (
            min(self.by_id[cue_id].start_ms for cue_id in delivered),
            max(self.by_id[cue_id].end_ms for cue_id in delivered),
        )

    # Notes and diagnostics ------------------------------------------------------

    def _finish_notes(self) -> list[NoteItem]:
        notes: list[NoteItem] = []
        order = ("song_lyrics_without_voice", "minor_timing_adjustments", "short_cues", "fast_reading_speed")
        keys = sorted(self.notes, key=lambda key: (order.index(key) if key in order else len(order), key))
        for key in keys:
            bucket = self.notes[key]
            numbers = sorted(self.position[cue_id] + 1 for cue_id in bucket.cue_ids if cue_id in self.position)
            title, detail, count = self._note_text(key, bucket, numbers)
            notes.append(NoteItem(
                id=f"N{len(notes) + 1}", kind=key, title=title, detail=detail, count=count,
                srt_numbers=numbers, raw_flags=sorted(set(bucket.raw_flags)), raw_style=sorted(set(bucket.raw_style)),
            ))
        return notes

    def _note_text(self, key: str, bucket: _Bucket, numbers: list[int]) -> tuple[str, str, int]:
        if key == "song_lyrics_without_voice":
            passages = _runs(numbers)
            spans = ", ".join(self._passage_label(run) for run in passages[:12])
            more = f" and {len(passages) - 12} more" if len(passages) > 12 else ""
            count = len(numbers) or len(bucket.cue_ids)
            plural = "passage" if len(passages) == 1 else "passages"
            return (
                "Song captions kept at your timing",
                f"{count} song-lyric captions (♪) in {len(passages)} {plural} have no matching voice in the "
                f"dub audio and were kept at your original text and timing: {spans}{more}.",
                count,
            )
        if key == "minor_timing_adjustments":
            count = len(bucket.raw_flags)
            return (
                "Minor boundary adjustments",
                f"{count} cue boundaries were nudged by less than {int(LARGE_START_SHIFT_SECONDS * 1000)} ms (start) "
                f"or {int(LARGE_END_SHIFT_SECONDS * 1000)} ms (end) to match the detected speech.",
                count,
            )
        if key == "short_cues":
            count = len(numbers) or len(bucket.raw_flags) + len(bucket.raw_style)
            return (
                "Short cues",
                f"{count} cues are shorter than the style minimum because the speech is short and the next line "
                f"follows closely: {srt_label(numbers) or 'see the JSON report'}.",
                count,
            )
        if key == "fast_reading_speed":
            count = len(bucket.raw_flags)
            measured = [
                (self.flags[index].confidence or 0.0, cue_id)
                for index in bucket.raw_flags
                for cue_id in self.flags[index].cue_ids[:1]
                if cue_id in self.position
            ]
            listed = ", ".join(
                f"#{self.position[cue_id] + 1} ({value:.0f} cps)" for value, cue_id in sorted(measured, reverse=True)[:5]
            )
            return (
                "Fast reading speed",
                f"{count} cues read faster than the style limit; they follow the speech. Fastest: {listed or 'n/a'}.",
                count,
            )
        spec = KIND_REGISTRY.get(key)
        message = clean_customer_text(bucket.messages[0]) if bucket.messages else ""
        count = len(bucket.raw_flags) + len(bucket.raw_style)
        title = spec.title if spec is not None else key.replace("_", " ").capitalize()
        detail = message if count == 1 else f"{count} times. {message}"
        return title, detail, count

    def _passage_label(self, run: list[int]) -> str:
        first, last = self.cues[run[0] - 1], self.cues[run[-1] - 1]
        label = srt_label(run)
        return f"{label} ({_short_time(first.start_ms)}–{_short_time(last.end_ms)})"

    def _finish_diagnostics(self) -> list[DiagnosticItem]:
        items: list[DiagnosticItem] = []
        for key in sorted(self.diagnostics):
            bucket = self.diagnostics[key]
            base_kind = key.split(":not_delivered")[0]
            if key.startswith("style:"):
                spec = STYLE_REGISTRY.get(key[len("style:"):])
            else:
                spec = KIND_REGISTRY.get(base_kind)
            title = _DIAGNOSTIC_TITLES.get(key) or (spec.title if spec is not None else key)
            if key.endswith(":not_delivered"):
                title = f"{title} (later undone; not in the delivered SRT)"
            items.append(DiagnosticItem(
                id=f"D{len(items) + 1}",
                kind=key,
                title=title,
                count=len(bucket.raw_flags) + len(bucket.raw_style),
                severity=bucket.severity if bucket.severity in ("info", "warning", "error") else "info",
                message=clean_customer_text(bucket.messages[0]) if bucket.messages else "",
                cue_ids=sorted(bucket.cue_ids),
                raw_flags=sorted(set(bucket.raw_flags)),
                raw_style=sorted(set(bucket.raw_style)),
            ))
        return items

    # Counts -------------------------------------------------------------------

    def _counts(
        self,
        review: list[ReviewItem],
        changes: list[ChangeItem],
        notes: list[NoteItem],
        diagnostics: list[DiagnosticItem],
    ) -> dict[str, int | float]:
        review_cues = {cue_id for item in review for cue_id in item.cue_ids if cue_id in self.position}
        raw = len(self.flags) + len(self.style_issues)
        return {
            "raw_finding_count": raw,
            "review_item_count": len(review),
            "review_error_count": sum(1 for item in review if item.severity == "error"),
            "review_warning_count": sum(1 for item in review if item.severity == "warning"),
            "review_cue_count": len(review_cues),
            "review_cue_ratio": round(len(review_cues) / len(self.cues), 4) if self.cues else 0.0,
            "change_count": len(changes),
            "text_change_count": sum(1 for item in changes if item.change != "timing"),
            "timing_change_count": sum(1 for item in changes if item.change == "timing"),
            "note_count": len(notes),
            "diagnostic_count": sum(item.count for item in diagnostics),
        }


def review_cue_ids(review: QCReview) -> set[int]:
    return {cue_id for item in review.review for cue_id in item.cue_ids}


def _verdict(review: list[ReviewItem], counts: Mapping[str, int | float]) -> Verdict:
    if any(item.severity == "error" for item in review):
        return "attention"
    if (
        int(counts.get("review_cue_count", 0)) >= REVIEW_CUE_COUNT_ATTENTION_FLOOR
        and float(counts.get("review_cue_ratio", 0.0)) > REVIEW_CUE_RATIO_ATTENTION
    ):
        return "attention"
    return "check" if review else "clean"


def _spec(kind: str) -> KindSpec:
    if kind.startswith("style:"):
        return STYLE_REGISTRY.get(kind[len("style:"):], _review(kind, _LISTEN, 80))
    return KIND_REGISTRY.get(kind, _review(kind.replace("_", " ").capitalize(), _LISTEN, 80))


def _priority(kind: str) -> int:
    return _spec(kind).priority


def _ranked(group: Sequence[_Candidate]) -> list[_Candidate]:
    return sorted(group, key=lambda candidate: (_SEVERITY_RANK[candidate.severity], _priority(candidate.kind)))


def _primary_kind(group: Sequence[_Candidate]) -> str:
    return _ranked(group)[0].kind


def _customer_severity(spec: KindSpec, flag: QCFlag) -> str:
    if spec.severity is not None:
        return spec.severity
    return "error" if flag.severity == "error" else "warning"


def _max_severity(left: str, right: str) -> str:
    return left if _SEVERITY_RANK.get(left, 9) <= _SEVERITY_RANK.get(right, 9) else right


def _flag_duration(flag: QCFlag, window_ms: tuple[int, int] | None) -> float | None:
    if flag.start is not None and flag.end is not None:
        return flag.end - flag.start
    if window_ms is not None:
        return (window_ms[1] - window_ms[0]) / 1000.0
    return None


def _flag_window_ms(flag: QCFlag) -> tuple[int | None, int | None]:
    start = round(flag.start * 1000) if flag.start is not None else None
    end = round(flag.end * 1000) if flag.end is not None else None
    if start is not None:
        start = max(0, start)
    if end is not None:
        end = max(0, end)
    return start, end


def _parse_seconds_window(text: str | None) -> tuple[float, float] | None:
    if not text or "-->" not in text:
        return None
    left, _, right = text.partition("-->")
    try:
        return float(left.strip()), float(right.strip())
    except ValueError:
        return None


def _timing_label(window: tuple[float, float]) -> str:
    start, end = (max(0, round(value * 1000)) for value in window)
    return f"{format_timestamp(start)} --> {format_timestamp(end)}"


def _short_time(ms: int) -> str:
    seconds, _ = divmod(max(0, int(ms)), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _runs(numbers: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for number in sorted(set(numbers)):
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return runs
