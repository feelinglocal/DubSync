from __future__ import annotations

import pytest

from dubsync import aligner
from dubsync.models import Cue, TokenMatch, Word
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import tokenize_cues


def _continuous_clause():
    """Captured 2A Scribe words 229..246; customer times are deliberately late."""
    cues = [
        Cue(index=33, start_ms=44100, end_ms=44933, lines=["半世紀も"]),
        Cue(index=34, start_ms=44933, end_ms=46466, lines=["南山県にいらっしゃり"]),
        Cue(index=35, start_ms=47300, end_ms=48700, lines=["龍神谷に"]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=1, speaker_id="speaker_2")
             for text, start, end in [
        ("半", 43.48, 43.56), ("世", 43.64, 43.78), ("紀", 43.78, 43.88), ("も", 43.88, 44.08),
        ("南", 44.08, 44.26), ("山", 44.26, 44.32), ("研", 44.32, 44.52), ("に", 44.52, 44.54),
        ("い", 44.56, 44.66), ("ら", 44.66, 44.74), ("し", 44.74, 44.82), ("た", 44.82, 44.94),
        ("り", 44.94, 45.16), ("、", 45.16, 45.161), ("龍", 45.32, 45.52), ("神", 45.52, 45.54),
        ("谷", 45.72, 45.76), ("に", 45.86, 46.00),
    ]]
    return cues, words


def test_captured_short_internal_particle_does_not_lock_the_continuous_clause_to_bad_source_time():
    cues, words = _continuous_clause()
    alignment = aligner.align_cues_to_words(cues, words)

    assert 34 not in alignment.diagnostics.missing_audio_cue_ids
    omission = next(span for span in alignment.divergence_spans if span.srt_text == "っ")
    assert (omission.start, omission.end) == (44.08, 45.16)
    assert alignment.cue_word_indices[34] == [4, 5, 7, 8, 9, 10, 12]
    rebuilt, flags = rebuild_cues(cues, words, alignment, StyleProfile(fps=30))
    first, second = rebuilt[1:]
    assert first.lines == cues[1].lines and second.lines == cues[2].lines
    assert abs(first.start_ms - 44080) < 34
    assert first.end_ms >= 45160
    assert first.end_ms <= second.start_ms
    assert second.start_ms <= 45320
    assert not any(flag.kind == "timing_evidence_held" for flag in flags)


@pytest.mark.parametrize("fault", [
    "first_placeholder", "last_placeholder", "all_placeholders", "shared_word", "invalid_internal",
    "nonfinite", "long_word", "large_gap", "low_confidence", "mixed_speaker",
])
def test_internal_short_word_exception_keeps_unreliable_or_shared_clause_windows_closed(fault):
    cues, words = _continuous_clause()
    alignment = aligner.align_cues_to_words(cues, words)
    omission = next(span for span in alignment.divergence_spans if span.srt_text == "っ").model_copy(
        update={"start": 44.74, "end": 44.74},
    )
    matches = list(alignment.token_matches)
    if fault == "first_placeholder":
        words[4] = words[4].model_copy(update={"end": words[4].start + .001})
    elif fault == "last_placeholder":
        words[12] = words[12].model_copy(update={"end": words[12].start + .001})
    elif fault == "all_placeholders":
        for offset, index in enumerate([4, 5, 7, 8, 9, 10, 12]):
            words[index] = words[index].model_copy(update={"start": 44.70 + offset * .002, "end": 44.701 + offset * .002})
    elif fault == "shared_word":
        matches.append(TokenMatch(cue_id=35, srt_token_index=14, asr_word_index=7, score=1))
    elif fault == "invalid_internal":
        words[7] = words[7].model_copy(update={"end": words[7].start})
    elif fault == "nonfinite":
        words[7] = words[7].model_copy(update={"end": float("inf")})
    elif fault == "long_word":
        words[7] = words[7].model_copy(update={"end": words[7].start + 3})
    elif fault == "large_gap":
        words[12] = words[12].model_copy(update={"start": 48, "end": 48.2})
    elif fault == "low_confidence":
        words[7] = words[7].model_copy(update={"confidence": .2})
    elif fault == "mixed_speaker":
        words[7] = words[7].model_copy(update={"speaker_id": "speaker_8"})

    assert aligner._mostly_matched_omission_windows([omission], matches, cues, tokenize_cues(cues), words) == [omission]
