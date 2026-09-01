# Unfinished Audio Cue Safety — TDD Evidence

## Requirement

When an editor supplies an unfinished dubbed track, a source cue whose speech is
missing must not borrow words or timing from another sentence. Reliable surrounding
dialogue must continue to use acoustic word timing. Source-only gaps and strongly
inconsistent matches are therefore held at their original SRT text and timing.

## User journeys

- A missing middle or trailing cue stays at its exact source timing instead of being
  interpolated between neighboring speech.
- A repeated partial phrase cannot steal the complete word window of a later cue.
- A short or full-sentence match far outside a reliable episode timing model is
  rejected and held for review.
- Valid uniform episode shifts and ordinary spoken-text divergences still synchronize.
- A held cue cannot later be moved by adjudication, forced alignment, VAD refinement,
  overlap merging, or final output cleanup.
- Cached alignment artifacts created before this guard are rejected on resume.

## RED evidence

The first focused run failed five new safeguards: outlier rejection, forced-alignment
protection, VAD protection, final-output protection, and adjudication isolation.

```text
5 failed
```

An adversarial follow-up then exposed and reproduced two more gaps: a complete
three-word phrase matched roughly 47 seconds outside the episode model, and a distant
two-word match with only one trustworthy local anchor.

```text
2 failed
```

Both RED runs occurred before their corresponding production fixes.

## Final verification

```powershell
py -3 -m pytest --cov=src/dubsync --cov-report=term-missing
```

```text
682 passed, 7 deselected
Total coverage: 86.73%
```

The deterministic CLI integration case also proves that a missing middle cue remains
at `00:00:02,000 --> 00:00:03,000`, is recorded in alignment diagnostics, receives a
`keep_srt` decision, and never reaches the LLM adapter.

## Limits

No unfinished customer audio example was supplied, so this run proves the code path
with deterministic SRT, ASR-word, forced-alignment, and VAD fixtures. Playback and
level-matched listening remain required before claiming perceptual delivery quality.
