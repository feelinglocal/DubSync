# Long-Audio Bounded Sync — TDD Evidence

## Requirement

A TV-length episode must complete without materializing every adjudication clip at
once. Audio-backed timing remains the source of truth, bracketed visual text stays
out of speech matching, and cues whose dub audio is missing retain their exact
source text and timing instead of borrowing another sentence.

## Root cause

The 43:33 fixture completed ASR but failed before output because eager extraction of
all adjudication snippets crossed the 32 MiB per-job work-file budget. The failure
was `Audio snippets would exceed the job storage budget`; upload size and ASR were
not the limiting stages.

## RED evidence

Focused tests were written before each production change and reproduced:

- eager snippet extraction exhausting the job storage budget;
- an ASR-only insertion anchored inside a missing-audio cue reaching adjudication;
- a protected cue being moved again during later timing refinement;
- transient snippet/provider failures being cached and reused;
- a Gemini snippet `Path.read_bytes` failure escaping instead of failing closed;
- packed sparse scenes lacking structural scene identity.

The first adversarial RED run contained four failures, followed by two explicit
scene-isolation failures. A final review then added four more failing cases for
cleanup masking and degraded-cache reads. The release review added two failing
punctuation-cache retry cases. Each failed before its corresponding implementation.

## Implementation contract

- `BoundedAudioSnippetBatchSource` creates only the current provider batch under
  `audio-snippets/batch-NNNN`, records hashes and fallback provenance, and removes
  the transient batch after the provider call.
- The maximum episode duration is 90 minutes and the snippet cap matches the
  25-span adjudication batch, so a 45-minute episode does not silently lose all
  audio double-checking.
- Snippet or provider degradation preserves source-backed decisions and is not
  written to the reusable adjudication cache.
- A transient punctuation-provider failure is neither written to nor trusted from
  the reusable punctuation cache, so a recovered provider is retried.
- Packed adjudication and punctuation payloads include explicit scene ID and
  position metadata, preventing evidence from an unrelated scene being treated as
  local context.
- Missing-audio protection includes source cue IDs and neighboring anchor cue IDs.
  A final restoration pass reapplies the exact source start, end, and text after
  timing refinement.

## Representative Long-Form Fixture

A representative episode-length audio/SRT pair was hashed before and after the run
to confirm the inputs were unchanged. It exceeded 40 minutes and contained hundreds
of subtitle cues, including visual-only and mixed bracketed cues.

The controlled live-provider run completed bounded adjudication batches. Peak
active snippet storage stayed below 4 MiB and well below the configured per-job
budget, while total snippets processed across the run exceeded that budget. No
transient batch directory remained afterward.

The normal local web job path reached `complete` at 100% rather than the former
storage-budget failure. A subsequent current-core rebuild applied the final
missing-audio restoration acceptance checks below.

Acceptance checks passed:

- every detected missing-audio source cue retained exact source timing and text;
- every visual-only bracket cue retained exact source timing and text;
- mixed visual/dialogue cues retained their bracketed lines while only spoken
  residue participated in sync;
- the adversarial stray-word case was restored exactly;
- cue IDs are sequential, timestamps are chronological, durations are positive,
  and the result stays inside media duration;
- bounded audio-only ad-libs were preserved without displacing source cues.

The mechanical report retained a small set of stacked and visual overlaps for
review rather than moving protected source cues. This run does not claim perceptual
approval without level-matched listening.

## Verification

```text
.venv\Scripts\python.exe -m pytest --cov=dubsync --cov-report=term-missing
712 passed, 7 deselected
Total coverage: 86.96%
Required test coverage of 80.0% reached.
```

`python -m compileall -q src` and `git diff --check` also passed. SRT-pair and
audio-energy audits completed successfully; those are mechanical diagnostics, not
substitutes for human listening.
