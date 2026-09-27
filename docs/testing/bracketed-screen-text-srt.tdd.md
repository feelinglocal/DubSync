# Bracketed Screen Text SRT TDD Evidence

## Source

Journeys were derived from a TV-series subtitle requirement for bracketed non-spoken screen text, such as document titles, locations, and character labels.

## User Journeys

- As a DubSync operator, I want bracketed screen-text cues excluded from spoken alignment, so that title cards and document descriptions do not lower alignment confidence or shift dialogue timing.
- As a DubSync operator, I want bracket-only cues preserved in the exported SRT at source timing, so that visual text remains available without being treated as dubbed speech.
- As a DubSync operator, I want mixed cues to keep their original text while aligning only spoken lines, so that a visual annotation does not corrupt precise dialogue timing.

## Task Report

| Behavior | RED Evidence | GREEN Evidence | Guarantee |
| --- | --- | --- | --- |
| Bracketed text ignored for cue-to-word alignment | `py -3 -m pytest tests/test_alignment_and_recue.py -q -k bracketed` failed: `assert 0.4545 == 1.0` for `alignment.anchor_coverage` | `py -3 -m pytest tests/test_alignment_and_recue.py tests/test_pipeline_cli.py -q -k "bracket or screen_text"` passed: `6 passed` | Bracketed screen text is not tokenized as spoken dialogue for alignment. |
| Bracket-only cues keep source timing | Same RED run showed bracket text was counted as unmatched source text | `.venv\Scripts\python.exe -m pytest tests\test_screen_text_isolation.py -q` passed: `20 passed` | Rebuild, VAD refinement, forced alignment, and final overlap cleanup preserve bracket-only visual cues at source timing. |
| Mixed cues keep visual labels while aligning/editing only speech | Initial focused suite failed in adjudication, punctuation, line splitting, timing refinement, forced alignment, speaker mapping, and QC paths | `.venv\Scripts\python.exe -m pytest tests\test_screen_text_isolation.py tests\test_alignment_and_recue.py tests\test_forced_alignment.py tests\test_timing_refinement.py tests\test_output_order.py tests\test_punctuation_pipeline.py tests\test_cue_segmentation.py tests\test_speaker_mapping.py tests\test_pipeline_cli.py -q` passed | Mixed cues such as `[label]` plus spoken text retain the label, newline structure, and display text while speech-only residue drives timing and text decisions. |
| Resume artifacts reject stale pre-feature alignment | The pre-feature `AlignmentResult` schema had no bracket-exclusion provenance, so an old `align.json` could be reused for bracketed files | Regression fixtures now reject empty/mismatched provenance and accept the complete annotated cue-id set | Resuming from adjudicate, rebuild, or verify fails closed when an old alignment artifact lacks matching bracket-exclusion provenance. |
| CLI sync writes clean artifacts | Covered by the added CLI regression after the unit RED | `.venv\Scripts\python.exe -m pytest --cov=dubsync --cov-report=term-missing` passed: `712 passed, 7 deselected`, total coverage `86.96%` | `align.json`, exported SRT, and QC summary exclude bracketed screen text from speech alignment accounting. |

## Coverage

Full backend coverage passed the repository gate:

 ```text
.venv\Scripts\python.exe -m pytest --cov=dubsync --cov-report=term-missing
712 passed, 7 deselected
Total coverage: 86.96%
Required test coverage of 80.0% reached.
```

`git diff --check` passed, with only Git's normal LF-to-CRLF working-copy warnings. Ruff was not installed in the virtual environment; the attempted `.venv\Scripts\python.exe -m ruff check src tests` exited with `No module named ruff`.

## Representative Source Check

A non-mutating check on a representative TV-series subtitle file confirmed that:

- bracket-only descriptions are classified as visual text;
- mixed visual/dialogue cues retain the bracketed line while exposing only spoken residue to alignment;
- in-memory write/parse round-tripping preserves every cue semantically;
- input bytes are not modified.

## Known Gap

The bracket-classification check did not include paired audio. Audio-backed timing behavior is covered by deterministic alignment, VAD, forced-alignment, and pipeline fixtures; perceptual delivery still requires listening.
