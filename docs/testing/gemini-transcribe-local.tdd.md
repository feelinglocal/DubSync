> **Retired:** Gemini 3.5 Transcribe ASR is disabled in CLI, web, UI, and queued jobs. This file is retained only as historical TDD evidence; do not use the setup below. Production transcription uses ElevenLabs Scribe v2.

# Gemini 3.5 Transcribe Local Pilot Evidence

## Scope and sources

Historical context: this retired pilot added Gemini 3.5 Transcribe as an opt-in local comparison provider. It never replaced the ElevenLabs Scribe v2 default and did not change Render or web deployment configuration.

The request and response contract was checked against Google's official documentation:

- https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe
- https://ai.google.dev/gemini-api/docs/transcribe
- https://ai.google.dev/gemini-api/docs/interactions-overview
- https://ai.google.dev/gemini-api/docs/pricing

## TDD evidence

The RED checkpoint is commit `7303850` (`test: define local Gemini transcription contract`). Before implementation, the contract tests failed on missing SDK annotation unwrapping, uploaded-file cleanup, request-limit validation, unknown-confidence handling, and unsafe `word_timestamps` / `store` overrides.

Additional RED hardening checks caught and drove fixes for:

- `google-genai` 2.10.0 silently omitting the new transcription fields; the cloud dependency now requires `google-genai>=2.20,<3`.
- API keys appearing in provider error text and chained tracebacks.
- Audio with an unverifiable duration reaching the paid API.
- A model substitution instead of the requested `gemini-3.5-transcribe` model.
- Gemini provider aliases bypassing cost metering.
- A cost-normalization regression for the existing `whisper-1` alias.

## Archived Verification

The results below describe the retired pilot at its historical commit; they are not current product capability claims.

- Focused Gemini adapter, local routing, CLI integration, artifact, confidence, cost, and documentation tests passed.
- `python -m pytest --cov=dubsync --cov-report=term-missing` passed: 605 passed, 8 deselected, 86.10% total coverage.
- `python -m pip check` passed. `pip-audit` separately reported 29 advisories across 9 packages already installed in the broader local ML/development environment; `google-genai` and its declared direct dependencies were not among the reported packages.
- The opt-in live smoke passed with a real 30-second German WAV and non-empty word annotations.
- A full 49.4-second local episode 032 sync passed with 125 Gemini words, 2 speakers, and a metered cost of `$0.004117`.
- The final live adapter deletes the uploaded Google file in `finally`, rejects `store: true`, redacts the configured key from provider errors, and suppresses an unredacted chained exception.

## Archived 30-second comparison

Both providers processed the same normalized first 30 seconds of episode 032, with `--no-llm`, fresh caches, and separate work directories.

| Measurement | Gemini 3.5 Transcribe | ElevenLabs Scribe v2 |
|---|---:|---:|
| Words | 78 | 78 |
| Detected speakers | 2 | 2 |
| Contiguous speaker turns | 4 | 4 |
| Normalized token edits versus source SRT | 3 / 78 (3.85%) | 3 / 78 (3.85%) |
| Unknown word-confidence values | 78 | 0 |
| Metered ASR cost | `$0.002500` | `$0.001833` |
| Fresh wall time on this machine | 9.796 s | 3.965 s |

The two ASR streams matched exactly on 75 of 78 aligned tokens. Among those matching tokens, their timestamps differed by 74 ms at word starts and 65 ms at word ends on average; median differences were 79 ms and 39 ms respectively. The source SRT is editorial guidance rather than an acoustic ground-truth transcript, so its token edit distance is useful for this pilot but is not a definitive WER score.

Artifacts are ignored local files under:

- `work/asr-compare-20260827/gemini/`
- `work/asr-compare-20260827/scribe/`

## Historical Pilot Guarantees (Superseded)

These rows document the retired adapter's former contract. Current code rejects this ASR provider before any client or upload is created.

| Behavior | Former pilot guarantee |
|---|---|
| Default provider | `provider.yaml` remains ElevenLabs Scribe v2. |
| Local-only enforcement | Gemini ASR is rejected unless the caller explicitly uses `--local`. |
| Exact model | The adapter rejects any model other than `gemini-3.5-transcribe`. |
| API mode | Requests use the Interactions API, uploaded audio, verbatim mode, word timestamps, optional speaker diarization, and `store=False`. |
| Timing contract | Native `word_info` annotations map into DubSync word text, start, end, and speaker fields. |
| Unknown confidence | Missing Gemini confidence remains `None`; downstream scoring marks fully unknown confidence as unscored instead of inventing certainty. |
| Duration safety | Timestamp/diarization requests fail closed when duration is unknown and fail before upload above 1,800 seconds. |
| Cost | Gemini ASR aliases meter at `$0.30/hour`, matching the documented `$0.005/minute` basis. |

## Known gaps

- This is one pilot sample, not evidence that Gemini is already a production upgrade.
- Correct automatic chunking above 30 minutes is intentionally absent. It requires timestamp rebasing, overlap deduplication, speaker stitching, cache design, and cost attribution.
- Gemini's tested word annotations did not include per-word confidence.
- The broader local environment's dependency-audit findings remain outside this scoped provider pilot and should be triaged separately before treating that environment as production-ready.
- Promotion requires a broader, manually reviewed German corpus with acoustic ground truth, especially names, compounds, overlaps, whispers, noise, and multi-speaker scenes.
