# MAI-Transcribe 2 integration and Scribe v2 comparison

At the September 5 checkpoint, MAI-Transcribe 2 via OpenRouter was the local default for new DubSync jobs, with ElevenLabs Scribe v2 retained in the model picker. The September 10 update makes Scribe v2 the default and retains MAI for single-file and batch Sync/Generate workflows. The measurements below describe the original local benchmark before deployment.

## Matched German dialogue benchmark

Three complete existing clips total 348.81875 seconds (5m48.82s): fantasy dialogue, relationship drama, and customer-corrected dialogue. Each model processed each clip three times, for nine calls per model. Requests alternated model order, used the same SHA-256-verified 16 kHz mono PCM WAV files, German language hint, diarization enabled, and no terminology hints. No DubSync transcription cache or LLM editing was used. Latency includes client setup, upload, provider processing, and response parsing; normalization is excluded. Repeated audio can be affected by provider warm-up or internal caching, which was not controlled.

| Measure | MAI-Transcribe 2 | Scribe v2 |
|---|---:|---:|
| Successful calls | 9/9 | 9/9 |
| Median request latency | 1.789 s | 6.164 s |
| Observed p95 latency (small sample) | 5.002 s | 9.050 s |
| Total request latency | 22.174 s | 61.448 s |
| Reference word disagreement (WER) | 4.194% | 5.338% |
| Reference character disagreement (CER) | 1.404% | 1.832% |
| Nine-call cost | $0.029167 reported | $0.063950 estimated |

MAI's median was 3.45 times faster and its reported cost was 54.39% below Scribe's estimate on this sample. MAI reported 1,050 billed seconds; repeated source duration was 1,046.45625 seconds. Whole-second provider rounding explains the small difference. Current catalog anchors are $0.10/audio-hour for [MAI on OpenRouter](https://openrouter.ai/microsoft/mai-transcribe-2) and $0.22/audio-hour for [Scribe v2](https://elevenlabs.io/pricing/api). MAI's launch rate is time-limited; provider-reported cost takes precedence over the configurable estimate. Scribe subscription discounts, taxes, credits and invoice adjustments were not verified.

| German clip | Duration | MAI median | Scribe median | MAI reference WER | Scribe reference WER |
|---|---:|---:|---:|---:|---:|
| Fantasy dialogue | 168.919 s | 4.882 s | 8.766 s | 4.803% | 6.550% |
| Relationship drama | 70.100 s | 1.530 s | 5.352 s | 3.353% | 4.536% |
| Customer-corrected dialogue | 109.800 s | 1.732 s | 6.164 s | 4.206% | 4.673% |

The references contain 612 unique words, not 1,836 independent words: repetitions multiply the scoring denominator. Neither model was fully deterministic. Raw data and exact per-clip metrics are in `work/mai-scribe-benchmark-20260905/paired/benchmark.json`; normalized audio, per-call words, usage and generation IDs remain in that ignored directory.

## What the text scores establish

These are agreement scores against supplied source/customer SRTs, not independently verified recognition accuracy. No new full audition or verbatim ground truth was created. Both models can disagree with a reference that paraphrases or differs from the performed audio. Word timing and speaker-label checks were structurally valid for all German calls, but no human word-timing or speaker reference was available to measure timing error or diarization accuracy.

Normalization removes subtitle markup, bracket annotations, uppercase speaker labels and punctuation, folds case, removes internal apostrophes, converts hyphens to spaces, and leaves numbers as written. Number and compound formatting affect WER: fantasy's four extra Scribe edits per run come from `100` versus `hundert` and hyphenated versus joined compound spelling. The relationship clip includes `geriet ich` versus Scribe's `riet ich`; a character name also varies. In the customer clip, MAI's `Raphael` differs from reference/Scribe `Rafael`, and Scribe has the lower CER despite its higher WER. These examples identify review points without declaring which spelling or wording is audible.

## Additional Portuguese long-audio check

A separate 328.67-second excerpt from the existing long-form Portuguese sample exercised MAI's two-request chunk path. The source SRT includes opening song/credits and is unsuitable as unquestioned verbatim ground truth. One final paired run completed successfully:

| Measure | MAI-Transcribe 2 | Scribe v2 |
|---|---:|---:|
| Request latency | 6.351 s | 7.581 s |
| Reference WER | 36.086% | 35.168% |
| Reference CER | 30.052% | 29.728% |
| Cost | $0.009194 reported | $0.020085 estimated |

Scribe was slightly closer to this reference. The large disagreement for both models prevents an overall accuracy-winner claim. This is one stress sample, not a reliable Portuguese latency distribution.

The first long MAI attempt exposed a final timestamp of 29.68 seconds on a 29.67-second chunk. The same 10 ms overrun appeared in a diagnostic replay. DubSync now clamps only endpoint rounding of at most 20 ms to the known audio boundary and records the original timing in an informational QC flag. Larger overruns, missing timestamps and starts at or beyond EOF still fail. The failed attempt and diagnostic responses remain preserved separately from the successful final run in `long-form/`, `long-form-diagnostic/`, and `long-form-final/` under the evidence directory. Their extra diagnostic charges are excluded from the comparison tables.

## Integration and validation

- OpenRouter uses the dedicated [audio transcription endpoint](https://openrouter.ai/docs/guides/overview/multimodal/stt), with word timestamps, diarization and explicit verbatim style. Phrase hints use Azure's documented `phraseList` option.
- Fixed endpoint, disabled redirects, bounded request/response sizes, sanitized errors, and server-only credentials. The supplied key is configured in ignored local `.env`; its prior content was preserved in an ignored backup. No credentials are included in code, browser configuration, or this report.
- Long WAVs stream in five-minute chunks with one-second context on either side. Overlapping lexical/timing records are matched before ownership to avoid duplicate or dropped words from small timestamp differences. Original provider timing is retained except the explicitly flagged endpoint rounding. Speaker IDs remain scoped to each independent request; cross-chunk speaker identity is not inferred.
- Model selection persists per job and batch child. A transactional SQLite migration preserves rows, indexes, triggers, views, and foreign keys. Historical `default` jobs retain Scribe behavior. Unknown and retired models are rejected; no silent provider fallback occurs.
- Successful OpenRouter charges are marked `audio_billed`; known charges from uncertain partial failures use `audio_billed_partial`. Sanitized failed ASR usage and costs persist in `asr_failure.json`. Cache hits preserve provenance and QC flags and incur no new charge.
- Live pipeline smoke: Generate and Sync passed for both models on the same 15-second sample, with real provider calls and LLM editing disabled to isolate transcription. Output SRTs and cost files were inspected.
- Frontend: 91 tests passed, 90.82% statement coverage and 89.47% branch coverage; typecheck and production build passed. All 18 Playwright tests passed, including model selection, uploads/downloads and widths from 320 to 1,440 px. A later text-only availability-label correction passed its six affected UI tests and all seven backend design-contract checks.
- Backend final: **933 tests passed, seven opt-in live tests deselected, 87.66% combined line/branch coverage** against the unchanged 80% minimum. The separate paid comparisons and pipeline smokes above were run explicitly. Two existing optional runtime/deprecation warnings remained; no test failures remained. Exact totals are recorded in `work/mai-scribe-benchmark-20260905/backend-tests.log` and `backend-coverage.json`.
- Independent implementation review found and verified fixes for factory/language composition, chunk-boundary jitter, configured-key availability, failed-request accounting, and bounded timestamp rounding. No remaining material blocker was found. The attached key was absent from all 30 changed/new delivery files checked.

To reproduce, load the server environment without echoing credentials, then run:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_transcription_models.py --manifest work/mai-scribe-benchmark-20260905/manifest.json --workdir work/mai-scribe-benchmark-20260905/paired --repetitions 3
```

Existing matching run evidence is reused without paid calls. Use a fresh ignored workdir for a new benchmark; `--force` explicitly repeats calls and replaces selected run evidence. `--prepare-only` validates and normalizes input without calling providers. No dependencies were added.

## Local server authentication recovery

The first user-submitted local batches failed because the server inherited an outdated `OPENROUTER_API_KEY`. The normal settings loader deliberately preserves deployment environment values over `.env`, whereas the benchmark explicitly loaded the checkout's `.env` first. A read-only OpenRouter key check returned HTTP 401 for the inherited key and HTTP 200 for the supplied local key. No key values are retained in this report.

Local startup now uses `python scripts/run_local.py`, which anchors the working directory to this checkout, loads its `.env` with local precedence, and binds to `127.0.0.1`. Production entry points retain deployment environment precedence. Known provider authentication, configuration, credit and rate-limit failures now produce fixed actionable public messages; arbitrary exception text and upstream response bodies remain private.

After backing up SQLite metadata, the latest nine stored jobs were recovered with their original IDs, tokens and uploads. All nine completed through the normal MAI sync pipeline with the configured LLM stages enabled: **336 cues**, with nonempty SRT, QC JSON and QC HTML artifacts. Every output SRT parsed with positive cue durations and a cue count matching its job record. SHA-256 hashes and per-job results are in `work/mai-local-auth-recovery-20260905/output-verification.json`. MAI reported **$0.026085** in transcription charges for this recovery; that figure excludes the separate LLM stages. The earlier duplicate failed batch was retained without additional paid processing.

The existing browser session recovered all nine downloads after refresh, and a browser-requested SRT download returned HTTP 200. Final local health returned HTTP 200, with no new job failures in the restarted server log. The outputs contain 20 QC errors and 187 QC warnings requiring review; successful processing does not establish subjective timing or editorial quality, and no full audition was performed.

Recovery regression validation: **951 tests passed, seven opt-in live tests deselected, 87.70% combined line/branch coverage**, above the unchanged 80% minimum. The misleading failure messages and missing local launcher were demonstrated before their fixes. Independent review found no remaining material issue in local environment precedence or safe error propagation. Logs and coverage evidence are in `work/mai-local-auth-recovery-20260905/`.
