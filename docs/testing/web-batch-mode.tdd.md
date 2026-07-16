# Web Batch Mode TDD Evidence

Date: 2026-07-16

## Acceptance contract

- Accept 1 to 10 audio files in one browser submission.
- In sync mode, match every audio/SRT pair by the same normalized filename stem.
- Process child jobs strictly one by one while preserving independent status, token, and failure state.
- Name each SRT download from its original audio stem plus `-dubsync-synced.srt`.
- Show the simple pairing instruction `Match names: 001.wav + 001.srt. Up to 10 pairs.`
- Show `Part of Feels Local` and use `rey@feelslocal.com` for contact.

## RED and GREEN record

- RED commit: `4066cb8 test: add batch upload and naming regressions`.
- The initial backend, frontend, and Playwright regressions failed because the web route accepted one job only, the browser stored one token only, and downloads used generic names.
- GREEN adds transactional batch admission, a single-worker sequential batch queue, per-child tokens and recovery, strict normalized-stem pairing, source-derived filenames, and the requested identity copy.
- Review iterations added source-attributed per-child recovery errors, strict single-job multipart parsing, one-at-a-time upload intake, a 512 MiB request limit, predicted PCM reservations, duration/output/workdir ceilings, a 4 GiB retained commitment quota, Unicode key parity, bounded subprocesses, stale-job terminalization, cancellation-safe intake rollback, startup orphan reconciliation, and structurally bounded SRT parsing.

## Final local verification

| Command | Result |
| --- | --- |
| `python -m pytest --cov=dubsync --cov-report=term-missing` | 309 passed, 5 deselected; 85.22% coverage |
| `npm run test:coverage` | 54 passed; 91.14% statements, 94.02% lines |
| `npm run typecheck` | PASS |
| `npm run build` | PASS |
| `npm run test:e2e` | 10 passed |
| `npm audit --omit=dev` | 0 vulnerabilities |
| `python -m pip check` | No broken requirements |

The normal suites use fixture processors and do not spend provider credits. No additional paid-provider smoke was run for this release.
