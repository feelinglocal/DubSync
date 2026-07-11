# Commercial web MVP TDD evidence

Date: 2026-07-11

This record covers the new Audio-to-SRT workflow, commercial FastAPI surface, browser job recovery, and retention cleanup. Paid-provider smoke tests are deliberately excluded from normal automated runs.

## Audio-to-SRT core

RED:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_transcription.py -q
```

The initial tests failed because `dubsync.transcription` and the `generate` CLI command did not exist.

GREEN:

- Added deterministic cue grouping by silence, known speaker change, duration, sentence ending, and line capacity.
- Added frame-snapped SRT generation, fixture/live ASR integration, guarded punctuation, QC artifacts, and cost artifacts.
- Added `dubsync generate`.
- Focused result: 3 tests passed.

## Web API and job service

RED:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_web_app.py -q
```

The initial tests failed because the web package, job API, protected downloads, persistence, and cleanup behavior did not exist.

GREEN:

- Added streamed and bounded multipart uploads.
- Added secret job tokens with only hashes persisted.
- Added generic unauthorized/expired lookup behavior.
- Added SQLite-backed status, restart recovery, protected downloads, and independent cleanup.
- Added fixture-backed generate processing through the real API route.
- Focused result: 4 tests passed.

## Frontend recovery and responsive workflow

RED:

```powershell
Set-Location web
npm test
```

The refresh-recovery test initially failed because active job credentials were not restored after a component remount. The responsive end-to-end test also established the mobile overflow and next-section viewport requirements before final styling.

GREEN:

- Added tab-scoped active-job persistence and cleanup.
- Added responsive Sync and Audio-to-SRT workspace behavior.
- Added direct legal routes and fixture-backed upload/process/download Playwright coverage.
- Focused component result: 3 tests passed.
- Focused Playwright result: 3 tests passed.

## Release hardening

RED:

- Audio preview replacement retained a revoked object URL when the selected file changed.
- The web language selector was persisted but not forwarded to ElevenLabs Scribe.
- Direct local web startup did not load the current workspace `.env`.

GREEN:

- Object URLs are now created and revoked per selected file.
- Explicit ISO language selections are forwarded to Scribe and included in the ASR cache configuration; `auto` keeps provider detection.
- `WebSettings.from_env()` loads `.env` from the current working directory without overriding existing environment variables.
- Unicode ellipsis cue splitting is explicitly covered as a regression guard.

## Final suite

Final release results:

- Python: 195 passed, 5 paid/live tests deselected, 84.73% branch coverage.
- Frontend: 18 passed; 92.63% statements, 90.41% branches, 92.45% functions, 95.78% lines.
- Playwright: 4 passed, covering generate, sync, token protection, refresh recovery, legal routes, and mobile overflow.
- TypeScript typecheck and Vite production build: passed.
- npm audit: 0 vulnerabilities.
- Python dependency check: no broken requirements.
- Render Blueprint: valid against `https://render.com/schema/render.yaml.json`.

The repository-level coverage threshold remains 80%; it was not reduced.
