# Gemini 3.5 Transcribe Web Testing Toggle Evidence

## Source Plan

No plan file was supplied. The journey was derived from the request: as a DubSync operator, I want Gemini 3.5 Transcribe available as an explicit web/production-flow test toggle so I can compare it without replacing the default ASR provider.

## User Journeys

- As a web user, I can see the Gemini 3.5 Transcribe testing toggle only when the server enables it and a Gemini key is configured.
- As a web user, I can opt a sync or audio-to-SRT job into `gemini-3.5-transcribe`, and unchecked jobs keep the default provider.
- As an operator, queued Gemini jobs fail closed if the testing flag is later disabled.

## Task Report

| # | What is guaranteed | Test file or command | Test type | Result | Evidence |
|---|--------------------|----------------------|-----------|--------|----------|
| 1 | The public config hides Gemini testing by default, hides it when the key is missing, and does not expose secret text. | `tests/test_web_app.py` | Integration | PASS | `python -m pytest tests/test_web_app.py tests/test_web_batches.py tests/test_transcription.py tests/test_audio_and_providers.py tests/test_documentation_acceptance.py -q` -> `167 passed` |
| 2 | Single sync, generate, and batch requests accept only the exact `gemini-3.5-transcribe` selection when enabled. | `tests/test_web_app.py`, `tests/test_web_batches.py` | Integration | PASS | Same focused pytest command -> `167 passed` |
| 3 | Gemini web intake rejects files over 1,800 seconds before provider processing. | `tests/test_web_app.py::test_gemini_transcribe_web_intake_enforces_1800_second_per_file_boundary` | Integration | PASS | Same focused pytest command -> `167 passed` |
| 4 | The frontend hides the toggle when unavailable, sends the provider only when checked, and keeps it available for sync, generate, and batch submissions. | `web/src/App.test.tsx` | Unit/UI | PASS | `npm --prefix web test -- src/App.test.tsx` -> `37 passed` |
| 5 | Frontend types and production bundle are valid. | `npm --prefix web run typecheck`; `npm --prefix web run build` | Build | PASS | TypeScript completed; Vite built `web/dist` successfully |
| 6 | The running local production-style app exposes the toggle without leaking key material. | `Invoke-RestMethod http://127.0.0.1:8001/api/config` | Smoke | PASS | `gemini_transcribe_testing_available=True`, `gemini_transcribe_max_audio_seconds=1800`, `HasSecretText=False` |

## Security Notes

- The web API accepts only `default` or `gemini-3.5-transcribe`; arbitrary provider strings are rejected before job creation.
- Gemini web testing requires both `DUBSYNC_ENABLE_GEMINI_TRANSCRIBE_WEB_TESTING=1` and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- Job storage persists only the enum value and SQLite constrains it to `default` or `gemini-3.5-transcribe`.
- The worker rechecks the feature flag before replaying a persisted Gemini job.
- The Gemini adapter continues to require `store: false`, exact model, word timestamps, and uploaded-file cleanup in `finally`.

## Coverage And Known Gaps

- `npm --prefix web audit --audit-level=high` reported `0 vulnerabilities`.
- `python -m pip_audit --format columns` reported existing Python dependency vulnerabilities in the active environment. This web-toggle change did not add dependency files; those findings should be triaged separately before treating the Python environment as production-clean.
- No paid Gemini web job was run in this validation. The local server was started on `http://127.0.0.1:8001` with the feature flag set only for that server process.
