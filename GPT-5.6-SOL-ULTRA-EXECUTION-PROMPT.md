# GPT-5.6 SOL Ultra execution prompt: finish and release DubSync

Use the following prompt in a new GPT-5.6 SOL Ultra coding session opened at the root of this `SRT Sync` workspace.

This prompt is organized around the desired result, source context, concrete deliverables, boundaries, and a final verification pass, following the OpenAI prompting guidance at https://learn.chatgpt.com/docs/prompting.

---

## Desired result

Finish, harden, and prove the existing DubSync commercial early-access MVP. The result must be a functional, professional web product that:

1. Synchronizes a customer-supplied target-language SRT to dubbed dialogue audio.
2. Generates a timed SRT from dialogue audio when no original SRT exists.
3. Uses ElevenLabs Scribe for acoustic word timestamps and Gemini only for bounded language decisions.
4. Runs as one Dockerized FastAPI/React service on Render with persistent storage.
5. Has no accounts, subscriptions, payment collection, Supabase dependency, or browser subtitle editor in this release.
6. Includes complete landing content, provisional pricing, contact information, Terms, Privacy, secure job access, QC downloads, and 24-hour deletion.
7. Is proven by automated tests, coverage, a production build, Docker/Render validation, and desktop/mobile browser evidence.

Do the work in the repository. Do not stop after proposing a plan. Inspect what is already implemented, preserve correct work, make only grounded changes, and continue until all locally achievable acceptance checks pass.

## Read first

Read these sources in order before editing:

1. `AGENTS.md` and any more-specific repository instructions.
2. `PLAN.md` for the immutable subtitle-engine contract and quality targets.
3. `docs/COMMERCIAL_PLAN.md` for the current product scope, pricing policy, Render architecture, roadmap, and launch gates.
4. `README.md` for setup, provider configuration, and the currently claimed feature surface.
5. `pyproject.toml`, `provider.yaml`, `providers.example.yaml`, `Dockerfile`, and `render.yaml`.
6. `src/dubsync/`, especially `transcription.py` and `web/`.
7. `web/src/`, `web/e2e/`, `tests/test_transcription.py`, and `tests/test_web_app.py`.
8. `docs/design/` for visual direction. Treat concepts as direction, not a reason to overwrite a better working interface.

Use current official provider and Render documentation when an API, price, retention policy, or deployment field may have changed. Do not infer current SDK parameter names or prices from memory.

## Source precedence

Resolve conflicts in this order:

1. Security, privacy, and data-integrity requirements.
2. `PLAN.md` engine truths.
3. `docs/COMMERCIAL_PLAN.md` commercial decisions.
4. Existing tested behavior.
5. This prompt.
6. Visual concepts.

Do not weaken the acoustic-timing contract for convenience.

## Product truths

These constraints are binding:

- Timing comes only from acoustic evidence: ASR word timestamps and optional forced alignment.
- Gemini may adjudicate wording, punctuation, and speaker context, but it must never create or adjust timestamps.
- In sync mode, preserve customer cue text, line breaks, and segmentation wherever the words are unchanged.
- Do not merge speech from two known speakers into one cue.
- Do not cut spoken beginnings or endings merely to satisfy subtitle style. Extend within real speech/gaps and QC-flag conflicts.
- A punctuation pass must reject any alphanumeric word change.
- Flag uncertainty instead of silently guessing.
- Cache paid provider stages by content and configuration so retries do not repeat charges unnecessarily.
- Never expose provider keys, raw job tokens, transcripts, filenames, or provider payloads in general logs.

## Existing implementation to verify

The repository is expected to contain these implemented surfaces. Verify them in the live tree and tests; do not assume this list is accurate:

- `dubsync sync`, `batch`, `profile`, `report`, and `generate` CLI commands.
- Audio-only cue generation using silence, duration, line capacity, sentence endings, and known speaker changes.
- FastAPI endpoints for health, public configuration, job creation, token-protected status, and token-protected downloads.
- Streaming upload limits and strict extension/content-type checks.
- A SQLite job store, one background executor, restart recovery, and independent retention cleanup.
- Secret random job tokens with only hashes persisted.
- A React workspace with Sync and Audio-to-SRT modes, file controls, options, audio preview, status polling, refresh recovery, and result downloads.
- Features, pricing, FAQ, contact, Terms, and Privacy pages.
- A strict Content Security Policy and other security headers.
- A multi-stage Docker image and Render Blueprint with one Starter service, one instance, Singapore region, and a 10 GB persistent disk.
- Fixture-backed unit, API, component, and Playwright tests.

If any claim is absent or broken, either implement it with tests or correct the documentation. Never report a feature merely because it appears in a plan.

## Commercial scope

Keep the first release intentionally small:

- No sign-up or sign-in.
- No subscriptions or recurring billing.
- No automatic payment collection. Prices are early-access/manual quote prices.
- No persistent customer history beyond the 24-hour job window.
- No Supabase.
- No collaboration workspace.
- No browser subtitle editor.

Published provisional prices must stay consistent across API and UI:

| Workflow | Price | Minimum |
|---|---:|---:|
| Audio to SRT | $0.12/min | $3 |
| Sync existing SRT | $0.18/min | $5 |
| Precision processing | $0.25/min | $10 |

Do not present Precision as generally available until a live forced-alignment validation passes. Every completed job must keep `cost.json` as the measured provider-cost source of truth.

## High-value feature order

First make the implemented MVP correct and deployable. Only after all core checks pass, address useful additions in this order, and keep each addition small and tested:

1. Preflight duration and quote: validate media with `ffprobe`, show duration, estimated price, and deletion time before paid processing.
2. Optional names and keyterms: accept a bounded list, pass it to Scribe safely, and include the documented surcharge in the quote/cost record.
3. Optional customer style sample: derive a per-job style profile and show the detected FPS/line constraints before submission.
4. Safe retry: resume from persisted stages without repeating successful paid calls.
5. Batch delivery: isolated matched audio/SRT jobs with a ZIP summary, where one failure does not discard successful results.

Do not implement accounts, payments, or scalable shared infrastructure merely because they are common SaaS features. Add them only when the commercial plan's scaling or revenue triggers are met.

## Backend requirements

- FastAPI owns the same-origin API and serves the built frontend.
- Validate every form field at the API boundary.
- Stream uploads to disk; never read a multi-gigabyte upload fully into memory.
- Do not trust filenames for storage paths.
- Use generic 404 responses for missing, expired, or unauthorized jobs so IDs cannot be enumerated.
- Use constant-time token verification.
- Store only token hashes.
- Recover queued/processing jobs safely after process restart.
- Run retention cleanup independently of incoming requests.
- Keep one worker thread for the disk-backed MVP unless the storage architecture changes.
- Ensure processing failure is visible, actionable, and does not leak secrets.
- Keep generated artifacts within the job directory and prove download paths cannot escape it.
- Keep API responses uncached.
- Health checks must verify the persistent metadata store, not only return a static 200.

## Frontend requirements

Build the actual workspace as the first screen, not a marketing-only hero.

- Use Inter and this palette: deep teal `#042F34`, charcoal teal `#16232B`, mint `#B5F2DB`, pale blue-gray `#E4EEF0`, white `#FFFFFF`, warm yellow `#FFC933`.
- Keep the UI modern, quiet, minimal, and professional, similar in information density and clarity to ElevenLabs without copying it.
- Use familiar Lucide icons for icon controls and accessible labels/tooltips.
- Use compact controls and cards with at most 8 px radius.
- Do not add gradient blobs, ornamental SVGs, oversized marketing type, nested cards, or purple/blue SaaS styling.
- Make both modes obvious and preserve user selections when appropriate.
- Provide clear idle, uploading, queued, processing, completed, expired, failed, and validation states.
- Recover an active job after refresh without persisting its bearer token beyond the browser tab.
- Keep Terms, Privacy, pricing, FAQ, and contact reachable from normal navigation.
- Use `rey@feelslocal.com` for product contact.
- At 390 px mobile width, there must be no horizontal scrolling or clipped controls.
- At desktop and mobile first viewport, the app must show the primary workflow and a visible hint of the next content section.
- Do not place visible tutorial prose or keyboard-shortcut instructions inside the product.

## Legal and privacy requirements

- Keep Terms and Privacy readable and linked from the workspace and footer.
- Terms must cover eligibility, customer rights to uploaded media, acceptable use, third-party providers, human review, early-access manual pricing/refunds, intellectual property, disclaimers, liability limits, termination, governing law, changes, and contact.
- Privacy must accurately disclose Render, ElevenLabs, Gemini, local retention, provider retention caveats, security practices, and customer rights/contact.
- Do not claim zero data retention unless the deployed provider account and configuration actually qualify.
- State clearly in release notes that the legal text needs qualified counsel review before paid launch.

## Render architecture

Use Render as the primary backend host.

For the MVP, keep:

- One Docker web service.
- Starter plan.
- Singapore region.
- One instance.
- One 10 GB persistent disk at `/var/data`.
- One background worker thread inside the service.
- SQLite on the disk.
- Health check at `/api/health`.
- Provider credentials as Render secrets.

Do not introduce Supabase. Document the one-instance and deployment-downtime limitation. If sustained concurrency requires scaling, propose Render Postgres plus shared object storage and a durable queue/workflow as a later migration; do not silently change the MVP architecture.

Validate `render.yaml` against the current official Blueprint schema or Render CLI/API. Build and smoke-test the Docker image when Docker is available. If external deployment is blocked by the absence of a real Git remote or Render authorization, prepare everything locally and report the exact blocker. Do not create or mutate external Render resources without explicit approval.

## Work method

1. Inspect the live workspace and current changes before editing. The workspace may contain user work; never reset or revert unrelated files.
2. Write or update a failing test before each behavior change.
3. Implement the smallest change that makes the test pass.
4. Refactor only when it improves the requested surface.
5. Run focused tests after each change.
6. Review every changed file for correctness, security, privacy, and stale documentation.
7. Run the complete acceptance suite.
8. Start the final local server and verify the actual browser workflow.

Do not manufacture test evidence, provider results, deployment status, costs, or screenshots. Distinguish fixture-backed proof from paid live-provider proof.

## Acceptance commands

Use the workspace virtual environment when present. On PowerShell, run at least:

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest --cov=dubsync --cov-report=term-missing

Set-Location web
npm audit --audit-level=high
npm run test:coverage
npm run typecheck
npm run build
npm run test:e2e
Set-Location ..
```

Required results:

- All normal Python tests pass; live paid-provider tests remain opt-in.
- Python branch coverage is at least 80%.
- Frontend statement, branch, function, and line coverage are each at least 80% for the tested application surface. Do not lower thresholds to pass.
- TypeScript typechecking and Vite production build pass.
- Playwright covers both workflows, protected status/download behavior, refresh recovery, legal routes, and mobile overflow.
- `npm audit --audit-level=high` reports no high/critical production vulnerability.
- No committed secret-like values appear outside ignored local environment files.
- `render.yaml` validates against current official Render fields.
- The Docker image builds and `/api/health` answers when Docker is available.

Run a fixture-backed browser flow end to end:

1. Open the app at 1440 x 900.
2. Submit an Audio-to-SRT job.
3. Observe queued/processing/completed state.
4. Download and inspect the generated SRT.
5. Refresh during or after the job and verify recovery.
6. Switch to Sync mode and verify the original SRT field and validation.
7. Open Terms and Privacy directly and through navigation.
8. Repeat layout checks at 390 x 844.
9. Confirm no console errors, broken assets, overlap, clipping, or horizontal scroll.
10. Save screenshots as evidence.

If credentials are available, do not spend them automatically. Ask for explicit approval before running paid live-provider jobs. If approved, run exactly one short generate job and one short sync job through the web route, record measured `cost.json`, inspect the SRT/QC outputs, and stop on any privacy or provider-contract uncertainty.

## Security review checklist

Before declaring the work complete, verify:

- No hardcoded secrets or secrets in logs/build assets.
- Upload limits are enforced during streaming.
- File extensions, content types, form options, and empty files are rejected correctly.
- Job IDs alone grant no access.
- Expired jobs grant no access and are deleted by the timer.
- Download paths remain inside the job directory.
- CSP does not require unsafe inline scripts or styles.
- Error responses do not expose paths, provider payloads, or credentials.
- The service runs as a non-root container user.
- SQLite and job directories are writable by that user.
- Rate limiting is documented as single-instance protection, not an abuse-proof distributed control.

Fix critical and high findings before continuing. Record lower-risk launch caveats in `docs/COMMERCIAL_PLAN.md` rather than hiding them.

## Completion report

Your final response must be concise and evidence-based. Include:

1. What was already correct and what you changed.
2. The exact test/build/security results, including counts and coverage.
3. Browser and Docker/Render verification evidence.
4. Any skipped paid-provider checks and why.
5. The exact local URL left running.
6. External launch blockers, especially Git/Render authorization and legal review.
7. The next three commercially useful features, selected from measured need rather than generic SaaS convention.

Do not claim the product is deployed unless a Render service was actually created and its public URL was tested. Do not claim it is ready to accept paid customer media until the paid-provider web smokes, provider data settings, alerts, operator legal details, tax/refund policy, and counsel review are complete.

