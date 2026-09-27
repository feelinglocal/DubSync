# Portuguese Language Option TDD Evidence

## User journey

As a DubSync web user, I can select Portuguese so that sync and generation jobs send the provider language code `pt`.

## RED

`npm test -- src/App.test.tsx -t "offers Portuguese and submits the pt language code"` executed the new integration test and failed because no accessible `Portuguese` option existed.

Checkpoint: `15c5850 test: define Portuguese language support`.

## GREEN

The language selector now contains `Portuguese` with value `pt`, and the same focused test passes while verifying that submitted `FormData` contains `language=pt`.

Checkpoint: `3bc3e4a feat: add Portuguese language option`.

## Validation

| Guarantee | Evidence | Result |
|---|---|---|
| Portuguese appears and submits `pt` | Focused `App.test.tsx` integration test | PASS |
| Frontend regressions remain green | `npm run test:coverage` | 64 passed; statements 90.89%, branches 87.63%, functions 91.26%, lines 94.12% |
| TypeScript remains valid | `npm run typecheck` | PASS |
| Backend language forwarding remains valid | Focused `test_web_app.py` / `test_web_batches.py` language tests | PASS |
| Production frontend bundle builds | `npm run build` | PASS |
| Running local app serves the new bundle | `/api/health`, `/api/config`, asset-hash comparison, and headless browser selection | Health `ok`; jobs available; `Portuguese` selected as `pt` |

## Secret handling and scope

The attached text file was treated only as credential data. The replacement Gemini key was written only to the Git-ignored local `.env`; it was not added to source, tests, commits, or deployment configuration. No Render or production change was made.
