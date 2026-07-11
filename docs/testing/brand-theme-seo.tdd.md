# Brand, Theme, and SEO TDD Evidence

Date: 2026-07-11
Branch: `codex/brand-seo-theme`

## Source

No implementation plan file was supplied. User journeys and acceptance criteria were derived from the branding, system theme, logo integration, and SEO goal in this task.

## User journeys

1. As a visitor, I want DubSync to follow my system theme until I choose light or dark mode, so the interface is comfortable and my choice persists.
2. As a visitor, I want a distinctive DubSync mark in the browser and shared site identity, so I can recognize the product at every scale.
3. As a search visitor, I want clear SRT sync, dubbing, audio-to-SRT, auto-captioning, and subtitle-QC language, so I can tell whether the product solves my workflow.
4. As a crawler or link unfurler, I want correct canonical metadata, structured data, crawler files, MIME types, and 404 behavior, so the public site is indexed without duplicate or misleading surfaces.
5. As a legal-page visitor, I want stable direct URLs and canonical redirects, so shared and inbound links do not break.

## RED and GREEN task report

### Initial brand, theme, and crawler contract

- RED commit: `fbcbf6c test: define brand theme and SEO contracts`
- RED evidence:
  - `npm test -- --reporter=verbose src/components/components.test.tsx src/App.test.tsx` ran 21 tests with 4 intended failures for missing theme state and route metadata.
  - `python -m pytest tests/test_web_app.py::test_frontend_serves_crawler_assets_and_rejects_unknown_routes tests/test_web_design_contract.py::test_brand_and_crawler_assets_are_declared_and_shippable -q` failed because crawler assets returned HTML and required files/tags did not exist.
  - `npm run test:e2e -- --grep "brand, theme"` failed because the page title was still `DubSync`.
- GREEN commit: `e7faaa6 feat: add owned brand theme and SEO foundation`
- GREEN evidence:
  - The same focused frontend tests passed 21 of 21.
  - The focused backend and design tests passed 4 of 4.
  - The same Playwright brand/theme scenario passed.

### Raw legal metadata and installed icon hardening

- RED commit: `e0a0099 test: cover legal metadata and installed brand assets`
- RED evidence:
  - Focused pytest failed because raw legal responses lacked `X-Robots-Tag`, route-specific heads, opaque maskable assets, Twitter image alt text, and deterministic font rendering.
- GREEN commit: `89ce468 fix: harden crawler and installed brand surfaces`
- GREEN evidence:
  - Raw `/terms`, `/privacy`, and `/payments` responses now have unique titles, descriptions, canonicals, Open Graph URLs, `noindex, follow`, and no homepage schema.
  - Slash variants return permanent 308 redirects to canonical no-slash URLs.
  - Transparent icons are declared only as `any`; separate opaque padded Apple and maskable PNGs pass dimension and PNG color-type assertions.
  - The social card embeds the bundled Inter WOFF2 and waits for `document.fonts.ready`.

### Static-file containment

- RED commit: `e92247c test: cover crawler symlink containment`
- RED evidence:
  - `python -m pytest tests/test_web_app.py::test_frontend_refuses_allowlisted_file_that_resolves_outside_static_root -q` returned 200 instead of the expected 404 when containment was forced false.
- GREEN commit: `e75a03a fix: enforce crawler asset containment`
- GREEN evidence:
  - The same test passed after the allowlisted-file branch required `_inside(requested_file, static_dir)`.

## Test specification

| # | What is guaranteed | Test file or command | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | System dark mode is used when no override exists | `components.test.tsx: uses the system theme` | Unit | PASS |
| 2 | Manual theme choice updates the document and persists | `components.test.tsx: persists an override` | Unit | PASS |
| 3 | Saved theme overrides the system on reload | `components.test.tsx: restores a saved theme` | Unit | PASS |
| 4 | Header identity uses the owned SVG mark | `components.test.tsx` and `app.spec.ts` | Unit and E2E | PASS |
| 5 | Homepage and legal routes apply correct titles and canonicals | `App.test.tsx` and `test_web_app.py` | Integration | PASS |
| 6 | Raw legal HTML is noindex and omits homepage schema | `test_web_app.py::test_frontend_serves_crawler_assets_and_rejects_unknown_routes` | Integration | PASS |
| 7 | Trailing-slash legal URLs permanently redirect | `test_web_app.py::test_frontend_serves_crawler_assets_and_rejects_unknown_routes` | Integration | PASS |
| 8 | Robots, sitemap, favicon, and brand assets have real MIME types | `test_web_app.py` and `app.spec.ts` | Integration and E2E | PASS |
| 9 | Unknown, hidden, and API-shadowing paths return 404 | `test_web_app.py` and `app.spec.ts` | Security integration | PASS |
| 10 | Allowlisted crawler files cannot resolve outside the static root | `test_web_app.py::test_frontend_refuses_allowlisted_file_that_resolves_outside_static_root` | Security integration | PASS |
| 11 | Metadata, schema, manifest, icons, and social card ship in the build contract | `test_web_design_contract.py` | Contract | PASS |
| 12 | Mobile layout has no horizontal overflow | `app.spec.ts: mobile first viewport` | E2E | PASS |

## Final verification

| Command | Result |
| --- | --- |
| `npm run test:coverage` | 28 passed; 90.74% statements, 87.23% branches, 91.5% functions, 92.79% lines |
| `npm run typecheck` | PASS |
| `npm run build` | PASS |
| `npm run test:e2e` | 8 passed |
| `python -m pytest -q` | 219 passed |
| `npm audit --audit-level=high` | 0 vulnerabilities |
| Isolated `pip-audit` for base plus `cloud` and `web` requirements | No known vulnerabilities |
| Raw built HTTP probe | Home 200/indexable; legal routes 200/noindex; slash route 308; robots and sitemap 200 with correct MIME; unknown route 404 |

## Known gaps and scope notes

- The implementation is committed locally but has not been pushed or deployed. Production remains unchanged until a separate approved publish step.
- No paid provider smoke job was run because branding, theme, and metadata do not change the processing pipeline.
- The broad developer virtual environment contains advisories in optional local-ML packages. The isolated deployed dependency surface is clean and this task did not change Python dependencies.
- Current market research found another active product using the exact `DubSync` name at `dubsync.app`. This is a strategic discoverability risk, not a test failure or legal conclusion.

## Merge evidence

The RED/GREEN sequence is preserved in commits `fbcbf6c`, `e7faaa6`, `e0a0099`, `89ce468`, `e92247c`, and `e75a03a`. If these commits are later squashed, copy this sequence into the pull-request or squash-commit body.
