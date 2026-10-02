# Accuracy upgrade evidence — 2026-10-01

Status: **owner accepted the final v10f local candidate with its documented limits on October 2, 2026**, rebuild policy **35**, on `upgrade/sync-accuracy-2026-10`. The implementation commit is `a29eae4c62c3fb3c76853c78b2f4f9ca8ea38093`; the acceptance documentation commit is recorded in `work/upgrade-20261001/RESUME.md`. No push, merge or deployment is authorized or claimed. Technical checks and native model hearing do not certify subjective quality or every accuracy gate.

## Owner decisions and measurement contract

The October 2 instruction chooses wording that actors actually say, including improvisations and performed register. `adjudication.register_policy: spoken` is the default; explicit `script` and no-LLM preservation remain. The follow-up requires correction of badly timed Japanese customer SRTs and a global maximum of **two physical lines**, with sensible wrapping or cue splitting. Unsupported evidence remains reviewable; ordinary processing and downloads continue.

The earlier ten-overlap exception list is superseded. Final native evidence leaves only two genuine simultaneous-speech pairs, which the original objective permits when flagged once. They do not need an invented sequential boundary. Numerical written-reference WER and timing gates are reported separately and are not silently declared passed under the actor-spoken policy.

After receiving the final files and the explicit WER, timing and readability limits below, the owner replied, **"i accept it for now."** This accepts the current candidate and its stated exceptions: episode 11 WER of 4.56%/4.18% against the original 4.11%/4.15% targets, population-dependent acoustic comparisons, two supplementary 33 ms end-error increases, and the documented caption/lyric reading tradeoffs. The measurements remain unchanged and the unmet numerical targets are not relabeled as passes. This decision closes local upgrade acceptance; it does not authorize a push, merge or deployment, or accept future regressions.

Episode 11/17 references inherit 94–99% of their timestamps from previous outputs. Actually human-edited edges are measured separately; inherited timing is not independent acoustic truth. `new bug/001 fixed customer.srt` is a text reference. Native model factual hearing is identified as model evidence, not personal human audition.

Private inputs and immutable evidence remain outside git under `work/upgrade-20261001/`, with generated caches on A:. Final evidence is `scratch/codex-final-v10f/`; the prior v9, v10b, v10c, v10d and v10e packets remain intact. Promoted/R1g controls used an older matcher with positive unmatched fallback; safe v2 and subsequent controls share a stricter matcher. Unsupported saved answers remain confidence-zero holds.

## R1–R6 disposition

| Package | Implemented behavior | Measured limit |
|---|---|---|
| R1 timing / ASR | MAI default with selectable Scribe; shared word repair/ownership, adaptive VAD, phrase edges, protected caps, punctuation guards, frame-safe export, lyric handling, retries and cost provenance | Sparse, collapsed or separated evidence can remain held |
| R2 wording | Flash 3.8 medium default; optional Lite high + Flash medium; native literal hearing, spoken/script policy, bounded deterministic decisions and per-case cache | Names and uncertain equivalence remain reviewable; hearing is not an accuracy guarantee |
| R3 measurement | Portable manifest-driven corpus/golden/timing tools, immutable source/input/harness bindings and network-denied replay | Historical reuse and fresh native evidence are distinct populations |
| R4 generation / ingestion | Exact ownership, balanced wrapping, silence-only reading-time extension, robust style inference and SRT encoding/parsing | Readability pressure remains; unspecified profanity policy is preserved |
| R5 dual ASR | Optional/off, separate cache/provenance/cost; ordinary secondary wording support leaves primary timing intact | Earlier paired routes had unchanged SRT/WER and 3–5 fewer reviews; no general gain is established |
| R6 reconciliation / layout | Whole-cue/residue hearing, bounded secondary acoustic correction, accepted-anchor omissions, interruption handling, caption composition and global two-line output | Positive native routes need actual complete evidence; fixed offline unanswered cases remain held |

Whole missing-cue removal requires complete anchored audio, clear native absence and no independent speech. Recovered wording uses one bounded speech chain whose internal pauses are strictly below **0.2 s**; the model never supplies timestamps. Secondary acoustic rescue preserves the raw primary words and their unique ownership. Residue edits target exact source tokens. New model questions, native failures and unavailable offline answers cannot borrow unrelated historical approval.

Accepted-anchor omission also checks the current neighboring wording and exact primary/secondary boundary mapping. Both streams must establish the same bounded gap, with no raw VAD activity, foreign owner or competing cue. Complete native clip receipts bind the actual WAV bytes, case key, span, decision and verified adjudication-audio identity. Partial cache refresh retains only verifiable cached receipts; legacy or altered manifests cannot become new proof. Rebuild verifies the bound outcomes and guard sets before reuse.

A retained one-letter accent collision at an edit boundary, such as source `E` versus ASR `É`, can now enter the same complete audio question when it completes its own cue and has exclusive, contiguous acoustic ownership. Matches and raw words remain unchanged. Only the expanded question loses its old cache key. Saved answers bind the full scope, retained matches and decisions; an interrupted refresh cannot promote a partial answer during rebuild or verify. Unaffected legacy cases remain resumable. Across the 13 actual runs, only episode 11 MAI77 and Scribe75 qualify.

The source-pair route requires complete ordered utterances, same/different voice evidence, no intervening speech, no clipping and an exact candidate clip inside wider context. Different voices produce two dash lines with no shared speaker/character label. The collapsed-singleton route requires accepted anchors, native wording and an independent bounded activity interval. Both routes retain raw timestamps for audit.

Semantic output preserves valid authored layout. Overwide dialogue is wrapped at linguistic boundaries; additional cues split at supported owned-word boundaries. Composed captions use separate pages with explicit source lineage. Spoken characters, punctuation, order and ownership remain intact, and every delivered cue has at most two physical lines. Japanese wrapping balances legal kana/punctuation clusters and avoids isolated trailing particles. Episode 11's interruption split preserves source217's entire word set around generated945.

## Concrete audio corrections

- **Japanese 2A Scribe source34/35:** sequential cues at **44.067–45.200 / 45.300–46.000 s**, correcting the false overlap identified by the owner. This is separate from source50 (`三分？`), now **64.367–65.200 s** around VAD **64.375–65.135 s**.
- **Japanese 2B Scribe source17:** **21.700–23.134 s**, covering the final `て` and VAD **21.715–23.075 s**. The erroneous raw primary word104 remains unchanged; exact secondary mapping supplies the bounded correction.
- **Japanese 1B, both models:** verified two-voice phrase/laugh exchange at **41.334–42.534 s**, with two dash lines. The earlier candidate missed a chuckle and was rejected. The v2 candidate contains the complete 17,680-frame PCM interval; previous negative hearings remain preserved.
- **Episode 11 Scribe source601, `É.`:** **1646.967–1647.300 s**, covering VAD **1646.975–1647.245 s**. Raw word2505 remains **1648.630–1648.631 s**. The scoped prompt now explicitly preserves source punctuation for `keep_srt`; two earlier responses containing `É` instead of `É.` were invalid exact-keep answers, not missing-confidence responses.
- **Episode 11 source441, both models:** removed only after complete native absence and matching accepted-anchor/secondary evidence. This removes the false **143 ms** overlap while retaining the raw streams and all other owners.
- **Episode 11 MAI source222:** the old partial question allowed `E é sério isso.` by repeating its retained first word. A fresh complete-scope hearing produces `É sério isso.` once, with the original three word owners. Scribe's corresponding complete-scope hearing also produces `É sério isso.`; its timing and three word owners remain unchanged.
- **Japanese layout:** six orphan lines are reflowed without timing or wording changes. Examples become `大口を叩いて / くれるじゃねぇか` and `テーブルを / ひっくり返したのは`.
- **Japanese 1B MAI QC:** a failed early hearing for source3 remains in raw diagnostics after a later complete native check confirms omission. It no longer asks the customer to fix a cue absent from the output. Mixed, retained, partial, uncertain and unrelated timing warnings remain actionable.

## Execution checks

| Check | Final evidence |
|---|---|
| Backend | **3,690 passed, seven live tests deselected, two existing warnings, 178.89 s; 91.35% combined statement/branch coverage** |
| Frontend | **100 tests in six files, 7.83 s; 91.28% statements / 89.16% branches / 91.77% functions / 93.92% lines**; byte-identical v10 inputs reused |
| Typecheck / build | Passed on the same **54 unchanged web files**, verified against current and final snapshots |
| Playwright | **24 passed in 26.6 s** against the final frozen backend |
| Frozen source and tests | **365 hashes, policy 35; all 340 non-Markdown inputs match the checkout, including all 186 test files** |
| Offline executions | **181 runs**, with final corpus, golden, timing and generation evidence described below |
| Actual exported artifacts | **13 SRTs, 3,400 cues and 134 current review items** inspected; six layout deltas reread, all remaining cue signatures revalidated; zero integrity failures |
| Independent review | Integrated runtime, provenance/cache integration and final four-file source/test delta cleared; unchanged reviewed hashes revalidated |

The **80% application coverage minimum** and stricter frontend thresholds remain unchanged. Intentional test updates preserve selectable providers, explicit script fixtures, strict native transport, invalid-response rejection, bounded timing, complete laughs, accepted anchors, interruption ownership, caption lineage and the two-line rule. The CLI fixture's invalid placeholder WAV was replaced with valid PCM after the new provenance recorder correctly rejected it; all nine original assertions remain. The QC registry scanner follows both conditional result branches without treating a comparison literal as an emitted flag kind.

Existing test expectations intentionally changed in this follow-up:

| Test file | Reason |
|---|---|
| `test_alignment_and_recue.py` | Pins missing-audio guard 8 instead of 7 for the changed whole-clause question scope |
| `test_partial_missing_audio_regressions.py` | Moves the 1/20 ms internal-word cases into positive whole-clause-question tests; raw word times must remain unchanged and unreliable outer anchors still reject |
| `test_pipeline_annotation_composition.py` | Replaces the expected three-line warning with exact two-line composition and stable fresh/cache/rebuild/verify results |
| `test_pipeline_cli.py` | Supplies valid decoded PCM for the new actual-clip provenance check |
| `test_protected_region_adlibs.py` | Permits lyric line reflow while requiring exact wording, timing and every non-layout field |
| `test_qc_review.py` | Registry discovery reads both emitted conditional branches without mistaking the condition's voice label for a flag kind |
| `test_song_caption_guard.py` | Expects the new informational reflow flag while preserving lyric wording/timing and retaining real readability warnings |

Independent reviews cover the integrated runtime, cache/proof boundaries and Japanese/QC changes and the final boundary-anchor correction. Source, tests and documentation are pinned separately where necessary; a green earlier snapshot is not presented as validation of later source. Final documentation receives content/diff checks after runtime freeze.

## Fixed offline measurements

The fixed suite comprises **162 corpus + three golden + four timing + 12 generation runs**. The corpus covers 60 episodes with both models: 120 no-LLM and 42 saved-reply runs. Fresh Japanese acceptance inputs and positive native corrections belong to the separate 13-output live-origin packet.

All **181 executions succeed**. Against v10e, **178 SRTs are byte-identical**. The Scribe episode 11 saved-corpus, golden and timing outputs keep source cues222/223 pending a fresh answer for the expanded boundary question; cue222 ends 4 ms earlier. Their former partial answer cannot authorize the retained first word. These three historical holds are distinct from the separately refreshed actual-native outputs. All 181 QC error identities and cost records remain equal. Four alignment-bound receipt hashes change, while all 72 supplemental question/decision/flag/outcome sets remain identical.

| Matrix group | Errors, safe v2 → final | Held cues | Source-timed dialogue | Fragments |
|---|---:|---:|---:|---:|
| No LLM | 459 → 451 | 440 | 435 | 2 |
| Saved replies | 247 → 287 | 361 | 138 | 10 |

Strict reuse is **1,101 exact / 16 unique-text decisions**, with **331 unmatched holds**. The saved +40 errors versus safe v2 are 44 unavailable-hearing errors and one editorial guard, offset by three removed missing-source errors, one missing-time error and one overlap error. On the original promoted 87-run population, error totals are 206→227 no-LLM and 215→287 saved; against R1g, 231→227 / 245→287. These historical matcher/population differences and explained increases preclude a uniform improvement claim. Exact identities remain in `offline/error-change-accounting.json`.

There are **138 matrix supplemental questions**: 41 confidence-zero replies and 97 pending answers. Golden/timing add 19/22 confidence-zero questions. All **72 receipt artifacts / 179 question occurrences** bind to the actual audio, source, words, regions and rebuild: **82 confidence-zero decisions, 97 pending, zero positive answers**. The fixed suite contains no positive source-pair/singleton/accepted-anchor path and zero ordinary native audio receipts among 3,882 cache entries. It cannot establish fresh hearing or positive cache-provenance behavior.

| Saved-decision stream | Delivered WER / false-edit spans | Speech-only WER / false spans | Human-edited start/end MAE |
|---|---:|---:|---:|
| Episode 11 MAI | 4.91% / 101 | 4.70% / 96 | 65.5 / 84.3 ms |
| Episode 11 Scribe | 4.72% / 94 | 4.53% / 89 | 67.9 / 99.5 ms |
| Episode 17 MAI | 1.56% / 26 | 1.66% / 26 | 183.2 / 74.9 ms |

Safe v2 speech-only WER/false spans were 4.34%/63, 3.87%/52 and 2.18%/25. Delivered scoring includes caption presentation; speech-only scoring removes balanced bracket text consistently from every input. The associated timing populations differ and are not interchangeable.

| Acoustic stream | Starts within 40 ms, raw / effective | Raw end median / mean absolute error |
|---|---:|---:|
| Episode 11 MAI | 88.0% / 91.3% | 64 / 78 ms |
| Episode 11 Scribe | 88.3% / 91.6% | 67 / 90 ms |
| Episode 17 MAI | 90.4% / 93.6% | 60 / 63 ms |
| Testlong Scribe | 84.7% / 89.1% | 60 / 70 ms |

The Scribe boundary hold removes one eligible end from each residual population: raw end n431→430 and within80ms 78.7→78.6%; effective end n672→671 and within80ms 80.1→80.0%. Median/mean errors and start distributions above remain unchanged. This denominator change is not a timing improvement.

The preserved handover artifacts independently reproduce all four original raw-ASR timing results. Pairing the identical audio, cue, boundary word and acoustic burst in both versions finds **1,950 common starts**, all unchanged, and **1,248 common ends**, with 1,247 unchanged and one improved testlong end. Thus the original raw-ASR comparison supports nonregression on common measurable edges. Eligible populations differ: the original/current raw start-within40ms rates are **88.2→88.0%, 88.2→88.3%, 90.0→90.4%, 84.7→84.7%**. The lower first aggregate is not silently called a passed numerical threshold. Supplementary effective-word measurements have two episode 11 MAI end errors that grow by 33 ms (**24→57 ms**, cue61; **47→80 ms**, cue712); both acquire an additional owned word. They are outside the stricter identical-full-ownership cohort. All paired mean absolute errors remain nonworse, but a universal per-edge raw/effective nonregression claim is false. Exact paired and excluded populations are in `review/paired-acoustic-distributions/paired-acoustic-comparison.json`.

Qualified timing has zero measured first-word cuts and short-with-room counts **3/1/0/2**. Seven timing overlaps remain explicitly held, in counts **1/3/0/3**; the broader matrix has 64 overlaps, including nine without that audit's hold tags. These are not the live packet's two simultaneous-speech pairs. All 12 generation cases retain exact one-time ordered ownership, unchanged raw words, zero overlaps/late starts/end clips beyond one frame and zero short cues with verified free room. Each Scribe episode 11 preset retains eight reviews including three errors.

## Actual native runs and exported artifacts

| Output | Cues / reviews | Overlap pairs | Original primary ASR USD | Capture/recovery lineage USD |
|---|---:|---:|---:|---:|
| German MAI | 48 / 4 | 0 | 0.003056 | 0.098286 |
| German Scribe | 48 / 5 | 0 | 0.006710 | 0.126277 |
| Japanese 1A MAI | 60 / 2 | 0 | 0.003278 | 0.451401 |
| Japanese 1A Scribe | 60 / 2 | 0 | 0.007184 | 0.317735 |
| Japanese 1B MAI | 57 / 9 | 0 | 0.003278 | 0.437628 |
| Japanese 1B Scribe | 57 / 5 | 0 | 0.007188 | 0.363319 |
| Japanese 2A MAI | 79 / 3 | 0 | 0.003417 | 0.370505 |
| Japanese 2A Scribe | 79 / 1 | 0 | 0.007474 | 0.384313 |
| Japanese 2B MAI | 79 / 8 | 0 | 0.003417 | 0.232707 |
| Japanese 2B Scribe | 79 / 4 | 0 | 0.007474 | 0.291064 |
| Episode 11 Scribe | 996 / 36 | 1 | 0.181223 | 3.265362 |
| Episode 17 MAI | 752 / 20 | 0 | unknown (reused) | 1.876261 |
| Episode 11 MAI | 1006 / 35 | 1 | 0.082889 | 4.591998 |

The actual packet contains **3,400 cues and 134 review items**, with zero integrity failures and a maximum of two lines. Costs above are recorded historical capture/recovery lineages, not prices for a single production job. Original primary ASR is already included where known. Shared capture ledgers must not be counted twice; the selected packet spans **67 distinct capture/recovery ledgers totaling $12.806856**, a subset of whole-task spend. Secondary ASR dependencies, individual recovery-stage charges, missing historical prices and all ledger hashes are recorded separately in `live/selected-capture-cost-provenance.json`.

All final runs replay exact captured replies with network disabled and **zero new paid calls**. Exported text/timing matches rebuild IDs; raw/effective/VAD identities, unique word ownership, receipt bindings, caption lineage and actual QC references are checked. German, Japanese and episode 11 ASR originated in this session; episode 17 reuses prior HV8 MAI ASR. Fresh native hearing on prior ASR is described separately from fresh transcription.

The only remaining live overlaps are episode 11 MAI IDs **708/984** (233 ms greetings) and Scribe IDs **513/969** (independent simultaneous speech). Native factual evidence supports genuine concurrency; each pair has one grouped customer review. No Japanese or caption-generated overlap remains. Raw boundary candidates are retained with their exact explanation; corrected raw timestamps are never silently relabeled as provider truth.

| Actual native stream | Delivered WER, v9 → final | Speech-only WER, v9 → final | Delivered false spans, v9 → final |
|---|---:|---:|---:|
| Episode 11 MAI | 4.77% → 4.56% | 4.29% → 4.36% | 91 → 95 |
| Episode 11 Scribe | 4.58% → 4.18% | 4.09% → 3.97% | 98 → 92 |
| Episode 17 MAI | 1.76% → 1.51% | 1.60% → 1.60% | 21 → 20 |

MAI's speech-only score regresses **0.07 percentage point**, with false spans 87→90; Scribe false spans fall 94→87 and episode 17 stays 20. Reduced repeated caption text explains part of delivered-WER improvement and is not a wording-accuracy claim. Final delivered human-edited start/end MAEs are **65.5/110.2, 57.1/85.3, 183.2/74.8 ms**, unchanged from v9 in paired populations of **29/52, 28/52 and 4/8**. Speech-only end MAEs are **129.5/104.5/293.1 ms**; episode 17 uses ten paired ends. Original reference hashes are unchanged.

These actual-native metrics are separate from fixed saved-decision results. The original episode 11 written-reference targets are not achieved: **MAI 4.56% versus 4.11%**, and **Scribe 4.18% versus 4.15%**. Increased MAI false-edit spans remain visible. Actor-spoken policy does not constitute a silent numerical waiver.

An independent comparison reproduces the original handover's human-edited-edge measurements and pairs the exact same reference edges in both outputs. All six common-edge MAEs are nonworse: episode 11 MAI starts **67.9→67.9 ms (n28)** and ends **87.5→86.1 ms (n48)**; Scribe starts **91.4→55.5 ms (n27)** and ends **101.4→85.0 ms (n51)**; episode 17 starts **183.2→183.2 ms (n4)** and ends **74.9→74.8 ms (n8)**. No common MAI edge worsens. One Scribe start and two ends worsen individually by 32–33 ms despite the aggregate gain. One prior Scribe end becomes an internal word boundary after a merge, so it is unpaired; the words remain present. Newly paired MAI ends, including a simultaneous greeting, explain the larger unpaired headline end MAE. The aggregate human-edited-edge gate passes on common measurable edges with these coverage limits; this does not establish acoustic-distribution nonregression. Exact pairs and input hashes are in `review/paired-human-edges-final.json`.

## Caption visibility and readability

The two-line ceiling does not prove comfortable reading. Episode 17 caption32 retains **1,040 of 1,900 ms visibility (54.737%)**, starts **860 ms late**, and displays **63 characters at 60.577 continuous CPS**. Its entire wording is preserved. This measured tradeoff recurs in matrix, golden and timing outputs: the global line cap takes precedence over continuous caption coverage.

In the actual episode 11 output, caption478 shows the name for **620 ms**, then the message for **700 ms** after a **620 ms delay**, at **62.86 visual-width units/s**. Its full message is not displayed simultaneously. Caption-page duration and visibility remain part of the QC evidence.

Native evidence at the reviewed episode 17 lyric passage found instrumental audio followed by `Uhum.`, rather than the proposed sung line. The unowned source lyric yields to speech and retains **207 ms** display with an approximately **96.6 CPS** warning. This is a picture-text/minimum-duration tradeoff, not a claimed acoustic correction or readability pass. Final caption-page intervals and warnings remain in the actual artifact audit.

## Remaining measured recommendations

- Partial timing: historical inventory found 112 source-timed cues, 14 with owned words, nine held partial cases and zero sole bursts inside independent anchors. The new bounded positive routes do not establish broad rescue accuracy.
- Transpositions: 20/18/10/9/5 candidate pairs in episode 11 MAI/Scribe, episode 17 MAI, episode 02 Scribe and testlong Scribe. Candidate counts are not actor truth; joint-region/name hints have no measured general adoption gain.
- Full mixes: historical energy activity is 87%, longest region 140 s; genuine chorus stays energetic. Delivery mix is unknown. Spectral/model alternatives remain optional pending matched accuracy and runtime evidence.
- Early starts: 462 raw candidates; actually human-edited edges provide one support, two rejections and 210 unknowns. Five source-timed half-matched cues lack hold flags. These inventories do not validate a high-precision detector.
- German articles: two lexical collisions; naive strict-key coverage falls 98.43→97.64% and 97.14→96.67%, with no demonstrated wording gain. Performed inflection remains the chosen policy; bounded word-to-word treatment preserving word-to-digit normalization remains unmeasured.
- Japanese: actual fresh-ASR/native outputs now exist and the reported continuity, laugh, tail and line-layout defects are corrected. Ambiguous names and uncertain physical boundaries remain review limits. No language hard gate or unrequested profanity terms are introduced.

Original inventories: `scratch/codex-r3/r6-evidence.json`, `r6-article-ablation.json`, `scratch/codex-r1-5/`, `scratch/codex-r1-7/`, `scratch/codex-r5/REPORT-R5.md`.

## Cost and acceptance

Paid validation is **closed at $22.998764 of $24**, leaving **$1.001236**, with **$0 reserved**. Breakdown: provider-reported ASR **$0.378197**, catalog-rate ASR **$0.614095**, recorded native input/output/thinking usage × rates **$18.090013**, conservative failed/interrupted upper estimates **$3.916459**. These are recorded costs, not invoice reconciliation. Failed attempts and excluded validation runs remain charged; copied native replies are counted only in their originating capture.

The final boundary-scope validation adds **$0.036706**: MAI **$0.018079** and Scribe **$0.018627**. Its first partial refresh correctly refused legacy unbound omission proof; those intermediate outputs remain excluded. A zero-network replay attempt stopped safely at zero cost when its exact batch was unavailable. Fresh bound-anchor hearings then add **$0.081780** (MAI **$0.031605**, Scribe **$0.050175**), preserving cue441's omission and producing four actual clip receipts per final episode 11 run. Total final follow-up spend is **$0.118486**, with existing ASR and unrelated native answers reused. Against v10e, exactly three cue texts change: both source222 corrections and Scribe source440 punctuation/case (`vem cá. mais perto.` → `Vem cá mais perto.`); all cue times, ownership, speaker and character fields stay identical. The ledger retains each originating request once.

The fixed offline timing fixtures emit **$0.504960 modeled ASR estimates** while using local `FixtureASRAdapter` data. They made no network requests and are not new billed spend. Earlier script-route hybrid recovery improved reference WER 4.37→4.11% / 2.33→2.14%, costing $0.109174/$0.061625; the wrong-language interrupted $0.747955 is retained in cost and excluded from quality claims.

R1–R4 and the approved bounded R5/R6 work are implemented, measured and accepted for this local candidate. Unmeasured R6 proposals remain recommendations. Backend/frontend/E2E and artifact-integrity checks pass; corpus increases are itemized rather than suppressed. The original episode 11 written-reference WER targets remain unmet, despite the already chosen actor-spoken policy. Common human-edited-edge average errors and the original raw-ASR common-edge distributions are nonworse with the pairing limits above; changed aggregate populations and the two supplementary end-error increases remain explicit. The owner's October 2 acceptance covers these current measured limits. Push, merge and deployment still require separate authorization; no release action has occurred.

The local review package `work/upgrade-20261001/review-v10f.zip` contains all 13 final SRTs and 26 matching QC reports. Each copied file and archived payload was checked against its audited SHA-256. Earlier packages and excluded intermediate runs remain intact. The latest snapshot is validated through immutable source/test hashes; the final commit proof verifies that the committed content is exactly that tested content, with documentation checked separately.

Evidence: final backend/E2E logs, `frontend-reuse-verification.json`, `offline/REPORT.md` and its hash/receipt/semantic audits, `live/final-v10f-acceptance-metrics.json`, actual-native golden comparisons, independent review receipts and the review ZIP manifest. The exact scoped local commit and final dirty-state boundary are recorded in `RESUME.md`.
