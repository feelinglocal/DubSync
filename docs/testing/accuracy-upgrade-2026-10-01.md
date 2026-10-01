# Accuracy upgrade evidence — 2026-10-01

Status: **final v9, policy 26 candidate on `upgrade/sync-accuracy-2026-10`; owner acceptance of timing/overlap exceptions pending**. The exact committed revision belongs in delivery/`RESUME`. No production push or deployment occurred; this report is not subjective audio-quality certification.

## Owner decision and evidence contract

On **October 2, 2026**, the owner chose captions matching what actors actually say, including improvised sentences and performed register. Default `adjudication.register_policy: spoken` implements that choice. Explicit `script` and no-LLM script preservation remain. Written-reference WER remains visible; a reference's register convention does not override the actor-spoken goal.

Automatic processing continues through known ambiguous passages and delivers actionable QC. This does not waive the original timing/zero-overlap acceptance requirement: the final live artifacts retain **ten explicit overlap pairs**, detailed below. **Owner disposition of these exceptions is pending.** No new automatic job gate is introduced.

Human episode 11/17 references inherit 94–99% of their timing from earlier outputs. Text and actually human-edited edges are measured separately; inherited timestamps are not independent acoustic truth. `new bug/001 fixed customer.srt` is text-only. Saved provider hearing is not an independent audition.

Private inputs, snapshots and full hash-bound evidence remain under `work/upgrade-20261001/`. All final paths below are relative to `scratch/codex-final-v9/` unless stated otherwise. Promoted baseline/R1g used an older matcher with positive unmatched fallback; the safe v2 and final controls share the strict matcher. Unsupported decisions stay confidence-zero holds.

## R1–R6 package disposition

| Package | Implemented behavior / disposition | Measured limit |
|---|---|---|
| R1 timing / ASR | MAI default, Scribe selectable; no silent switch. Shared word repair/ownership, streaming adaptive VAD, phrase edges, held-source caps, lexical punctuation guards, frame-safe export, lyric protection, retries and true cost provenance | Sparse, collapsed or separated acoustic evidence remains reviewable; zero physical uncertainty is not established |
| R2 wording | Flash 3.8 medium default; Lite high + Flash medium opt-in. Native v12 literal `heard_text`/evidence, language/register policy, finite deterministic predecisions and per-case cache | Names/orthographic equivalents can remain held; clear model hearing is evidence, not proof |
| R3 docs / measurement | Portable manifest-based corpus, golden and raw/effective timing tools; offline replay denies network and pins source/input/harness identity | Exact, unique-text and unmatched decision reuse are reported separately |
| R4 generation / ingestion | Exact ownership, balanced wrapping, silence-only reading-time extensions, style inference, robust SRT parsing/encoding and warned empty-cue omission | Style/ambiguity reviews remain; profanity terms unknown, existing explicit policy preserved |
| R5 dual ASR | Optional/off, secondary wording only, primary timing, separate cache/provenance/cost and strict resume | Recent paired routes had unchanged SRT/WER and only 3–5 avoided reviews; no measured current-route accuracy gain |
| R6 reconciliation / captions | Whole missing-cue and exact-residue native questions; independent speech chains, narrow actor interruption split and pure bracket-caption composition | Broad structural/name/full-mix proposals remain recommendations without claimed gain |

Missing-cue omission requires complete anchored audio, native clear absence and no detected speech activity. Recovered wording uses one independent speech chain whose internal pauses are **strictly below 0.2 s**; the LLM supplies no timestamps. Residue edits target exact source tokens; different wording without owned timing remains held. No-LLM and unavailable native answers preserve source rather than fabricating successful hearing.

Final v9 verifies Japanese native laughs, whole-parent holds for distant word groups, retained word tails, lexical starts after punctuation and stale-QC cleanup. The narrow episode 11 MAI interruption splits source217 around generated945 into IDs217/945/998 while preserving the sentence's entire owned-word set. Genuine greeting/chorus context remains.

## Execution checks and intentional test updates

| Check | Final evidence |
|---|---|
| Backend | **3,100 passed, seven live deselected, two existing warnings, 162.72 s; 91.02% combined statement/branch coverage** — `backend-coverage.log`, `coverage.json` |
| Frontend | **100 tests, six files, 5.89 s; 91.28% statement / 89.16% branch / 91.77% functions / 93.92% lines** — `frontend-coverage.log`, coverage summary |
| Typecheck / build | Passed v8; all **54 web files** are byte-identical to tested v3/v8/v9/current, so those results are reused with hash proof |
| Playwright | **24 passed, 22.7 s** — `frontend-e2e.log` |
| Frozen offline suite | **162 corpus + three golden + four timing + 12 generation**, zero execution failures; **339 snapshot hashes, rebuild policy 26** |
| Inspection / review | All **13 actual SRT/QC outputs, 3,403 cue texts and 143 review items** read; independent review resolved; root verifies **335 non-documentation files** still match the tested snapshot |

The **80%** application coverage minimum and frontend thresholds remain unchanged. Test/execution success does not establish every acceptance gate or 100% acoustic accuracy. Final documentation is checked separately after the immutable application snapshot.

Intentional test updates retain selectable providers and hybrid controls while pinning MAI/Flash defaults. Script-specific fixtures explicitly request `script`; native fixtures supply v12 `heard_text`/evidence and strict `AdjudicationResponseBatch` transport. Malformed legacy confidence/quotation approval still rejects. Hybrid cases distinguish clear hearing from ambiguous short/negation passages. Generation expectations preserve starts and repeated-word ownership while applying verified-silence reading time and balanced wrapping. Held source dialogue preserves authored inline lines when safe speaker boundaries are unavailable. Targeted timing/QC regressions pin the repaired real Japanese, punctuation, laugh, interruption and ownership cases. Earlier R3 removed unused `reports.write_changes_diff` after finding no application callers, together with its exclusive four test functions/six parameterized cases; the active `write_change_log` path remains covered.

## Matched offline measurements

The fixed matrix contains **60 episodes with both models**: 120 no-LLM and 42 saved-decision runs. New Japanese 1A/1B/2A acceptance inputs are separate. All **181 SRTs** and compared artifacts/semantics equal v8; both golden comparators reran on v9. Acoustic metric reuse has matching output, four WAV and measurement-script hashes.

| Group | Errors, safe v2 → final v9 | Held cues | Source-timed dialogue | Fragments |
|---|---:|---:|---:|---:|
| No LLM | 459 → 459 | 442 → 442 | 437 → 437 | 2 → 2 |
| Saved decisions | 247 → 269 | 345 → 349 | 139 → 139 | 10 → 10 |

The saved +22 errors comprise **21 unavailable native/residual hearing questions and one rejected historical malformed quotation**. Replay covers **1,099 exact and 16 unique-text decisions**, with **330 explicit unmatched holds**. All **71 supplemental matrix questions** remain unassessed: 53 pending, 18 confidence-zero unconfirmed. They receive no invented successful hearing.

For the original 87-run promoted population, no-LLM errors/holds/source-timed dialogue are **206→231 / 212→235 / 211→242**; saved are **215→269 / 372→349 / 112→139**. These older-matcher controls are historical, not a strict matched comparison. Both add 32 timing-evidence holds; net unresolved-overlap errors add 8/7. The error accounting lists every added/removed kind, cue ID and message, including unavailable audio, invalid stored verdicts and minimum-duration changes. This is not uniform improvement over the promoted baseline.

All **169 sync outputs** preserve spoken lines and unique ownership versus v8. The earlier 17 word-tail extensions remain; one existing testing3-033 cues30/31 hold grows 68→201 ms versus v2. Against v7, only French `batch2-test2.mai.nollm` cue74 end 108.500→108.534 and cue75 start 108.500→108.634 change, with one added timing warning. No new error identity, overlap pair, text, line-break or owner change. Six combined-caption three-line warnings remain.

| Stream | Speech-only WER, safe v2 → final | False-edit spans | Delivered WER / false spans | Human-edited start/end MAE |
|---|---:|---:|---:|---:|
| Ep11 MAI | 4.34% → 4.70% | 63 → 96 | 5.19% / 100 | 65.5 / 84.3 ms |
| Ep11 Scribe | 3.87% → 4.34% | 52 → 89 | 4.81% / 93 | 67.9 / 99.5 ms |
| Ep17 MAI | 2.18% → 1.66% | 25 → 26 | 1.81% / 27 | 183.2 / 74.9 ms |

Delivered WER includes intentional repeated caption text. Speech-only scoring strips balanced bracket text equally from source, reference and outputs. These saved-decision reference scores expose mixed outcomes and register conflict; they do not prove audible errors or validate fresh hearing. Original edited-edge and speech-only populations differ and their MAEs are not interchangeable.

| Acoustic stream | Overlaps | Starts within 40 ms, raw / effective | Raw end median / mean absolute |
|---|---:|---:|---:|
| Ep11 MAI | 1 | 88.0% / 91.3% | 64 / 78 ms |
| Ep11 Scribe | 3 | 88.3% / 91.6% | 67 / 90 ms |
| Ep17 MAI | 1 | 90.4% / 93.6% | 60 / 63 ms |
| Testlong Scribe | 3 | 84.7% / 89.1% | 60 / 70 ms |

The qualified timing population has zero measured first-word cuts/duplicate owners; its eight overlaps are seven explicit holds and one lyric pair. Short speech with available room remains 3/1/0/2. Unowned/held evidence is not automatic physical proof. All **12 generation outputs** keep exact one-time ordered ownership, unchanged raw words, zero overlaps/late starts/end clips beyond one frame and zero short cues with verified free room. Ambiguous input retains a hold; each Scribe ep11 preset retains eight reviews including three errors.

Evidence: `offline/REPORT.md`, `final-summary.json`, `v8-equality-proof.json`, `metric-reuse-provenance.json`, `measurement-provenance.json`, `final-evidence-verification.json` and `reused-v8-audits/error-change-accounting.json`.

## Live per-model actual artifacts

All **13** captured runs were strictly replayed with zero new calls. Actual exported text/times bind uniquely to rebuild IDs; raw/effective/VAD hashes, unique ownership, grouped review references and **2,553 unchanged-source line-break instances** verify. All **12/12 native supplemental questions** have decisions: six omissions, four utterances and two held outcomes. No audio was auditioned. German/Japanese/ep11 ASR was fresh in the originating session; ep17 reuses prior **HV8 MAI ASR**.

| Output | ASR | Cues | Reviews | Overlap pairs | Lexical late / end candidates > frame | Capture/recovery lineage USD |
|---|---|---:|---:|---:|---:|---:|
| Japanese 1A | MAI | 60 | 3 | 1 | 0 / 0 | $0.404104 |
| Japanese 1A | Scribe | 60 | 2 | 0 | 0 / 0 | $0.302299 |
| Japanese 1B | MAI | 59 | 11 | 2 | 0 / 0 | $0.389435 |
| Japanese 1B | Scribe | 59 | 8 | 3 | 0 / 0 | $0.301933 |
| Japanese 2A | MAI | 79 | 3 | 0 | 0 / 0 | $0.359632 |
| Japanese 2A | Scribe | 79 | 3 | 1 | 1 / 1 | $0.268093 |
| Japanese 2B | MAI | 79 | 9 | 0 | 0 / 0 | $0.223823 |
| Japanese 2B | Scribe | 79 | 5 | 0 | 1 / 1 | $0.174352 |
| Ep11 | MAI | 1,005 | 38 | 1 | 0 / 0 | $3.161595 |
| Ep11 | Scribe | 997 | 33 | 1 | 0 / 1 | $2.853967 |
| German short | MAI | 48 | 4 | 0 | 0 / 0 | $0.098286 |
| German short | Scribe | 48 | 5 | 0 | 0 / 0 | $0.119374 |
| Ep17 | MAI | 751 | 19 | 1 | 0 / 0 | $1.803561 |

Per-output costs above include the originating capture and required recovery attempts; they are **lineage costs, not hypothetical single-run prices**. Deduplicated total is **$10.460454**; full task spend is reported below.

The ten actual overlaps are seven uncertain Japanese pairs (1A MAI#11/12; 1B MAI#3/4,#11/12; 1B Scribe#3/4,#11/12,#35/36; 2A Scribe#34/35), Ep11 MAI#764/765 (IDs708/982, 233 ms greeting/chorus), Ep11 Scribe#546/547 (IDs513/970, 1,034 ms collapsed actor-owner candidate), and Ep17#571/572 (IDs565/752, 500 ms lyric/`Uhum.`). **Zero pure bracket-caption overlaps remain.** All ten have actual grouped QC. Actor labels do not independently prove simultaneous speech.

Remaining literal boundary candidates are protected: 2A Scribe#34 853 ms start lag; #50 a far 分 5,694 ms past its 800 ms held parent; 2B Scribe#17 585/607 ms lag/end difference; Ep11 Scribe#641 a 1 ms `É.` placeholder with 1,401 ms end difference. Additional >frame display leads remain held/reviewable, including ep17#534's incomplete native clip. They are not all physically proven cuts. Four repaired MAI punctuation leads are now 1/5/26/25 ms; repaired 1B Scribe#58 and 2A/2B#65/#66 tails/laughs no longer have their prior late/cut/overlap candidates. One frame uses fallback **30 fps**, not verified video FPS.

German Scribe's ordinary malformed-quotation response remains actionable QC; names, spelling equivalence, sparse words and readability pressure remain explicit. Ep17 includes 13 unchanged name-only cues without lexical owners or individual review; preserved lyric/unowned timing limits blanket acoustic claims.

Evidence: `live/final-v9-completed-manual-audit.md` and JSON manifest, `final-v9-acceptance-metrics.json`, per-run audit files, and [the exact review files](../../work/upgrade-20261001/scratch/codex-final-v9/REVIEW-FILES.md). Older v6 findings are preserved separately.

## R6.3 Dialogue and annotation overlaps

Default `output.no_overlaps: true` composes pure bracketed captions without owned words, appending caption lines while retaining every spoken ID, original line sequence, timestamp, metadata and owner set. Caption-only prefixes/gaps/suffixes retain coverage; only caption text repeats. Lyrics are never repeatable caption tracks. Explicit `false` preserves original segmentation.

Final corpus composition covers **25 tracks across ten runs**, removing **28 eligible pairs** without repeating speech. Composition alone has **0 ms speech timing delta**. Corpus maximum added caption visibility is **1,483 ms early / 1,160 ms late**; six three-line warnings remain.

| Fresh ep11 caption | Caption-only retained piece | Added early / late visibility | Occurrences | Continuous CPS |
|---|---|---:|---:|---:|
| `[Luan Nian]`, near 539 s | 574 ms prefix | 0 / 914 ms | 2 | 6.989 |
| `[Pequim→Zurique]` | 1,340 ms suffix | 676 / 0 ms | 2 | 6.140 |
| Announcement | 620 ms prefix, 200 ms suffix | 0 / 0 ms | 3 | 41.667 |
| `[Luan Nian]`, near 2,104 s | 1,020 ms suffix | 783 / 0 ms | 2 | 5.358 |

The 55-character announcement remains continuous for **1,320 ms**. Counting already visible text as newly appearing gives 275 raw suffix CPS and 118 with `Hum`; continuous rate is 41.667 CPS and still needs readability review. Ep17#30 has three lines/65-character width/72.67 raw CPS. A mixed testlong caption gains 1,430 ms early/970 ms late across untouched speech and a 640 ms gap; email gains 1,160 ms tail. These are explicit picture-text/readability tradeoffs, not acoustic improvements or subjective readability proof.

Evidence: `scratch/codex-annotation-merge/annotation-composition-handoff.md`, final offline annotation audit and actual live composition/coverage checks.

## Remaining measured recommendations

- Partial timing: 112 source-timed cues, 14 with owned words, nine held partial cases, **zero** sole bursts confined to independent anchors. Further rescue needs repaired evidence and measured edited-edge benefit.
- Transpositions: 20/18/10/9/5 split-span pairs in ep11 MAI/Scribe, ep17 MAI, ep02 Scribe and testlong Scribe. Candidate counts are not actor truth; joint-region and chapter/name hints have no measured adoption gain. Approved optional cost is not a requirement to adopt them.
- Full mix: historical energy activity 87%, longest region 140 s; genuine chorus stays energetic. Customer delivery mix is unknown. Spectral/model alternatives remain optional pending matched accuracy and runtime measurement.
- Early-start proposal: 462 raw candidates; actual human-edited edges yield one support, two rejections, 210 unknown. Five source-timed half-matched cues lack hold flags; neither inventory proves a high-precision detector.
- German articles: two lexical collisions; naive strict-key coverage falls 98.43→97.64% and 97.14→96.67%, with no demonstrated wording gain. Performed inflection is the chosen policy; bounded word-to-word treatment preserving word-to-digit normalization remains unmeasured.
- Japanese: originating fresh-ASR/native runs and final actual artifacts exist, including repaired laughs/tails. Listening validation for ambiguous names and held physical timing remains a separate limit. No language hard gate is recommended.

Evidence: `scratch/codex-r3/r6-evidence.json`, `r6-article-ablation.json`, `scratch/codex-r1-5/`, `scratch/codex-r1-7/`, `scratch/codex-r5/REPORT-R5.md`. Profanity terms remain unknown; preserve existing explicit policy without adding unrequested terms.

## Cost and acceptance handoff

The closed task ledger is **$20.509561 of $24**, leaving **$3.490439**, with **$0 reserved**: provider-reported ASR **$0.378197**, catalog ASR **$0.613869**, Gemini usage × configured rates **$15.601036**, uncertain failed/interrupted charges **$3.916459**. Excluding uncertainty, known/estimated usage is **$16.593102**. Matrix spend is $0.668768 and all other validation $19.840793. These figures are not invoice-reconciled.

All paid calls are complete; the total includes old route comparisons, failed/interrupted attempts, fresh ASR and successful exact-request recoveries. Final replays made no new calls. The live packet's deduplicated **$10.460454 / 35 capture ledgers** is only a subset of task spend; zero replay charge does not make originating processing free.

Earlier script-route exact recovery improved hybrid written-reference WER 4.37→4.11% / 2.33→2.14%; direct Flash was 3.97/2.03%. Two/one recovery requests cost $0.109174/$0.061625. The interrupted wrong-language $0.747955 is conservatively charged and excluded from quality scoring. Those script-route comparisons do not certify the actor-spoken product goal.

Technical execution, final actual-artifact inspection and independent review are complete. Written-reference metrics are mixed, native legacy questions remain unassessed offline, and the **ten live overlap exceptions plus caption visibility/readability tradeoffs await owner disposition**. The original strict acceptance requirement is not represented as passed. No subjective audition or production release is claimed. Publishing, pushing, merging or deployment requires explicit release authorization.

Final evidence: `root-checks.json`, `paid-summary.json`, backend/frontend/E2E logs, offline `REPORT.md`/`final-summary.json`, live manual audit/manifest and `REVIEW-FILES.md`. Documentation content/diff checks accompany the handoff.
