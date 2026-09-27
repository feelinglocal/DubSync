# DubSync quality and durability audit, 2026-09-05

The local app has reproducible quality and reliability fixes ready for review. This audit does not establish 100% spoken-text/timing accuracy or a production release. Some supplied VO is missing dialogue entirely. Those passages retain source subtitles and explicit review findings.

## Scope and architecture

Audited checkout: `E:\Work Files\SRT Sync`, branch `codex/two-line-gemini37-20260814`, starting commit `ade6b8fda6aa8d4e508f28adc10fb941896973fe`. Existing untracked customer/test folders were preserved. No commit, push, provider-policy switch, or deployment was performed during the audit phase. The user subsequently authorized pushing and deploying these changes to the existing Render service.

| Layer | Current behavior and audit focus |
| --- | --- |
| React workspace | Single/batch uploads, per-job secret tokens, restore/polling, protected artifact downloads, native audio player and waveform |
| FastAPI intake | Access-code admission, upload/SRT structure bounds, audio probe, storage reservations, per-IP limits |
| Job service | SQLite metadata, atomic claims, serial children within each batch, bounded parallel independent jobs, retention and restart recovery |
| Audio/ASR | FFmpeg normalization, streaming content hashes, word timestamps/speaker tags, cached provider results |
| Sync pipeline | Source normalization, bounded lexical alignment, divergence review, indexed text reconciliation, acoustic re-cueing |
| Language passes | Batched adjudication, punctuation with editorial guards, optional speaker mapping; no LLM-generated timestamps |
| Generation | Word grouping by lexical/speaker ownership, configured VAD and duration clamping, acoustic boundary refinement |
| Verification/export | Optional forced alignment, overlap/media bounds, immutable source holds, cue scores and JSON/HTML/SRT change reports |
| Operations | One Docker/Render service, persistent disk, one SQLite-owning process; no horizontal scaling or checked-in CI workflow |

## Fixed defects and acceptance examples

| Area | Failure before the change | Result |
| --- | --- | --- |
| Improv ownership | A cross-cue replacement produced an isolated 80 ms `Tenho.` before its sentence | Replacement text and word timing belong to the surviving sentence: `Tenho uma negociação comercial à tarde,`, 1243.040–1244.440 |
| Improv confidence | A below-gate rewrite or deletion was still applied | Source span is kept; proposed wording, confidence, verdict, and reason remain in QC. Timing holds do not erase separately approved insertions |
| Similar words | Whole-sentence similarity could hide a meaningful `door`/`gate` change | Actual lexical changes reach batched adjudication; case/punctuation equality can still bypass it |
| Text formatting | Punctuation could alter censor masks or break tags | Existing editorial signatures reject those mutations |
| Speaker ownership | Multi-token ASR rows and unknown speaker tags could move text across speakers | Exact lexical group boundaries preserve words and known speaker transitions |
| Forced alignment | Positional provider-row slicing let contractions/percentages/Korean text consume the next cue's word | Normalized lexical-stream mapping validates complete ownership; invalid/incomplete/duplicate rows are held and excluded from forced-confidence scoring |
| Short/overlapping speech | Minimum duration created silence tails or collapsed same-onset cues | Padding stops at acoustic evidence; actual simultaneous speech remains positive and reviewable |
| Generation | Generation ignored configured VAD and could stretch a 1-second recording's last subtitle to 2.8 seconds | Uses the shared boundary policy, exact grouped-word provenance, and media bounds; CPS produces QC instead of manufactured speech timing |
| Export | Fuzzy overlap deduplication removed distinct sentences/speakers; frame-snapped repetition could vanish | Duplicate repair is restricted; generation preserves every grouped occurrence. Protected spoken cues also appear in overlap QC |
| Evaluation | Good starts with wrong ends or many unmatched cues could pass the timing target | Target requires start and end tolerances plus complete cue matching. Invalid duration has explicit QC |
| Resume | Timing stages could silently use changed source audio or an unrelated leftover normalized WAV | Hash validation runs before writes; old policy checkpoints and newly insufficient confidence require the correct rebuild stage |
| Artifact durability | Interrupted normalization/cache writes destroyed prior usable artifacts | Unique temporary files and atomic publication preserve previous completed files; source/destination audio aliases are rejected |
| Disk/worker lifecycle | Admission ignored other jobs' future space; shutdown started more queued work | Global future reservations count at admission; active calls finish and queued work remains recoverable. Exceptional lifespan exits clean up locks/workers |
| Server latency | ZIP compression blocked the async request loop; old rate-limit clients accumulated | ZIP creation uses the bounded worker pool; expired limiter entries are pruned |
| Browser durability | Overlapping polls regressed job state; storage failures crashed restore | One request per child, finite refresh deadline, in-memory token fallback and explicit recovery guidance |
| Browser memory | Long audio fully decoded merely for a waveform | Metadata-first preview; over 5 minutes or 16 MB retains native playback/upload without full waveform decoding |
| Visible quality | A completed job looked ready even with review errors | Validated QC summaries distinguish warnings/errors, information-only notes, and unavailable summaries |
| Test isolation | Browser ASR fixtures could fall through to live text-model defaults | Both fixture providers are explicit and test-server cloud keys are disabled |

Focused tests demonstrated failures before repair, including additional interactions found by independent review. Existing tests that expected low-confidence edits, fabricated CPS duration, or a fixed unsafe temporary filename were updated to assert the corrected behavior; coverage thresholds were retained.

## Evidence and reproducibility

Local evidence lives under `work/app-audit-20260905/`; it is intentionally excluded from Git with other customer evidence.

- Baseline: 719 backend tests passed, 87.05% combined line/branch coverage; 71 frontend tests passed, 90.89% statements and 87.73% branches. The required coverage minimum is 80%.
- Frontend final: 82 tests passed, 90.64% statements, 88.95% branches, 93.56% lines; typecheck and Vite production build passed. A recovered token store cannot be overwritten after an initial restore failure.
- Browser final: all 18 Playwright tests passed against isolated fixture-backed FastAPI at port 8766, including context isolation, uploads/downloads, refresh recovery, batching and responsive layouts.
- Additional rendered desktop/mobile checks verified blocked browser storage, in-memory-token downloads, visible QC errors, and zero waveform decodes for a 301-second file while short audio still decoded.
- `pip check` found no broken installed requirements. Optional local TorchCodec/pyannote decoding emitted a pre-existing DLL/runtime warning. External vulnerability-database scanning and live production verification were not performed.
- Backend final: **822 passed, 7 live tests deselected**, 87.28% combined line/branch coverage; no failures. Exact totals are recorded in `final-tests.xml` and `final-coverage.json`.

The first interrupted browser run unexpectedly made fixture-text Gemini requests because the existing fixture omitted `llm.provider`. Completed job cost artifacts meter $0.034249 for that run. It was stopped, the isolation defect was fixed, and the final browser run is explicitly offline. Real-media replays denied network sockets and used no provider spend. These were test-text calls, not customer-media uploads.

### Real media

`corpus/inventory.json` covers 22 source/media fixtures totaling 80m44.77s and 1,895 source cues. There are 37 historical source/result pair audits. Three representative fixtures were replayed through both a Git-archived starting implementation and the current production pipeline, using identical source, normalized audio, and ASR hashes. Saved language decisions are matched by exact span content and word/cue indices, not case IDs alone.

| Fixture | Evidence |
| --- | --- |
| Customer-fixed new bug | 48 cues; correct `aus Angst, einen Laut von mir zu geben.` stays together; output identical to the good baseline |
| testing4 raw ASR | 112 cues, all tokens preserved; missing/untrustworthy dialogue and existing source overlaps stay visible for review |
| Long | The orphan replacement becomes one 1.4-second sentence. A discovered guard interaction was corrected so independent approved `você roubou` wording survives a neighbouring uncertainty hold |

Run the preserved recipes only into their separate work directories:

```powershell
.\.venv\Scripts\python.exe work\app-audit-20260905\corpus\replay_corpus.py --phase current
.\.venv\Scripts\python.exe work\app-audit-20260905\corpus\compare_replays.py
```

Full cue comparisons, exact input/output hashes, provider-replay limitations, silence evidence and review clips are in `corpus/CORPUS-REVIEW.md`, `current-comparison.json`, and each fixture's `replay-evidence.json`. Cached-model replay proves deterministic code behavior; it does not validate a fresh model decision or replace listening.

All three final replays conserve the baseline normalized token sequence. The long output has 803 cues and SHA-256 `fd87fa83f396f05ea02cf7018e6a1d92991ea428f58294cfcd5300854e099bb5`. Its 127 error flags include necessary source holds and newly exposed overlaps; it is not an error-free deliverable. After the last replay, the only changed Python module was generation's `transcription.py`, removing raw uncensored ASR words from the public generation metadata; the sync replay path is unchanged and the final full test suite covers that correction.

### Performance evidence

The ASR region-envelope lookup no longer copies every remaining region for every word. A same-output 20,000-word/region microbenchmark measured 0.2237s before and 0.0247s after, about 9x for that operation. This is a single-run synthetic probe, not an end-to-end speed claim. Generation carries direct word-group ownership and avoids scanning the full word stream per cue. Disk streaming remains bounded.

Forced-aligner row grouping was checked against the provider's [primary implementation](https://raw.githubusercontent.com/MahmoudAshraf97/ctc-forced-aligner/main/ctc_forced_aligner/text_utils.py), alongside installed-code tests; a live model execution was not substituted by that source review.

Corpus processing took roughly 0.1s, 0.3s and 3.5s locally, excluding normalization and all provider inference. Removing the fuzzy-text shortcut sends two additional long-fixture spans to review (252 to 254 adapter lookups); batching is unchanged. Fresh provider latency, quality and cost effects remain unmeasured.

## Remaining limitations

1. The long fixture contains digital silence from 1203.8–1212.6 seconds while six source cues contain dialogue. Missing speech cannot provide timing or wording truth. Neither ASR confidence values nor a green test suite establishes 100% precision.
2. There is no complete independent verbatim/phoneme-labelled corpus or full audition. Some source-held cues remain outside trustworthy audio, too brief, or overlapping. Error counts can rise because those problems are now reported honestly.
3. Real MMS model execution and current-cloud adjudication/punctuation were not validated. Model dependencies, language coverage, words spoken simultaneously, noise, and ASR hallucinations remain material limitations.
4. Production remains one SQLite-backed process. Running Python threads cannot be forcibly cancelled; shutdown waits for active provider work, and dead-lettering does not stop the running call. Separate OS temporary-volume capacity is not independently reserved.
5. Atomic writes protect each file, not a multi-file transaction. Legacy ASR checkpoints without provenance remain readable with a warning; regenerate from ASR when identity is uncertain.
6. The waveform limits reduce ordinary VO memory use, but browser metadata/decoder behavior is not a strict resource bound for malformed or unusually multichannel input.
7. At audit completion these changes were local and uncommitted. Deployment, exact live SHA, post-deploy logs, and real provider acceptance require separate release evidence; the audit checks above do not establish those results.
