# DubSync accuracy upgrade: handover (2026-10-01)

This document hands the unfinished accuracy upgrade to the next engineer or agent. It is self-contained for planning. Supporting material (analysis notes, benchmark scripts, test media, cached transcripts) lives only on the owner's Windows machine under `E:\Work Files\SRT Sync\`, mostly in git-ignored folders.

Note (2026-10-03): this brief records the state on October 1, 2026. Two statements that were wrong at the time are corrected inline (overlap errors, detector saturation). Later behaviour, including the cue-start rule and repeat handling, is in [the accuracy-upgrade report](../testing/accuracy-upgrade-2026-10-01.md).

Read sections 1 to 6 before changing code. Sections 7 and 8 are the remaining work and the order to do it in.

---

## 1. Goal and owner priorities

The owner wants DubSync to be the best SRT synchronization app available. In order of importance:

1. Accurate cue timing: each cue starts and ends with the real speech.
2. Correct words when actors improvised: the subtitle follows what was actually said.
3. Correct word order.
4. Far fewer QC warnings and errors shown to the customer after a sync.
5. Both transcription models must work well: Microsoft MAI-Transcribe 2 (via OpenRouter) and ElevenLabs Scribe v2. The owner believes MAI is better, and the measurements agree.

Languages that matter: German, Portuguese, Japanese. Most test material is German (55 episodes); 4 long episodes are Portuguese; 1 is French; there is no real Japanese audio.

---

## 2. Decisions already made (do not re-ask)

Owner-confirmed:

| Topic | Decision |
|---|---|
| Default transcription model | Let benchmarks decide. Result: MAI-Transcribe 2 is now the default; Scribe v2 stays selectable; no silent fallback between them. |
| Paid API testing | Allowed up to about $25 total (ElevenLabs, OpenRouter, Gemini). About $0.57 is spent. Track and report spend. |
| QC presentation | Customers see only actionable items. Routine notices go to a separate change log. Never hide a real problem. |
| Git | Work on branch `upgrade/sync-accuracy-2026-10`. Commit at verified milestones. Do not push, merge to `main`, or deploy without the owner's explicit approval. |
| LLM stages | Accuracy first. Stronger models or extra passes are acceptable when measured; report the cost change. |
| Ground truth | Only three files are human-corrected: `test fix/011-final-fixed by human.srt`, `test fix/more testing long eps/017 fixed by human.srt`, `new bug/001 fixed customer.srt` (text only; its timings are unsynced source timings). The `*-dubsync-synced.srt` result folders are unverified reference points. |
| Production host | Render service `srv-d98vqev7f7vs739hn3hg`, Starter instance: 512 MB RAM, 0.5 CPU, one instance, 10 GB disk. `render.yaml` has `autoDeployTrigger: commit`, so pushing `main` deploys. |

Evidence-based defaults chosen during the upgrade (each has a switch; the owner has been told and has not objected):

| Topic | Default | How to reverse |
|---|---|---|
| Cue start | The speech-burst onset floored to a frame, about 20 ms before the acoustic onset. | `timing.phrase_edge_snap: false` |
| Short cues | Extend into verified silence up to the minimum duration, never past the next cue or other speech. | `timing.min_duration_policy: acoustic` |
| Song lyric (`♪`) cues absent from the voice track | Keep source text and timing, one informational note per cue, no error. | none |
| A lyric cue overlapping a spoken line | The lyric cue ends where the spoken line starts (delivered cues do not overlap). The human editor merged such lines into one cue instead; that is not built. | none |
| Spelling and register variants | Keep the script for abbreviations (`Sr.`, `Srta.`, `Dr.`) when the actor says the full word. | none |
| Low-confidence or audio-unavailable adjudication | Keep the source text, but time the cue from the words it owns. | one set in `pipeline._source_timing_held_cue_ids` |

Constraints that follow from the host:

- Production code must stay pure Python with no new runtime dependencies. No numpy, torch or onnxruntime in the production path. Audio is processed by streaming, not loaded whole.
- Timing comes only from acoustic evidence (ASR word timestamps, energy analysis, optional forced alignment). LLMs decide wording only and never produce timestamps.

---

## 3. Current state

- Branch `upgrade/sync-accuracy-2026-10`, HEAD `971c7ed`, 71 commits ahead of `main` (`5e29527`). Not pushed. Not deployed. Production still runs `5e29527`.
- Backend tests: 2,327 pass, 7 opt-in live tests deselected (1,793 before the upgrade).
- Frontend: 95 unit tests, typecheck and production build passed after the first wave. Playwright end-to-end tests have not been run since the default model changed.
- Weekly Claude usage ran out before the roadmap was finished; that is why this handover exists.

### 3.1 Measured results

Offline replay of 87 runs from the test corpus through the current code, with stored LLM verdicts (`saved` mode), 0 failures:

| Measure | Before (`5e29527`) | Now |
|---|---|---|
| QC findings per 100 cues | 90.1 | 45.1 |
| QC errors | 723 | 215 |
| Dialogue cues left at source timing | 412 | 112 |
| Cues under a confidence hold | 664 | 248 (all are harness "no saved decision" cases) |
| Overlap errors (`output_overlap_unresolved`) | 127 | 12 (corrected 2026-10-03: the preserved 87-run bench records 12, not 10) |
| `timing_refined` warnings | 1,775 | about 71 |

Against the human-corrected files (replays with stored LLM decisions):

| Measure | Before | Now |
|---|---|---|
| Episode 11, MAI: word error | 5.07% | 4.11% |
| Episode 11, Scribe: word error | 4.30% | 4.15% |
| Episode 17, MAI: word error | 1.84% | 1.87% (one token: a lyric cue the old tool had overwritten is now kept) |
| Episode 11, MAI: end error on edges the human corrected | 108 ms | 88 ms |
| Episode 11, MAI: start error on edges the human corrected | 76 ms | 68 ms |

Against the audio itself (10 ms energy onsets and offsets, four long-episode runs, measured after the timing package):

- Starts within 40 ms of the onset: 70 to 77% before, 85 to 90% after. No cue starts after its first word (was 2 and 11 cues on two Scribe runs).
- End minus speech offset, median: 85 to 96 ms before, 60 to 67 ms after (40 ms tail plus frame rounding).

Customer-facing QC, on stored reports: episode 11 (MAI) 1,030 raw findings become 47 review items; episode 17 becomes 21; 13 real web jobs go from all "review needed" to 6 clean, 5 check, 2 attention.

Live runs with real providers on short German clips: both models complete; MAI and Scribe outputs for the same clip agree within one frame on 89% of starts and 87% of ends.

### 3.2 A warning about the human-corrected files

Episodes 11 and 17 were corrected by editing an earlier DubSync output (11 from a Scribe run, 17 from a MAI run). 94 to 99% of their timestamps equal the old tool's times. Consequences:

- Agreement on edges the human did not touch measures similarity to old code, not accuracy. It dropped after the upgrade (episode 17 start error 3 ms to 32 ms) and that is expected.
- Use these files for word accuracy, for the edges the human actually changed (the `human_edit_subset` block in the golden benchmark), and for conventions: zero overlapping cues, ends about 33 to 40 ms after the last word, lyric cues kept at source timing, sub-500 ms cues accepted.
- They cannot rank MAI against Scribe on timing. The energy-based measurements can.
- "One frame early" in those files is an import artifact: the old export truncated 30 fps grid times to milliseconds, and the customer's tool floors to frames. Export now rounds frame boundaries up to the millisecond. Do not add a lead-in to compensate.

---

## 4. What changed in the code

Pipeline order for a sync job (`src/dubsync/pipeline.py`, `sync_episode`): ingest, ASR, align, deterministic holds, LLM adjudication, confidence gate, ad-lib and word ownership, text edits, segmentation and speaker-turn split, speech evidence (VAD plus word-edge repair), rebuild timing, settle steps, overlap policy, punctuation, then verify (refinement, restores, final ordering and overlap resolution, QC report).

| Area | Files | What it does now |
|---|---|---|
| ASR adapters | `mai_transcribe.py`, `providers.py`, `cache.py`, `cost.py` | MAI: drops re-decoded word runs, collapses doubled countdown numbers, pairs chunk-boundary words spelled differently, stitches speaker labels across the 300 s chunks (unlinked speakers keep a name in the same scope), per-chunk retries, diarization fallback, tolerates one invalid word. Scribe: retries, typed errors, `logprob` and audio events saved as evidence. Language codes normalised to ISO-639-1. MAI is the default (`provider.yaml`, `web/app.py: DEFAULT_TRANSCRIPTION_PROVIDER`, `web/src/types.ts`). |
| Speech detection | `vad.py`, `silence.py` | 10 ms hop, per-file adaptive thresholds with hysteresis, integer frame math. Setting `threshold_dbfs` or `window_ms` selects the old fixed detector. |
| Word-edge repair | `asr_timing.py`, `timing_refinement.py` (`SpeechEvidence`, `speech_evidence_for_words`) | Stretched ASR word edges are moved onto the speech burst they belong to, once, before cues are timed. Rebuild and verify share the repaired words. Alignment, adjudication and text edits still see raw words (see 7, item R1.8). |
| Cue timing | `recue.py`, `timing_refinement.py`, `style_profile.py` | One word-window rule, burst-based ends, one minimum-duration policy, frame-safe rounding. `_enforce_monotonic` is gone. |
| Overlaps | `output_order.py`, `overlap.py` | Overlaps are resolved without cutting anyone's words; a held cue is never clipped to a sliver; genuine simultaneous speech stays and is flagged once. |
| Alignment | `aligner.py`, `alignment_windows.py`, `tokenize.py`, `text_metrics.py`, `source_quality.py` | Anchor-guided band with a soft unique-pair check (long episodes survive offset, drift, an unspoken opening). ASR words tokenise like SRT text. Inline markup is not a token. Portuguese, Spanish and French number words. Compound spellings match. Unheard lyric cues are held as blocks. |
| Holds and adjudication | `pipeline.py`, `adjudication.py`, `hybrid_adjudication.py`, `audio_snippets.py`, `adjudication_snippets.py`, `observability.py` | Unknown ASR confidence is unknown, not zero. Deterministic keeps are not gated. No 90-minute cap. Spans up to 120 s get one covering clip. Song caption guard. Swapped-cue reconciliation. Review-provider failure is transient, not cached. |
| New modules | `speaker_evidence.py`, `edit_consistency.py`, `detached_speech.py`, `qc_review.py` | Speaker-change evidence across label scopes; text and timing applied together or held together; words spoken far from a cue placed at their own time; the customer-facing QC layer. |
| Text edits | `changes.py`, `cue_segmentation.py` | Replacement words go to the cue they were spoken in; re-decoded copies are not inserted; apostrophes, quotes and punctuation survive edits; collapsed ad-libs are padded, merged or dropped. |
| Ingest | `pipeline.py` | Duplicate or broken cue numbers get internal sequential ids; unusable source durations are repaired with a flag. |
| QC presentation | `qc_review.py`, `reports.py`, `evaluation.py`, `verify.py`, `web/app.py`, `web/src` | Raw flags are classified at report time into review, changes, notes and diagnostics, deduplicated per cue or block, with delivered SRT numbering. Verdict tiers clean, check, attention. `changes.diff.srt` is a true change log. |

Things that will trip you up:

- `tests/test_qc_review.py::test_every_emitted_flag_kind_is_registered` fails when any new `QCFlag` kind is not in `qc_review.KIND_REGISTRY`. Register every new kind with its bucket.
- Emitter severities and kind names are pinned by about 30 tests and double as an internal message bus. Do not change them to reduce noise; fix false triggers at the emitter or classify in `qc_review.py`.
- `tests/test_documentation_acceptance.py` pins sentences in `README.md`, `provider.yaml`, `providers.example.yaml` and `docs/COMMERCIAL_PLAN.md`.
- Policy versions: `_REBUILD_POLICY_VERSION = 10`, `_ADJUDICATION_POLICY_VERSION = 3` (`pipeline.py`), `HYBRID_POLICY_VERSION = 2`, `SNIPPET_BATCH_STRATEGY_VERSION = "bounded_batches_v3"`, prompt versions in `llm_providers.py` (`adjudication-v11-...`). Bump the relevant one when behaviour changes, because caches and resume checks key on them. `--resume verify` from artifacts written by older code is rejected by design.
- `pyproject.toml` sets `addopts = "-q"`; adding another `-q` hides the pass count. Use `-o addopts=""`.
- The shell on this machine has a stale `OPENROUTER_API_KEY` that overrides `.env`. Unset it for CLI runs. `scripts/run_local.py` already handles this for the web app.
- Windows console encoding is cp1252. Set `PYTHONIOENCODING=utf-8` before printing subtitle text.
- Scribe is not deterministic, even with temperature 0 and a seed (probed). MAI is deterministic and returns no per-word confidence.
- Corrected 2026-10-03: the adaptive energy detector does not saturate on full mixes. On episode 02 its thresholds reach the -40 dBFS cap and it fragments speech into syllable-size bursts, so word repair can cut cue endings without a cue-level finding.

---

## 5. Where everything is

Paths are relative to `E:\Work Files\SRT Sync\`. Everything under `work/` is git-ignored and exists only on this machine.

| What | Where |
|---|---|
| Short status file | `work/upgrade-20261001/RESUME.md` |
| Analysis notes (12 areas, with file and line references to `5e29527`, reproductions and measurements) | `work/upgrade-20261001/understand/*.md`; index in `_digest.md`; structured list in `_results.json` (141 bugs, 107 improvements) |
| Independent review of the first wave | `work/upgrade-20261001/understand/review-wave1.md`, scripts in `scratch/review/` |
| Package reports (tasks done, skipped, follow-ups, before and after numbers) | `work/upgrade-20261001/wave1-reports.json`, `wave2-reports.json` |
| Test corpus manifest (60 episodes, cached transcripts per model) | `work/upgrade-20261001/corpus-manifest.json` |
| Corpus benchmark and replay tools | `work/upgrade-20261001/scratch/pipeline-impl/` (`bench_all.py`, `show.py`, `replay_offline.py`, `replay_head.py`, `golden_bench.py`) |
| Benchmark baselines | same folder: `bench-before/` (base commit), `bench-final3/` (current), `golden-before.json`, `golden-final3.json` |
| Timing-versus-audio tools | `scratch/timing/` and `scratch/timing-impl/` (`bench_metrics.py`, `bench_replay.py`, `sweep.py`) |
| QC layer benchmark | `scratch/qc-impl/bench_qc.py` |
| Word-placement tools | `scratch/placement-impl/tools/` |
| Live run outputs | `work/upgrade-20261001/live/` |
| Human-corrected files | section 2 |
| Run directories with full stage artifacts for the long episodes | listed at the end of `understand/timing.md` |

Caveat on the scratch tools: several put the main checkout's `src` first on `sys.path`. When comparing against the base commit, export a copy with `git archive` and point `DUBSYNC_SRC` and `PYTHONPATH` at it.

---

## 6. How to verify

Git Bash syntax; run from the repository root.

```bash
# Backend suite (about 50 s). Expect 2327 passed, 7 deselected before new work.
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -o addopts="" -q

# Frontend
cd web && npx tsc -b && npx vitest run && npm run build && cd ..

# Corpus benchmark: 87 offline replays, no network, about 40 s
S="work/upgrade-20261001/scratch/pipeline-impl"
export PYTHONPATH="E:/Work Files/SRT Sync/src" DUBSYNC_SRC="E:/Work Files/SRT Sync/src" PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -B "$S/bench_all.py" --out "$S/bench-<tag>" --jobs 4
.venv/Scripts/python.exe "$S/show.py" "$S/bench-final3/summary.json" "$S/bench-<tag>/summary.json" saved nollm

# Human-reference benchmark
REPLAY_TAG=<tag> .venv/Scripts/python.exe -B "$S/replay_head.py" all
REPLAY_TAG=<tag> .venv/Scripts/python.exe -B "$S/golden_bench.py" --json "$S/golden-<tag>.json"

# Live run (paid; a 4-minute clip costs about $0.05 to $0.20, mostly Gemini)
env -u OPENROUTER_API_KEY .venv/Scripts/python.exe -m dubsync sync "testing 2/003.srt" "testing 2/003.wav" \
  -o work/upgrade-20261001/live/out.srt --providers provider.yaml --workdir work/upgrade-20261001/live/wd-<tag>
```

Acceptance gates for every change:

1. A failing regression test first, then the fix. Tests go in `tests/`, pytest style, matching neighbours.
2. Full backend suite green.
3. Corpus benchmark: no replay failures; errors, held cues and fragment cues do not rise without an explained reason.
4. Human-reference benchmark: word error does not rise on episodes 11 and 17; error on human-edited edges does not rise.
5. For timing changes: start-minus-onset and end-minus-offset distributions on the four long runs do not get worse (`scratch/timing-impl/bench_metrics.py`).
6. For anything touching providers or prompts: one live run per model before declaring it done.

---

## 7. Remaining work

Ordered by value for the owner's priorities. Each item names the evidence, the files, the concrete change and how to accept it.

### R1. Close the known defects (small, do first)

| # | Defect | Where | Change | Accept when |
|---|---|---|---|---|
| R1.1 | Lyric cues marked with only an opening or only a closing `♪` stay error-level holds. The aligner treats a note anywhere as a lyric; the pipeline guard needs a note at both ends. | `aligner._is_song_lyric_cue`, pipeline guard near `is_song_caption_cue` (about `pipeline.py:2570`) | One shared predicate, or convert any block the aligner classified as lyric. | `scratch/review/case_lyric_partial.py` gives two `song_lyric_source_kept` notes and no error. |
| R1.2 | Minimum-duration extension in refinement ignores a following held cue and delays it by up to `min_cue_dur`. | `timing_refinement.py` near line 247 (`dialogue_cues` excludes protected cues) | Use protected non-screen-text cues as end caps, as rebuild's `_next_acoustic_start_by_cue` does. | `scratch/review/case_minext_held.py`: the hold keeps its 1.300 s start. |
| R1.3 | Joining a one-letter residue before the next cue gives "e Ele ainda ...". Also after a prefix insertion: "Eu Vou chamar a polícia.", "Cuidado, Olha pra frente.", "sem A checagem". | `cue_segmentation.py` near line 269; prefix insertion in `changes.py` | Lower-case the following word when it is no longer sentence-initial, unless it is a name or the matched ASR word is capitalised; capitalise the new first word when the cue started a sentence. Several tests pin the current casing; update them deliberately. | The three examples come out correctly cased; the punctuation validator still passes. |
| R1.4 | Episode 17 still has a lone "E" cue. MAI word ends are tight, so "E" (ends 234.639) and "qual" (starts 234.84) are 0.201 s apart and miss the 0.2 s attach rule. | ad-lib attach rule in `pipeline.py` (note P-11 in `understand/pipeline.md`) | Decide attachment from the speech burst: attach when both words are in one burst or the gap contains no silence, instead of a fixed 0.2 s on word times. | No one-letter cue in the episode 17 replay; `fragment_cues` in the corpus benchmark falls from 10. |
| R1.5 | Words in a different order than the human file on episode 11 (MAI) went from 10 to 13 after the word-placement package; ownership shifts went from 8 to 7. Not investigated. | `detached_speech.py`, `changes.py` distribution | Diff the 13 against the 10 with `scratch/placement-impl/tools/` and `golden_bench.py`'s moved-token list. Fix if they are real regressions; record the reason if the human file is the outlier. | A written explanation, and a fix or a test for each real case. |
| R1.6 | A confident `keep_srt` across several cues still splits the owned words by equal count; one cue in episode 11 (Scribe, cue 403) is timed around an "Ah," spoken 21 s later. | `pipeline._alignment_with_decision_words` (rebuild note BUG-10) | Partition kept words by acoustic gaps and the cues' own matched tokens. | No `timing_outlier_trimmed` on that cue. |
| R1.7 | A re-decoded copy with a gap above 0.1 s is not treated as a copy ("Feliz Feliz ano novo", 0.14 s). Doubled non-numeric runs from MAI are kept and flagged. | `mai_transcribe.py`, `detached_speech.py` | Use the 10 ms energy envelope: a copy whose interval has under half of its frames active is not real speech. Genuine repetitions ("Schnell, schnell!", "nein, nein") must survive. | The 11 observed cases plus the chorus case are single; the pinned genuine-repetition tests still pass. |
| R1.8 | Word-edge repair runs only before rebuild. Alignment, ad-lib attach and text rules still see raw Scribe words, so the two models behave differently there. | `pipeline.sync_episode` | After R1.4, move `speech_evidence_for_words` to directly after ASR and give every stage the repaired words. An earlier attempt was backed out because Scribe then produced more separate interjection cues under the fixed 0.2 s rule. | Corpus and human-reference benchmarks do not get worse; Scribe and MAI cue counts for episode 11 move closer. |
| R1.9 | Generate mode does not read `timing.phrase_edge_snap` or the model id; it uses default repair windows. | `transcription.py` | Pass the provider config and model id as sync does. | A unit test with a per-model override. |
| R1.10 | `style:min_duration` still warns on cues the minimum-duration rule no longer reports, including cues within one frame. `missing_audio_source_cue_held` duplicates `missing_audio_timing_held` on non-lyric cues. `asr_word_clamped` is one flag per word. | `verify.py`, `pipeline.py` near 2191, `qc_review.py` | Align the lint with the rule; fold the duplicates in `qc_review.py` (emitters stay). | Review-item counts in `bench_qc.py` do not rise; raw duplicates are grouped. |
| R1.11 | Smaller items: `adapter_from_config()` still defaults an `asr` section without a provider to ElevenLabs; Scribe words with `start` or `end` of `None` raise `TypeError`; with language `auto`, MAI's per-chunk detection flipped to Catalan on episode 17 (infer the language from the source SRT); MAI has no per-chunk cache, so a late failure discards paid chunks; the cache key includes non-semantic parameters such as `timeout_seconds`; forced alignment (disabled by default) has its own minimum-duration rule and no score gate; `_qc_result_metadata` re-parses the QC JSON on every status poll; quote balancing covers double quotes only; spans above 120 s stay held (none in the corpus). | as named | Fix opportunistically with tests. | Suite green. |

### R2. Adjudication quality (largest remaining gain for priority 2)

Evidence, from an offline evaluation of 632 real MAI cases on episodes 11 and 17 against the human files (`understand/adjudication.md`):

- Gemini 3.8 Flash with local context beats 3.5 Flash-Lite on the same cases: 151 against 224 errors, and 57 against 150.
- The current hybrid (Lite first, Flash review) scored 313 / 135; "always follow ASR" scored 420 / 223. Self-reported model confidence carries no signal.
- 123 of 317 escalations to the review model (39%) are deterministic classes: colloquial reductions (`pra`/`para`, `tá`/`está`; the human split 35 / 37), misheard source names (about 50 of 54 should keep the source), spacing-only differences and abbreviations (16 of 16 keep the source).
- When MAI and Scribe agree on a substitution, the human sided with the ASR 94% of the time (see R5).

Plan:

1. **Deterministic pre-decisions before any LLM call** (`adjudication.py`): spacing-only and hyphenation-only differences keep the source; abbreviation against its spoken form keeps the source; a source proper name against a near-homophone ASR spelling keeps the source (build the name lexicon from capitalised tokens that recur in the source SRT); register reductions follow a config knob `adjudication.register_policy: script | spoken`, default `script`. These decisions carry confidence 1.0 and are never gated.
2. **Model routing** (`provider.yaml`, `hybrid_adjudication.py`): evaluate Gemini 3.8 Flash at medium thinking for all remaining cases against the current hybrid. Expected cost is 20 to 40% more adjudication spend. Adopt it if the live comparison on episodes 11 and 17 confirms the offline result.
3. **Prompt v12** (`llm_providers.py`, bump `_ADJUDICATION_PROMPT_VERSION`): state the language; mark the divergent span inside its full cue text; state the policies from step 1 explicitly; ask for `heard_text` plus an evidence enum (`heard_clearly`, `heard_unclear`, `not_audible`) instead of a free confidence number; keep "never move words between scenes, never return timestamps".
4. **Risk-based escalation**: accept a primary answer without review only when it equals the ASR hypothesis and the case is not in a risky class (names, numbers, negations, single-word substitutions).
5. **Per-case caching** with a stable key, so one changed case does not invalidate a batch.
6. **Optional**: send proper names from the source SRT as Scribe `keyterms` and MAI `phraseList`. This raises Scribe cost by about 20%; ask the owner first.

Accept when: on fresh live runs of episodes 11 and 17 with MAI, word error against the human files is lower than 4.11% and 1.87%; words wrongly changed (the benchmark's `false` count) do not rise; cost per episode is reported from `cost.json`. Budget: a 49-minute episode costs roughly $1 to $2 in Gemini calls per full run; plan for two to four runs.

Owner question that belongs here: should detached interjections ("Hã?", "Ei,", "Hum.") become their own cues? The tool inserts about 35 per episode; the human deleted 13 and merged 9 into neighbours. A knob `generation.interjection_policy: keep | merge | drop` with default `merge` (attach to the adjacent cue of the same speaker when within one burst, otherwise keep) is the proposal; confirm with the owner before changing the default.

### R3. Documentation and tooling

1. `README.md`: the QC section (review, changes, notes, diagnostics; verdict tiers; `changes.diff.srt` as a change log; cue scores only with real evidence), the adaptive speech detector, `timing.min_duration_policy`, `timing.phrase_edge_snap`, MAI as default, the new flags, the retry behaviour, and the Japanese and long-audio notes. Keep `tests/test_documentation_acceptance.py` passing.
2. A dated report under `docs/testing/` with the numbers from section 3 and the limits from section 3.2.
3. Promote the benchmark into the repository: `scripts/replay_offline.py`, `scripts/bench_corpus.py`, `scripts/golden_bench.py`, `scripts/timing_vs_audio.py`, reading the corpus manifest from a path argument, with a small fixture-backed test each. The manifest and media stay outside git.
4. Remove `reports.write_changes_diff` and its four tests once nothing calls it.
5. Run the Playwright suite (`cd web && npm run test:e2e`) with the new default model and fix what it finds.

### R4. Audio-to-SRT generation (`understand/generation.md`)

1. Neighbour cap: the tail plus independent floor and ceil rounding creates overlap errors between generated cues (31 with Scribe on one episode, 181 without VAD). Cap each end at the next start. Reuse the sync overlap resolver.
2. Style presets' `min_cue_duration` and `max_cps` are only flagged, never applied (the broadcast preset gives 143 errors and 293 warnings on 646 cues). Apply them with the same extend-into-silence rule as sync, start stays acoustic.
3. Line wrapping: greedy wrapping matches the German house line breaks 17% of the time, balanced wrapping 78%. Make balanced, orphan-aware wrapping the default.
4. Sentence boundaries: do not split after ordinals, abbreviations or mid-sentence ellipsis.
5. SRT parser: accept period timestamps, empty cues and non-UTF-8 input with a clear message; do not silently merge cues that lack a blank line. `write_srt` must never emit a blank line inside a cue.
6. Profanity masking is language-blind and its lexicon disagrees with customer data (it masks `idiot`, `dummkopf`, `trottel`, which customers leave; it misses `Abschaum`). Gate it by language and tier the lexicon. Ask the owner which terms and languages to mask.
7. Sample-derived style: use robust percentiles instead of raw extremes, and detect fps.

Accept when: generated output for two corpus episodes per model has zero overlaps, no cue under the preset minimum with free room, and the QC review list is short.

### R5. Optional dual-model cross-check (ask the owner first; adds about $0.22 per audio hour)

Evidence (`understand/asr_compare.md`, `understand/asr.md`): the two models agree on 96 to 98.6% of tokens; words both produced match the reference 96.6% of the time, words only one produced 36 to 48%. On episode 11, in 17 MAI spans Scribe agreed with the script, and the LLM still applied MAI's wording in 7.

Design:

- `asr.cross_check: {provider: elevenlabs, model_id: scribe_v2}` runs a second transcription. MAI stays the timing source.
- Align the two word streams (the aligner already does this for SRT tokens).
- A substitution both models heard is pre-accepted without an LLM call. A substitution only the primary heard, where the secondary matches the script, is kept as script unless the LLM confirms it with `heard_clearly`. An insertion only one model heard goes to review instead of becoming a cue.
- Expose a web toggle; record both costs in `cost.json`.

First step either way (about $1): transcribe the 19 corpus episodes that have no cached transcript and the missing model for the others, so every episode has both. The corpus manifest lists what is missing.

### R6. Remaining structural items (do after R1 to R4, each needs measurement first)

1. Cues still at source timing with partial evidence (112 dialogue cues in the corpus): time them from matched words plus the enclosing speech burst when exactly one burst lies between the confidently timed neighbours (`recue.timing_evidence_issue`, note "Improvement F" in `understand/timing.md`).
2. Transposition-aware spans: an actor swapping two phrases becomes two unrelated cases today (`understand/aligner.md`, U6).
3. Merge spoken dialogue into an overlapping lyric or bracketed screen-text cue as the human editor does, instead of trimming or leaving the overlap (2 to 4 screen-text overlaps per episode remain unflagged).
4. Full mixes with music: the energy detector cannot see pauses (episode 02). A spectral detector would need a model file and a runtime the host cannot afford by default; make it an optional extra and ask the owner whether deliveries are ever full mixes.
5. QC precision: add two detectors with high precision against the human edits: a cue that starts more than 150 ms before its first matched word, and a cue kept at source timing although its words matched. The largest remaining review class is "wording moved between cues" (10 to 15 per long episode, 50 to 67% precision).
6. German article inflections (`ein`, `eine`, `einen`) share one alignment key; making them distinct exposes real improvisations but lowers coverage on two episodes. Product decision.
7. Japanese: no real audio exists locally. Ask the owner for one clip with its subtitle file, then run both models end to end.

### R7. Release

1. All gates in section 6 green, plus Playwright.
2. One live sync per model on a German clip and on a Portuguese long episode; inspect the SRT and the QC report by hand.
3. Summarise for the owner: results, behaviour changes (section 2), cost per episode, open decisions.
4. Only with the owner's approval: confirm `OPENROUTER_API_KEY` is set on Render (MAI is the default and there is no fallback), merge to `main`, and push. Pushing `main` deploys. Then check `/api/health`, `/api/config` (`jobs_available: true`), and run one short paid job through the web UI.

---

## 8. Questions for the owner

Ask these together, once, when the work that depends on them is next:

1. Interjections: own cue, merged into the neighbour, or dropped? (R2)
2. Register: follow spoken reductions (`pra`, `tá`) or keep the script? Current default keeps the script. (R2)
3. May names from the source SRT be sent to the ASR as key terms (about 20% more Scribe cost)? (R2)
4. Is a second transcription per job acceptable for the cross-check? (R5)
5. Profanity: which languages and terms are masked? (R4)
6. Are deliveries always voice-only stems, or sometimes full mixes with music? (R6.4)
7. When the voice track is silent where the script has dialogue: keep the cue at source timing with a review item (current), drop it, or fail the job?
8. Is 31 to 35 characters per second acceptable when the timing is acoustically right? The human accepted it; the review threshold is 35.
9. Which subtitle tool and frame rate does the customer's editor use? Both corrected files look like a floor-based 30 fps tool.
10. A Japanese sample clip with its subtitle file. (R6.7)
