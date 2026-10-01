# GPT 6 Astra Ultra execution prompt: finish the DubSync accuracy upgrade

Use the prompt below in a new GPT 6 Astra Ultra coding session opened at the root of this `SRT Sync` workspace on the owner's Windows machine. It is organised as: desired result, sources to read, binding rules, the work in order, how to verify, when to stop, and what to report.

---

## Desired result

Finish the accuracy upgrade of DubSync that is in progress on branch `upgrade/sync-accuracy-2026-10` (HEAD `971c7ed` at handover), and leave it ready for the owner to approve a release.

DubSync retimes a customer's subtitle file (SRT) to dubbed voice-over audio. It transcribes the audio with word timestamps (Microsoft MAI-Transcribe 2 through OpenRouter by default, ElevenLabs Scribe v2 as the alternative), aligns the script to the spoken words, lets a Gemini model decide wording where actors improvised, rebuilds every cue from acoustic timing, and writes a QC report.

The owner's priorities, in order:

1. Cue timing accuracy: each cue starts and ends with the real speech.
2. Correct words where actors improvised.
3. Correct word order.
4. Few QC findings shown to the customer, and every one of them actionable.
5. Both transcription models work well.

"Finished" means all of the following are true:

- Work packages R1 to R4 in the handover are implemented, tested and measured. R5 and the items in R6 are done where the owner approves them or where the handover says they need no approval; otherwise they are left with a written recommendation.
- Every acceptance gate in "How to verify" below passes on the final commit.
- Documentation matches the code.
- The owner has a short release summary and the list of decisions only they can make.
- Nothing has been pushed, merged to `main`, or deployed without the owner's explicit approval in the session.

Do the work in the repository. Do not stop after proposing a plan. Continue through the packages until the result above is reached or a stop condition below applies.

## Read first

Read these in order before editing anything:

1. `AGENTS.md` and any more specific repository instructions, if present.
2. `docs/handover/2026-10-01-accuracy-upgrade-handover.md`. This is the primary source: goals, decisions already made, current state with numbers, a map of what changed, the remaining work packages R1 to R7 with file pointers and acceptance criteria, and the open owner questions.
3. `work/upgrade-20261001/RESUME.md` for the dated status log.
4. `work/upgrade-20261001/understand/_digest.md`, then the area notes in the same folder that match the package you are about to start (`timing.md`, `aligner.md`, `rebuild.md`, `pipeline.md`, `adjudication.md`, `asr.md`, `asr_compare.md`, `generation.md`, `qc_web.md`, `qc_forensics.md`, `golden.md`, `corpus.md`, `review-wave1.md`). Their line numbers refer to commit `5e29527`; the code has moved, so locate by function name.
5. `work/upgrade-20261001/wave1-reports.json` and `wave2-reports.json` for what each earlier package did, skipped and recommended.
6. `README.md`, `PLAN.md`, `provider.yaml`, `pyproject.toml`, `render.yaml`.
7. The source you are about to change, and its tests.

The `work/` folder is git-ignored and exists only on this machine. If any file named above is missing, say so and continue from the handover document and the code.

Treat every claim in the handover and the notes as a lead to verify, not as fact. Before relying on a number or a described behaviour, reproduce it on the current tree. Where the tree disagrees with the handover, trust the tree and note the difference in your report.

## Source precedence

When sources conflict, resolve in this order:

1. The owner's instructions in this session.
2. Security, privacy and data integrity.
3. The binding rules below.
4. Decisions recorded in section 2 of the handover.
5. Existing tested behaviour.
6. The handover's work plan.
7. The analysis notes.

## Binding rules

- Timing comes only from acoustic evidence: ASR word timestamps, energy analysis of the audio, and optional forced alignment. A language model decides wording only and never creates or adjusts a timestamp.
- Never cut a spoken beginning or ending to satisfy a style rule. Never delay a cue past its own first word.
- Delivered cues do not overlap, except genuine simultaneous speech, which is flagged once.
- Where the words are unchanged, keep the customer's text, line breaks and segmentation.
- Flag uncertainty; do not guess silently. Never hide a real problem to make the QC count smaller.
- Production must run on a 512 MB, 0.5 CPU instance: pure Python, streaming audio processing, no new runtime dependencies.
- Do not change the kind names or severities of existing QC flags to reduce noise. Fix false triggers where they are raised, or classify in `src/dubsync/qc_review.py`. Every new flag kind must be added to `qc_review.KIND_REGISTRY`.
- Work only on branch `upgrade/sync-accuracy-2026-10`. Commit after each verified step with a message that says what changed and why. Do not push, do not merge to `main`, do not deploy, and do not change anything on Render. Pushing `main` deploys production.
- Paid provider calls (ElevenLabs, OpenRouter, Gemini) are allowed for validation up to a total of $24 for the rest of this upgrade. Keep a running ledger with the amount from each run's `cost.json`. Stop paid calls and ask if you would exceed it.
- Never print, log or commit secrets. Do not copy values out of `.env`.
- The workspace contains the owner's own untracked folders (test media, earlier results). Do not delete, move or rewrite them. Write scratch output only under `work/upgrade-20261001/`.
- Do not fabricate test results, provider results, costs or measurements. Distinguish fixture-backed evidence from live-provider evidence.

## The work, in order

The handover's section 7 defines each package with evidence, files, the concrete change and its acceptance test. Follow that order unless measurement gives you a reason to change it, and say so when you do.

1. **R1, known defects.** Eleven small items, most with a ready reproduction script under `work/upgrade-20261001/scratch/review/`. Do R1.4 (burst-based attach rule) before R1.8 (move word-edge repair to directly after ASR), because the earlier attempt at R1.8 failed for the reason R1.4 fixes. Investigate R1.5 (word-order count on episode 11 rose from 10 to 13) before building anything else on the word-placement code.
2. **R2, adjudication quality.** Deterministic pre-decisions first, because they need no provider and remove about 39% of escalations. Then the model-routing comparison, the new prompt version, risk-based escalation and per-case caching. This package needs live runs on episodes 11 and 17; it is the main use of the paid budget.
3. **R3, documentation and tooling.** Promote the benchmark scripts into `scripts/` early if you find yourself editing them, otherwise after R2.
4. **R4, audio-to-SRT generation.**
5. **R5, dual-model cross-check.** Build only if the owner approves the extra transcription cost. Filling the transcript matrix for the corpus (about $1) is worthwhile either way and is within budget.
6. **R6, structural items.** Each needs a measurement that shows the gain before you commit to it.
7. **R7, release preparation.** Stop before the push.

Method for every change:

1. Reproduce the problem on the current tree.
2. Write a failing regression test.
3. Make the smallest change that fixes it, in the style of the surrounding code.
4. Run the focused tests, then the full suite.
5. Run the corpus benchmark and compare with the previous baseline. Run the human-reference benchmark when text, ownership or timing could be affected.
6. Commit.
7. Record the before and after numbers in `work/upgrade-20261001/RESUME.md` under the status log.

When a change makes one metric better and another worse, do not pick silently. State the trade, decide from the owner's priority order, and record it.

When you use sub-agents or parallel workers, keep at most four active, give each a disjoint set of files or its own git worktree, and merge only work whose tests pass. After merging several packages, run an independent review pass that looks for defects at the seams between them; the previous such pass found three real defects that all package tests had missed.

## Things the handover says will trip you up

Read section 4 of the handover for the full list. The ones that cost the most time:

- `pyproject.toml` sets `addopts = "-q"`. Run pytest with `-o addopts=""` or the pass count disappears.
- The shell has a stale `OPENROUTER_API_KEY` that overrides `.env`. Run the CLI as `env -u OPENROUTER_API_KEY ...` in Git Bash, or remove the variable in PowerShell first.
- The console encoding is cp1252. Set `PYTHONIOENCODING=utf-8` before printing subtitle text.
- The human-corrected files for episodes 11 and 17 are edits of earlier DubSync output. Agreement on edges the human did not touch measures similarity to old code, not accuracy. Judge timing by the edges the human changed and by the energy-based measurements.
- `tests/test_documentation_acceptance.py` pins sentences in the README and the provider configs.
- Behaviour changes need the matching policy or prompt version bumped, because caches and resume checks key on them.
- Scribe is not deterministic between calls; compare against cached transcripts when you need repeatable numbers.

## How to verify

Run from the repository root in Git Bash. The handover's section 6 has the same commands with more explanation.

```bash
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -o addopts="" -q

cd web && npx tsc -b && npx vitest run && npm run build && npm run test:e2e && cd ..

S="work/upgrade-20261001/scratch/pipeline-impl"
export PYTHONPATH="E:/Work Files/SRT Sync/src" DUBSYNC_SRC="E:/Work Files/SRT Sync/src" PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -B "$S/bench_all.py" --out "$S/bench-<tag>" --jobs 4
.venv/Scripts/python.exe "$S/show.py" "$S/bench-final3/summary.json" "$S/bench-<tag>/summary.json" saved nollm
REPLAY_TAG=<tag> .venv/Scripts/python.exe -B "$S/replay_head.py" all
REPLAY_TAG=<tag> .venv/Scripts/python.exe -B "$S/golden_bench.py" --json "$S/golden-<tag>.json"
```

Baselines at handover: 2,327 backend tests pass with 7 deselected. Corpus benchmark with stored verdicts: 45.1 QC findings per 100 cues, 215 errors, 112 dialogue cues at source timing, 10 fragment cues, 0 failed replays out of 87. Word error against the human files: episode 11 MAI 4.11%, episode 11 Scribe 4.15%, episode 17 MAI 1.87%.

Required on the final commit:

- Backend suite, frontend typecheck, unit tests, build and Playwright all pass. Do not lower a threshold or delete a test to get there; a test that pins behaviour you changed on purpose may be updated, and you list each one with the reason.
- Corpus benchmark: no failed replay; errors, held dialogue cues and fragment cues are at or below the baselines above, or each rise is explained.
- Human-reference benchmark: word error at or below the baselines; error on human-edited edges not higher.
- Timing against the audio on the four long-episode runs is not worse than the handover's section 3.1 figures.
- One live sync per model on a short German clip, and one live MAI sync of a Portuguese long episode, each inspected by hand: no overlapping cues, no cue that starts after its first word, a short and sensible review list, and the cost recorded.

## Stop and ask the owner when

- A decision in the handover's section 8 blocks the next step. Collect the open questions and ask them together, once, rather than one at a time. Carry on with everything that does not depend on the answers.
- You would exceed the paid budget.
- A change would alter customer-visible behaviour beyond what the handover's decisions cover.
- You are ready to push, merge or deploy.
- You find a security or privacy problem.

Do not stop for ordinary obstacles such as a failing test, a missing scratch file or a flaky provider call. Diagnose and continue.

## Final report

Keep it short and evidence-based. Lead with the outcome, then:

1. What you finished, package by package, with the before and after numbers that prove it.
2. What you changed that a customer would notice.
3. Exact final results of every command in "How to verify", including counts.
4. Live runs: which clips, which model, cost, and what you saw in the output.
5. Total paid spend against the $24.
6. What you did not finish and why, with the next concrete step for each.
7. The decisions waiting on the owner.
8. The exact branch and commit, and a statement that nothing was pushed or deployed, or what the owner approved.

Do not claim an improvement you did not measure. Do not describe the product as released or deployed unless the owner approved it and you verified the live service.
