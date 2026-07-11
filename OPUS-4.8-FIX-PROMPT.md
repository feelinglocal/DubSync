# Execution prompt for Claude Opus 4.8 — DubSync v1.1: bug fixes + smarter pipeline

> Paste everything below the line into a fresh Opus 4.8 agent session opened in the `SRT Sync` workspace.

---

## ROLE

You are the senior post-production engineer who owns DubSync (`src/dubsync/`). The studio ran `dubsync batch` on a real season (`work/batch 1/`) and the editors found systematic timing and ordering bugs while conforming the output in Premiere Pro. Every bug below has been root-caused against the real pipeline artifacts in `work/batch 1_work/<ep>/` — do not re-litigate the diagnosis, but do reproduce each one with a test before fixing it. `PLAN.md` §7 remains the authoritative algorithm spec.

## GROUND RULES (unchanged from v1)

- Timing comes only from acoustic evidence (ASR word timestamps, VAD, forced alignment). LLMs never move timestamps.
- Preserve customer cue segmentation when text is unchanged.
- Flag, don't guess. Silent wrong output is the worst failure.
- Deterministic core, pure functions, pytest green before every commit. Unit tests never hit live APIs; build regression fixtures from the real artifacts listed below (copy the relevant slices of `asr.json` / `align.json` / source SRT into `tests/fixtures/` — do not depend on `work/` at test time).
- Fixture provenance: episode 7 = out-of-order + duplicate case; episode 9 = early-end cases; episode 10 = giant-cue case.

## CONFIRMED BUGS — fix all, in this order

### BUG 1 (P0) — Duplicate cue text stacked at the same time; improvised lines inserted alongside kept originals

**Symptom:** `work/batch 1/7.synced.srt` cues 36 & 37 both read "seine Gefühle beeinflussen," and both start at `00:01:12,166` (Premiere renders them stacked as a 4-line caption; editors must delete/swap by hand).

**Root cause chain (verified):**
1. Source `7.srt` contains cue 42 "seine Gefühle beeinflussen," timed `00:01:12,166` but placed **at the end of the file, out of chronological order**. The aligner (`aligner.py::align_cues_to_words`) tokenizes cues in **file order**, so the monotonic Needleman–Wunsch can never match cue 42's tokens to ASR words that occur mid-stream: `align.json` shows `unmatched_cue_ids: [42]`.
2. The same spoken words surface as an insert-only divergence (`adjudicate.json` case-6 → `use_audio` "seine Gefühle beeinflussen,") → `pipeline.py::_adlib_cue_ids_by_case` + `changes.py::apply_adjudication_decisions` mint a **new ad-lib cue** with that text.
3. `recue.py::rebuild_cues` keeps the unmatched cue 42 **with its original source timing** (drop_policy `keep_flagged`) → output contains both the ad-lib cue and the original — same text, same time.
4. The LLM also adjudicated cue 42's delete-span separately (case-8 `keep_srt`) with no awareness that case-6 had already inserted the same text — the batches have no cross-case reconciliation.

**Required fixes:**
- **a. Sort cues chronologically before alignment.** In the ingest stage, stable-sort parsed cues by `(start_ms, index)` before tokenization, keep the original index for traceability, and emit a `source_out_of_order` QC flag listing the moved cues. This alone lets ep7 cue 42 match normally and prevents the whole cascade. Renumber on output (already done).
- **b. Reconcile ad-lib insertions against unmatched source cues before creating new cues.** After adjudication, for every would-be ad-lib insertion, search unmatched/dropped source cues whose original timing window overlaps the span (±3 s) or whose normalized text similarity to `final_text` is ≥ 0.8 (`rapidfuzz`, on `alphanumeric_signature`). On a hit: reuse the source cue (its index and, if `keep_srt`-equivalent, its exact text), time it from the span's ASR words, emit a single `adlib_reconciled` flag — never two cues.
- **c. Never emit source timings into the synced output.** Unmatched cues that survive `drop_policy: keep_flagged` must be re-timed by interpolation between their matched neighbors (proportional position between previous cue's last matched word end and next cue's first matched word start), then snapped to the nearest VAD speech region inside that window when one exists; flag `interpolated_timing` with `severity: warning`. A cue with customer timing inside an otherwise re-timed file is guaranteed to land wrong.
- **d. Duplicate-text guard in verify.** Any two output cues whose normalized texts are identical (or ≥ 0.9 similar) and whose time ranges overlap → auto-merge into the earlier/longer one and flag `duplicate_cue_merged` (error severity if it still required judgment).

### BUG 2 (P0) — Cue ends while the actor is still speaking

**Symptom:** `9.synced.srt` cue 18 "Das Ding ist mindestens Level 15." ends `00:00:53,166` but the actor speaks "fünfzehn" until 53.86 s. Cue 20 "Du bist ein Level-1-Versager" ends `00:00:58,366` (0.5 s min-dur floor) while speech runs to 59.62 s. (Editors marked both in Premiere.)

**Root cause chain (verified in `work/batch 1_work/9/align.json`):**
1. Cue 18's matched words stop at "Level" (53.159); "fünfzehn." vs "15" became divergence case-4 because `tokenize.py::_NUMBER_WORDS` only maps 0–12, so `fünfzehn` ≠ `15`.
2. Cue 20's matched words are only "du bist ein" (ends 58.279); "Level-eins-Versager" vs "Level 1 Versager" became case-5.
3. Both cases resolve to `keep_srt` — and `pipeline.py::_alignment_with_decision_words` only merges span ASR word indices into `cue_word_indices` for `use_audio`/`hybrid` verdicts. **`keep_srt` spans contribute no timing**, so `recue.py::_cue_timings` computes `end = max(matched word.end)` over a truncated word set.

**Required fixes:**
- **a. Timing must cover the span words for every verdict.** In `_alignment_with_decision_words`, also attach `span.asr_word_indices` to the owning cue(s) for `keep_srt` decisions (text unchanged; timing evidence only). The ASR words are what was spoken in that slot regardless of which text wins.
- **b. Widen number normalization.** Replace `_NUMBER_WORDS` with a proper per-language cardinal/ordinal normalizer covering at least 0–100 plus common compounds for the target languages in use (German first: `fünfzehn`, `zwanzig`, `fünfunddreißig`, ordinals like `dritte`), or normalize digits→words with a small rule table per language. Also strip hyphens in normalization so `Level-1-Versager` tokenizes like `Level 1 Versager`. This kills a whole class of false divergences (cheaper LLM bills, fewer keep_srt timing holes).
- **c. End snapping must not cut speech.** In `_cue_timings`, snap the end with `snap_ceil`, not `snap_floor` (starts keep `snap_floor`).
- **d. VAD end-extension:** if the cue's last timing word ends inside a VAD speech region and the region continues, extend the end to `min(region end + end_pad, next cue start)` with no arbitrary 300 ms cap on trims/extensions toward真 speech (keep a cap only against extending into the *next* cue's region). Rework `timing_refinement.py::_refined_end_ms` accordingly (see BUG 3c).

### BUG 3 (P0) — Cue stays on screen long after speech ended (up to 19 s)

**Symptom:** `10.synced.srt` cue 16 "Los geht's." spans `00:00:48,266 --> 00:01:07,400` — 19.1 s for two words. Source cue was 0.57 s.

**Root cause chain (verified):**
1. **ASR word-duration outlier:** `work/batch 1_work/10/asr.json` word 62 is `"geht's."` with `start=48.599, end=67.379` — ElevenLabs stretched the word across an 18-second silence. `vad.json` proves speech actually ends at 48.9 s.
2. `recue.py::_cue_timings` trusts `max(word.end)` blindly — no per-word duration sanity, no cue-span plausibility check.
3. `timing_refinement.py` failed to save it: `_regions_overlapping_cue` picks the **last** region overlapping the (bloated) cue — the 67.0–69.0 s region that belongs to ambience before the *next* line — so the trim logic never fires.

**Required fixes:**
- **a. Clamp ASR word durations at ingest** (in the ASR adapter normalization layer): if `word.end - word.start > max_word_duration` (config, default 2.0 s), clamp end to the end of the VAD region containing `word.start` (fallback: `start + max_word_duration`) and flag `asr_word_clamped`. Applies to all ASR adapters — this is a provider quirk, handle it centrally.
- **b. Cue-span plausibility check in `_cue_timings`:** compute `expected_dur = display_width(cue.plain_text) / max_cps` (config, default max_cps 30) … `expected_dur_hi = width / min_cps` (default 4). If the matched-word span exceeds `expected_dur_hi` by more than a frame or contains an inter-word gap > `max_intra_cue_gap` (default 1.5 s), keep only the **largest dense cluster** of matched words (gap-based clustering), re-time from that cluster, and flag `timing_outlier_trimmed`. Same guard for divergence-span timing.
- **c. Rewrite VAD boundary refinement region selection:** pick the speech region(s) that contain (or are nearest to) the cue's **word timestamps**, never "whatever overlaps the possibly-bloated cue rectangle". Trim ends down to `region_end + end_pad` whenever the cue extends beyond its own speech region into silence, regardless of how large the excess is (remove the `max_trailing_silence`-only-trims asymmetry). Preserve `min_cue_dur` afterwards.
- **d. CPS sanity net in verify:** flag any output cue with CPS > 30 (`impossible_cps_fast` — text can't fit) or < 2 with duration > 2 s (`impossible_cps_slow` — cue lingering over silence). These two rules alone would have caught bugs 2 and 3 automatically; wire them into `qc_report` summary counts.

### BUG 4 (P1) — Output ordering not globally enforced

`changes.py::apply_adjudication_decisions` sorts merged cues by **original/customer** `start_ms` while ad-lib cues carry **ASR** times — apples vs oranges; and after `rebuild_cues` + refinement re-time everything, nothing re-sorts. `write_srt` writes list order.

**Required fix:** add a final deterministic ordering pass immediately before `write_srt` in `_run_verify_stage`: stable-sort by `(start_ms, end_ms, index)`, then resolve residual same-speaker overlaps (`_enforce_monotonic` semantics, but global and after ALL timing passes), re-snap, and assert monotonic non-decreasing starts (raise on violation — this is a pipeline invariant, not a QC flag). Remove the misleading intermediate sort in `apply_adjudication_decisions` (keep positional insertion of ad-libs next to their span's neighbor cues instead).

## MAKE IT SMARTER (P1/P2 upgrades, after the bugs are fixed and tests are green)

1. **Ad-lib and replacement placement contract in the adjudication schema.** Add optional `replaces_cue_ids: list[int]` and `insert_after_cue_id: int | null` to `AdjudicationDecision`; extend `_adjudication_prompt` so each case carries its neighboring cases and nearby unmatched source cues, with the instruction: "If the spoken insertion corresponds to a nearby source cue (dropped, out-of-order, or paraphrased), return `replaces_cue_ids` instead of allowing a duplicate insertion." Deterministic code (BUG 1b) remains the safety net; the LLM signal just improves the first pass.
2. **Mini-alignment for multi-cue replacements.** `pipeline.py::_partition_contiguous` splits span words across cues by equal count; `changes.py::_split_text_for_cues` splits text by width. Replace both with one mini Needleman–Wunsch between `final_text` tokens and the span's ASR words (reuse `aligner._align_tokens`), so each rebuilt cue owns exactly the words it displays, and cue boundaries land at word boundaries with correct per-cue timing.
3. **Re-flow into extra cues instead of overflowing lines.** `changes.py::flow_text_to_lines` stuffs overflow into the last line (exceeds `max_chars_per_line`, lint noise). When wrapped lines exceed `max_lines_per_cue`, split into additional cues following the customer's segmentation density (PLAN.md §7.5), each timed from its own words via upgrade 2.
4. **Silero VAD adapter** (`vad.py`): optional `provider: silero` (torch hub, CPU-friendly, ~1 MB) as the default when installed, energy VAD as fallback. The energy detector counted crowd/ambience at 67–69 s as "speech" in ep10; Silero separates speech from effort noises/walla far better. Keep the fixture adapter for tests.
5. **Confidence floor for adjudication evidence:** Scribe returned `confidence: 1.0` for the bloated word — do not weight ASR confidence as meaningful when the provider saturates it; prefer VAD agreement + span-duration plausibility as the confidence signal in `DivergenceSpan.confidence`.
6. **Premiere-friendly output mode:** config `output.no_overlaps: true` (default on) — after final ordering, clip overlapping cue pairs at the midpoint of their overlap (different speakers) unless `overlap_policy: stack` is explicitly set; identical-text overlaps were already merged by BUG 1d. Premiere caption tracks render stacked cues as the 4-line pileup the editors photographed.
7. **QC report upgrades:** group flags by family with counts at the top (timing / text / ordering / source); add per-cue "delta vs source" (start/end shift in frames) so editors can eyeball the 3 biggest movers; write `review.csv` (cue, in, out, old text, new text, flags) that imports cleanly into a spreadsheet.
8. **Golden regression harness:** add `tests/regression/` with the frozen slices of episodes 7, 9, 10 (source cues + asr words + vad + expected output cues). `pytest -m regression` re-runs the full deterministic pipeline (fixture ASR/VAD/LLM adapters) and diffs against the expected SRT. Each fix above lands with its episode-derived case: ep7 → exactly one "seine Gefühle beeinflussen," cue; ep9 cue 18 end ≥ 53.86 s − 1 frame and cue 20 end ≥ 59.62 s − 1 frame; ep10 cue 16 end ≤ 49.0 s + pad.

## CONFIG ADDITIONS (all with defaults; document in README)

```yaml
timing:
  max_word_duration: 2.0        # s, ASR word clamp
  max_intra_cue_gap: 1.5        # s, dense-cluster split threshold
  max_cps: 30                   # impossible-fast net
  min_cps: 2                    # impossible-slow net (duration > 2s)
output:
  no_overlaps: true
vad:
  provider: silero              # falls back to energy if torch missing
```

## DO NOT

- Do not change the public CLI surface or artifact filenames (editors have scripts on them).
- Do not re-run live ASR/LLM for tests; the `work/` artifacts already contain everything needed to build fixtures.
- Do not "fix" the punctuation validator or SRT writer style rules — they are behaving.
- Do not let any fix regress the byte-fidelity round-trip test on `Examples/srt test.srt`.

## FINAL ACCEPTANCE (self-check before you report done)

1. `pytest -q` fully green, including the new `regression` marker suite; list test counts before/after.
2. Re-run the pipeline offline (fixture adapters fed by the frozen ep7/9/10 slices) and show: no duplicate-text overlapping cues anywhere; every output cue's end within 1 frame of `max(speech end within cue, min-dur)`; global start-time monotonicity assertion passes.
3. Zero `impossible_cps_fast` / `impossible_cps_slow` flags on the regression episodes after fixes (they must fire on the *pre-fix* artifacts in a dedicated test proving the net works).
4. Update `README.md` (new config keys, Silero optional install, ordering invariant) and `PLAN.md` §7.5/§8 rows that changed.
5. Report: per-bug before/after cue timings for the four photographed cases (ep7 36/37, ep9 18, ep9 20, ep10 16), plus the top 3 remaining risks.
