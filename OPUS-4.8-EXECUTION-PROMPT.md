# Execution prompt for Claude Opus 4.8 — build "DubSync"

> Copy everything below the line into a fresh Opus 4.8 agent session opened in the `SRT Sync` workspace.
> Prerequisites in the workspace before you start the agent: `PLAN.md`, `Examples/srt test.srt`, and a `.env` you create with whichever keys you have (`ELEVENLABS_API_KEY`, `GEMINI_API_KEY`, optional `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Recommended but optional: one real test pair in `Testdata/` (`EP01.srt` + `EP01.wav`).

---

## ROLE

You are a senior post-production engineer and Python architect. You have shipped subtitle/dubbing tooling used on streaming-platform localization pipelines, and you know that in post-production a silently wrong output is worse than a loudly flagged one. You are building a production tool, not a demo.

## MISSION

Build **DubSync**, a Windows-friendly Python 3.11+ CLI app that synchronizes a customer-supplied target-language SRT to the dubbed VO-only audio (WAV/MP3) of a drama episode, and reconciles the SRT text with what the actors actually said. Read `PLAN.md` in this workspace first — it is the authoritative plan (architecture §6, algorithms §7, milestones §10, metrics §11). This prompt restates the binding constraints; where details differ, `PLAN.md` wins.

Inputs per episode: `episode.srt` (text mostly right, timing wrong, some lines improvised by dub actors) + `episode.wav|mp3` (clean dialogue-only dub track).
Outputs: `episode.synced.srt` + `qc_report.html` + `qc_report.json` + `changes.diff.srt` + resumable stage artifacts in a `workdir/`.

## PRODUCT TRUTHS (never violate)

1. **Timing comes only from acoustic models** — ASR word timestamps and (optional) MMS forced alignment. LLMs never produce or adjust timestamps; research shows LLM audio timestamps drift by seconds.
2. **LLMs do language reasoning only**: improvisation adjudication (keep SRT / use audio / hybrid), speaker→character mapping, punctuation/casing. The punctuation pass must be guarded by a validator that diffs alphanumeric content before/after and rejects any word change.
3. **Preserve the customer's cue segmentation** wherever text is unchanged: same cue count, same line breaks, same text — only retimed. Sentences stay deliberately split across multiple short cues (see house style). Never collapse to one-sentence-per-cue.
4. **Flag, don't guess.** Every text change, dropped line, overlap, low-confidence adjudication, and source-SRT anomaly must appear in the QC report. Confidence gate default 0.7.
5. **Deterministic core.** Parsing, alignment, re-cueing, frame-snapping, and style enforcement are pure, unit-tested Python. API calls are cached on disk keyed by content-hash (audio SHA-256 + model + params) so re-runs are free and tests never need live APIs.

## HOUSE STYLE (derived from `Examples/srt test.srt` — German dub-script standard; verify against the file yourself in M1)

- Timestamps snap to a frame grid; the example is 30 fps (multiples of 33.33 ms, floor-truncated to ms: `,033/,066/,466/,666`). fps must be auto-detected from the input SRT, configurable (23.976/24/25/29.97/30).
- Max 2 lines per cue; max ~26 characters per line (config `max_chars_per_line`, default 26); long compounds may hyphen-split across lines ("Drachen-\nEvolutionssystem").
- Cue durations observed 0.5–3.5 s; enforce `min_cue_dur` 0.5 s; no hard max — timing follows speech.
- Zero-gap chaining between consecutive cues is legal (`11,666 --> 11,666` boundary); cues of the same speaker never overlap; continuation punctuation carries sentences across cues (comma at cue end), terminal `. ? !`, ellipses `...` for trailing/interrupted lines.
- No speaker names or dashes in the output SRT (speaker info lives in the QC report), unless `overlap_policy: dash` merges a true simultaneous exchange into one two-line dashed cue.
- The example file contains real-world dirt: trailing spaces, and cues 33–35 are a scrambled-text source error. Your parser must tolerate the former; your pipeline must survive and QC-flag the latter (`source_error`).
- Implement `dubsync profile <sample.srt>` that re-derives all of the above into `style_profile.yaml` so a new customer standard is a config change, not a code change.

## STACK (fixed)

Python 3.11+, `uv` for env/deps, Typer + Rich CLI, `pysubs2` for subtitle I/O, `ffmpeg` via subprocess for audio normalize (16 kHz mono WAV), `rapidfuzz` for token similarity, `pydantic` for artifact schemas, `pytest` for tests. Providers behind adapters:
- `ASRAdapter`: **ElevenLabs Scribe v2** primary (`model_id="scribe_v2"`, `timestamps_granularity="word"`, `diarize=True`, keyterm prompting with character names when provided) → normalize to `WordStream` = list of `{text, start, end, confidence, speaker_id}`. Optional extras: `whisperx` (local fallback), `assemblyai`, `openai whisper-1` (`verbose_json` + `timestamp_granularities:["word"]`, no diarization).
- `LLMAdapter`: **Gemini 3.5 Flash** (`gemini-3.5-flash`, GA) is the default for **both** the adjudication and punctuation passes — structured JSON output, 1M context (whole episode + ASR JSON in one call), native audio input for optional snippet double-checks; set thinking level `low` for punctuation batches and default/dynamic for adjudication; use context caching for the episode transcript across passes. Include OpenAI and Anthropic implementations of the same interface, and make the model per-pass configurable in `providers.yaml` (e.g. escalate adjudication to a Pro-tier model on difficult episodes). All calls request JSON conforming to pydantic schemas; on invalid JSON retry once, then degrade to `keep_srt` + QC flag.
- Optional extras (guard imports; app must run without them): `ctc-forced-aligner` (MMS) for the precision verify pass; `pyannote.audio` 4.x community-1 for local overlap detection.

## PIPELINE (implement exactly; artifacts as JSON in `workdir/<episode>/`)

`ingest → asr → align → adjudicate → rebuild → verify`

- **align**: normalize tokens (lowercase, strip punct, NFC, per-language number handling, CJK falls back to char-level); band-limited weighted Needleman–Wunsch (monotonic) SRT-tokens ↔ ASR-words, match = rapidfuzz ratio (anchor grade ≥ 0.85), band from coarse cue-order pre-alignment. Output anchors (runs of ≥3 matches), divergence spans (bounded by anchors, carrying both texts + confidences + speaker ids + inherited boundary timestamps), unmatched-cue list.
- **adjudicate**: heuristics first (punctuation-only diff → keep SRT; tiny Levenshtein ASR noise → keep SRT; SRT cue over silence → dropped-line flag; speech with no SRT → ad-lib case). Rest → LLM in per-scene batches (scene = gap > 4 s or configurable) with 2 cues of context each side. Contract per case: `{case_id, verdict: keep_srt|use_audio|hybrid, final_text, confidence, speaker, character, reason}`.
- **rebuild**: unchanged cues keep text verbatim, retimed: start = first word start − `lead_in`(0 ms), end = last word end + `tail`(40 ms), floor-snap to grid, extend to `min_cue_dur` into available gaps, monotonic per speaker, allow 0-gap chaining. Changed spans re-flow to house style mimicking original segmentation density (split at clause boundaries; ≤2×26; balanced lines), each new cue timed from its own words. `drop_policy: keep_flagged` default. `overlap_policy: stack` default (`dash`, `flag_only` also implemented). Then the punctuation LLM pass (scene-level, words frozen, validator-enforced).
- **verify**: VAD/silence gate (no cue on <−45 dBFS span); optional MMS forced-align of final text to refine timestamps; per-cue score; style lint (grid, line rules, durations, overlaps); emit QC report (HTML + JSON: every flag with cue id, old→new text, timestamps, reason, confidence) and `changes.diff.srt`.

## MILESTONES — execute in order, each is one commit with green tests before moving on

Work through M0–M8 from `PLAN.md` §10: M0 scaffold → M1 SRT engine + style profile → M2 audio/ASR adapters + cache → M3 aligner → M4 re-cue engine → M5 LLM adjudication + punctuation → M6 speakers/overlap → M7 verify/QC → M8 E2E + batch + README + cost meter. Exit criteria per milestone are listed there; treat them as acceptance tests and write them as pytest where possible.

Testing rules:
- Unit tests never hit live APIs — use recorded fixture JSON (create `tests/fixtures/` with synthetic `WordStream`s and a miniature SRT modeled on the example).
- Build synthetic E2E fixtures: (a) same text, shifted/warped timing → expect 100% anchor coverage, cue count preserved, MAE < 1 frame; (b) injected improvised span → expect exact span isolation and `use_audio` path; (c) interleaved speakers with overlap → expect `stack` and `dash` outputs; (d) cue over silence → expect dropped-line flag.
- `Examples/srt test.srt` must round-trip byte-identically through the parser/writer (modulo normalized trailing whitespace — preserve a `--strict-roundtrip` test proving intentional normalization only).
- One `--live` smoke test per provider, skipped unless the key is present.
- If `Testdata/EP01.srt` + `EP01.wav` exist, run the full pipeline on them at M8 and include the QC report in your final summary; if keys are missing, run the fully-local mode and say so.

## CLI SURFACE

```
dubsync sync <srt> <audio> [-o out.srt] [--style style_profile.yaml] [--providers providers.yaml] [--workdir DIR] [--local] [--no-llm] [--fps N] [--resume STAGE]
dubsync batch <folder> [same flags]
dubsync profile <sample.srt> [-o style_profile.yaml]
dubsync report <workdir>
```
`--no-llm` = timing-only mode (still full QC); `--local` = WhisperX + no cloud calls. Print a cost meter at the end of every run (API seconds/tokens × published prices).

## DO NOT

- Do not use LLM-reported timestamps for anything.
- Do not merge or re-segment cues whose text is unchanged.
- Do not let the punctuation pass touch words (validator must hard-fail the batch).
- Do not invent SDK parameter names — verify against current provider docs before wiring each adapter (Context7/web).
- Do not add a GUI, database, or web service in v1 (Review UI is M9, out of scope unless everything else is done and verified).
- Do not swallow errors: a failed stage stops that episode with a clear message and leaves resumable artifacts.

## FINAL ACCEPTANCE (self-check before you report done)

1. `pytest -q` fully green; list the test count.
2. `dubsync sync` on synthetic fixture (a) meets: ≥90% cue starts within ±1 frame, cue count preserved, 0 style-lint violations.
3. Improv fixture (b): span detected, text replaced, QC lists old→new with confidence.
4. `dubsync profile Examples/"srt test.srt"` reproduces the house-style table from `PLAN.md` §2.
5. README.md: Windows quickstart (uv, ffmpeg, .env), provider matrix, config reference, cost table, troubleshooting.
6. Report: what works, what's stubbed, measured timings/costs, and the top 3 risks you'd tackle next.
