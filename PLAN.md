# DubSync — SRT Re-Timing & Improvisation Reconciliation for Dubbed Drama

**Plan v1.0 — 2026-07-02**
**Deliverable of this document:** a complete, executable construction plan for an app that takes (a) a customer-supplied target-language SRT with wrong timing and (b) the VO-only dubbed audio (WAV/MP3), and outputs a frame-accurate, house-style-compliant SRT whose text matches what the actors actually said — including improvised lines, multi-speaker scenes, and context-correct punctuation.

---

## 1. Executive summary

The customer's SRT text is *mostly* right but its timing is wrong and some lines were improvised by the dub actors. Off-the-shelf subtitle sync tools (ffsubsync, alass) only shift/stretch cues globally — they cannot re-time each cue individually, cannot detect changed dialogue, and cannot reason about speakers or punctuation. 

DubSync solves this with a five-stage pipeline:

1. **ASR** the VO-only audio with word-level timestamps + speaker diarization (ElevenLabs Scribe v2 primary; WhisperX local fallback).
2. **Anchor-align** the SRT text to the ASR word stream with a fuzzy dynamic-programming aligner → every SRT word gets an audio timestamp; divergent spans are isolated as *improvisation candidates*.
3. **Adjudicate** each divergent span with an LLM (**Gemini 3.5 Flash** default; GPT-5.x / Claude / Gemini Pro pluggable) that sees scene context, both text versions, ASR confidence, and (optionally) the raw audio snippet → decides "SRT is right (ASR error)" vs "actor improvised (use spoken text)" vs "hybrid".
4. **Re-cue deterministically**: rebuild each cue's start/end from its words' timestamps, snap to the frame grid, enforce house style (max 2 lines, ~25 chars/line, min duration, chaining), preserving the customer's original cue segmentation wherever text is unchanged.
5. **Verify & report**: optional forced-alignment pass (MMS, 158+ languages) on the *final* text for maximum precision, then emit the synced SRT + a QC report that flags every low-confidence cue, overlap region, and text change for human review.

Timing always comes from acoustic models (ASR word timestamps / forced alignment). LLMs are used **only** for language reasoning (improv adjudication, punctuation, speaker/character attribution) — never as the timing source, because research confirms LLM audio timestamps drift by seconds.

Cost per 45-min episode: **≈ $0.35–0.55** in API calls (ASR ≈ $0.21 + Gemini 3.5 Flash ≈ $0.15–0.30). A fully local free mode (WhisperX + local aligner) is included for sensitive content.

---

## 2. Ground truth: what `Examples/srt test.srt` defines (the house style)

Extracted programmatically-verifiable rules from the provided German example (68 cues, ~2.8 min):

| Property | Observed value | Rule for the app |
|---|---|---|
| Timestamp grid | Every value is a multiple of 33.33 ms, truncated to ms (`,033`, `,066`, `,466`, `,666`…) | Snap all output times to a **30 fps frame grid** (auto-detect fps from input; configurable 23.976/24/25/29.97/30) |
| Cue duration | min 0.50 s (cue 18), max 3.50 s (cue 57), typical 1–2.5 s | Enforce configurable `min_cue_dur` (default 0.5 s); no artificial max — timing follows speech |
| Lines per cue | 1–2, never 3 | Hard max 2 lines |
| Line length | ≤ ~26 chars incl. spaces (e.g. "dass du Großes erreichst." = 25) | Config `max_chars_per_line` (default 26); long compounds may hyphen-split ("Drachen-\nEvolutionssystem") |
| Sentence structure | One sentence deliberately **split across 2–4 consecutive cues**, joined by continuation punctuation ("Doch dieses begehrte Talent / brachte mir nur Verrat / von der ganzen Welt.") | **Never merge cues into one-sentence-per-cue.** Preserve the customer's cue segmentation; re-time each piece |
| Inter-cue gap | 0 ms chaining allowed (cue 5→6: `11,666 → 11,666`); typical gaps 33–500 ms | Allow zero-gap chaining; never negative overlap between consecutive cues of the same speaker |
| CPS | Up to ~25 CPS (cue 17) | CPS is **not** a timing constraint (this is a dub script — timing mirrors speech). Report CPS in QC only |
| Punctuation | Continuation commas at cue end, terminal `.`/`?`/`!`, ellipses for trailing/interrupted speech ("Ich...", "Diese Energie...") | LLM punctuation pass must reproduce these conventions |
| Speaker labels | None in the SRT (no dashes, no names) | Speaker/character tracking is internal + QC-report only, unless `overlap_policy` says otherwise |
| Data quirks | Trailing spaces on some lines; cues 33–35 contain a scrambled-text error in the source | Parser must be whitespace-tolerant; pipeline must survive (and flag) source-SRT errors |

A `style_profile` module will re-derive this table automatically from any sample SRT, so a new customer standard = drop in a new example file.

---

## 3. The five hard problems (and the strategy for each)

| # | Problem | Strategy |
|---|---|---|
| P1 | **Per-cue re-timing** (global offset tools can't) | Word-level anchor alignment SRT-text ↔ ASR-words; cue start = its first matched word's start, cue end = its last matched word's end (+ configurable pad, then frame-snap) |
| P2 | **Improvised lines** (spoken ≠ SRT text) | Divergence spans from the aligner → LLM adjudication with scene context + ASR confidence + optional audio snippet double-check (Gemini audio-native) |
| P3 | **Multiple people speaking together** | Diarization with overlap detection (Scribe speaker IDs; pyannote `community-1` overlap regions as fallback). Overlapping cues get overlapping time ranges or flags per `overlap_policy` |
| P4 | **Distinguish characters/speakers** | Stable diarization cluster IDs → LLM maps clusters to character names from conversational context (names used in dialogue); optional voice-reference matching (OpenAI `known_speaker_references`, ElevenLabs speaker library) |
| P5 | **Context-correct punctuation** | Whole-scene LLM pass that re-punctuates the final text with the house conventions (continuation commas across split cues, `?`/`!` from semantics, ellipses for interruptions), constrained to *not* change words |

---

## 4. AI landscape research (July 2026)

### 4.1 ASR with word-level timestamps (the timing backbone)

| Provider / model | Word timestamps | Diarization | Languages | Price | Verdict |
|---|---|---|---|---|---|
| **ElevenLabs Scribe v2** (`scribe_v2`) | ✅ precise, built for subtitle sync | ✅ up to 32 speakers, `words[].speaker_id` | 90+ (incl. id, zh, ja, ko, de, es, pt…) | $0.22/hr (+$0.05/hr keyterm prompting) | **Primary.** Purpose-built for subtitling; keyterm prompting takes character names; audio-event tags; multi-language auto-detect |
| **WhisperX** (local, faster-whisper + wav2vec2 alignment + pyannote) | ✅ sub-100 ms after forced alignment | ✅ via pyannote | ~99 ASR / 35+ alignment models | Free (GPU recommended) | **Local/free fallback**; also the offline mode for sensitive content |
| **AssemblyAI** Universal-3 Pro / Universal-2 | ✅ | ✅ word-level (+$0.02/hr) | U3-Pro: 6 (en/es/fr/de/it/pt); U2: 99+ | $0.21/hr / $0.15/hr | Strong alternative for European targets; U3-Pro accepts 1,500-word natural-language prompts |
| **Deepgram Nova-3** | ✅ | ✅ | ~40 | ≈$0.26/hr batch (verify at signup) | Fast; fewer languages; fine as 3rd adapter |
| **OpenAI `whisper-1`** | ✅ (`verbose_json` + `timestamp_granularities:["word"]`) | ❌ | 99 | $0.006/min ($0.36/hr) | Usable, no diarization |
| **OpenAI `gpt-4o-transcribe-diarize`** | ❌ (segment-level only) | ✅ + `known_speaker_references[]` (map up to 4 named speakers from 2–10 s voice refs) | 99 | $0.006/min | **Not** a timing source; useful auxiliary for character-name mapping |
| **Gemini 2.5/3.x audio-native** | ❌ (second-level at best; documented drift of 1–3 s+, worse on long files) | prompt-based only | wide | $1/M audio tokens (~32 tok/s) | **Never for timing.** Reserved for reasoning + audio snippet verification |

Sources: elevenlabs.io/docs + pricing pages, developers.openai.com/api/docs/guides/speech-to-text, assemblyai.com/pricing + docs, ai.google.dev/gemini-api/docs (audio, pricing), github.com/m-bain/whisperX, Google AI dev forum threads on Gemini timestamp drift.

### 4.2 Forced alignment (precision re-timing of *known* text)

| Tool | Coverage | Notes |
|---|---|---|
| **`ctc-forced-aligner`** (MMS-300M) | 158–1,130 languages (ISO 639-3, `--romanize` for non-Latin) | Word-level CTC alignment of final corrected text → audio; low memory; the "gold" pass after text edits. |
| torchaudio `MMS_FA` / wav2vec2 pipelines | major languages | Same idea, heavier integration |
| Montreal Forced Aligner | per-language dictionaries | Too heavyweight/ops-y for this app; skip |

### 4.3 Diarization / overlap

- **pyannote `community-1`** (open-source, pyannote.audio 4.0): best OSS diarization; `get_overlap()` gives simultaneous-speech regions; "exclusive diarization" mode simplifies word↔speaker reconciliation.
- **pyannoteAI Precision-2** (API): ~28% more accurate, confidence scores, voiceprints — optional paid upgrade.
- Scribe v2's built-in diarization is usually sufficient since VO-only audio is clean; pyannote is the overlap-detection backstop.

### 4.4 LLM reasoning layer (pluggable)

| Model | Price (in/out per M) | Role fit |
|---|---|---|
| **Gemini 3.5 Flash** (`gemini-3.5-flash`, GA May 2026) | $1.50 / $9 ($0.15 cached input) | **Default for all LLM passes.** 1M context fits a whole episode + ASR JSON in one call; native audio input for snippet double-checks; structured outputs; thinking on by default (use `low` for punctuation batches, higher for adjudication) |
| **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`, GA July 2026) | $0.30 / $2.50 | Low-cost option for high-throughput punctuation and speaker-mapping passes; retain each pass's configured thinking level |
| Gemini 3.1 Pro / 3.5 Pro (when GA) | $2 / $12 | Optional quality upgrade for adjudication on difficult episodes |
| Claude Opus/Sonnet (Anthropic) | higher | Alternative adjudicator; no audio input → text-only usage |
| GPT-5.x (OpenAI) | comparable | Same role; structured outputs solid |

Design rule: **provider-agnostic `LLMAdapter`** with structured-output JSON schemas, so the studio can swap by API-key availability. Default: **Gemini 3.5 Flash for both adjudication and punctuation** (one model, one key, audio-capable); config can route each pass to a different model.

### 4.5 Prior art (why we must build)

- **ffsubsync**: FFT correlation of VAD signals → one global offset/framerate fix. Cannot re-time individual cues, cannot change text.
- **alass**: dynamic programming with split points → handles ad-break shifts. Still no per-cue timing, no text awareness.
- Both fail on dubbing: every cue may drift differently (dub takes are re-paced per line) and improvised text breaks any audio↔text assumption they make. They remain useful as a *coarse pre-pass sanity check* only.

---

## 5. Recommended stack

- **Language/runtime**: Python 3.11+ (Windows-first; the studio runs Windows), packaged with `uv`.
- **CLI**: Typer + Rich (progress, tables). Batch mode over folders of episodes.
- **Subtitle I/O**: `pysubs2` (or `srt` lib) with a strict round-trip test-suite; whitespace/BOM/CRLF tolerant.
- **Audio**: `ffmpeg` (bundled instructions) → 16 kHz mono WAV for all model inputs.
- **ASR adapters**: `elevenlabs` SDK (primary), `whisperx` (optional extra, local), `assemblyai`, `openai` — behind one `ASRAdapter` interface returning a normalized `WordStream` (word, start, end, confidence, speaker_id).
- **Alignment**: custom weighted Needleman–Wunsch over normalized tokens with `rapidfuzz` similarity (no heavy deps); optional embedding assist for paraphrase spans.
- **Forced alignment (optional precision pass)**: `ctc-forced-aligner` (torch) as an optional extra `[precision]`.
- **Diarization backstop**: `pyannote.audio` 4.x community-1 as optional extra `[diarize-local]`.
- **LLM adapters**: `google-genai` (default: `gemini-3.5-flash`), `openai`, `anthropic` behind `LLMAdapter` with JSON-schema structured outputs.
- **Config**: `style_profile.yaml` (house rules) + `providers.yaml` + `.env` for keys. Style profile can be auto-derived from a sample SRT.
- **Caching**: content-hash (audio SHA-256 + model + params) → cached ASR/diarization JSON on disk. Re-runs are free.
- **Review UI (later milestone)**: FastAPI + single-page review app — table of flagged cues, waveform snippet playback, accept/reject per change, re-export.

---

## 6. Architecture & data flow

```
                       ┌──────────────────────────────────────────────────┐
 customer.srt ────────►│ 1 INGEST   parse SRT, detect fps grid, build     │
 audio.wav/mp3 ───────►│           style profile, ffmpeg → 16k mono wav   │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────────┐
                       │ 2 ASR      Scribe v2 (word ts + speaker_id)      │
                       │            [cache] [fallback: WhisperX local]    │
                       │            + optional pyannote overlap regions   │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────────┐
                       │ 3 ALIGN    normalize tokens → weighted NW DP     │
                       │            SRT words ↔ ASR words (monotonic)     │
                       │            → anchors, divergence spans,          │
                       │              unmatched-cue list                  │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────────┐
                       │ 4 ADJUDICATE (LLM, batched per scene)            │
                       │   improv?  SRT-wins / audio-wins / hybrid        │
                       │   speaker→character map, overlap resolution      │
                       │   [optional Gemini audio-snippet double-check]   │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────────┐
                       │ 5 REBUILD  re-flow changed text to house style   │
                       │            (2 lines, ≤26 ch, split at clauses)   │
                       │            re-time every cue from word ts        │
                       │            frame-snap, min-dur, chaining rules   │
                       │            LLM punctuation pass (words frozen)   │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────────┐
                       │ 6 VERIFY   [optional] MMS forced-align final     │
                       │            text → refine ts; per-cue score;      │
                       │            style lint; QC report + review file   │
                       └──────────────┬───────────────────────────────────┘
                                      ▼
              synced.srt  +  qc_report.html/json  +  changes.diff.srt
```

Every stage writes its artifact to a `workdir/` (JSON), so any stage can be re-run independently and the pipeline is debuggable/resumable.

---

## 7. Core algorithms

### 7.1 Token normalization
Lowercase, strip punctuation, NFC-normalize, expand digits→words per language (or normalize both sides the same way), map unicode ellipsis, keep a pointer back to the original cue index + char span for every token.

### 7.2 Anchor alignment (SRT tokens ↔ ASR words)
Weighted Needleman–Wunsch (monotonic, global) over the two token sequences:
- match score = `rapidfuzz.ratio(a,b)` scaled; ≥0.85 similarity counts as anchor-grade
- gap penalties tuned so short function-word mismatches don't break anchors
- band-limited DP (Sakoe–Chiba around a coarse pre-alignment from cue order + cumulative duration) to keep it O(n·k), episodes align in seconds
- output: for each SRT token → matched ASR word (with its timestamps) | INSERT | DELETE

Contiguous runs of ≥N matched tokens (default 3) = **anchor regions** (timing is trusted). Runs of mismatch bounded by anchors = **divergence spans** — each becomes an adjudication case carrying: original SRT text, ASR hypothesis, ASR word confidences, speaker IDs, and the bounding timestamps inherited from surrounding anchors.

### 7.3 Divergence classification (before spending LLM tokens)
Cheap heuristics triage spans: identical-after-normalization (punctuation-only diff → auto-keep SRT), tiny Levenshtein (ASR spelling noise → keep SRT), SRT span with zero ASR speech in window (line dropped by actors → flag), speech with no SRT text (ad-lib insertion → LLM), everything else → LLM.

### 7.4 LLM adjudication contract (structured output)
Per scene batch, the model returns for each case:
```json
{ "case_id": "...", "verdict": "keep_srt | use_audio | hybrid",
  "final_text": "...", "confidence": 0.0-1.0,
  "speaker": "cluster_3", "character": "Luna | unknown",
  "reason": "one sentence" }
```
Low confidence (< threshold, default 0.7) ⇒ QC flag, never silent. Optional second opinion: ship the audio snippet (±2 s pad) to Gemini audio-native and ask "what is literally said?" — costs ~$0.001/snippet.

### 7.5 Re-cue rules (deterministic, unit-tested — no LLM)
- Source cues are sorted chronologically before alignment using `(start_ms, original_index)` while preserving original cue ids; moved cues emit `source_out_of_order` QC.
- Unchanged-text cues: keep exact original text and line breaks; only re-time. start = first word start minus `lead_in` (default 0 ms) then floor-snap, end = last word end + `tail` (default 40 ms) then ceil-snap, enforce `min_cue_dur` by extending into available gap, enforce monotonic non-overlap per speaker, allow 0-gap chaining.
- `keep_srt` adjudication keeps source wording but still attaches divergent ASR word indices to cue timing, so numeric/spelling preferences cannot cut off the actor's last word.
- Changed-text spans: re-flow into cues mimicking the original segmentation density (target ≈ original cue count for that sentence; split at clause/phrase boundaries; ≤2 lines × ≤26 chars; balanced lines; compound hyphenation last resort), then time each new cue from its own words.
- Dropped lines: configurable `drop_policy: remove | keep_flagged` (default `keep_flagged` with zero-length warning in QC, since a human must decide).
- Overlaps (P3): per `overlap_policy: stack` (overlapping cue times, default) | `dash` (merge into one 2-line dashed cue) | `flag_only`.
- VAD boundary refinement uses matched cue word timestamps when available, not only the broad cue rectangle. ASR word durations longer than `timing.max_word_duration` are clamped to the containing speech region and flagged as `asr_word_clamped`.
- The final output pass sorts by `(start_ms, end_ms, index)`, merges duplicate overlapping captions as `duplicate_cue_merged`, resolves residual same/unknown-speaker overlaps when `output.no_overlaps: true`, and asserts monotonic cue starts before `write_srt`.

### 7.6 Punctuation pass
One LLM call per scene with the *final* word sequence, cue boundaries marked, speaker/character labels attached. Instruction: adjust punctuation/casing only — a validator diffs alphanumerics before/after and rejects any word change. Applies house conventions from §2.

---

## 8. Hard-case playbook

| Case | Behavior |
|---|---|
| Actor improvises a whole line | Divergence span → LLM verdict `use_audio` → text replaced, re-flowed, re-timed; QC lists old→new |
| Actor slightly rephrases ("Na klar" → "Na gut, klar") | `hybrid`/`use_audio` per LLM; hybrid keeps SRT wording where ASR confidence is low |
| Two characters overlap | Diarization overlap region; words split by speaker_id; per `overlap_policy`; always QC-flagged |
| Crowd/walla ("SSS! SSS!") | Audio-event/low-confidence cluster; if SRT has a cue there, time to the energy envelope; else ignore |
| Line dropped in the dub | SRT cue with no matched speech → `drop_policy`, QC-flagged |
| ASR hallucination in silence | VO-only audio minimizes it; VAD gate: no cue may sit on <-45 dBFS silence |
| Source SRT scrambled/out of chronological order | Source ingest sorts by time and emits `source_out_of_order`; duplicate-overlap guard merges any remaining repeated captions before export |
| Impossible ASR word span or impossible display speed | Clamp long ASR word duration to VAD region and emit `asr_word_clamped`; verify emits `impossible_cps_fast` / `impossible_cps_slow` for QC |
| Non-Latin targets (zh/ja/ko/th…) | Tokenizer switches to char/morpheme level; MMS aligner with `--romanize`; line-length rules from style profile (full-width counting) |

---

## 9. App shape

- **v1 = CLI**: `dubsync sync EP01.srt EP01.wav -o EP01.synced.srt --style style_profile.yaml --providers providers.yaml` (+ `dubsync batch <folder>`, `dubsync profile <sample.srt>` to derive style, `dubsync report <workdir>`).
- **v1.5 = Review UI**: local FastAPI server, one screen: flagged-cue table → click = hear snippet, see SRT vs spoken text, accept/edit/reject → re-export. This mirrors the human QC pass the studio already does, just 10× faster.
- Everything runs on Windows without GPU by default (API mode); GPU optional extras enable full-local mode.

---

## 10. Build milestones (each = one PR-sized step with cold-start context brief + exit criteria)

| # | Step | Depends on | Exit criteria |
|---|---|---|---|
| M0 | Scaffold: `uv` project, Typer CLI skeleton, config loading, `.env`, workdir artifacts, logging | — | `dubsync --help` runs; CI-style `pytest -q` green on empty suite |
| M1 | SRT engine: tolerant parser/writer, fps-grid detection, `style_profile` auto-derivation from sample SRT | M0 | Round-trip byte-fidelity test on `Examples/srt test.srt`; profile output matches §2 table |
| M2 | Audio + ASR adapters: ffmpeg normalize, `ASRAdapter` interface, ElevenLabs Scribe v2 impl, disk cache; WhisperX adapter stub behind extra | M0 | Given a fixture WAV: normalized `WordStream` JSON with word ts + speaker ids; cache hit on 2nd run; unit tests use recorded fixture JSON, no live API |
| M3 | Anchor aligner + divergence spans (pure algorithm) | M1, M2 | Synthetic fixtures: shifted timing → 100% anchors; injected improv span → correctly isolated; property tests for monotonicity |
| M4 | Re-cue engine + SRT writer integration | M1, M3 | On a fixture where text is unchanged: output cue count == input, all times frame-snapped, min-dur & chaining enforced, style lint clean |
| M5 | LLM adapters + adjudication + punctuation pass (structured outputs, scene batching, word-freeze validator) | M3 | Mocked-LLM unit tests; one live smoke test behind `--live` flag; invalid JSON → retry+degrade path tested |
| M6 | Speakers & overlap: speaker_id propagation, character mapping call, overlap policies, pyannote backstop (optional extra) | M2, M5 | Fixture with interleaved speakers produces correct `stack`/`dash` outputs; QC flags emitted |
| M7 | Verify & QC: optional MMS forced-align pass, silence/VAD gate, per-cue score, `qc_report.html/json`, `changes.diff.srt` | M4, M5 | Report renders; every changed/flagged cue listed with reason + timestamps; forced-align improves fixture MAE |
| M8 | E2E + batch + docs: `dubsync sync` end-to-end on a real episode, batch mode, README, cost meter (prints $ per run) | all | One real episode processed under target metrics (§11); README quickstart works on clean Windows machine |
| M9 (opt) | Review UI (FastAPI + SPA) | M7 | Accept/reject round-trip re-exports valid SRT |

Parallelizable: M1 ∥ M2 after M0; M5 ∥ M4 after M3.

---

## 11. Evaluation & QC metrics (definition of "good")

Build a golden set from past delivered episodes (unsynced SRT + VO audio + final human-synced SRT). Targets for v1:

- **Timing**: ≥90% of cue starts within ±1 frame of golden; ≥98% within ±3 frames; MAE < 50 ms.
- **Improv detection**: precision ≥0.9 / recall ≥0.85 on spans that humans changed.
- **Structure**: 0 style-lint violations (line count/length, grid, chaining); cue count preserved for unchanged text.
- **Review burden**: ≤10% of cues flagged for human review on a typical episode.
- QC report always errs toward flagging — silent wrong output is the worst failure mode in post-production.

---

## 12. Cost model (per 45-min episode, API mode)

| Item | Cost |
|---|---|
| Scribe v2 ASR + diarization | $0.165 (0.75 h × $0.22) |
| Keyterm prompting (character names) | +$0.04 |
| LLM adjudication + punctuation (**Gemini 3.5 Flash**: ~60–100k tok in × $1.50/M + ~15–25k out × $9/M, thinking `low` on punctuation) | $0.15–0.30 (context caching of the episode transcript cuts repeat-pass input to $0.15/M) |
| Audio-snippet double-checks (~30 × 20 s) | ~$0.02 |
| Forced-align + pyannote (local) | $0 |
| **Total** | **≈ $0.35–0.55** (fully-local mode: $0) |

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ASR weak on a specific target language/accent | Adapter architecture → benchmark per language in M2 with a 5-min sample; keyterm prompting with character/world names; WhisperX fallback comparison |
| Diarization confuses similar voices (same VA voicing 2 roles) | Character mapping is advisory-only; overlaps always QC-flagged; optional voiceprint refs |
| LLM "fixes" text it shouldn't | Word-freeze validator on punctuation pass; adjudication only inside divergence spans; confidence gate |
| Non-Latin cue-length rules differ (CJK width) | Style profile stores per-script counting rules; derive from customer sample |
| Long-file API limits (25 MB OpenAI, chunking) | Chunk audio at silence boundaries with overlap stitching (only needed for fallback adapters; Scribe handles long files) |
| Hallucinated ASR text in music/pauses | VO-only input + VAD gate + confidence threshold |
| API outage mid-batch | Stage artifacts on disk; resumable pipeline; cache |

---

## 14. Defaults chosen (change in config, don't re-litigate in code)

1. Primary ASR **ElevenLabs Scribe v2**; local mode WhisperX. 
2. Default LLM **Gemini 3.5 Flash** (`gemini-3.5-flash`) for both adjudication and punctuation (thinking `low` for punctuation batches); OpenAI/Anthropic adapters included, Pro-tier escalation configurable per pass. 
3. Frame grid auto-detected, fallback 30 fps (matches the example). 
4. `overlap_policy: stack`, `drop_policy: keep_flagged`, adjudication confidence gate 0.7. 
5. Output preserves customer cue segmentation unless text changed. 
6. All timing from acoustic models; LLMs never move timestamps.

**Open questions for the studio (answers slot into config; defaults above apply meanwhile):**
- Which target languages ship first? (affects M2 language benchmark matrix)
- Typical episode length & monthly volume? (affects batch/cost tuning)
- Is a GPU machine available for local mode?
- Golden pairs available for the eval set (§11)? — highest-value asset for tuning
- House convention for on-screen simultaneous dialogue (stacked cues vs dashed merged cue)?

---

*Companion file: `OPUS-4.8-EXECUTION-PROMPT.md` — the ready-to-paste prompt that instructs Claude Opus 4.8 to build this app milestone-by-milestone.*
