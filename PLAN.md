# DubSync — SRT Re-Timing & Improvisation Reconciliation for Dubbed Drama

**Plan v1.1 — updated 2026-10-02**
**Deliverable of this document:** a complete, executable construction plan for an app that takes (a) a customer-supplied target-language SRT with wrong timing and (b) the VO-only dubbed audio (WAV/MP3), and outputs a frame-accurate, house-style-compliant SRT whose text matches what the actors actually said — including improvised lines, multi-speaker scenes, and context-correct punctuation.

The construction milestones and provider research below retain historical planning context. Current behavior follows `provider.yaml`, the [README](README.md), and the [October 1 accuracy-upgrade report](docs/testing/accuracy-upgrade-2026-10-01.md). That report records technical checks, matched offline measurements, actual SRT/QC inspections and unresolved evidence limits. The October 2 follow-up requires correction from audio and a global two-line ceiling. Local validation and approval to publish remain separate.

---

## 1. Executive summary

The customer's SRT text is *mostly* right but its timing is wrong and some lines were improvised by the dub actors. Off-the-shelf subtitle sync tools (ffsubsync, alass) only shift/stretch cues globally — they cannot re-time each cue individually, cannot detect changed dialogue, and cannot reason about speakers or punctuation. 

DubSync solves this with a five-stage pipeline:

1. **ASR** the VO-only audio with word-level timestamps + speaker diarization (Microsoft MAI-Transcribe 2 via OpenRouter by default; ElevenLabs Scribe v2 stays selectable; WhisperX is an optional local route). A provider failure never silently switches MAI and Scribe.
2. **Anchor-align** the SRT text to the ASR word stream with a fuzzy dynamic-programming aligner → every SRT word gets an audio timestamp; divergent spans are isolated as *improvisation candidates*.
3. **Adjudicate** each divergent span: first resolve proven equivalents under the actor-spoken default policy, then ask **Gemini 3.8 Flash** at medium thinking with focused audio and local ownership/context. Native prompt v12 returns literal `heard_text` and hearing evidence. Lite high followed by Flash medium review remains an opt-in YAML route. Gemini 3.7 Flash handles bounded text-only punctuation with medium thinking; Luna stays on speaker mapping.
4. **Re-cue deterministically**: rebuild each cue's start/end from its words' timestamps, snap to the frame grid, enforce house style (max 2 lines, ~25 chars/line, min duration, chaining), preserving the customer's segmentation where compatible with confirmed speech and the two-line ceiling. Splits use complete words and semantic boundaries while retaining acoustic ownership.
5. **Verify & report**: optional forced alignment of final text, shared acoustic edge refinement, final ordering/overlap checks, then the synced SRT and QC report. Actionable items determine `clean`/`check`/`attention`; successful changes, episode notes, and operator diagnostics remain separately inspectable. `changes.diff.srt` logs applied wording and line-layout changes.

Timing always comes from acoustic models (ASR word timestamps / forced alignment). LLMs are used **only** for language reasoning (improv adjudication, punctuation, speaker/character attribution) — never as the timing source, because research confirms LLM audio timestamps drift by seconds.

Long-form API costs depend on the number of adjudication batches and audio-context configuration. Use the [September 14 episode comparison](docs/testing/flash-lite-adjudication-2026-09-14.md) for measured provider usage; the original $0.35–0.55 planning assumption does not describe the measured full-context runs. A fully local free mode (WhisperX + local aligner) is included for sensitive content.

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
| P2 | **Improvised lines** (spoken ≠ SRT text) | Divergence spans from the aligner → Gemini 3.8 Flash medium adjudication with local context + acoustic ownership + focused audio snippets |
| P3 | **Multiple people speaking together** | Diarization with overlap detection (Scribe speaker IDs; pyannote `community-1` overlap regions as fallback). Overlapping cues get overlapping time ranges or flags per `overlap_policy` |
| P4 | **Distinguish characters/speakers** | Stable diarization cluster IDs → LLM maps clusters to character names from conversational context (names used in dialogue); optional voice-reference matching (OpenAI `known_speaker_references`, ElevenLabs speaker library) |
| P5 | **Context-correct punctuation** | Whole-scene LLM pass that re-punctuates the final text with the house conventions (continuation commas across split cues, `?`/`!` from semantics, ellipses for interruptions), constrained to *not* change words |

---

## 4. Historical AI landscape research (July 2026, defaults updated October 1)

Prices and advertised capabilities in this section are historical research, not a current quote. Runtime billing evidence and configured estimates are recorded separately in `cost.json`.

### 4.1 ASR with word-level timestamps (the timing backbone)

| Provider / model | Word timestamps | Diarization | Languages | Price | Verdict |
|---|---|---|---|---|---|
| **Microsoft MAI-Transcribe 2 via OpenRouter** (`microsoft/mai-transcribe-2`) | ✅ required word timestamps | ✅ requested; bounded retry without it | Provider auto-detection or explicit language hint | September 5 catalog estimate $0.10/hr; returned `usage.cost` takes precedence | **Current default.** Bounded overlapping chunks, duplicate cleanup, no silent Scribe fallback |
| **ElevenLabs Scribe v2** (`scribe_v2`) | ✅ precise, built for subtitle sync | ✅ up to 32 speakers, `words[].speaker_id` | 90+ (incl. id, zh, ja, ko, de, es, pt…) | $0.22/hr (+$0.05/hr keyterm prompting) | **Selectable alternative.** Keyterms, audio-event tags, multi-language auto-detect |
| **WhisperX** (local, faster-whisper + wav2vec2 alignment + pyannote) | ✅ sub-100 ms after forced alignment | ✅ via pyannote | ~99 ASR / 35+ alignment models | Free (GPU recommended) | **Local/free fallback**; also the offline mode for sensitive content |
| **AssemblyAI** Universal-3 Pro / Universal-2 | ✅ | ✅ word-level (+$0.02/hr) | U3-Pro: 6 (en/es/fr/de/it/pt); U2: 99+ | $0.21/hr / $0.15/hr | Strong alternative for European targets; U3-Pro accepts 1,500-word natural-language prompts |
| **Deepgram Nova-3** | ✅ | ✅ | ~40 | ≈$0.26/hr batch (verify at signup) | Fast; fewer languages; fine as 3rd adapter |
| **OpenAI `whisper-1`** | ✅ (`verbose_json` + `timestamp_granularities:["word"]`) | ❌ | 99 | $0.006/min ($0.36/hr) | Usable, no diarization |
| **OpenAI `gpt-4o-transcribe-diarize`** | ❌ (segment-level only) | ✅ + `known_speaker_references[]` (map up to 4 named speakers from 2–10 s voice refs) | 99 | $0.006/min | **Not** a timing source; useful auxiliary for character-name mapping |
| **Gemini 2.5/3.x audio-native** | ❌ (second-level at best; documented drift of 1–3 s+, worse on long files) | prompt-based only | wide | $1/M audio tokens (~32 tok/s) | **Never for timing.** Reserved for reasoning + audio snippet verification |

Sources: elevenlabs.io/docs + pricing pages, developers.openai.com/api/docs/guides/speech-to-text, developers.openai.com/api/docs/models/gpt-5.6-luna, assemblyai.com/pricing + docs, ai.google.dev/gemini-api/docs (audio, pricing), github.com/m-bain/whisperX, Google AI dev forum threads on Gemini timestamp drift.

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
| **OpenAI GPT-5.6 Luna** (`gpt-5.6-luna`) | $1 / $6 ($0.10 cached input) | Default for speaker mapping. Responses API structured outputs; `medium` reasoning for this text-only pass. Text/image input only, so it is not the default adjudicator when audio snippets are enabled |
| **Gemini 3.7 Flash** (`gemini-3.7-flash`) | $0.75 / $3.75 recorded standard paid tier through 2026-12-31; $1.50 / $7.50 from 2027-01-01 | **Default punctuation model**, using `medium` thinking |
| **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`, GA July 2026) | $0.30 / $2.50; $0.03 cached input; $1/M cached tokens/hour storage | Optional hybrid primary, using `high` thinking, local source context, focused audio snippets, and v12 hearing evidence |
| **Gemini 3.8 Flash** (`gemini-3.8-flash`) | $0.75 / $3.75 standard paid tier through 2026-12-31; $1.50 / $7.50 from 2027-01-01 | **Default adjudicator** at `medium` thinking; also the optional reviewer for flagged Lite cases |
| **Gemini 3.5 Flash** (`gemini-3.5-flash`, GA May 2026) | $1.50 / $9 ($0.15 cached input) | Optional higher-cost alternate adjudicator with native audio snippet verification |
| Gemini 3.1 Pro / 3.5 Pro (when GA) | $2 / $12 | Optional quality upgrade for adjudication on difficult episodes |
| Claude Opus/Sonnet (Anthropic) | higher | Alternative adjudicator; no audio input → text-only usage |
| GPT-5.x (OpenAI) | comparable | Same role; structured outputs solid |

Design rule: **provider-agnostic `LLMAdapter`** with structured-output schemas. Default: **Gemini 3.8 Flash medium thinking** for adjudication with local source context and focused audio snippets; **Gemini 3.7 Flash medium thinking** for punctuation; **OpenAI GPT-5.6 Luna medium reasoning** for optional speaker mapping. The October script-route comparison was repeated after exact-request recovery of truncated hybrid replies. Those written-reference results do not certify the actor-spoken goal. The October 2 follow-up corrects recoverable source-timing errors from bounded audio evidence and enforces at most two display lines. The accuracy-upgrade report records the current artifact checks and remaining evidence limits. Hybrid remains configurable: use Lite high as primary and enable Flash medium fallback. Both wording passes use the v12 evidence policy and language/register context. The optional reviewer receives selected clips, nearby source cues, owned ASR words, and explicit review reasons; it receives no full episode audio or episode cache. Full episode audio remains configurable for supported alternatives. Unresolved wording preserves source text; only missing or ambiguous acoustic ownership requires retaining source timing. Known ambiguous passages remain reviewable while processing and downloads continue. Agreement between models does not guarantee correctness.

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
- **ASR adapters**: MAI through OpenRouter (default), `elevenlabs` SDK (selectable), `whisperx` (optional local extra), `assemblyai`, `openai` — behind one `ASRAdapter` interface returning word, start, end, optional confidence, and speaker ID.
- **Alignment**: custom weighted Needleman–Wunsch over normalized tokens with `rapidfuzz` similarity (no heavy deps); optional embedding assist for paraphrase spans.
- **Forced alignment (optional precision pass)**: `ctc-forced-aligner` (torch) as an optional extra `[precision]`.
- **Diarization backstop**: `pyannote.audio` 4.x community-1 as optional extra `[diarize-local]`.
- **LLM adapters**: `google-genai` (default adjudicator: `gemini-3.8-flash`), `openai`, `anthropic` behind `LLMAdapter` with JSON-schema structured outputs.
- **Config**: `style_profile.yaml` (house rules) + `providers.yaml` + `.env` for keys. Style profile can be auto-derived from a sample SRT.
- **Caching**: content-hash ASR JSON plus validated LLM batch and independent case decisions. Case keys include local source context, acoustic ownership/audio provenance, language/register/name policy, prompt/policy versions, models, and non-secret settings. Cache hits create no new provider charge; changed or transiently failed cases can require a new call.
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

Contiguous runs of matched tokens form **anchor regions**. Mismatch runs become **divergence spans** carrying source text, the ASR hypothesis, optional word confidence, speaker evidence, exact owned indices, and acoustic windows. Repaired word evidence is shared downstream; uncertain words spanning multiple plausible bursts cannot establish a unique cue boundary.

### 7.3 Divergence classification (before spending LLM tokens)
Deterministic predecisions handle punctuation/casing and finite language-scoped spacing, abbreviation, recurring-name, and register equivalents. The default `adjudication.register_policy: spoken` follows performed wording; proven whole-span register reductions take the ASR's register words with the script's case and punctuation. Punctuation-only ASR insertions and marked Japanese names are decided without a paid question; a confident rewrite that spells two or more source kanji in kana is held. Explicit `script` preserves authored equivalents, and no-LLM processing retains script wording. Unknown languages get no language-specific rules. Near-homophones, ambiguous names, real lexical compounds, changed numbers, and negations remain review cases. An empty ASR span alone does not prove that an actor dropped a line.

Eligible missing dialogue receives a native audio question whose editable indices cover exactly one whole missing cue between independent matched anchors. Exact neighboring source-token residue can receive its own question without making other words editable. A whole-cue omission requires complete clear hearing and no speech activity in the anchored gap. Recovered wording uses one independent speech chain with all internal pauses strictly below the existing 0.2-second threshold; no LLM supplies timestamps. Multiple chains, anchor-crossing activity, or changed residual wording without owned word times preserve source timing for review. No-LLM processing preserves these passages. Saved legacy verdicts cannot answer new native questions; unanswered cases are explicit confidence-zero holds, rather than fabricated 0.95 confirmations. Processing continues through these review items.

### 7.4 LLM adjudication contract (structured output)
Native prompt v12 returns one evidence decision per case:
```json
{ "case_id": "...", "verdict": "keep_srt | use_audio | hybrid",
  "final_text": "...", "heard_text": "...",
  "evidence": "heard_clearly | heard_unclear | not_audible",
  "speaker": "cluster_3", "character": "Luna | unknown",
  "reason": "one sentence" }
```
The language and register policy are explicit prompt inputs. `final_text` replaces only the editable span; padded audio and neighboring source/ASR words are read-only. Clear hearing must pass deterministic wording validation; unclear/inaudible or unavailable audio keeps source wording and creates review. Legacy stored confidence decisions still use the configured gate (default 0.7). The configured hybrid reviewer receives selected clips and local evidence, with actual usage recorded per model; no fixed per-snippet cost is promised.

### 7.5 Re-cue rules (deterministic, unit-tested — no LLM)
- Source cues are sorted chronologically before alignment using `(start_ms, original_index)` while preserving original cue ids; moved cues emit `source_out_of_order` QC.
- Unchanged-text cues retain source wording and line breaks when compatible with the two-line ceiling. Crowded cues use semantic reflow or independently timed splits while preserving spoken words and ownership. Starts floor-snap from the first owned acoustic word/burst; ends ceil-snap from the last owned speech edge plus the configured tail. `min_duration_policy: extend_into_silence` permits a display tail only through verified silence before another sound or cue, including held dialogue. `acoustic` disables that extension. Frame-grid export rounds upward to milliseconds; no timestamp comes from an LLM.
- `keep_srt` adjudication keeps source wording but still attaches divergent ASR word indices to cue timing, so numeric/spelling preferences cannot cut off the actor's last word.
- Changed-text spans: re-flow into cues mimicking the original segmentation density (target ≈ original cue count for that sentence; split at clause/phrase boundaries; ≤2 lines × ≤26 chars; balanced lines; compound hyphenation last resort), then time each new cue from its own words.
- Missing dialogue: `drop_policy: remove | keep_flagged`, default `keep_flagged`; source text/timing stays available for review when no trustworthy speech supports a retime. No zero-duration dialogue is exported.
- Overlaps (P3): per `overlap_policy: stack` (overlapping cue times, default) | `dash` (merge into one 2-line dashed cue) | `flag_only`.
- Streaming adaptive energy VAD uses a 10 ms hop. Shared phrase-edge repair handles stretched ASR edges against their burst before cue timing; `timing.phrase_edge_snap: false` disables it and model-specific limits remain configurable. In a recording whose phrase starts lag as a rule, a start up to `lagging_start_advance_ms` (700 ms) after its burst onset moves back when the level track hears the phrase from the onset on; collapsed tokens stay. A cue that still starts after its own speech onset raises `cue_starts_after_speech_onset`. Ambiguous multi-burst evidence stays held for review.
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
| ASR hallucination in silence | Speech evidence checks supported additions; unsupported source dialogue stays held with a review item rather than fabricating an acoustic time |
| Source SRT scrambled/out of chronological order | Source ingest sorts by time and emits `source_out_of_order`; duplicate-overlap guard merges any remaining repeated captions before export |
| Impossible ASR word span or impossible display speed | Clamp long ASR word duration to VAD region and emit `asr_word_clamped`; verify emits `impossible_cps_fast` / `impossible_cps_slow` for QC |
| Non-Latin targets (zh/ja/ko/th…) | Character-based comparison, Japanese width/kana-safe normalization, grouped acoustic words, and visual-width line rules; optional MMS requires separate model validation |

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

Build a golden set from source SRT, VO audio, and independently corrected SRT. The current episode 11/17 references are edits of earlier DubSync output: most edges are inherited, so full timestamp agreement measures old-output similarity. Evaluate their text and `human_edit_subset`, then compare acoustic onset/offset distributions from the audio. These v1 targets remain objectives, not a claim that the upgrade has met them:

- **Timing**: ≥90% of cue starts within ±1 frame of golden; ≥98% within ±3 frames; MAE < 50 ms.
- **Improv detection**: precision ≥0.9 / recall ≥0.85 on spans that humans changed.
- **Structure**: 0 style-lint violations (line count/length, grid, chaining); cue count preserved for unchanged text.
- **Review burden**: ≤10% of cues flagged for human review on a typical episode.
- Status (independent review of the accepted October candidate, delivered episode 11 MAI / Scribe and episode 17 MAI files, against references that inherit 94–99% of their timestamps): starts within ±1 frame 55.4/58.4/46.3% and within ±3 frames 92.7/94.2/95.6%, unmet; start MAE 54.3/48.7/47.4 ms, unmet for episode 11 MAI; improvisation precision 0.47/0.53/0.76 and recall 0.66/0.70/0.78, unmet; style-lint violations 123/126/115, unmet; review burden 4.6–5.2%, met. See the [requirement dispositions](docs/testing/accuracy-upgrade-2026-10-01.md#requirement-dispositions).
- Review precision and recall must be measured alongside count reduction. Raw findings remain in JSON; routine successful changes are a log, while real uncertainty and acoustic overlap stay actionable.

---

## 12. Cost model (per 45-min episode, API mode)

| Item | Cost |
|---|---|
| MAI-Transcribe 2 (default) | Historical catalog estimate $0.075 for 45 minutes; provider-reported billing takes precedence |
| Scribe v2 ASR + diarization (selectable) | Historical configured estimate $0.165 (0.75 h × $0.22) |
| Keyterm prompting (character names) | +$0.04 |
| LLM adjudication + punctuation | Flash medium adjudication and punctuation need representative per-episode measurement with v12 and deterministic predecisions; the earlier August `testing 4` run is historical evidence for different adjudication settings |
| Optional second ASR cross-check | Disabled by default; extra billed/estimated audio cost is reported separately when enabled |
| Forced-align + pyannote (local) | $0 |
| **Total** | **Not yet established for a 45-minute episode after the complete-context upgrade** (fully-local mode: $0) |

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

1. Primary ASR **Microsoft MAI-Transcribe 2 via OpenRouter**; **ElevenLabs Scribe v2 remains selectable**, without silent provider fallback. Local mode uses WhisperX.
2. Default adjudication LLM **Gemini 3.8 Flash** (`gemini-3.8-flash`) with `thinking_level: medium`, local source context, v12 hearing evidence, and focused audio snippets. The optional YAML hybrid uses Lite high as primary and enables Flash medium `fallback` for flagged cases. Full episode audio is disabled by default and prohibited for the hybrid reviewer. Punctuation uses **Gemini 3.7 Flash** with `thinking_level: medium`; speaker mapping uses **OpenAI GPT-5.6 Luna** (`gpt-5.6-luna`) through the Responses API with `reasoning_effort: medium`; alternate adapters remain configurable.
3. Frame grid auto-detected, fallback 30 fps (matches the example). 
4. `overlap_policy: stack`, `drop_policy: keep_flagged`, adjudication confidence gate 0.7. 
5. Output has at most two physical lines. Under a source-derived style, keep a one- or two-line customer cue byte-for-byte; its width is a style finding. Only a cue over two lines, or an explicitly chosen or stricter style, uses semantic wrapping, complete-word timed splits or ordered caption pages. Do not invent internal timing when acoustic ownership is uncertain.
6. All timing from acoustic models; LLMs never move timestamps.
7. Actor-spoken register is the default (`adjudication.register_policy: spoken`), including confirmed improvised sentences. Explicit `script` preserves authored equivalents; no-LLM processing retains script wording. Optional dual-ASR cross-check remains off unless requested; primary words and ownership remain intact. A separately confirmed whole utterance may use unique secondary evidence to identify an existing acoustic region, with explicit provenance.
8. Generation applies selected minimum-duration/CPS targets only within verified silence, with balanced wrapping and shared ownership-based overlap handling. Parsing supports comma/period timestamps and missing blank separators; legacy byte encodings need the reader's explicit notice or encoding selection.
9. With default `output.no_overlaps: true`, pure bracketed captions without owned words compose around spoken intervals and paginate within the two-line ceiling. Only caption text repeats. Every spoken word and its ownership remain preserved through semantic splits. A narrow interruption rule may partition a sentence around another actor's independently owned insertion. A complete, sequential spoken-line/laugh exchange can share two dialogue lines after structured hearing verifies both its wider context and exact candidate excerpt. Genuine simultaneous speech remains reviewable. Lyrics are not repeatable caption tracks; an eligible unowned lyric may yield its tail to speech. Caption delay, coverage gaps and reading pressure remain explicit. `output.no_overlaps: false` preserves separate annotation tracks while retaining the two-line ceiling.

**Open questions for the studio (answers slot into config; defaults above apply meanwhile):**
- Which target languages ship first? (affects M2 language benchmark matrix)
- Typical episode length & monthly volume? (affects batch/cost tuning)
- Is a GPU machine available for local mode?
- Golden pairs available for the eval set (§11)? — highest-value asset for tuning
- House convention for on-screen simultaneous dialogue (stacked cues vs dashed merged cue)?

---

*Companion file: `OPUS-4.8-EXECUTION-PROMPT.md` — the ready-to-paste prompt that instructs Claude Opus 4.8 to build this app milestone-by-milestone.*
