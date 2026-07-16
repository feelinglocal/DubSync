# DubSync

DubSync is a Windows-friendly Python 3.11+ CLI for retiming a customer-supplied target-language SRT to dubbed VO-only audio while preserving the customer's subtitle segmentation and flagging dialogue changes for QC.

Timing comes from acoustic data only: ASR word timestamps and optional forced alignment. LLM adapters are used only for language decisions such as improv adjudication and punctuation, never for timestamps.

## Commercial Web MVP

DubSync also includes a responsive React/FastAPI application with two customer workflows:

- **Sync existing SRT:** upload one file pair or a batch of up to 10 matched audio/SRT pairs, then download synchronized SRT and QC artifacts for each source. Batch children run one by one, and cue presentation rules are derived from each uploaded SRT.
- **Audio to SRT:** upload one audio file or a batch of up to 10, choose a built-in subtitle preset, enter custom line/timing/CPS rules, or upload an example SRT to derive them, then generate acoustically timed SRT and QC artifacts.

The language selector defaults to provider auto-detection. An explicit language is forwarded to ElevenLabs Scribe and becomes part of the cached ASR configuration.

The first commercial release intentionally has no customer accounts, subscriptions, or Supabase dependency. Manual quotes issue a rotating job access code before paid processing, and every accepted child job receives a separate secret browser-held result token. Uploads and results expire 24 hours after each child finishes, and the API limits job creation per source IP. Production job intake fails closed when the access code is not configured. See `docs/COMMERCIAL_PLAN.md` for the product scope, provisional pricing, deployment limits, roadmap, and paid-launch gates.

Local web setup:

```powershell
python -m pip install -e ".[dev,cloud,web]"
Set-Location web
npm ci
npm run build
Set-Location ..
dubsync-web
```

Open `http://127.0.0.1:8000`. The server reads `provider.yaml`, `style_profile.yaml`, and API keys from `.env` by default. The DubSync default generation preset honors that configured style profile; every other web generation style is resolved per job. API documentation is disabled unless `DUBSYNC_ENABLE_DOCS=1`.

Job intake defaults to five submissions per source IP per hour (`DUBSYNC_MAX_SUBMISSIONS_PER_HOUR`) and ten outstanding child jobs (`DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS`). Single and batch request payloads are capped at 512 MiB, retained job commitments are capped at 4 GiB, and only one request may pass upload intake at a time so concurrent copies cannot exhaust the 10 GB disk. SRT files are capped at 2 MiB, 60,000 lines, and 20,000 cues, with bounded line lengths and an incremental parser so structurally hostile subtitle files fail before expensive processing. Before acceptance, non-fixture audio is probed with a 15-second deadline and its predicted 16 kHz PCM plus work allocation is reserved. Production also enforces a four-hour audio limit, a 1 GiB per-job ceiling, bounded normalized/snippet outputs, and 2 GiB of minimum free disk. Configure these bounds with the `DUBSYNC_MAX_*` and `DUBSYNC_MIN_FREE_STORAGE_BYTES` variables shown in `.env.example`. Existing deployments that only set `DUBSYNC_MAX_JOBS_PER_HOUR` retain that value as a fallback. Queued or processing jobs with no state update for 24 hours are dead-lettered on startup or periodic cleanup, then retained for the normal terminal retention window; configure that deadline with `DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS`. Every FFmpeg subprocess has a finite 1,800-second default timeout controlled by `DUBSYNC_FFMPEG_TIMEOUT_SECONDS`.

For frontend development, run `npm run dev` inside `web` and run the FastAPI service separately. The production Docker image builds the frontend and serves it from the same origin as the API.

### Render Deployment

`Dockerfile` and `render.yaml` define the initial production architecture: one Starter web service in Singapore, one background processing thread, and a 10 GB persistent disk mounted at `/var/data`. SQLite metadata and job files live on that disk. This keeps the early-access system small, but it also means one instance, deployment downtime, and no horizontal scaling.

To deploy:

1. Put this workspace in a real private Git repository and connect that repository to Render.
2. Create a Blueprint from `render.yaml`.
3. Enter `ELEVENLABS_API_KEY`, `GEMINI_API_KEY`, and a strong `DUBSYNC_JOB_ACCESS_CODE` as Render secrets. Never commit `.env`.
4. Confirm `/api/health`, `/api/config` reports `jobs_available: true`, and the deployed commit matches the release SHA.
5. Run one short paid-provider generate job through the web UI. The fixture-backed E2E suite covers sync behavior without provider spend.

Local schema check after installing the development extra:

```powershell
.venv\Scripts\python.exe -c "import json, urllib.request, yaml; from jsonschema import Draft7Validator; data=yaml.safe_load(open('render.yaml', encoding='utf-8')); schema=json.load(urllib.request.urlopen('https://render.com/schema/render.yaml.json')); errors=list(Draft7Validator(schema).iter_errors(data)); print('VALID' if not errors else errors); raise SystemExit(bool(errors))"
```

Supabase is not required for this MVP. Move metadata to a shared database and media to object storage before enabling multiple service instances or persistent customer history. Rotate `DUBSYNC_JOB_ACCESS_CODE` whenever it is shared outside an accepted quote or a customer engagement ends.

## Windows Quickstart

Prerequisites:

- Python 3.11 or newer
- `uv` for normal project setup
- `ffmpeg` on `PATH` for audio normalization

Setup:

```powershell
uv venv
uv pip install -e ".[dev,cloud,web]"
python -m dubsync --help
```

This machine did not have `uv` installed during implementation, so verification used:

```powershell
python -m pip install -e ".[dev,cloud,web]"
python -m pytest --cov=dubsync --cov-report=term-missing
```

Live provider smoke tests are opt-in because they can spend API credits:

```powershell
python -m pytest --live tests/test_live_smoke.py
```

Create `.env` as needed:

```text
ELEVENLABS_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
ASSEMBLYAI_API_KEY=...
HUGGINGFACE_ACCESS_TOKEN=...
```

## Commands

```powershell
python -m dubsync profile Examples\"srt test.srt" -o style_profile.yaml

python -m dubsync sync episode.srt episode.wav `
  -o episode.synced.srt `
  --style style_profile.yaml `
  --providers providers.yaml `
  --workdir workdir

python -m dubsync generate episode.wav `
  -o episode.generated.srt `
  --providers providers.yaml `
  --workdir workdir

python -m dubsync batch . --providers providers.yaml --workdir workdir
python -m dubsync report workdir\episode
python -m dubsync report workdir\episode --synced episode.synced.srt --golden episode.golden.srt --fps 30
```

`--no-llm` runs timing-only mode. It still emits full QC for divergences, unmatched cues, style issues, and overlaps.

`--resume asr` reloads persisted ingest/style artifacts before rerunning ASR. `--resume align` and later stages reuse `workdir/<episode>/asr.json` instead of calling ASR again. `--resume adjudicate` reloads persisted ingest and alignment artifacts before rerunning adjudication. `--resume rebuild` reloads persisted ingest, alignment, and adjudication artifacts before re-cueing. `--resume verify` reloads `align.json` and `rebuild.json`, so verification/report generation starts from the persisted rebuilt subtitle artifact instead of recomputing earlier stages from the source SRT.

`--local` forces the ASR provider to WhisperX and disables LLM calls. If `dubsync[local]` is not installed, it fails with a clear WhisperX optional-extra error rather than requesting cloud credentials.

## Provider Matrix

| Role | Provider | Status | Config |
|---|---|---|---|
| ASR primary | ElevenLabs Scribe v2 | Implemented optional adapter | `asr.provider: elevenlabs`, `model_id: scribe_v2`, `diarize: true`, optional `keyterms` / `character_names` |
| ASR fallback | OpenAI Whisper | Implemented optional adapter, no diarization | `asr.provider: openai`, `model: whisper-1` |
| ASR fallback | AssemblyAI | Implemented optional adapter | `asr.provider: assemblyai`, `model: universal-3-pro` or `universal-2`, `speaker_labels: true` |
| ASR local | WhisperX | Implemented optional adapter; requires `dubsync[local]` | `asr.provider: whisperx` |
| Test/offline | Fixture wordstream | Implemented | `asr.fixture_path: path/to.wordstream.json` |
| LLM default | Gemini | Implemented optional adapter | `llm.provider: gemini`, `model: gemini-3.5-flash` |
| LLM alt | OpenAI | Implemented optional adapter | `llm.provider: openai`, `model: gpt-5.5` |
| LLM alt | Anthropic | Implemented optional adapter | `llm.provider: anthropic` |
| Test/offline | Fixture decisions | Implemented | `llm.provider: fixture` |
| Precision verify | Fixture forced alignment | Implemented | `forced_alignment.fixture_path: path/to.forced-align.json` |
| Precision verify | MMS / ctc-forced-aligner | Implemented optional adapter; requires `dubsync[precision]` and model runtime | `forced_alignment.provider: mms` |
| Overlap backstop | Fixture overlap regions | Implemented | `overlap_detection.fixture_path: path/to.overlap.json` |
| Overlap backstop | pyannote community-1 | Implemented optional adapter; requires `dubsync[diarize-local]` and model access | `overlap_detection.provider: pyannote` |
| Speech activity | Fixture VAD regions | Implemented | `vad.fixture_path: path/to.vad.json` |
| Speech activity | Energy threshold VAD | Implemented deterministic adapter | `vad.provider: energy` |
| Speech activity | Silero VAD | Implemented optional local adapter; falls back to energy if unavailable | `vad.provider: silero` |

## Config Reference

`style_profile.yaml`:

```yaml
fps: 30.0
max_lines_per_cue: 2
max_chars_per_line: 26
min_cue_dur: 0.5
allow_zero_gap: true
lead_in_ms: 0
tail_ms: 40
overlap_policy: stack
drop_policy: keep_flagged
```

`providers.yaml`:

```yaml
asr:
  provider: elevenlabs
  model_id: scribe_v2
  diarize: true
  keyterms:
    - Drachen-Evolutionssystem
  character_names:
    - Luna
    - Matthew

llm:
  provider: gemini
  model: gemini-3.5-flash
  # Optional: reuse an existing Gemini explicit cache resource.
  # cached_content: cachedContents/your-episode-context-cache
  # Optional per-pass overrides inherit provider/api key unless changed:
  adjudication:
    confidence_gate: 0.7
    scene_gap_seconds: 4.0
    audio_snippet_double_check:
      enabled: false
      pad_seconds: 2.0
      max_duration_seconds: 20.0
  punctuation:
    model: gemini-3.1-flash-lite
    scene_gap_seconds: 4.0
    thinking_level: low
  speaker_mapping:
    model: gemini-3.1-flash-lite
  # Optional for providers/models without built-in defaults:
  # input_per_million: 2.0
  # output_per_million: 10.0

forced_alignment:
  provider: mms
  language: deu
  romanize: true
  batch_size: 4

overlap_detection:
  provider: pyannote
  model: pyannote/speaker-diarization-community-1

vad:
  provider: energy
  threshold_dbfs: -45.0
  window_ms: 100
  min_region_ms: 100
  min_coverage: 0.2
# Optional neural VAD when torch/Silero are available; otherwise use energy.
# vad:
#   provider: silero
#   sampling_rate: 16000

timing:
  max_word_duration: 2.0
  max_intra_cue_gap: 1.5
  max_cps: 30
  min_cps: 2

output:
  no_overlaps: true

speaker_mapping:
  # Use either fixture mapping for deterministic runs:
  fixture:
    SPEAKER_00: Luna
    SPEAKER_01: Matthew
  # Or use the configured llm provider to infer from cue context:
  # provider: llm
```

Forced-alignment artifacts preserve the provider's raw rows in `forced_align.json`, while exported cues and QC review windows are clamped to valid non-negative, frame-snapped subtitle times.

For deterministic tests or no-key demos:

```yaml
asr:
  fixture_path: tests/fixtures/episode.wordstream.json
llm:
  provider: fixture
  responses: {}
```

## Cost Model

The CLI writes `cost.json` and prints a cost meter. Fixture, local, resumed, and cached ASR paths record zero API cost. Uncached cloud ASR calls are metered from WAV duration and the configured provider price. Live LLM calls record token costs when the provider response exposes usage metadata and either a built-in Gemini price or explicit `input_per_million` / `output_per_million` pricing is available. `llm.adjudication`, `llm.punctuation`, and `llm.speaker_mapping` can override provider/model/pricing per pass.

| Item | Planned cost basis |
|---|---|
| Scribe v2 ASR | audio seconds x provider hourly price (`$0.22/hr`, or `$0.27/hr` when `keyterms` or `character_names` enable keyterm prompting) |
| AssemblyAI ASR | audio seconds x provider/model hourly price (`$0.21/hr` for `universal-3-pro`, `$0.15/hr` for `universal-2`, plus `$0.02/hr` when `speaker_labels` is enabled; enabled by default) |
| LLM adjudication/punctuation | input/output tokens x model price |
| Audio snippet double-checks | inline snippet audio duration is included in Gemini input usage when provider metadata is available |
| Local forced alignment/diarization | zero API cost |

## What Works Now

- SRT parser/writer with strict round-trip tests proving only CRLF and trailing text whitespace are normalized.
- Style profile derivation from `Examples/srt test.srt`.
- `profile` rejects malformed sample SRT files with clear CLI errors instead of raw parser exceptions.
- Malformed `--providers` and `--style` YAML files are rejected with clear CLI errors that name the config file.
- Invalid style-profile values such as `fps: 0` are rejected with clear CLI errors that name the file and field.
- Non-mapping provider config sections such as `vad: []`, `forced_alignment: []`, `overlap_detection: []`, and `speaker_mapping: []` are rejected instead of being silently ignored.
- `sync` and `batch` load `.env` from the current working directory before resolving provider keys, without overwriting already-set environment variables.
- Fuzzy monotonic, band-limited SRT-token to ASR-word alignment with anchor regions and divergence spans persisted in `align.json`.
- Delete-only divergence spans inherit the surrounding matched-word window, so dropped-line/adjudication cases have concrete boundary timestamps when bounded by anchors.
- Alignment normalization maps common digit strings and English/German number words to the same canonical tokens, avoiding false divergences such as `2` vs `two`.
- Source cues are sorted chronologically before alignment while preserving original cue ids; moved cues are reported as `source_out_of_order`.
- Deterministic re-cueing from ASR word timestamps, frame snapping, min duration, and zero-gap chaining.
- Cue starts floor-snap and cue ends ceil-snap, so fractional model timings cannot truncate the final spoken syllable.
- Min-duration padding extends only into available same-speaker display gaps; if the next cue starts too soon, the cue remains short and is surfaced by style lint instead of shifting speech timing.
- Frame-grid ceiling never snaps fractional model timings backward when enforcing minimum duration or forced-alignment ends.
- Configured cue lead-in is clamped at zero so early speech cannot produce invalid negative SRT timestamps.
- Fixture-backed ASR/LLM path for offline E2E tests.
- Web audio generation resolves explicit presets, custom rules, and uploaded-example subtitle styles per job while preserving the configured profile for the DubSync default preset; the selected line, timing, gap, lead/tail, and CPS rules are recorded in `generate.json` and applied during output finalization.
- Web sync derives its style from the user-supplied source SRT instead of applying the server's global generation profile.
- Web batch intake accepts up to 10 matched audio/SRT pairs, matches them by case-insensitive filename stem, and submits them as one rate-limited request to the single sequential worker.
- Browser-held access recovers every child in a submitted batch after refresh, while each child keeps an isolated token and failure state.
- Downloaded SRT names preserve the validated source stem and append `-dubsync-synced.srt`.
- ElevenLabs Scribe v2 ASR forwards configured keyterms and character names as `keyterms` while still requesting word timestamps and diarization.
- Opt-in `--live` pytest smoke tests for Gemini, Anthropic, ElevenLabs, OpenAI Whisper, and AssemblyAI are deselected from normal offline test runs.
- Gemini LLM calls use the installed `google-genai` `models.generate_content` API with JSON response schemas.
- Gemini thinking-level controls are wired through `thinking_level` (`minimal`, `low`, `medium`, `high`) using `thinking_config.thinking_level`; punctuation defaults to `low` when using Gemini through `llm.punctuation`.
- Gemini explicit context-cache reuse is wired through `cached_content`, which may be set at `llm.cached_content` or overridden per pass. DubSync reuses an existing Gemini cache resource but does not create or delete remote caches automatically.
- Optional adjudication audio-snippet double-checks extract padded WAV snippets from the local audio, persist `audio_snippets.json`, include snippet hashes in the LLM cache key, and send Gemini inline audio parts with `types.Part.from_bytes` when `llm.adjudication.audio_snippet_double_check.enabled: true`.
- Improv replacement path with QC flags and acoustic timing from spoken ASR words.
- ASR-only ad-lib spans can be accepted by adjudication, inserted as new acoustically timed cues, and QC-flagged as `adlib_inserted`.
- Exported SRT files are sequentially renumbered in playback order, including ad-lib cues inserted between existing source cues.
- `keep_srt` adjudication still attaches divergent ASR word indices to timing, so kept source spelling/numbers do not cut off the actor's spoken span.
- Final output sorting merges duplicate overlapping captions as `duplicate_cue_merged`, resolves residual same/unknown-speaker overlaps when `output.no_overlaps: true`, and asserts monotonic starts before writing SRT.
- Multi-cue improv replacements are distributed once across the original cue count instead of duplicating the replacement text into every cue.
- Multi-cue improv timing partitions the accepted spoken word indices across affected cues, so rebuilt changed cues do not all inherit the full span timing.
- Heuristic adjudication keeps source SRT for punctuation/casing-only differences and tiny ASR spelling noise without spending an LLM call.
- Adjudication LLM spans carry up to two cue texts before and after the divergent span as structured context.
- Fixture-backed punctuation pass with a validator that rejects word changes.
- Punctuation word-freeze validation rejects digit-to-word substitutions such as `2` -> `two`; number normalization remains limited to alignment.
- LLM adjudication retries invalid structured output once before falling back to `keep_srt` with a QC flag.
- `adjudicate.json` persists adjudication decisions and adjudication-stage QC flags, so `--resume rebuild` preserves low-confidence or invalid-response warnings instead of silently dropping them.
- Validated LLM adjudication decisions, speaker mappings, and punctuation outputs are cached in `workdir/<episode>/llm-cache`, keyed by input payload, model, and non-secret request params, so repeat non-resume syncs avoid recomputing the same LLM pass.
- QC JSON/HTML report, `changes.diff.srt`, and a verify-stage `verify.json` artifact.
- `changes.diff.srt` is emitted as a parseable SRT review file with one cue per text-changing flag, preserving the QC timestamp window and old/new text lines.
- Per-cue verification scores and CPS are written to `qc_report.json` and rendered in `qc_report.html`; scores use forced-alignment confidence when present, otherwise ASR word confidence.
- QC HTML flag rows include cue ids, timestamps, confidence, and old/new review text for changed or flagged cues.
- The verify stage writes `verify.json` with the finalized summary, cue scores, QC flags, and style issues for resumable/debuggable stage inspection.
- ASR cache keyed by audio SHA-256, model, and non-secret params; credential fields are stripped before cache metadata is written.
- Uncached cloud ASR calls add audio-duration cost items to the cost meter; ElevenLabs keyterm/character-name prompting includes the plan's `$0.05/hr` surcharge; AssemblyAI uses the plan's Universal-3 Pro / Universal-2 rates plus the default speaker-label surcharge unless `speaker_labels: false`; cache hits remain free.
- Live LLM adapters retain provider usage metadata and add token cost items for Gemini defaults or configured model prices.
- LLM provider/model config can be overridden per pass for adjudication, punctuation, and speaker mapping, and cost items use the resolved pass model.
- The adjudication confidence gate defaults to `0.7` and can be raised or lowered with `llm.adjudication.confidence_gate`.
- Live adjudication prompts receive the resolved confidence gate, so provider-side reasoning and local QC flagging use the same threshold.
- Adjudication sends LLM cases in scene batches split by `llm.adjudication.scene_gap_seconds` instead of one episode-wide batch.
- Punctuation sends cue batches split by `llm.punctuation.scene_gap_seconds`, with the same word-freeze validator applied after each proposed change.
- Valid punctuation proposals preserve cue line breaks instead of flattening two-line subtitles into one line.
- Punctuation prompts include speaker cluster and mapped character labels as context while keeping those labels out of the output SRT.
- Deterministic 16-bit WAV silence gate for cues sitting below the configured dBFS threshold.
- VAD-backed unmatched cues with insufficient speech coverage are QC-flagged as dropped-line candidates.
- Unmatched and policy-removed source cues carry their original timestamp windows in QC for review.
- Source-error detector for adjacent duplicated/scrambled cue fragments, including the named dirty block in `Examples/srt test.srt`, with affected timestamp windows in QC.
- Speaker ID propagation from matched ASR words into rebuilt cues, enabling overlap policy decisions.
- Speaker-aware monotonic enforcement: same-speaker cues are chained, while different-speaker overlaps can remain stacked.
- Style lint allows known different-speaker stacked overlaps while still treating same/unknown-speaker overlap as invalid.
- Overlap policy QC flags include the actual overlapping timestamp window for review, including overlaps where speaker IDs are still unknown.
- `overlap_policy: dash` merges only known different-speaker overlaps; unknown-speaker overlaps stay separate and are QC-flagged for human review.
- Fixture-backed and LLM-backed speaker-to-character mapping writes `speaker_map.json` and QC entries without adding names to the SRT.
- LLM speaker mapping uses cue text context only, never timestamps, and its token usage is included in the cost meter when usage metadata is available.
- `--resume asr` reuses persisted ingest/style artifacts while rerunning ASR; `--resume align` reuses persisted ASR artifacts; `--resume adjudicate` reuses persisted ingest and alignment artifacts; `--resume rebuild` reuses persisted ingest, alignment, and adjudication artifacts; `--resume verify` reuses persisted ASR, alignment, and rebuild artifacts.
- `--local` routes to WhisperX/no-LLM mode instead of cloud providers.
- WhisperX diarization accepts `HUGGINGFACE_ACCESS_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN`, matching the `.env` quickstart and pyannote overlap backstop token aliases.
- `batch` accepts the same core execution flags as `sync`, including `--local`, `--fps`, `--resume`, and `--no-llm`, and prints the output path, artifact path, and cost meter for each processed episode.
- Batch mode ignores generated SRT artifacts such as `*.synced.srt`, `*.changes.diff.srt`, and `changes.diff.srt` so reruns do not recursively process review/output files as new episode inputs.
- Batch exits non-zero if every source SRT is skipped because no matching WAV/MP3 exists, avoiding a silent successful no-op.
- Fixture-backed forced alignment can refine final cue timings and writes `forced_align.json`.
- MMS forced alignment is wired through `ctc-forced-aligner`'s Python API and reduces word-level timestamps back to cue-level timing refinements.
- Fixture-backed overlap detection writes `overlap.json` and QC-flags cues intersecting detected simultaneous-speech regions.
- Optional pyannote community-1 overlap backstop is wired behind `dubsync[diarize-local]`; it derives overlap regions from local diarization turns.
- Fixture-backed and energy-threshold speech activity detection writes `vad.json` and QC-flags cues with insufficient speech-region coverage.
- VAD-backed boundary refinement uses matched cue word timestamps when available; ASR words longer than `timing.max_word_duration` are clamped to the containing speech region and flagged as `asr_word_clamped`.
- Optional `vad.provider: silero` uses local Silero VAD when available and falls back to the deterministic energy VAD if the model/runtime cannot be loaded.
- Verify emits `impossible_cps_fast` and `impossible_cps_slow` QC flags using `timing.max_cps` and `timing.min_cps`.
- `report --synced --golden` computes the PLAN §11 timing/review metrics: cue counts, start MAE, within-1/3-frame ratios, source-aware improv precision/recall, review burden, and target booleans. When `ingest.json` is present, source-vs-golden text defines the actual changed cues; the improv target requires at least 0.9 precision and 0.85 recall.
- The timing target boolean requires all PLAN §11 timing gates: at least 90% of starts within 1 frame, at least 98% within 3 frames, and start MAE below 50 ms.
- `report` refuses a parent workdir containing multiple episode reports unless a specific episode workdir is provided, avoiding silent selection of the wrong QC report.
- `report` rejects malformed `qc_report.json` and malformed comparison SRTs with clear CLI errors instead of raw parser exceptions.
- `drop_policy: remove` drops unmatched source cues while QC-flagging the removed text; `keep_flagged` remains the default.
- CJK/Thai/Hangul/Japanese tokenization falls back to character-level units, and style profile/lint/reflow use visual display width for full-width text.
- Changed-text reflow hyphen-splits over-wide unspaced compounds so replacements can satisfy the two-line house style when possible.

## Readiness Report

Current status: the CLI and commercial web MVP are implemented, the default Render domain is healthy, fixture-backed automated tests cover both customer workflows, and the approved production ElevenLabs plus Gemini smoke job completed. The web surface includes per-job generation styles, source-derived sync styling, gated job creation, polling, refresh recovery, protected downloads, legal and payment policies, retention cleanup, and commit-aware Render health checks.

Still unverified or intentionally outside this release: real WhisperX/pyannote/MMS model execution in this workspace, production Silero model quality, language-specific morphological tokenizers, customer accounts, automatic payment collection, and a browser cue editor.

### Measured Timings And Costs

Latest local offline verification in this workspace:

| Command | Result | Runtime / cost evidence |
|---|---|---|
| `python -m pytest --cov=dubsync --cov-report=term-missing` | `309 passed, 5 deselected`, coverage `85.22%` | Normal offline suite; paid/live smoke tests deselected |
| `npm run test:coverage` | `54 passed`; statements `91.14%`, lines `94.02%` | React workflow, sequential batch behavior, recovery, generation style controls, access gate, API client, session, legal, error, and media lifecycle tests |
| `npm run test:e2e` | `10 passed` | Generate, SRT-derived style, sync, sequential batch naming, access code, token protection, refresh recovery, legal routes, decoded waveform pixels, responsive layout, select-icon inset, and feature-grid alignment |
| `npm run typecheck` and `npm run build` | PASS | TypeScript and Vite production bundle |
| Production web `generate` smoke | PASS | 3.444-second WAV, 1 cue, 0 QC flags, `$0.000376` recorded provider cost on Render commit `5c79356` |
| Render JSON Schema validation | PASS | `render.yaml` validates against Render's published schema |
| Production dependency audit | PASS | `npm audit` and isolated `pip-audit` for `.[cloud,web]` report no known vulnerabilities |
| `python -m dubsync --help` | PASS | Exposes `sync`, `batch`, `generate`, `profile`, and `report` |
| `python -m dubsync profile Examples\"srt test.srt" -o tmp_profile_smoke.yaml` | PASS | Reproduces the 30 fps, 2-line, 26-char, 0.5s min-duration house profile |
| Fixture-backed sync tests | PASS | Cost meter records fixture/local/resumed paths as zero API cost |

On 2026-07-11, the single approved paid web smoke ran through `https://dubsync.onrender.com` in `generate` mode. ElevenLabs Scribe produced one cue and the configured Gemini punctuation pass completed without a provider error. The protected result reported `$0.000376` total provider cost, and all three artifacts downloaded successfully. QC reported zero flags and one line-length warning; that warning exposed a punctuation-stage reflow defect, which is now covered by unit and generate-pipeline regression tests. A second paid run was not made because approval covered one provider-backed job.

`render.yaml` was schema-validated, but the Docker image was not built locally because Docker is unavailable on this machine. The connected GitHub repository and Render service provide the production Docker build evidence.

### Top 3 Risks

1. Live-provider drift: the ElevenLabs plus Gemini generate path has one production smoke result; OpenAI, Anthropic, AssemblyAI, WhisperX, pyannote, and MMS still rely on deterministic coverage until separately authorized live tests are run.
2. Real-episode quality: synthetic fixtures prove timing, improv replacement, overlap, dropped-line, and source-error paths, but the PLAN targets need a golden episode set to measure cue-start MAE, improv precision/recall, and review burden on actual delivered material.
3. Language quality beyond generic handling: CJK/full-width behavior is covered, but production quality for Japanese/Thai/Chinese/Korean and code-switching will benefit from language-specific tokenization and per-language house-style samples.

## Known Gaps

- The approved ElevenLabs plus Gemini smoke covered commit `5c79356`; the punctuation reflow correction that followed is verified offline and was not given a second paid run.
- WhisperX local transcription is wired through the documented Python API: `load_model`, `load_audio`, `transcribe`, `load_align_model`, `align`, and optional diarization.
- Live pyannote execution was not smoke-tested; it requires `dubsync[diarize-local]`, accepted model terms, and a Hugging Face token or local model path.
- MMS forced alignment is implemented behind `dubsync[precision]`, but real model execution was not smoke-tested in this workspace.
- Deterministic energy VAD is wired; optional Silero VAD is available with energy fallback, but production quality should be validated on a golden set.
- CJK/Thai/Hangul/Japanese text now uses character-level tokenization and visual-width line checks; language-specific morphological tokenizers remain a future quality upgrade.
- Live LLM speaker-to-character inference is implemented through the configured LLM adapter, but was not smoke-tested against real provider responses in this workspace.
- Live Gemini punctuation completed in the production web smoke; OpenAI and Anthropic usage metering remains covered only by deterministic response-shape tests.
- Audio-snippet double-checks are implemented for Gemini inline audio, but were not live-smoke-tested against the real Gemini API in this workspace. Automatic Gemini context-cache creation/deletion remains unimplemented because it creates third-party resources and can incur storage billing; provide `cached_content` to reuse a cache created outside DubSync.
- OpenAI and Anthropic token prices require explicit config overrides until model-specific billing defaults are confirmed.
- The commercial web workspace supports submission, status, and downloads; a browser cue editor remains intentionally out of scope until customer QC behavior proves it is needed.

## Troubleshooting

- If `dubsync.exe` is not on `PATH`, use `python -m dubsync`.
- If `uv` is unavailable, use `python -m pip install -e ".[dev]"`.
- If ffmpeg fails, confirm `ffmpeg -version` works in the same PowerShell session. A timeout reports explicitly; increase `DUBSYNC_FFMPEG_TIMEOUT_SECONDS` only for validated long-running media.
- If cloud providers fail, check `.env` keys and install `.[cloud]`.
- If a punctuation pass changes words, DubSync rejects the batch and leaves a QC flag path for review.
- If adjudication returns invalid structured output twice, DubSync preserves the source SRT text and emits an `invalid_llm_response` QC flag.
- WhisperX adapter wiring was checked against the official project README: https://github.com/m-bain/whisperx
- pyannote community-1 adapter wiring follows the official README/model card `Pipeline.from_pretrained(...); pipeline("audio.wav")` flow: https://github.com/pyannote/pyannote-audio and https://huggingface.co/pyannote/speaker-diarization-community-1
- ctc-forced-aligner wiring follows the official README Python API (`load_alignment_model`, `load_audio`, `generate_emissions`, `preprocess_text`, `get_alignments`, `get_spans`, `postprocess_results`): https://github.com/MahmoudAshraf97/ctc-forced-aligner
