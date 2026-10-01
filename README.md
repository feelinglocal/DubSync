# DubSync

DubSync is a Windows-friendly Python 3.11+ CLI for retiming a customer-supplied target-language SRT to dubbed VO-only audio, preserving source segmentation where compatible with confirmed dialogue and splitting clearly identified actor turns. The wording goal follows what the actors actually say, including improvised sentences and performed register. Dialogue changes and uncertain mappings remain visible in QC.

Timing comes from acoustic data only: ASR word timestamps, streaming speech-energy analysis, and optional forced alignment. LLM adapters are used only for language decisions such as improv adjudication and punctuation, never for timestamps.

Automatic processing produces reviewable subtitles, not a guarantee of perfect transcription or phoneme timing. Uncertain wording edits keep the source text and remain in QC; the cue can still be timed from its uniquely owned words. Missing or ambiguous acoustic evidence preserves source timing for review. Independently approved edits elsewhere in that cue are retained. Processing continues through known ambiguous passages, which remain reviewable in the delivered artifacts. Real speech overlaps and readability pressure remain visible. Completed web jobs show the QC review state next to their downloads. See [the October 1 accuracy-upgrade report](docs/testing/accuracy-upgrade-2026-10-01.md) for final v9 measurements and explicit acceptance exceptions, and [the September quality audit](docs/testing/app-quality-audit-2026-09-05.md) for historical results.

## Commercial Web MVP

DubSync also includes a responsive React/FastAPI application with two customer workflows:

- **Sync existing SRT:** upload one file pair or a batch of up to 10 matched audio/SRT pairs, then download synchronized SRT and QC artifacts for each source. Batch children run one by one, and cue presentation rules are derived from each uploaded SRT. The web sync form can optionally cap cues to a chosen maximum line count; overlong source-backed cues are split only when aligned ASR word timing provides a safe acoustic split point, otherwise they are kept and QC-flagged for review.
- **Audio to SRT:** upload one audio file or a batch of up to 10, choose a built-in subtitle preset, enter custom line/timing/CPS rules, or upload an example SRT to derive them, then generate acoustically timed SRT and QC artifacts.

The language selector defaults to Auto-detect and includes **Japanese 日本語** for both workflows. An explicit language is forwarded to the configured ASR provider and becomes part of the cached ASR configuration. Sync can derive a conservative hint from clearly identifiable source dialogue when no concrete language is set; short, mixed, or unrecognized text leaves provider detection in control. An inferred hint appears in QC and can be overridden explicitly. Audio-to-SRT generation retains provider auto-detection when no language is selected.

### Japanese subtitles

Japanese processing handles unspaced kanji/kana and mixed Latin text in both sync and generation. Comparison normalizes full-width/half-width variants while preserving voiced kana distinctions; authored subtitle text retains its original Unicode representation. Grouped ASR words are compared in the same character units as the source, with every match linked back to the original acoustic word timestamps. Exact matching transcripts use a linear alignment shortcut.

If one ASR timestamp spans multiple source cues, their internal boundary remains uncertain. DubSync preserves those source timings with a QC warning, including on verification resume; usable per-cue forced alignment can resolve them. Repeated characters and multiple accepted corrections are applied to their exact original text positions. Uncertainty does not prevent the job from completing or the SRT from being downloaded.

Generated subtitles join Japanese fragments without extra spaces and recognize Japanese sentence punctuation and closing quotes. Line wrapping follows best-effort [Japanese line-breaking conventions](https://www.w3.org/TR/jlreq/#line_breaking_rules). If a narrow custom width cannot fit an unbreakable cluster, output remains available and the existing style report identifies the overflow. Japanese requires no extra dictionary, mandatory language selection, or language-specific job gate; these text improvements also apply under Auto-detect. Custom profiles and SRT-derived styles remain available.

CLI `sync`, `batch`, and `generate` accept `--language ja` (also `jpn` or `ja-JP`) or `--language auto`. Omitting the option preserves the provider configuration. For example:

```powershell
dubsync sync episode.srt episode.wav -o episode.synced.srt --providers provider.yaml --language ja
dubsync generate episode.wav -o episode.generated.srt --providers provider.yaml --language ja
```

Language hints reach MAI-Transcribe 2, ElevenLabs, OpenAI Whisper, AssemblyAI, and WhisperX as ISO-639-1 codes: `deu`, `pt-BR`, and `ja-JP` are sent as `de`, `pt`, and `ja`, and codes without an ISO-639-1 form pass through unchanged. AssemblyAI's default model selection includes its documented multilingual fallback for Japanese or auto-detection. An explicit Japanese selection also sets an already configured MMS aligner to `jpn`; it does not enable or install that optional model. Automated tests use provider fixtures and mocked SDK calls; real Japanese audio transcription and optional model quality still need listening-based evaluation.

The first commercial release intentionally has no customer accounts, subscriptions, or Supabase dependency. Manual quotes issue a rotating job access code before paid processing, and every accepted child job receives a separate secret browser-held result token. Uploads and results expire 24 hours after each child finishes, and the API limits job creation per source IP. Production job intake fails closed when the access code is not configured. See `docs/COMMERCIAL_PLAN.md` for the product scope, provisional pricing, deployment limits, roadmap, and paid-launch gates.

Local web setup:

```powershell
python -m pip install -e ".[dev,cloud,web]"
Set-Location web
npm ci
npm run build
Set-Location ..
python scripts/run_local.py
```

Open `http://127.0.0.1:8000`. For local use, run `python scripts/run_local.py`; it loads this checkout's `.env` with precedence over inherited shell values, so a stale parent API key cannot replace your local key. The regular `dubsync-web`/Uvicorn production entry points continue to prioritize deployment environment variables. The server reads `provider.yaml` and `style_profile.yaml` by default. The DubSync default generation preset honors that configured style profile; every other web generation style is resolved per job. API documentation is disabled unless `DUBSYNC_ENABLE_DOCS=1`.

Job intake defaults to twenty submissions per source IP per hour (`DUBSYNC_MAX_SUBMISSIONS_PER_HOUR`) and twenty outstanding child jobs (`DUBSYNC_MAX_OUTSTANDING_CHILD_JOBS`), enough for two full ten-file batches. Single and batch request payloads are capped at 512 MiB, retained job commitments are capped at 4 GiB, and concurrent uploads reserve their combined inbound bytes before copying so overlapping requests cannot race the 10 GB disk admission check. SRT files are capped at 2 MiB, 60,000 lines, and 20,000 cues, with bounded line lengths and an incremental parser so structurally hostile subtitle files fail before expensive processing. Before acceptance, non-fixture audio is probed with a 15-second deadline and its predicted 16 kHz PCM plus work allocation is reserved. Production also enforces a four-hour audio limit, a 1 GiB per-job ceiling, bounded normalized/snippet outputs, and 2 GiB of minimum free disk. Configure these bounds with the `DUBSYNC_MAX_*` and `DUBSYNC_MIN_FREE_STORAGE_BYTES` variables shown in `.env.example`. Existing deployments that only set `DUBSYNC_MAX_JOBS_PER_HOUR` retain that value as a fallback. Queued or processing jobs with no state update for 24 hours are dead-lettered on startup or periodic cleanup, then retained for the normal terminal retention window; configure that deadline with `DUBSYNC_ACTIVE_JOB_TIMEOUT_HOURS`. Every FFmpeg subprocess has a finite 1,800-second default timeout controlled by `DUBSYNC_FFMPEG_TIMEOUT_SECONDS`.

For frontend development, run `npm run dev` inside `web` and run the FastAPI service separately. The production Docker image builds the frontend and serves it from the same origin as the API.

### Render Deployment

`Dockerfile` and `render.yaml` define the production architecture: one Starter web service in Singapore, two bounded background processing threads, and a 10 GB persistent disk mounted at `/var/data`. Independent submissions or batches may process concurrently, while children inside each batch remain strictly serial. SQLite metadata and per-job files live on that disk, queued work is claimed atomically so an accidental duplicate submission cannot process the same job twice, and a data-directory process lock rejects a second service process before it can requeue active work. This remains a single-process, single-instance design with deployment downtime and no horizontal scaling.

To deploy:

1. Put this workspace in a real private Git repository and connect that repository to Render.
2. Create a Blueprint from `render.yaml`.
3. Enter `OPENROUTER_API_KEY` for default MAI-Transcribe 2 transcription, `ELEVENLABS_API_KEY` for optional Scribe v2 transcription, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and a strong `DUBSYNC_JOB_ACCESS_CODE` as Render secrets. Never commit `.env`.
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

Gemini 3.5 Transcribe ASR is disabled. The CLI, web API, UI, and queued-job processor reject that transcription provider, including stale saved selections. Microsoft MAI-Transcribe 2 via OpenRouter is the default cloud ASR for new sync and audio-to-SRT jobs; ElevenLabs Scribe v2 remains selectable. `GEMINI_API_KEY` powers Gemini 3.8 Flash adjudication with medium thinking and the unchanged Gemini 3.7 Flash punctuation pass with medium thinking.

Create `.env` as needed:

```text
OPENROUTER_API_KEY=...
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

New ASR checkpoints record source and normalized-audio hashes. A timing-stage resume rejects changed/missing audio before overwriting existing artifacts; use `--resume asr` to regenerate acoustic evidence. Legacy checkpoints remain readable with an explicit unverified-provenance QC warning. Rebuild checkpoints predating the current text/timing safeguards must resume from `rebuild`; `verify` also rejects decisions that no longer satisfy the configured confidence gate. These policies reuse the saved source snapshot intentionally. Normalized audio, caches, and result artifacts replace existing files only after a complete new file has been written.

`--local` disables LLM calls and selects the WhisperX local-test ASR path. Normal cloud processing defaults to Microsoft MAI-Transcribe 2 through OpenRouter (`provider.yaml`). The web workspace has a transcription model picker for both Sync and Generate, including batches, and preselects MAI-Transcribe 2; ElevenLabs Scribe v2 remains available. Each job stores its selection, so changing the default does not change previously submitted jobs; queued jobs saved with the older `default` value still use Scribe v2. If the default model's key is missing, the picker shows it as unavailable and the job is not switched to the other model.

Set `OPENROUTER_API_KEY` in the server environment for MAI and `ELEVENLABS_API_KEY` for Scribe. Keys stay on the server; `/api/config` exposes only model names and availability. MAI requests word timestamps, diarization and verbatim text. Long normalized audio is sent in bounded chunks with overlapping context. Speaker IDs are linked across a chunk cut when the words both chunks heard in the overlap map one label to one label; otherwise they stay scoped to their chunk. Each chunk gets at most three requests for timeouts, HTTP 408/5xx and connection errors; a chunk that keeps timing out with diarization is retried once without it and its words carry no speaker labels. Re-decoded duplicate word runs and doubled countdown numbers are removed before timing is used, with informational QC flags. MAI cannot silently fall back to Scribe or fabricate timing if the provider omits word timestamps. Scribe requests get the same bounded retries.

`cost.json` records OpenRouter's reported `usage.cost` as `audio_billed`; when billing metadata is unavailable, the configured hourly rate is an estimate. Scribe costs use its configured hourly estimate. Cached transcription makes no new charge. MAI's September 5, 2026 catalog estimate is $0.10/audio-hour; set `asr.dollars_per_hour` if the rate changes.

See the [September 5, 2026 MAI/Scribe comparison](docs/testing/mai-transcribe-2-comparison-2026-09-05.md) for historical latency, reference-text agreement, costs, and validation limits. Failed requests retain sanitized billing evidence in `asr_failure.json`; known charges with an uncertain total are marked `audio_billed_partial`, and a failed or retried request that may still have been billed is estimated as `audio_uncertain_estimate`.

### Optional ASR wording cross-check

Sync can request the other transcription model as independent wording evidence; it is off by default. The primary alone supplies timing and word ownership. The web checkbox explicitly defaults off, including when YAML enables it. For a MAI primary, YAML uses `asr.cross_check: {provider: elevenlabs, model_id: scribe_v2}`; for Scribe, use `{provider: openrouter, model: microsoft/mai-transcribe-2}`. Generation does not use this option.

Only narrow, unambiguous time-local multiword agreement can bypass audio review. Names, numbers, negations, repeated/character-level tokens, conflicting ownership, and source/acoustic holds remain reviewable; agreement is not proof of hearing. Secondary cache, settings, audio provenance, and costs are separate. Resume after ASR requires matching saved secondary evidence and makes no new secondary request; a requested secondary failure is explicit and retains both cost records. Recent paired runs showed no incremental WER benefit, so optional extra transcription spend is not enabled automatically. See the [accuracy report](docs/testing/accuracy-upgrade-2026-10-01.md) for scope and limits.

### Gemini Audio Context

Gemini adjudication receives focused case audio clips and nearby read-only source context, with exact editable-span and acoustic word ownership. Prompt v12 also carries the resolved language and `register_policy`. The default `adjudication.register_policy: spoken` follows the clearly audible performed form, including equivalent Portuguese reductions such as `para`/`pra` and `está`/`tá`. Set `adjudication.register_policy: script` to preserve authored equivalents explicitly. Register shortcuts do not settle genuine changes in meaning, number, or negation; those require audio adjudication. Instructions prohibit borrowing neighboring words, moving words between scenes, translating dialogue, or returning replacement timestamps. No-LLM processing retains script wording.

The native v12 reply contains `heard_text` and an `evidence` value of `heard_clearly`, `heard_unclear`, or `not_audible`; it does not ask the model for a confidence number. Clear hearing is still checked against the bounded wording and editorial policy. Unclear or inaudible replies keep source wording with a review item. Legacy stored decisions retain the configurable confidence gate. This evidence contract limits automatic edits; it is not a guarantee that a model heard the words correctly.

Deterministic checks run before paid adjudication for punctuation/casing, finite language-scoped spacing and abbreviation equivalents, established source-name spellings, and equivalent register reductions. Under `spoken`, a proven whole-span register reduction uses exact ASR wording; formatting, markup, abbreviation, and name differences can still require audio review. Under explicit `script`, equivalent register variants preserve source text. Unknown languages receive no language-specific equivalence rules; ambiguous names, lexical compounds, changed numbers, and negations still require review. The [September 14 comparison](docs/testing/flash-lite-adjudication-2026-09-14.md) is historical evidence for the earlier prompt and routing, not validation of v12.

For an eligible whole cue absent from ASR, native adjudication asks a bounded audio question between independently matched neighboring anchors. A separate question can own the exact source-token residue in a neighboring cue; other words remain read-only. An empty ASR span never proves omission: removing the missing cue requires complete audio, clear native hearing that it is absent, and no detected speech activity in the anchored gap. Recovered wording needs one independent speech chain, with each internal pause strictly below 0.2 seconds, before acoustic timing can replace the source interval. Multiple separated chains, borrowed neighboring wording, or different residual wording without owned word timing stay reviewable. No-LLM processing preserves these source passages. Saved legacy decisions that do not answer the new questions also preserve them for review; they do not receive fabricated successful hearing evidence. A narrow interruption rule can partition an otherwise overlapping sentence around another actor's independently owned insertion when every word has an exact, unique partition. It retains the whole sentence and reaction, uses existing acoustic timestamps and preserves genuine greeting/chorus context; uncertain cases remain reviewable.

The default route uses Gemini 3.8 Flash at MEDIUM thinking after deterministic predecisions. The older October script-route comparison was repeated after recovering truncated hybrid replies; its written-reference conventions differ from the actor-spoken goal. Final v9 technical checks and all 13 actual SRT/QC inspections are complete. Ten explicit uncertain-source, actor-owner and lyric/dialogue overlap pairs remain for owner acceptance; the [accuracy-upgrade report](docs/testing/accuracy-upgrade-2026-10-01.md) records those exceptions, mixed reference scores, caption visibility tradeoffs and the closed validation cost.

The Lite-first hybrid remains opt-in through YAML: set `llm.adjudication.model: gemini-3.5-flash-lite`, `thinking_level: high`, and enable a `fallback` using `gemini-3.8-flash` with `thinking_level: medium`. Deterministic checks escalate invalid or uncertain replies, retained source text that conflicts with ASR, and proposed wording that differs from owned ASR words. The reviewer receives only padded clips, nearby cues, word ownership, and review reasons, with no full episode audio or shared episode cache. Missing, invalid, or uncertain replies preserve the source and produce QC findings. `hybrid_adjudication.json` records each route and reason; costs retain the actual model for each call. Disabling fallback while keeping Lite as primary selects Lite-only. Agreement between models does not prove correctness. The [September 14 comparison](docs/testing/flash-lite-adjudication-2026-09-14.md) remains historical evidence; verify any deployed commit separately.

If a required focused clip is unavailable or does not cover its case, that case retains source wording with a QC flag. Timing can still use uniquely owned acoustic words; missing or ambiguous word ownership preserves source timing. Other cases with complete audio can still proceed.

The optional `llm.adjudication.audio_context.enabled: true` route supplies full episode audio and ordered source subtitle context alongside those clips. It remains available for configurable alternatives such as Gemini 3.8 Flash. The default uses local clips and disables full episode audio. The transport and fallback rules below apply when this route is enabled.

For audio longer than 180 seconds, DubSync prepares one mono 24 kHz, 64 kbps MP3 and uploads it once through Gemini's Files API; already compact MP3s can be reused. Short inputs retain the normalized WAV. Focused case snippets remain WAV and carry their episode offsets. They are sent inline unless the estimated aggregate request, including base64 encoding, exceeds 18 MB, in which case their Files API URIs are used.

The episode audio and ordered source context are reused through a job-owned Gemini cache when eligible. Its TTL is capped at 900 seconds and renewed only within the job lifetime. Job completion or failure triggers cleanup of the owned cache and uploads; cleanup or billing uncertainty is reported in `gemini_audio_context.json` and QC. If cache creation or renewal fails, the existing audio URI can be used within a cumulative 256,000 uncached audio-token budget. Upload failure or budget exhaustion holds the affected source dialogue for QC rather than proceeding without the required context. Generation, cache creation, and cache storage costs are recorded in `cost.json` when available, with uncertainty made explicit.

On September 10, 2026, local preparation of the supplied long audio reduced the context upload from 71.568 MB to 23.724 MB in 5.968 seconds, with a measured duration difference of -0.005 seconds. This verifies compression and duration preservation only; it does not establish adjudication accuracy or subjective audio quality.

## QC and Acoustic Timing

`qc_report.html` and `qc_report.json` separate findings into four views: `review` contains actionable problems, `changes` records successful wording and timing operations, `notes` contains episode information, and `diagnostics` retains operator details. Related findings collapse into one review item with the delivered SRT cue numbers and timecodes. The raw `flags` and `style_issues` remain in JSON; fewer review items do not mean findings were deleted. Unknown warning/error kinds enter review by default.

The job verdict is `clean` with no review items, `check` with warnings, or `attention` with any review error. A warning-only job also becomes `attention` when at least ten cues need review and they exceed 10% of the delivered cues. These tiers describe review burden, not guaranteed subtitle accuracy.

`changes.diff.srt` is a change log with one block per applied wording edit, addition, or removal in playback order, including old/new text. Timing changes remain in the QC report. Invalid or unavailable change windows use explicitly labeled 1 ms diagnostic markers; those markers never define delivered dialogue timing. Cue scores are shown only when real ASR or forced-alignment confidence exists. Missing provider confidence is `null`/unscored, not zero.

Default energy VAD streams the normalized WAV at a 10 ms hop. Its thresholds adapt to the file's speech level and noise floor, with hysteresis for quiet edges. Set `vad.threshold_dbfs` for a fixed threshold or `vad.window_ms` for legacy non-overlapping analysis windows. Music beds can keep an energy detector active through pauses; full-mix quality needs separate validation.

`timing.phrase_edge_snap` repairs stretched ASR phrase edges against their speech burst before downstream timing. Set it to `false` to disable that repair, or supply `start_advance_ms`, `end_extension_ms`, and `models.<ASR model id>` overrides. Both sync and generation honor the selected model's settings. Multiple plausible bursts remain uncertain instead of forcing a unique word boundary.

`timing.min_duration_policy: extend_into_silence` lets a short cue reach its display minimum only while verified silence remains, capped by other speech and the next cue, including held dialogue. `acoustic` keeps speech-bound ends. Generation also uses verified silence for a selected CPS target, capped by media and maximum-cue duration; it never delays cue starts. A short or fast cue can remain when there is no room. Frame-grid values are rounded upward to milliseconds at export so editor frame flooring does not lose a frame.

Useful raw kinds include `song_lyric_source_kept` (note), `asr_duplicate_words_dropped` and `asr_word_clamped` (diagnostics), `source_cue_timing_repaired` (review), and `timing_evidence_held` (review). Absent lyric captions keep source text/timing, including cues with a single music-note edge. Genuine simultaneous speech remains a review item when a safe non-overlapping output cannot preserve all words.

With `output.no_overlaps: true` (default), pure bracketed screen captions without owned speech words are composed with intersecting speech displays. Every spoken cue retains its ID, interval, original line sequence, speaker metadata, and word ownership; caption lines are appended and may repeat across adjacent speech and caption-only segments. Lyrics remain timed text and are never repeated as caption tracks. Composition preserves caption coverage but can extend a caption before or after its authored screen edge; the [measured visibility tradeoffs](docs/testing/accuracy-upgrade-2026-10-01.md#r63-dialogue-and-annotation-overlaps) include zero movement of speech. Combined line count/width and genuinely fast caption reading remain reviewable. Set `output.no_overlaps: false` to preserve original annotation segmentation.

## Provider Matrix

| Role | Provider | Status | Config |
|---|---|---|---|
| ASR default | Microsoft MAI-Transcribe 2 via OpenRouter | Word timestamps, diarization, actual billing metadata | `asr.provider: openrouter`, `model: microsoft/mai-transcribe-2`, `diarize: true`, optional `keyterms` / `character_names` |
| ASR alternative | ElevenLabs Scribe v2 | Word timestamps and diarization | `asr.provider: elevenlabs`, `model_id: scribe_v2`, `diarize: true`, optional `keyterms` / `character_names` |
| ASR fallback | OpenAI Whisper | Implemented optional adapter, no diarization | `asr.provider: openai`, `model: whisper-1` |
| ASR fallback | AssemblyAI | Implemented optional adapter | `asr.provider: assemblyai`, `model: universal-3-pro` or `universal-2`, `speaker_labels: true` |
| ASR retired | Gemini 3.5 Transcribe | Disabled; provider selectors reject it in CLI, web, and queued jobs | No supported configuration |
| ASR local | WhisperX | Implemented optional adapter; requires `dubsync[local]` | `asr.provider: whisperx` |
| Test/offline | Fixture wordstream | Implemented | `asr.fixture_path: path/to.wordstream.json` |
| LLM text default | OpenAI GPT-5.6 Luna | Implemented adapter using the Responses API | `llm.provider: openai`, `model: gpt-5.6-luna`, per-pass `reasoning_effort` |
| LLM adjudication default | Gemini 3.8 Flash | Local source context, focused audio clips, v12 evidence decisions | `llm.adjudication.provider: gemini`, `model: gemini-3.8-flash`, `thinking_level: medium`, `fallback.enabled: false` |
| LLM adjudication alternative | Gemini 3.5 Flash-Lite | Optional Lite-first primary; focused audio clips and local context | `llm.adjudication.model: gemini-3.5-flash-lite`, `thinking_level: high` |
| LLM adjudication review | Gemini 3.8 Flash | Opt-in hybrid reviewer for flagged Lite cases | `llm.adjudication.fallback.enabled: true`, `model: gemini-3.8-flash`, `thinking_level: medium` |
| LLM punctuation default | Gemini 3.7 Flash | Word-preserving punctuation pass | `llm.punctuation.provider: gemini`, `model: gemini-3.7-flash`, `thinking_level: medium` |
| LLM alt | Anthropic | Implemented optional adapter | `llm.provider: anthropic` |
| Test/offline | Fixture decisions | Implemented | `llm.provider: fixture` |
| Precision verify | Fixture forced alignment | Implemented | `forced_alignment.fixture_path: path/to.forced-align.json` |
| Precision verify | MMS / ctc-forced-aligner | Implemented optional adapter; requires `dubsync[precision]` and model runtime | `forced_alignment.provider: mms` |
| Overlap backstop | Fixture overlap regions | Implemented | `overlap_detection.fixture_path: path/to.overlap.json` |
| Overlap backstop | pyannote community-1 | Implemented optional adapter; requires `dubsync[diarize-local]` and model access | `overlap_detection.provider: pyannote` |
| Speech activity | Fixture VAD regions | Implemented | `vad.fixture_path: path/to.vad.json` |
| Speech activity | Adaptive energy VAD | Streaming 10 ms bursts; fixed settings remain configurable | `vad.provider: energy` |
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
  provider: openrouter
  model: microsoft/mai-transcribe-2
  diarize: true
  keyterms:
    - Drachen-Evolutionssystem
  character_names:
    - Luna
    - Matthew

adjudication:
  # Follow performed wording; set script to preserve authored equivalents.
  register_policy: spoken

llm:
  provider: openai
  model: gpt-5.6-luna
  timeout_seconds: 90
  max_retries: 2
  # Per-pass overrides inherit the base settings unless provider/model changes:
  adjudication:
    provider: gemini
    model: gemini-3.8-flash
    confidence_gate: 0.7
    scene_gap_seconds: 4.0
    thinking_level: medium
    fallback:
      enabled: false
      provider: gemini
      model: gemini-3.8-flash
      thinking_level: medium
    audio_context:
      enabled: false
      compress_long_audio: true
      cache_enabled: true
      cache_ttl_seconds: 900
      max_uncached_audio_tokens: 256000
    audio_snippet_double_check:
      enabled: true
      pad_seconds: 2.0
      max_duration_seconds: 20.0
  punctuation:
    provider: gemini
    model: gemini-3.7-flash
    scene_gap_seconds: 4.0
    thinking_level: medium
  speaker_mapping:
    reasoning_effort: medium
  # Optional for providers/models without built-in defaults:
  # input_per_million: 1.0
  # output_per_million: 6.0

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
  # Omit threshold_dbfs/window_ms for adaptive 10 ms detection.
  # threshold_dbfs: -45.0  # optional fixed threshold
  # window_ms: 100         # optional legacy windows
  min_coverage: 0.2
# Optional neural VAD when torch/Silero are available; otherwise use energy.
# vad:
#   provider: silero
#   sampling_rate: 16000

timing:
  max_word_duration: 2.0
  max_intra_cue_gap: 1.5
  min_duration_policy: extend_into_silence
  # phrase_edge_snap: false  # disable phrase-edge repair
  # phrase_edge_snap:
  #   start_advance_ms: 200
  #   end_extension_ms: 300
  #   models:
  #     scribe_v2: {start_advance_ms: 200}
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

The CLI writes `cost.json` and prints a cost meter. Fixture, local, resumed, and cached ASR paths record zero API cost. Uncached cloud ASR calls are metered from WAV duration and the configured provider price. Live LLM calls record token costs when the provider response exposes usage metadata and a built-in GPT-5.6 Luna/Gemini price or explicit `input_per_million` / `output_per_million` pricing is available. `llm.adjudication`, `llm.punctuation`, and `llm.speaker_mapping` can override provider/model/pricing per pass.

| Item | Planned cost basis |
|---|---|
| MAI-Transcribe 2 ASR (default) | OpenRouter's reported `usage.cost` per request; `$0.10/hr` estimate when billing metadata is missing |
| Scribe v2 ASR | audio seconds x provider hourly price (`$0.22/hr`, or `$0.27/hr` when `keyterms` or `character_names` enable keyterm prompting) |
| AssemblyAI ASR | audio seconds x provider/model hourly price (`$0.21/hr` for `universal-3-pro`, `$0.15/hr` for `universal-2`, plus `$0.02/hr` when `speaker_labels` is enabled; enabled by default) |
| LLM adjudication/punctuation | input/output tokens x model price |
| Full episode context and focused audio clips | generation input, cache creation, and cache storage usage; URI reuse alone does not eliminate input-token charges |
| Local forced alignment/diarization | zero API cost |

## What Works Now

- SRT parsing accepts comma or period timestamps, empty cues, and a new cue header without a blank separator. Sync omits empty entries without merging neighboring cues, preserves them in ingest metadata, and adds `source_empty_cues_ignored` review; an entirely blank source fails before ASR. The writer removes interior blank lines and rejects wholly blank output cues, negative starts, and zero/reversed durations. Original input files are preserved. Byte readers support UTF-8, BOM UTF-16/32, and a warned Windows-1252 fallback; sync surfaces `source_encoding_converted` review. Ambiguous Japanese legacy bytes require an explicit encoding or conversion to UTF-8.
- Style profile derivation uses robust duration/CPS percentiles and confident sample FPS detection, with the configured FPS as fallback.
- `profile` rejects malformed sample SRT files with clear CLI errors instead of raw parser exceptions.
- Malformed `--providers` and `--style` YAML files are rejected with clear CLI errors that name the config file.
- Invalid style-profile values such as `fps: 0` are rejected with clear CLI errors that name the file and field.
- Non-mapping provider config sections such as `vad: []`, `forced_alignment: []`, `overlap_detection: []`, and `speaker_mapping: []` are rejected instead of being silently ignored.
- `sync` and `batch` load `.env` from the current working directory before resolving provider keys, without overwriting already-set environment variables.
- Fuzzy monotonic, band-limited SRT-token to ASR-word alignment with a bounded cue-time tie-breaker, anchor regions, and divergence spans persisted in `align.json`.
- Delete-only divergence spans inherit the surrounding matched-word window, so dropped-line/adjudication cases have concrete boundary timestamps when bounded by anchors.
- Alignment normalization maps common digit strings and English/German number words to the same canonical tokens, avoiding false divergences such as `2` vs `two`.
- Source cues are sorted chronologically before alignment while preserving original cue ids; moved cues are reported as `source_out_of_order`.
- Deterministic re-cueing from ASR word timestamps, frame snapping, min duration, and zero-gap chaining.
- Cue starts floor-snap and cue ends ceil-snap, so fractional model timings cannot truncate the final spoken syllable.
- Min-duration padding extends only into verified silence and stops before following speech or cues. Conflicting evidence that would reverse or collapse a cue's duration preserves its prior timing with `timing_refinement_held` QC; unresolved invalid intervals cannot be exported.
- Frame-grid ceiling never snaps fractional model timings backward when enforcing minimum duration or forced-alignment ends.
- Configured cue lead-in is clamped at zero so early speech cannot produce invalid negative SRT timestamps.
- Fixture-backed ASR/LLM path for offline E2E tests.
- Web audio generation resolves explicit presets, custom rules, and uploaded-example subtitle styles per job while preserving the configured profile for the DubSync default preset; the selected line, timing, gap, lead/tail, and CPS rules are recorded in `generate.json` and applied during output finalization.
- Generation uses exact word ownership and the shared overlap resolver, balanced wrapping for spaced text, and sentence boundaries that respect titles, ordinals, abbreviations, and continuing ellipses. Selected reading-time targets extend ends only into verified silence.
- Web sync derives its style from the user-supplied source SRT instead of applying the server's global generation profile, with an optional maximum-line override that uses aligned word timing instead of blind text-only splitting.
- Web batch intake accepts up to 10 matched audio/SRT pairs, matches them by case-insensitive filename stem, and submits each batch as one serial work unit. With two bounded workers, two batches may run concurrently while the children inside either batch remain sequential.
- Browser-held access recovers every child in a submitted batch after refresh, while each child keeps an isolated token and failure state.
- Downloaded SRT names preserve the validated source stem and append `-dubsync-synced.srt`.
- ElevenLabs Scribe v2 ASR forwards configured keyterms and character names as `keyterms` while still requesting word timestamps and diarization.
- Opt-in `--live` pytest smoke tests for OpenAI GPT-5.6 Luna, Gemini, Anthropic, ElevenLabs, OpenAI Whisper, and AssemblyAI are deselected from normal offline test runs.
- OpenAI LLM calls use the Responses API `responses.parse` structured-output path with `store: false`, bounded SDK retries/timeouts, refusal and incomplete-response handling, and explicit `reasoning.effort`. The production base text model is `gpt-5.6-luna`; speaker mapping uses `reasoning_effort: medium`. Gemini 3.7 Flash punctuation uses `thinking_level: medium`.
- Gemini LLM calls use the installed `google-genai` `models.generate_content` API with JSON response schemas.
- Gemini thinking-level controls use `thinking_config.thinking_level`; adjudication defaults to Gemini 3.8 Flash medium. Lite high with Flash medium review remains an opt-in YAML route. Punctuation remains Gemini 3.7 Flash medium.
- Gemini episode context uses a bounded job-owned Files API upload and explicit cache with cleanup. Externally supplied `cached_content` remains available when automatic episode context is disabled; DubSync does not replace or delete that external cache.
- Long-audio context is compressed once, and the complete source transcript is serialized losslessly for the shared cache. Batches contain at most eight cases with four concurrent requests; timed-out multi-case batches have one bounded individual-case recovery pass. See [the September 10 subtitle audit](docs/testing/subtitle-quality-fixes-2026-09-10.md) for measured evidence and limits.
- Default adjudication checks extract padded WAV snippets, persist `audio_snippets.json`, include source/snippet hashes and context settings in the LLM cache key, and use inline audio or Files API URIs according to the aggregate request size.
- Improv replacement path with QC flags and acoustic timing from spoken ASR words.
- ASR-only ad-lib spans can be accepted by adjudication and inserted as acoustically timed cues, while far-tail and highly repetitive music-like candidates are held as error-level QC findings instead of captioned.
- Exported SRT files are sequentially renumbered in playback order, including ad-lib cues inserted between existing source cues.
- `keep_srt` adjudication still attaches divergent ASR word indices to timing, so kept source spelling/numbers do not cut off the actor's spoken span.
- Final output sorting only deduplicates exact text at the same onset when speakers do not conflict. Acoustically supported overlaps remain visible for review instead of moving an actor's utterance; final starts are monotonic and every exported interval has positive duration.
- Multi-cue improv replacements are distributed once. When an explicitly mapped replacement consumes earlier cues and a following cue's prefix, leaving one replacement word before a surviving clause, that word's text and acoustic timing stay with the surviving clause instead of becoming an orphan cue.
- Multi-cue improv timing partitions the accepted spoken word indices across affected cues, so rebuilt changed cues do not all inherit the full span timing.
- Exact local word and speaker evidence separates actor turns even when a cue already fits its line limit. Each split retains its own spoken-word indices; ambiguous mappings remain intact with `speaker_turn_split_held` QC instead of assigning improvised clauses to an actor by token count.
- Deterministic adjudication resolves proven formatting, language-scoped spelling/abbreviation, recurring-name, and register equivalents without a paid call. The selected register policy controls proven reductions; ambiguous lexical differences remain review cases.
- Adjudication LLM spans carry up to two cue texts before and after the divergent span as structured context.
- Fixture-backed punctuation pass with validators that reject word changes and added/removed/restyled quotation delimiters, including German low-high dialogue quotes; word-identical proposals retain customer line-break positions.
- Punctuation word-freeze validation rejects digit-to-word substitutions such as `2` -> `two`; number normalization remains limited to alignment.
- LLM adjudication retries invalid structured output once before falling back to `keep_srt` with a QC flag.
- `adjudicate.json` persists adjudication decisions and adjudication-stage QC flags, so `--resume rebuild` preserves low-confidence or invalid-response warnings instead of silently dropping them.
- Validated batch adjudication, speaker mappings, and punctuation outputs use `workdir/<episode>/llm-cache`. Independent adjudication decisions also use `llm-case-cache`; when a batch cache misses, unaffected cases can be reused after another divergence changes. Keys include prompt/policy versions, language/register and source-name policy, local source context, acoustic ownership, audio provenance, resolved models, and non-secret settings. Transient audio/provider/review failures are not cached as lasting verdicts.
- QC JSON/HTML report, `changes.diff.srt`, and a verify-stage `verify.json` artifact.
- `changes.diff.srt` records applied wording changes from the classified change log, with delivered numbering and old/new text; timing-only adjustments stay in the QC report.
- Per-cue verification scores and CPS are written to `qc_report.json` and rendered in `qc_report.html`; scores use forced-alignment confidence when present, otherwise real ASR word confidence, and remain unscored when neither exists.
- QC HTML flag rows include cue ids, timestamps, confidence, and old/new review text for changed or flagged cues.
- The verify stage writes `verify.json` with the finalized summary, cue scores, QC flags, and style issues for resumable/debuggable stage inspection.
- ASR cache keyed by audio SHA-256, model, and non-secret params; credential fields are stripped before cache metadata is written.
- Uncached cloud ASR calls add audio-duration cost items to the cost meter; ElevenLabs keyterm/character-name prompting includes the plan's `$0.05/hr` surcharge; AssemblyAI uses the plan's Universal-3 Pro / Universal-2 rates plus the default speaker-label surcharge unless `speaker_labels: false`; cache hits remain free.
- Live LLM adapters retain provider usage metadata and add token cost items for Gemini defaults or configured model prices.
- LLM provider/model config can be overridden per pass for adjudication, punctuation, and speaker mapping, and cost items use the resolved pass model.
- The adjudication confidence gate defaults to `0.7` and can be changed with `llm.adjudication.confidence_gate`. Hybrid mode requires a finite value greater than zero and at most one. Native v12 replies use explicit hearing evidence and deterministic validation; legacy stored decisions still use their confidence gate.
- Adjudication sends LLM cases in scene batches split by `llm.adjudication.scene_gap_seconds` instead of one episode-wide batch.
- Punctuation sends cue batches split by `llm.punctuation.scene_gap_seconds`, with the same word-freeze validator applied after each proposed change.
- Valid punctuation proposals preserve cue line breaks instead of flattening two-line subtitles into one line.
- Punctuation prompts include speaker cluster and mapped character labels as context while keeping those labels out of the output SRT.
- Streaming 16-bit WAV energy and speech-activity checks share the timing evidence, with adaptive defaults and configurable fixed thresholds.
- VAD-backed unmatched cues with insufficient speech coverage are QC-flagged as dropped-line candidates.
- Unmatched and policy-removed source cues carry their original timestamp windows in QC for review.
- Source-error detector for adjacent duplicated/scrambled cue fragments, including the named dirty block in `Examples/srt test.srt`, with affected timestamp windows in QC.
- Speaker ID propagation from matched ASR words into rebuilt cues, enabling overlap policy decisions.
- Overlap resolution uses word ownership rather than moving spoken boundaries to enforce ordering. Known different-speaker overlap can remain stacked and visibly reviewed.
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
- Adjudicated wording is placed by word timing: words spoken more than `timing.max_intra_cue_gap` away from their cue become a cue of their own (`adlib_inserted`) instead of being shown early, a replacement covering several cues gives each spoken phrase to the cue at whose time it is heard, and a phrase the ASR provider decoded twice is shown once.
- Optional `vad.provider: silero` uses local Silero VAD when available and falls back to the deterministic energy VAD if the model/runtime cannot be loaded.
- Verify emits `impossible_cps_fast` and `impossible_cps_slow` QC flags using `timing.max_cps` and `timing.min_cps`.
- `report --synced --golden` first aligns predicted and golden cues monotonically by text, then computes the PLAN §11 timing/review metrics: cue counts, start MAE, within-1/3-frame ratios, source-aware improv precision/recall, review burden, and target booleans. When `ingest.json` is present, source-vs-golden text defines the actual changed cues; the improv target requires at least 0.9 precision and 0.85 recall.
- The timing target boolean requires all PLAN §11 timing gates: at least 90% of starts within 1 frame, at least 98% within 3 frames, and start MAE below 50 ms.
- `report` refuses a parent workdir containing multiple episode reports unless a specific episode workdir is provided, avoiding silent selection of the wrong QC report.
- `report` rejects malformed `qc_report.json` and malformed comparison SRTs with clear CLI errors instead of raw parser exceptions.
- `drop_policy: remove` drops unmatched source cues while QC-flagging the removed text; `keep_flagged` remains the default.
- CJK/Thai/Hangul/Japanese tokenization uses character-level comparison units, with Japanese width normalization, kana-voicing preservation, and natural fragment joining. Style profile/lint/reflow use visual display width for full-width text.
- Changed-text reflow hyphen-splits over-wide unspaced compounds so replacements can satisfy the two-line house style when possible.

## Offline Accuracy Benchmarks

The committed tools accept private manifest paths; corpus media, provider artifacts, human references, and generated run directories stay outside git. Paths resolve relative to a manifest's `repo_root`, or its directory when that field is omitted. No absolute owner-machine path is built into the tools. `scripts/replay_offline.py` substitutes recorded ASR/decisions and blocks network calls. `saved` mode measures replay compatibility with earlier verdicts; it does not test a new prompt's hearing quality. Its evidence reports exact, text-only, and unmatched decision reuse separately. `--no-loose` restricts reuse to exact ownership matches.

Run from the repository root in PowerShell, using fresh output directories:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& .\.venv\Scripts\python.exe scripts\replay_offline.py --manifest C:\private\corpus.json --episode clip --model mai --mode nollm --out work\bench-run
& .\.venv\Scripts\python.exe scripts\bench_corpus.py --manifest C:\private\corpus.json --providers provider.yaml --out work\bench-suite --jobs 4
& .\.venv\Scripts\python.exe scripts\golden_bench.py --manifest C:\private\golden-suite.json --json work\golden-metrics.json
& .\.venv\Scripts\python.exe scripts\timing_vs_audio.py --manifest C:\private\timing-suite.json --word-evidence raw --json work\timing-metrics.json
```

A corpus manifest has `episodes`, each with an `id`, `source_srt.path`, full `normalized_16k_candidates`, cached `asr.mai`/`asr.scribe_v2` transcripts (`id`, `coverage`, `fixture_path`), `stage_dirs`, and `offline_replay.commands` (`model`, `transcript`, `saved`). Saved replay additionally needs stage artifacts with `align.json` and `adjudicate.json`. The small portable example is constructed in [the benchmark tests](tests/test_offline_benchmark_tools.py).

A golden manifest has `golden_suite`, mapping labels to `source`, `golden`, `outputs` (label-to-SRT paths), and `timing_valid`; set `timing_valid: false` for text-only references. A timing manifest has `timing_runs`, a list of replay directories. Timing analysis streams WAV samples and accepts `--word-evidence effective` for repaired ownership alongside the default raw-provider comparison.

For another source revision, export it into a separate directory and set both `DUBSYNC_SRC` and `PYTHONPATH` to its `src` before running the same tools and comparator. Keep manifests, audio, saved decisions, and metric settings matched. The episode 11/17 human files mostly retain old app timestamps: compare word accuracy and the `human_edit_subset`, then use energy-based onset/offset measurements for acoustic timing. Source-timestamp equality alone does not establish a timing failure.

## Readiness Report

The CLI and commercial web MVP are implemented, with fixture-backed tests for both customer workflows. The web surface includes per-job generation styles, source-derived sync styling, gated job creation, polling, refresh recovery, protected downloads, legal and payment policies, retention cleanup, and commit-aware Render health checks. The [October 1 report](docs/testing/accuracy-upgrade-2026-10-01.md) separates upgrade checkpoints from final acceptance work. Historical results below do not establish the deployed state or quality of later changes; verify the exact Render commit for each release.

Still unverified or intentionally outside this release: real WhisperX/pyannote/MMS model execution in this workspace, production Silero model quality, language-specific morphological tokenizers, customer accounts, automatic payment collection, and a browser cue editor.

### Measured Timings And Costs

Historical verification snapshot, recorded through August 6, 2026; the production generate smoke was July 11, 2026. These counts are retained as historical evidence, not current test totals:

| Command | Result | Runtime / cost evidence |
|---|---|---|
| `python -m pytest --cov=dubsync --cov-report=term-missing` | `494 passed, 7 deselected`, coverage `85.88%` | 41.30s on 2026-08-06; paid/live smoke tests deselected |
| `npm run test:coverage` | `62 passed`; statements `91.05%`, lines `94.15%` | React workflow, sequential batch behavior, recovery, generation style controls, access gate, API client, session, provider disclosure, legal, error, and media lifecycle tests |
| `npm run test:e2e` | `11 passed` | Two-device shared-code isolation, generate, SRT-derived style, sync, sequential batch naming, token protection, refresh recovery, legal routes, decoded waveform pixels, responsive layout, select-icon inset, and feature-grid alignment |
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

1. Live-provider drift: GPT-5.6 Luna text-only structured calls and one ElevenLabs plus Gemini production generate path have historical live evidence. Gemini 3.7 Flash high adjudication and medium punctuation completed a local CLI replay on `testing 4` on August 14, 2026. That replay does not validate the September 10 Gemini 3.8 Flash medium route or its new audio context. Every provider/model rollout requires verification on its deployed commit.
2. Real-episode quality: synthetic fixtures prove timing, improv replacement, overlap, dropped-line, and source-error paths, but the PLAN targets need a golden episode set to measure cue-start MAE, improv precision/recall, and review burden on actual delivered material.
3. Language quality beyond automated checks: Japanese text handling and fixture-backed workflows are covered, but real provider accuracy for Japanese/Thai/Chinese/Korean and code-switching still needs representative audio, per-language house-style samples, and listening review.

## Known Gaps

- The approved ElevenLabs plus Gemini smoke covered commit `5c79356`; the punctuation reflow correction that followed is verified offline and was not given a second paid run.
- WhisperX local transcription is wired through the documented Python API: `load_model`, `load_audio`, `transcribe`, `load_align_model`, `align`, and optional diarization.
- Live pyannote execution was not smoke-tested; it requires `dubsync[diarize-local]`, accepted model terms, and a Hugging Face token or local model path.
- MMS forced alignment is implemented behind `dubsync[precision]`, but real model execution was not smoke-tested in this workspace.
- Deterministic energy VAD is wired; optional Silero VAD is available with energy fallback, but production quality should be validated on a golden set.
- CJK/Thai/Hangul/Japanese text uses character-level comparison and visual-width line checks. Japanese additionally supports kana-safe normalization, grouped ASR word matching, punctuation-aware generation, and best-effort line breaking; language-specific morphological tokenizers remain a possible future upgrade.
- Live LLM speaker-to-character inference is implemented through the configured LLM adapter, but was not smoke-tested against real provider responses in this workspace.
- Live Gemini punctuation completed in the historical production web smoke. Live GPT-5.6 Luna punctuation (`medium`) and text-only adjudication (`high`) structured-output calls completed on 2026-08-06 before adjudication was restored to Gemini for audio understanding; Anthropic usage metering remains covered only by deterministic response-shape tests.
- The August 14, 2026 paid local replay on `testing 4` used Gemini 3.7 Flash high adjudication and medium punctuation and returned zero QC errors. One cached Scribe transcription produced two independent energy-audit warnings on an unchanged cue; replaying those decisions against the clean cached transcription produced zero boundary findings. This is historical compatibility evidence, not validation of the new Gemini 3.8 Flash full-audio route or a deployed golden-quality pass.
- GPT-5.6 Luna token pricing has a built-in `$1/M` input and `$6/M` output default; Anthropic token prices still require explicit config overrides.
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
