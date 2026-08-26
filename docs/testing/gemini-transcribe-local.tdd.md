# Gemini Transcribe Local Option TDD Evidence

## Source

Journeys were derived from the local-only Gemini 3.5 Transcribe test request. API behavior was checked against:

- https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe
- https://ai.google.dev/gemini-api/docs/transcribe
- https://ai.google.dev/gemini-api/docs/interactions-overview
- https://ai.google.dev/gemini-api/docs/pricing

## User Journeys

1. As an operator, I can test Gemini 3.5 Transcribe locally without changing the production ElevenLabs ASR default.
2. As an engineer, I can compare ASR timing streams because Gemini word annotations map into DubSync `Word` objects.
3. As an operator, I get clear failures for unsupported long timestamp/diarization requests or missing word annotations.

## RED Evidence

- `python -m pytest tests/test_audio_and_providers.py::test_gemini_transcribe_adapter_maps_word_info_annotations tests/test_audio_and_providers.py::test_gemini_transcribe_requires_local_mode -q` failed before the adapter existed.
- `python -m pytest tests/test_audio_and_providers.py::test_gemini_transcribe_unwraps_sdk_unknown_annotations_and_deletes_upload tests/test_audio_and_providers.py::test_gemini_transcribe_deletes_upload_when_interaction_fails tests/test_audio_and_providers.py::test_gemini_transcribe_factory_accepts_deduplicated_custom_vocabulary tests/test_audio_and_providers.py::test_gemini_transcribe_rejects_invalid_request_limit tests/test_alignment_and_recue.py::test_divergence_with_unknown_asr_confidence_remains_unscored -q` failed 4 of 5 before SDK annotation unwrapping, upload cleanup, request-limit validation, and unknown-confidence scoring were implemented.
- `python -m pytest tests/test_audio_and_providers.py::test_gemini_transcribe_rejects_unsafe_local_contract_overrides -q` failed before `word_timestamps: false` and `store: true` were rejected.
- `python -m pytest tests/test_audio_and_providers.py::test_google_genai_version_supports_transcription_interaction_fields -q` failed with `google-genai` 2.10.0, which lacked the typed Transcribe interaction fields needed for this adapter.

## GREEN Evidence

- `python -m pytest tests/test_audio_and_providers.py -k "gemini_transcribe" tests/test_alignment_and_recue.py::test_divergence_with_unknown_asr_confidence_remains_unscored tests/test_verify_scores.py::test_score_cues_marks_unknown_asr_confidence_as_unscored tests/test_cache_and_cost.py::test_default_asr_prices_match_plan_cost_table tests/test_documentation_acceptance.py::test_cloud_dependencies_require_medium_thinking_compatible_google_genai -q` passed.
- `python -m pytest tests/test_audio_and_providers.py tests/test_alignment_and_recue.py tests/test_verify_scores.py tests/test_transcription.py tests/test_cache_and_cost.py tests/test_documentation_acceptance.py -q` passed.
- `python -m pytest tests/test_audio_and_providers.py tests/test_alignment_and_recue.py tests/test_verify_scores.py tests/test_transcription.py tests/test_pipeline_cli.py tests/test_cache_and_cost.py tests/test_documentation_acceptance.py -q` passed.
- `python -m pytest --cov=dubsync --cov-report=term-missing` passed with 603 passed, 8 deselected, and 84.23% total coverage.
- `python -m pytest --live tests/test_live_smoke.py::test_live_gemini_transcribe_smoke -q` passed with `DUBSYNC_LIVE_TRANSCRIBE_AUDIO=work/gemini-transcribe-smoke-032-30s.wav`.
- `python -m dubsync sync "testing 3\original srt\032.srt" "testing 3\audio\032.wav" -o "work\gemini-transcribe-compare\032.gemini.synced.srt" --providers providers.gemini-transcribe.local.example.yaml --workdir "work\gemini-transcribe-compare" --local --no-llm` passed and wrote ignored comparison artifacts.

## Implementation Guarantees

| Behavior | Guarantee |
|---|---|
| Production default | `provider.yaml` and normal sync/generate runs remain on ElevenLabs Scribe v2 unless the operator explicitly passes `--local` with a Gemini local override. |
| Local-only enforcement | `adapter_from_config(..., local_mode=False)` rejects `gemini_transcribe`. |
| Interactions request shape | Uploaded audio is sent to `client.interactions.create` with `model`, audio `input`, `generation_config.transcription_config`, verbatim mode, word timestamps, optional speaker diarization, and `store=False`. |
| Data retention | The adapter rejects `store: true` and best-effort deletes uploaded files in a `finally` block. |
| Word timings | `word_info` annotations are converted into DubSync `Word` text/start/end/speaker fields; missing provider confidence is represented as `None` and downstream scoring marks that source as unscored. |
| Cost metering | Uncached Gemini Transcribe ASR calls meter at `$0.30/hr`, matching Google's `$0.005/min` Transcribe price basis. |

## Live Comparison Artifact

The local Gemini run for episode 032 produced:

- ASR metadata: `provider=gemini_transcribe`, `model=gemini-3.5-transcribe`
- Word annotations: 125
- Speakers: 2
- Confidence values: 125 null values, expected because Gemini Transcribe word annotations do not currently provide per-word confidence in the tested response
- Cost meter: 49.4 seconds, `$0.004117`

SRT timing summary for the same episode:

| File | Cues | Start ms | End ms | Text chars |
|---|---:|---:|---:|---:|
| Original source SRT | 30 | 0 | 47000 | 715 |
| Existing synced SRT | 30 | 100 | 47066 | 732 |
| Gemini local output | 30 | 0 | 47066 | 715 |

## Known Gaps

- This is a local comparison path, not a production provider migration.
- Automatic chunking for audio longer than 30 minutes is intentionally not enabled yet. Correct chunking needs timestamp rebasing, overlap dedupe, speaker stitching, cache keys, and cost accounting.
- A fresh Scribe-vs-Gemini live comparison was not run in this pass because `ELEVENLABS_API_KEY` was not present in the shell environment. The current comparison uses Gemini output against the repo's existing episode 032 synced artifact.
