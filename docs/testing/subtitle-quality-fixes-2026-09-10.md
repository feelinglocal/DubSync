# Subtitle quality fixes — 10 September 2026

This local audit uses the original `test fix/11.srt` and 49:25.469 Portuguese dub `11.mp3`, the supplied `11-dubsync.srt`, and the separately human-edited SRT. The five supplied files are preserved with SHA-256 checks. The original output's provider and complete processing state were not available, so its errors are not attributed to MAI or a particular Gemini model.

## Changes

- Scribe v2 is the default for new Sync and Generate jobs. MAI remains selectable, and a stored job's explicit provider remains unchanged.
- Adjudication uses `gemini-3.8-flash` with medium thinking. Literal audible improvisation, pronouns, and reactions are retained when confirmed. Source spelling and missing/uncertain audio remain guarded; the model never supplies cue timestamps.
- Indexed replacements spanning several source cues retain every word outside the edited span. Replacement text and acoustic word ownership use the same partition; contractions remain intact. Invalid token spans are held for review.
- Accepted insertions are placed by their acoustic position. Speaker turns are split using unique local lexical mappings and word-level diarization, including short edge reactions beside complete phrases. Ambiguous or unstable speaker mappings remain flagged.
- Display padding cannot consume a subsequent actor's speech or extend a cue through a broad VAD region beyond its own reliable words. Original source-order inversions remain visible in QC before playback sorting.
- Collapsed or sparsely matched word timestamps preserve the original cue timing with approved wording and an explicit review flag. The check runs before speaker children are created, survives later verification, and preserves held cue starts as barriers to neighboring padding. Accepted whole-turn deletions remove their empty dialogue markers and opening separator residue.
- Timing refinement holds a conflicting proposal instead of producing zero/negative duration. SRT writing and finalization reject invalid intervals before publication, including protected source cues. Diagnostic changes files label untimed findings separately.

The reproduced `Eu` defect involved two stages: accepted speech at 1490.190–1490.550 was first moved after `Colho, sim.`, then its endpoint was shortened behind its new start. Both stages are now covered by regressions and the final export invariant.

## Audio context and latency controls

Short inputs up to 180 seconds retain lossless normalized WAV. Longer audio is prepared as mono 24 kHz MP3 at 64 kbps, uploaded once through the Gemini Files API, and shared with the complete source transcript through a job-owned native context cache. Each request also receives bounded lossless case clips, absolute episode offsets, and exact ASR word/speaker evidence.

For this episode, preparation reduced **71,568,387 to 23,724,333 bytes** (66.85%); the preparation benchmark took **5.968 seconds**, and the decoded duration differed by **5 ms**. The originals are untouched. The source transcript's lossless column/row serialization reduced **158,440 to 48,964 UTF-8 bytes** (69.10%) with exact roundtrip checks for IDs, order, times, lines, and identities. Byte savings are not equivalent to model-token or billed-cost savings.

Adjudication uses at most eight cases per batch and four concurrent batches. Timed-out multi-case requests receive one bounded recovery pass as individual cases. Authentication/rate errors and already-singleton timeouts do not expand into more attempts. Pending work is bounded, cache/file leases protect in-flight requests, and all batches share the existing snippet-storage cap. Owned files and caches are cleaned up; user-supplied external caches are never deleted.

Cache provenance includes the model/prompt/policy, source and normalized-audio hashes, context configuration, complete source context, and the exact local word text, timing, and speaker evidence. Changing acoustic evidence cannot reuse an incompatible adjudication result.

## Validation

The complete offline backend suite passed **1,318 tests**, with seven opt-in live tests deselected and **88.60% branch-inclusive coverage**, above the 80% gate. Frontend validation passed 91 tests, typecheck/build, and two focused browser tests for the provider-selection changes.

Independent review found and resolved three additional regressions: missing word-level cache provenance, floating-point acceptance of a duration-only timing clamp, and a suppressed source-order inversion warning. The full-drop regression contained an invalid hand-authored 11-index span for ten source tokens; the valid fixture was corrected, and a separate malformed-span hold regression was added.

The bounded real Gemini check used full episode audio plus 11 cases from the sentence-fragment, snow-exchange, `Eu`, and reaction examples. It completed in **56.344 seconds** and retained all clearly audible target fragments. Subsequent mechanical replay separates the snow speakers and `Hã?` / `Uau, que lindo!` / `Ah,`; it keeps `Eu` before `Colho, sim.` and preserves both complete improvised question lines.

Seven original decoded audio clips, totaling 83 seconds, also received factual audio review independent of the supplied subtitle text. Those observations support the speech sequence and actor separation; their model-estimated times are not frame-accurate timing labels. Exact file hashes, submitted coverage, raw usage, and provider-file cleanup records are retained with the audit artifacts.

## Complete episode run

The final live adjudication and rebuild processed the complete 49:25.469 episode in **632.437 seconds (10 min 32 sec)** with the previously measured Scribe result cached. The original Scribe transcription took 32.391 seconds plus 1.749 seconds for normalization; those measurements are separate, so the live run is not presented as an uncached end-to-end benchmark.

Gemini 3.8 Flash with medium thinking produced 367 decisions. Three timed-out batches were recovered as individual cases: 79 total generation attempts, 76 completed responses, and no remaining provider-unavailable decisions. Every attempt used the shared full-audio native cache. Preparation, upload, and cache creation took 6.328, 7.563, and 5.734 seconds in that run. Lossless source-context compaction reduced the actual cache size from 163,099 to 122,481 tokens; its audio component stayed at 94,858 tokens. Cleanup completed with no in-flight requests or warnings.

The saved conservative estimate for this final adjudication run is **$7.632663**, including reservations for the failed attempts. It excludes earlier diagnostic runs and the separately cached Scribe call. Inconsistent Gemini cached-token counters prevent treating this as a confirmed charge or claiming cache billing savings. Final mechanical fixes replay these same approved decisions with paid-provider calls explicitly disabled and the cost ledger unchanged.

The detailed local artifacts are under `work/subtitle-quality-20260910/`: `final-run-summary.json` records live timing and metering; `final-rebuild-summary.json` records the exact delivered SRT hash; `final-case-review.md` and `final-comparison.md` compare source, supplied output, human revision, and new cues. The final QC report preserves unresolved acoustic and editorial targets.

The delivered review copy is `test fix/11-dubsync-scribe-v2-gemini38-review-2026-09-10.srt`, with matching `.qc.json` and `.qc.html` companions. Its SHA-256 is `e0f210d51f810c4aff0f0a0419b8f7e7dddc36fc9dbfe77b3c7fb051f0d5441e`. The last mechanical rebuild took 7.890 seconds and retained every lexical token from the preceding reviewed version while correcting the final actor boundary. Earlier generated review copies are preserved under the audit work directory.

| Whole-file check | Supplied DubSync output | New review output |
|---|---:|---:|
| Cues | 955 | 975 |
| Zero or negative durations | 1 | 0 |
| Cues shorter than 100 ms | 0 | 0 |
| Empty dialogue turns | 2 | 0 |
| Cues above 20 nonspace characters/second | 158 | 147 |
| Adjacent overlaps | 15 | 20 |

The new file retains the ending through **49:22.780**. Added overlaps include preserved source timing and simultaneous speech, so a lower overlap count is not used as the acceptance criterion. Four collapsed/sparse timing holds retain approved dialogue for review. The complete `E o Yuanzhu vai pagar.` phrase is now one cue with its measured words; it is no longer split into isolated `E.` and `o.` fragments.

The final actor-boundary regression separates `devia aproveitar` (1900.333–1901.266) from `Hum.` (1902.533–1903.033). Their exact diarized word mapping had missed the speaker-locality limit by 10 ms. The locality check now permits one configured output frame, capped at 1/24 second; this is a bounded policy tolerance, not a claim of detected source-frame precision. Exact lexical ownership, speaker stability, timestamp reliability, and source-protection checks remain in force. Nineteen regressions cover the real case, both timing boundaries, several frame rates, and the low-frame-rate cap.

## Acceptance limits

The human reference is editorial comparison evidence, not perfect acoustic truth. Its duplicated 67 ms cue 334 is excluded only in the derived comparison; the original remains unchanged. Actual simultaneous speech, missing-audio source holds, and excessive reading speed remain reviewable rather than receiving guessed timing. No complete human audition or claim of perfect transcript/phoneme accuracy is made.

SequenceMatcher agrees with 4,152 of the derived human reference's 4,251 tokens in the new output, versus 4,202 in the supplied output. These agreement counts do not establish a whole-file accuracy improvement. The concrete acoustic cases and regression tests establish the narrower fixes above. Editorial punctuation, protected/uncertain actor mappings, and small discrepancies between Scribe and independently estimated speech onsets remain review targets. An authored standalone `Dr.` cue also remains ambiguous to the general sentence-boundary heuristic; the tested fix handles a copied abbreviation inside a longer source cue without introducing a language dictionary.

Gemini returned inconsistent cached-token metadata during live checks (cached input exceeded reported total input). Metering therefore records a conservative full-input estimate and retains the raw counters; it does not invent a cache discount or claim an invoice amount.

This is local implementation and validation. It is not a production deployment.
