# Flash-Lite adjudication and episode 11/17 comparison

Measurements were recorded locally. Full-context, focused-audio v4, and the approved hybrid comparisons are complete.
The v12 edit-template prompt was tested and not adopted after mixed results;
the active primary prompt remains v11. Protected-source speech recovery has
passed its focused tests and independent review. The approved clip-only hybrid
is enabled in both provider configuration files and passed final validation. Original subtitles, audio, human edits, and the September 10
Gemini 3.8 result remain unchanged.

## Configuration and evidence rules

- Primary adjudication uses `gemini-3.5-flash-lite` with `thinking_level: high`, the
  highest supported level. The enabled fallback uses `gemini-3.8-flash` at medium
  thinking for flagged cases, with selected clips and detailed local context only.
  Punctuation remains Gemini 3.7 Flash medium.
- Lite's final configuration uses focused audio snippets plus full ordered
  source text. Full episode audio/cache is disabled for Lite and remains an
  optional route for other models. Missing or incomplete required snippets
  preserve the affected source text and timing with explicit QC, including
  partial batches and a zero confidence gate. They cannot authorize a text-only
  correction. Adjudication cache policy 2 invalidates the old fallback behavior.
- The revised prompt follows a numbered audio decision workflow. It treats
  actor paraphrases as valid even when they differ completely from the script,
  distinguishes whole-cue divergences from partial replacements, preserves
  neighboring words, and retains uncertainty when the audio cannot resolve it.
- New testing uses MAI-Transcribe 2, Portuguese, and verbatim transcription.
  The full matched runs explicitly disable optional diarization after the same
  request succeeded without it and timed out with it. The application default
  is unchanged. The source SRT and audio are model inputs; human-edited and
  historical outputs are evaluation-only references.
- The saved September 10 episode 11 result used Scribe v2. Its comparison is
  historical; it does not isolate model differences from ASR differences.
- Meter native MAI `usage.cost` separately from Gemini token-based estimates,
  cache creation/storage, failed-request reservations, and validation expenses.
  A returned token estimate is not an invoice.

## Reproduced defects

An internal source omission between touching ASR words could mark an entire
otherwise well-aligned cue as missing audio. The narrow repair supplies a real
audio review window only when two exact consecutive word anchors on each side
and at least 75% exact cue coverage support it. It does not approve text changes
or invent timestamps. Source edges, sparse or unreliable evidence, screen text,
and globally unresolved alignment remain protected.

A replacement spanning the end of one source cue and the start of the next
could put its first spoken word in the previous cue. The repair uses retained
lexical anchors, actual acoustic separation, and speaker evidence to assign an
exact ASR prefix to its continuation. A longer replacement can split at one
unambiguous internal speech gap when every replacement token has matching word
evidence. Text and timing share the assignment; transfer into a protected
external cue is held as a complete edit. Ambiguous cuts, speaker conflicts,
punctuation-only word records, and incomplete evidence remain reviewable.
An unrelated uncertain span does not block an independently approved edit
inside the same cue.

Actual MAI evidence for episode 17 establishes the first confirmed case:
`embaixo` ends at 2075.479 s, while `Sem` starts at 2086.840 s and `pressa` at
2087.120 s. The baseline attaches `Sem` to `Estou aqui embaixo`; the repaired
replay places `Sem pressa` together. This replay uses explicit diagnostic
`use_audio` decisions and is not a completed final-model benchmark.

The final frozen candidate-v4 diagnostic replay also fixes `percebi`, `e que`,
the `E` prefixes before source cues 216 and 218, and the split question in w02.
Conservation evidence checks all 81 words across five repaired episode 17
groups exactly once, in acoustic order, with frame-snapped first/last-word
endpoints. These are diagnostic replays, separate from fresh model decisions.
For w02, a narrowly verified isolated `é`/`E` match joins two adjacent
divergences into one audio question. This gives the adjudicator the complete
performed phrase. Missing or stale decisions preserve the affected source.

An ordinary pause inside the accepted phrase `toda a minha força pra impedir`
also triggered an overbroad ownership hold. A failed continuation hypothesis
now falls back to a valid edit within the same cue unless positive evidence
requires a transfer. Read-only neighboring anchors cannot restore a cue that
was not actually edited.

Whole-cue edits spanning separated speech groups now require unambiguous
word ownership. Episode 11 case121 preserves `Tá bom, então vamos lá.` in its
own speech group and moves the trailing `Eu` across the gap to the following
cue, while retaining the independently approved `esperar` edit there. Both
saved model responses conserve all ten real words once and in order. This
requires the configured speech-gap threshold, a single trailing prefix word,
and exact retained anchors in the following speech sequence. Ambiguous
episode 11 case32 instead retains its original text/timing and displays the
proposed correction in QC. Full-episode inventories find only these two
whole-cue cases in episode 11 and none in episode 17.

Exact full-cue preference could steal a repeated short phrase from stronger
neighboring matches. It now preserves total exact support and other completely
matched cues. Actual short-window regressions cover both this failure and the
`A Lumi` duplicate-prefix case. Full episode 17 and 11 alignment ownership and
protection counts remain identical to the release baseline.

Partial empty edits spanning multiple cues now remove only the indexed source
tokens, retain each cue's audible residue, and do not acquire rejected ASR word
timings. Full-cue drop policy and annotation protections remain in force.
The final rebuild policy is 8 and missing-audio guard version is 6, including
the protected-source recovery described below; older assignments are invalidated.

MAI may return a successful empty transcription without a `words` field. The
adapter accepts only explicit empty text with well-formed empty-text segments;
malformed or nonempty speech responses still require real word timestamps.

## Completed full-context comparison

These runs use the same MAI words, prompt v11, alignment snapshot, output cap,
and full-audio/snippet configuration. Only adjudication model/thinking differs.
Token edits count substitutions, insertions, and deletions against spoken human
reference text; punctuation is ignored and disputed reference text remains scored.

| Episode | Supplied older result | Preserved Scribe/3.8 | Full-context Lite HIGH | Matched MAI/3.8 MEDIUM |
| --- | ---: | ---: | ---: | ---: |
| 11: reference token edits | 79 | 176 | 346 | 172 |
| 17: reference token edits | 21 | unavailable | 266 | 49 |
| 11: adjacent output overlaps | 15 | 20 | 21 | 11 |
| 17: adjacent output overlaps | 4 | unavailable | 12 | 3 |

The full-context Lite configuration fails the requested closer-to-human target.
Its model token/cache estimates were $2.03582255 and $1.40347770 for episodes
11 and 17; matched Flash estimates were $5.835144875 and $3.92950525. Episode
11 first-attempt elapsed time was 222.781 s for Lite and 495.609 s for Flash.
Episode 17 cumulative attempt time was 170.140 s and 352.531 s, including an
audited budget-reserve pause/resume; these are not clean cold-run latency figures.

There were 190 generation responses and six cache creations across the four
runs, with no provider failures. The $13.203950375 token/cache estimate is
separate from $47.699981275 conservative committed reservations. Inconsistent
Google metadata sometimes reports more cached input tokens than total input
tokens; the harness retains uncertainty rather than treating these as invoices.
The initial internal reservation caps were amended with an immutable record,
retaining every original request and avoiding duplicate generations on resume.

The focused-audio diagnostic kept exactly the same eight w02 spans, prompt
bytes, source/ASR context, schema, and snippet WAVs. Removing the full-audio
cache changed all eight Lite decisions from `keep_srt` at confidence 0.3 to the
correct literal improvisations at model-reported confidence 1.0. This was one
fresh generation, 16.078 s and an estimated $0.027237. It reported 3,190 thinking
tokens; the full-context counterpart omitted thought usage. Offline SDK wire
inspection verifies HIGH was serialized. Missing thought telemetry alone does
not prove thinking was disabled. Since cached source text moves inline too,
this tests a combined configuration change rather than isolating audio length.

The optimized full runs must be reported separately from this controlled model
comparison. Human reference files were never provider inputs.

## Completed focused-audio v4 comparison

Lite HIGH used focused clips and full inline source text. Flash MEDIUM retained
full audio/cache, clips and full source text. Both used prompt v11 and the same
MAI words and candidate-v4 code. Unchanged whole Flash requests reused their
exact native responses; changed requests received fresh decisions. This is an
optimized configuration comparison, not a controlled model-only experiment.

| Episode | Lite reference token edits | Flash reference token edits | Lite overlaps | Flash overlaps |
| --- | ---: | ---: | ---: | ---: |
| 17 | 150 | 44 | 9 | 3 |
| 11 | 253 | 170 | 15 | 11 |

Lite improves over full-context v2 but still misses the closer-to-human wording
target. Both outputs fix the tested episode 17 prefix assignments, the long gap
before `Sem pressa`, the case84 timing hold, and episode 11 case32's early name.
Fresh Lite omits episode 11's trailing `Eu`; Flash preserves its transfer.
Lite also returns whole sentences for several partial edits, repeating words
already retained by the pipeline, and makes unsupported confident source keeps.

Lite's 40 episode 17 generations cost an estimated $0.520992 and took 165.000 s;
its 55 episode 11 generations cost $0.766363 and took 234.765 s. Flash's logical
generation estimates are $3.719381 and $5.727974 respectively, counting reused
inference rather than treating it as free. Flash additionally needed a new
episode 17 cache estimated at $0.101716125. Prior cache history remains in the
original v2 ledger. Flash's 76.266 s and 12.890 s v4 runtimes include 38 and 55
response replays respectively and cannot measure fresh model speed.

V4 incurred $1.577639125 in new token/cache estimates in total: 95 fresh Lite
generations, two fresh Flash generations, and one cache creation. Conservative
committed reservations total $2.023924375, reflecting the two new Flash
responses' inconsistent cache counters. Punctuation was explicitly skipped by
the existing policy for audio longer than 1,800 seconds in every v2/v4 run.
The configured punctuation model was not charged for these episodes.

Experimental prompt v12 makes each case's edit operation explicit. Original indexed source
words are grouped into `retain` and `replace` slots; a verified omission inserts
an empty string instead of repeating retained neighboring dialogue. Unknown,
mismatched or disconnected source indices cannot produce a false template.
All original source, ASR and audio evidence remains intact. Three selected
batches covering 20 cases cost an estimated $0.051639 and took 42.984 s. They
fixed cases98/100 but introduced omissions and did not consistently fix the
other copied-context or false-keep cases. No fourth diagnostic or full v12 run
was sent. The active application therefore retains the full-tested v11 prompt;
v12 source, tests, requests and results remain saved as experimental evidence.

## Approved focused review route

The user approved a hybrid after the pure Lite comparisons: Lite 3.5 at HIGH
handles the first pass, and Gemini 3.8 reviews only flagged cases with their
focused clips and detailed local source/ASR/word ownership context. The review
adapter cannot receive an episode audio cache or whole-episode audio context.
Only selected case IDs are editable; accepted primary sibling clips are not
attached. The primary prompt remains v11. The review prompt is separately
versioned as `adjudication-review-v1-local-audio-ownership`.

The routing checks are triage, not an oracle. A primary decision is accepted
only with sufficient confidence and lexical agreement with its owned ASR
hypothesis (for a source keep, source and ASR must agree). Other decisions need
audio review. A confident rejection of a genuine improvisation therefore still
escalates. ASR and Lite can share an error, so this cannot detect every mistake.
Malformed, absent or uncertain review decisions preserve the source for review.
`hybrid_adjudication.json` records routes and triggers; `cost.json` attributes
each response to the actual configured primary or review model.

An offline fixed-rule counterfactual used the preserved native v4 replies,
without any provider calls or reference-based case selection. It routed 138/265
episode 17 and 180/369 episode 11 lexical cases to their recorded Flash answers.
Word differences fell from 150 to 56 and from 253 to 181, respectively. The old
Flash-only outputs scored 44 and 170. These are **recorded full-context answers**,
not evidence that the new clip-only review route has achieved those scores.
The final live review results are recorded separately below.

A narrowly validated alignment recovery also separates episode 11's third
ASR greeting hypothesis from the following protected lyrics. Only original case 313
changes; all 659 other cases across both episodes are unchanged. The new speech
child needs its own audio approval; old parent decisions cannot approve it.
Lyrics retain their original text and timing under removal and resume. Guard 6
and rebuild policy 8 invalidate the affected older checkpoints. Sixty-three
new tests pass and independent review cleared 209 focused checks.

## Completed hybrid comparison and delivery

Both full hybrid runs and one exact failed-request recovery per episode are
complete. The final outputs are labeled `hybrid-v8` / `recovery-1`; the original
transport-held outputs remain immutable. Human references were never model
inputs and no disputed human text was excluded to improve the scores.

| Episode | Supplied older result: word differences | Pure focused Lite HIGH | Preserved full-audio Flash MEDIUM | Final hybrid |
| --- | ---: | ---: | ---: | ---: |
| 17 | 21 | 150 | 44 | 67 |
| 11 | 79 | 253 | 170 | 185 |

These count word substitutions, insertions and deletions against the supplied
human dialogue, ignoring punctuation. They measure reference agreement, not
acoustic accuracy. The hybrid removes 83 and 68 differences relative to pure
Lite, but still has 23 and 15 more than the tested full-audio Flash setup. It
also remains farther from the human wording than the supplied older results,
which contain the reported timing and cue-boundary problems. This is a cost
and quality compromise, not achievement of Flash-level wording agreement.
Different prompts, source snapshots and audio context make this a configuration
comparison; the earlier matched v2 table is the controlled model comparison.

| Episode | Lite generation estimate | Focused Flash review estimate | Total adjudication estimate | Shared MAI billed cost | Flash audio union / episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | $0.520992 | $0.779107 | $1.300099 | $0.082139 | 557.854 s / 2938.560 s (18.98%) |
| 11 | $0.777096 | $1.025528 | $1.802624 | $0.082889 | 666.575 s / 2965.464 s (22.48%) |

The preserved v4 Flash generation estimates are $3.719381 and $5.727974, so
hybrid adjudication is about 65% and 69% cheaper on these episodes, or 67%
combined. This excludes full-audio baseline cache creation/storage and adds
neither shared MAI nor unknown failed-request charges to generation estimates.
Logical costs include reused responses exactly once; a replay is not a free
model decision. Fresh hybrid requests, including both recoveries, cost an
estimated $1.837444; the recoveries alone added $0.045906. Two original failed
requests retain $0.3187995 in unknown reservations, not confirmed charges.
Neither hybrid route created an episode audio cache. Punctuation was skipped
by the existing long-audio policy in both episodes.

Flash reviewed 138 of 265 LLM cases in episode 17: 136 resolved through Flash,
127 through Lite, and two remained held. Episode 11 had 181 of 369 cases
reviewed: 181 resolved through Flash and 188 through Lite. These case counts
are not fractions of all episode cues. Audio union counts overlapping clip
padding once; attached clip sums were 649.828 and 850.278 seconds. Source
cue neighborhoods, ASR word ownership, the untrusted Lite proposal and explicit
review reasons supplied context without sending the complete audio to Flash.

The original hybrid wall times were 224.046 and 304.219 seconds; the isolated
recovery reconstructions took 16.937 and 53.250 seconds. Almost all primary
requests reused proven identical Lite responses. Recovery reused only exact
successful hybrid responses, and the Flash baseline also contains replays.
These are not cold latency measurements and do not prove a speedup.

Both final SRTs have positive durations throughout. Episode 17 has three
adjacent overlaps and 89 cues above 20 CPS; episode 11 has 15 overlaps and 118
above 20 CPS. On exact one-to-one dialogue pairs, 578/579 starts and 579/579
ends in episode 17 are within 250 ms of the human file; episode 11 has 707/715
starts and 692/715 ends. Pair coverage varies with wording/segmentation; the
saved reports include common-pair comparisons. These are not timing scores
for every cue in the episode.

The final files have no prefix leakage in the tested focus cases. `percebi`,
`e que` and `Sem pressa` stay with their following phrases; the latter starts
at 2086.833 s while the previous line ends at 2075.533 s. Episode 11 retains
`Eu Vou esperar um pouco` at 1009.400–1010.133 s. The failed-request recoveries
changed only four episode 17 cues and three episode 11 cues inside their
authorized clip windows. Remaining concerns include name variants, retained
peach/annual-trip wording and dialogue overlaps. These results do not establish
that every improvised line or every false hold is now correct.

The third episode 11 greeting remains a concrete limit on the automatic judge.
Human and MAI text contain three `Feliz ano novo` phrases. Both fresh Lite and
fresh focused Flash explicitly rejected the third with confidence 1.0, although
the complete 2038.120–2043.039 s clip and all three ASR phrases were supplied.
The pipeline correctly honored that decision; it did not discard an approved
insertion. The output contains two greetings, and the three missing reference
words remain scored. This requires listening review. The model's confidence
does not establish truth, and the app cannot automatically detect every such
shared error without an independent reference or human review.

Separate SRT/QC/change/route copies and a comparison report are delivered in
`test fix/adjudication-comparison-2026-09-14/`. Source/human files are unchanged.
The final SRT hashes are
`e8f82844dbba951a6f5ef4fe46b528746d3e5903f7ad76e53144e1036dad1e18`
(17) and `ae449b1c40e813dcc684b6dd044c669e5836136a651f24e9c527bfb3e0ba9985`
(11). These measurements precede deployment; verify the live commit separately.

## Current validation and provider limitations

The combined hybrid/source-recovery candidate-v8 snapshot is
`0f87ea358c58af901c30e53f28e4f237d6824deed377a2bb0e097bcfb947dc34`.
It passes 1,623 backend tests with seven opt-in live tests deselected and
89.35% branch-aware coverage in 52.94 seconds. The full gate blocked outbound
network access. Independent hybrid review passed 240 focused checks and caught
two issues before the live run: native confidence values must reach strict
validation without coercing booleans/strings into approvals, and neighboring
ASR context must participate in cache invalidation. Both have failing-before,
passing-after regressions; malformed replies still retain their reported usage.

After activation in both provider YAML files, the final standard gate again
passed 1,623 tests, seven live deselections and two existing warnings, with
89.35% coverage in 49.90 seconds. An old configuration wire test initially
expected a text-only request; it now supplies the required clip and verifies
Lite HIGH plus Flash MEDIUM with no cache or full-audio upload. All 59 current
application source files still match the frozen candidate-v8 engine.

Frozen candidate-v4 passed 1,487 backend tests, with seven configured
deselections and 88.97% coverage, above the 80% gate. Two existing
environment/deprecation warnings remained. Independent review passed 148
focused tests with provider/network calls blocked and verified that the
reviewed source hashes match the immutable benchmark snapshot
`6b3dd6b186432979eab13a8d247eec651985a3b33decbf9534f2570d62c4fc25`.

The unadopted prompt-only candidate-v6 snapshot
`487017bdf72c90c30c798014c53e0bbec77f77b563678f154dd960e2a361a756`
passes 1,504 backend tests with seven deselections and 88.99% coverage.
Independent review passed 73 focused tests with provider and outbound network
access blocked. The four full-suite warnings comprise two existing environment
warnings and two expected malformed-index fixture serialization warnings.

Initial MAI attempts on the opening failed: a 301-second chunk exceeded a
90-second client timeout; the longer 240-second client timeout then received
an upstream HTTP 408, as did a 61-second chunk. A 13-second diagnostic also
timed out with diarization. That same clip succeeded in 1.422 seconds without
diarization, and the exact first 301-second request then succeeded in 6.922
seconds without it. A disjoint 60-second diarized spoken clip succeeded too,
so duration alone does not explain the observed failures.

Both full MAI transcripts completed: 3,011 words for episode 17 and 3,716 for
episode 11. Nineteen additional requests completed in 58 seconds with no failed
requests or retries; the successful first chunk was reused locally. Reported
full transcript charges, counting that seed once, are $0.082138889 and
$0.082888889 respectively. All MAI diagnostics plus full requests total
$0.170361111 reported, with $0.20 retained separately as unknown reservations
from four earlier failures. These reservations are not confirmed charges.

Episode 17 has 70 preserved source cues: 68 marked lyrics plus `Zhu.` and
`Claro,`. There are no newly held cues in the candidate. The fresh full-episode
alignment replay therefore does not demonstrate a reduction in missing-audio holds;
it demonstrates the specific word-ownership repairs. Full context resolves
the apparent cue 217 tail hold seen at a diagnostic clip boundary.

The matched full runs use snapshot
`10eb5177bef2ae8d7f054cc463ecf6a91c1c6cf2f9aa4720d30f7a525c1e97c5`
and pair `full-matched-v2`. The revised adjudication prompt is v11, punctuation
v8 is shared, and both models have the same 32,768-token output cap. The human
reference is never sent to the providers. Additional paid clip comparisons
were skipped after full MAI transcription became available.

A separate factual audio check using six clips completed with confirmed Gemini
3.8 Flash, no subtitle references, and no generation retries. It used 2,538
input tokens, 1,886 response tokens, and 823 thinking tokens in 16.484 seconds:
$0.01206225 at the published Standard rates. These are fallible phrase-level
observations, not precise word timing or ground truth. The earlier media-plugin
upload attempt failed without a transcript; one known upload was deleted and
another upload may exist. Its unresolved reservation is kept separately.

Two initial synthetic regression-test calls unexpectedly reached the configured
punctuation model before the test factories were fully mocked. They contained
only synthetic wording and cost an estimated $0.010656 in total. Their native
records are preserved separately under `review/unintended-test-calls/`; they
are excluded from the model comparison. Subsequent targeted review blocks
both the provider constructor and outbound network access.

Local artifacts are under `work/lite-adjudication-20260914/`: `corpus/` contains
hashes and comparison fixtures, `benchmark/` records exact provider attempts,
`word-ownership/` and `diagnosis/` contain reproduction evidence, and `acoustic/`
contains the independent audio observations. These private test materials are
not included in the application's source distribution.

## Verified provider references

- [Google thinking levels](https://ai.google.dev/gemini-api/docs/thinking)
- [Flash-Lite model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini usage metadata](https://ai.google.dev/api/generate-content#UsageMetadata)
- [MAI-Transcribe 2](https://openrouter.ai/microsoft/mai-transcribe-2)
