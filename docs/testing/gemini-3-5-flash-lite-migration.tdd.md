# Gemini 3.5 Flash-Lite Migration TDD Evidence

## Source and user journey

The production migration follows Google's stable [Gemini 3.5 Flash-Lite model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) and [latest-model migration guide](https://ai.google.dev/gemini-api/docs/latest-model). The operator journey is: all active Flash-Lite passes use `gemini-3.5-flash-lite`, existing thinking levels remain unchanged, and usage cost reports use the current published prices.

## RED evidence

| Guarantee | Command | RED result |
|---|---|---|
| Production, example, README, adapter fixtures, and cost accounting require the 3.5 model | `python -m pytest tests/test_llm_pass_config.py tests/test_documentation_acceptance.py tests/test_cache_and_cost.py tests/test_llm_usage_cost.py -q` | Six expected failures identified the remaining 3.1 configuration and missing 3.5 pricing. |
| Custom legacy configurations keep cost accounting during migration | `python -m pytest tests/test_cache_and_cost.py::test_llm_token_prices_use_plan_defaults_and_config_overrides -q` | One expected failure showed the legacy price lookup returned `None`. |

## GREEN evidence

| Guarantee | Test or command | Result |
|---|---|---|
| Production, example, README, adapter fixtures, and cost accounting use 3.5 | Focused migration suite | 38 passed |
| No retired model identifier remains as a literal in source, tests, configuration, or documentation | Repository acceptance test plus `rg` scan | PASS; zero matches |
| Existing thinking behavior is unchanged | Configuration acceptance tests | Adjudication `high`, punctuation `medium`, speaker mapping unspecified |
| Current token prices are applied | Cost unit and pipeline integration tests | 3.5 Flash-Lite uses `$0.30` input and `$2.50` output per million tokens |
| Legacy custom configuration cost accounting remains available | Focused cost regression test | PASS at `$0.25` input and `$1.50` output per million tokens |
| Full offline Python suite and coverage threshold | `python -m pytest --cov=dubsync --cov-report=term-missing` | 314 passed, 5 deselected; 85.08% coverage |
| Installed dependency consistency | `python -m pip check` | No broken requirements |
| Production dependency set | Isolated `pip-audit` of a fresh `.[cloud,web]` environment | No known vulnerabilities |

## Implementation

- `provider.yaml`, `providers.example.yaml`, and the README example now use `gemini-3.5-flash-lite` for punctuation and speaker mapping.
- Explicit thinking levels were not changed. The adapter continues to forward `thinking_config.thinking_level` through `generate_content`.
- The migration path does not send deprecated `temperature`, `top_p`, `top_k`, `candidate_count`, or `thinking_budget` fields.
- Built-in 3.5 Flash-Lite pricing is `$0.30/$2.50` per million input/output tokens.
- A versioned compatibility branch retains cost reporting for custom configurations that have not migrated yet without leaving the retired model identifier as a literal active reference.

## Review iteration

Two independent reviews initially found that removing legacy price lookup would silently omit cost events for custom configurations. RED coverage was added, legacy cost compatibility was restored, and both follow-up reviews confirmed the issue resolved with no remaining P1/P2 findings. The security review found no committed secrets or security regressions.

## Known gaps

No paid Gemini request was made. Provider behavior is covered through official API contract review, configuration tests, adapter tests, cost integration tests, and the full offline regression suite. Existing unrelated SQLite `ResourceWarning` messages remain in parts of the web test suite.

No Render deployment was triggered. The commits deliberately include `[skip render]` because this handoff is push-only and deployment remains manual.

## Merge evidence

- RED checkpoints: `ca8d5d1`, `6681834`
- GREEN checkpoints: `ea1c1c2`, `d948767`
