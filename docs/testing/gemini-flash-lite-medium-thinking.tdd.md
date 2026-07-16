# Gemini Flash-Lite Medium Thinking TDD Evidence

## Source and user journey

No plan file was supplied. The journey was derived from the request: as a DubSync operator, I want the Gemini 3.1 Flash-Lite punctuation pass to use medium thinking so punctuation receives more reasoning depth than the previous low setting.

## RED evidence

| Guarantee | Command | RED result |
|---|---|---|
| Production, example, and fallback punctuation configuration require `medium` | `python -m pytest tests/test_llm_pass_config.py tests/test_documentation_acceptance.py -q` | Three expected failures reported `low != medium`. |
| README documents the upgraded default | `python -m pytest tests/test_documentation_acceptance.py::test_readme_documents_gemini_thinking_level_controls -q` | Expected phrase `punctuation defaults to medium` was absent. |
| The installed and declared Google GenAI SDK support `ThinkingLevel.MEDIUM` | `python -m pytest tests/test_gemini_adapter.py::test_google_genai_sdk_supports_medium_thinking_level tests/test_documentation_acceptance.py::test_cloud_dependencies_require_medium_thinking_compatible_google_genai -q` | The installed 1.52.0 enum lacked `MEDIUM`, and `pyproject.toml` allowed incompatible versions. |

## GREEN evidence

| Guarantee | Test or command | Result |
|---|---|---|
| The punctuation fallback resolves to medium while explicit overrides remain supported | `tests/test_llm_pass_config.py` | PASS |
| Production and example configs use Flash-Lite with medium thinking | `tests/test_documentation_acceptance.py` | PASS |
| The adapter still forwards configured thinking levels to Gemini | `tests/test_gemini_adapter.py` | PASS |
| The installed SDK exposes `ThinkingLevel.MEDIUM` | Google GenAI 2.12.1 enum inspection and focused compatibility tests | PASS |
| Dev-only installs do not require the optional cloud SDK at collection time | Cloud-enabled focused test plus simulated missing-SDK run | PASS / SKIP as intended |
| Full offline Python suite and coverage threshold | `python -m pytest --cov=dubsync --cov-report=term-missing` | 312 passed, 5 deselected; 85.07% coverage |
| Production dependency set | Isolated `pip-audit` of a fresh `.[cloud,web]` environment using the Dockerfile's pip floor | No known vulnerabilities |

## Implementation

- `provider.yaml`, `providers.example.yaml`, and the README example now set punctuation `thinking_level: medium`.
- The punctuation fallback in `llm_providers.py` now returns `medium` when no explicit level is supplied.
- The cloud dependency floor is `google-genai>=1.56,<3`; wheel inspection confirmed 1.55 lacks `MEDIUM` and 1.56 adds it.
- Explicit per-pass values such as low, minimal, or high continue to override the fallback.

## Review iteration

Independent review found that the initial compatibility test imported the optional cloud SDK unconditionally. The test now uses `pytest.importorskip`, so a documented dev-only install can collect and run the suite without `google-genai`, while cloud-enabled runs still enforce the medium enum. Follow-up review confirmed the issue is resolved with no remaining P1/P2 findings.

## Known gaps

No paid Gemini request was made. Provider behavior is covered through configuration, adapter, SDK compatibility, and full offline regression tests. Existing unrelated SQLite `ResourceWarning` messages remain in parts of the web test suite.

## Merge evidence

- RED checkpoints: `065b436`, `dd8b896`, `472dd9a`
- GREEN checkpoints: `f7cb341`, `7f8733b`, `8ac9d71`
