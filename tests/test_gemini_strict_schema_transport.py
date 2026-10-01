import json
import sys
import types

import pytest
from pydantic import ValidationError

from dubsync.llm_providers import AdjudicationResponseDecision, GeminiLLMAdapter
from dubsync.models import DivergenceSpan


def test_native_strict_schema_omits_unsupported_gemini_transport_keyword(monkeypatch):
    captured = []
    class Client:
        def __init__(self, **kwargs):
            self.models = types.SimpleNamespace(generate_content=self.generate)
        def generate(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(text='{"decisions": []}')
        def close(self):
            pass
    google = types.ModuleType("google")
    google.genai = types.SimpleNamespace(Client=Client)
    monkeypatch.setitem(sys.modules, "google", google)
    adapter = GeminiLLMAdapter(api_key="synthetic", max_retries=0)
    adapter.adjudicate([DivergenceSpan(case_id="one", cue_ids=[1], srt_text="old", asr_text="new")])
    schema = captured[0]["config"]["response_schema"]
    if isinstance(schema, type):
        schema = schema.model_json_schema()
    assert "additionalProperties" not in json.dumps(schema)
    decision = schema["$defs"]["AdjudicationResponseDecision"]
    assert {"heard_text", "evidence"} <= set(decision["required"])


def test_transport_compatibility_does_not_accept_model_confidence():
    with pytest.raises(ValidationError):
        AdjudicationResponseDecision.model_validate({
            "case_id": "one", "verdict": "use_audio", "final_text": "new",
            "heard_text": "new", "evidence": "heard_clearly", "reason": "heard",
            "confidence": 1,
        }, strict=True)
