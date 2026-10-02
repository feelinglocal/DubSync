"""The public default follows performed register; script remains explicit."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dubsync import llm_providers, pipeline
from dubsync.adjudication import AdjudicationEngine, confidence_gated_decision
from dubsync.adjudication_policy import DeterministicAdjudicationPolicy
from dubsync.hybrid_adjudication import HybridAdjudicationAdapter
from dubsync.models import AdjudicationDecision, AudioSnippet, DivergenceSpan


def _span(source="Eu estou para sair.", asr="Eu tô pra sair!"):
    return DivergenceSpan(case_id="spoken-default", cue_ids=[1], srt_text=source,
                          asr_text=asr, start=1.0, end=2.0)


# The performed register words keep the script's punctuation (review F8).
_SPOKEN_TEXT = "Eu tô pra sair."


class NoCallAdapter:
    def adjudicate(self, spans):
        pytest.fail("A proven register equivalent must not call a provider")


@pytest.mark.parametrize("language,source,asr,expected", [
    ("pt", "Eu estou para sair.", "Eu tô pra sair!", _SPOKEN_TEXT),
    ("de", "Ich habe es gesagt.", "Ich hab es gesagt.", "Ich hab es gesagt."),
    ("en", "I am going to leave.", "I am gonna leave!", "I am gonna leave."),
])
def test_default_engine_returns_exact_performed_register(language, source, asr, expected):
    decisions, flags = AdjudicationEngine(NoCallAdapter(), language=language).adjudicate([_span(source, asr)])
    assert decisions[0].verdict == "use_audio"
    assert decisions[0].final_text == expected
    assert decisions[0].confidence == 1.0
    assert flags == []


def test_explicit_script_policy_keeps_authored_form_and_has_distinct_cache_identity():
    default = DeterministicAdjudicationPolicy(language="pt")
    script = DeterministicAdjudicationPolicy(language="pt", register_policy="script")
    assert default.decide(_span()).final_text == _SPOKEN_TEXT
    assert script.decide(_span()).final_text == _span().srt_text
    assert script.decide(_span()).verdict == "keep_srt"
    assert default.cache_context()["register_policy"] == "spoken"
    assert default.cache_context() != script.cache_context()


@pytest.mark.parametrize("filename", ["provider.yaml", "providers.example.yaml"])
def test_shipped_config_selects_spoken_register(filename):
    config = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
    assert config["adjudication"]["register_policy"] == "spoken"
    assert pipeline._adjudication_register_policy(config) == "spoken"


def test_omitted_pipeline_register_defaults_to_spoken_and_explicit_script_remains():
    assert pipeline._adjudication_register_policy({}) == "spoken"
    assert pipeline._adjudication_register_policy({"adjudication": {}}) == "spoken"
    assert pipeline._adjudication_register_policy({"adjudication": {"register_policy": "script"}}) == "script"


@pytest.mark.parametrize("adapter_class", [
    llm_providers.GeminiLLMAdapter, llm_providers.OpenAILLMAdapter, llm_providers.AnthropicLLMAdapter,
])
def test_native_adapter_context_and_prompt_default_to_spoken(adapter_class):
    adapter = adapter_class(api_key="synthetic")
    assert json.loads(adapter._adjudication_payload([_span()]))["register_policy"] == "spoken"
    adapter.set_adjudication_context(language="pt")
    assert json.loads(adapter._adjudication_payload([_span()]))["register_policy"] == "spoken"
    adapter.set_adjudication_context(language="pt", register_policy="script")
    assert json.loads(adapter._adjudication_payload([_span()]))["register_policy"] == "script"


def test_hybrid_context_defaults_to_spoken_and_forwards_explicit_script():
    contexts = []
    primary = SimpleNamespace(set_adjudication_context=lambda **kwargs: contexts.append(kwargs))
    adapter = HybridAdjudicationAdapter(primary, lambda **kwargs: ([], []))
    assert adapter.register_policy == "spoken"
    adapter.set_adjudication_context(language="pt")
    assert contexts[-1] == {"language": "pt", "register_policy": "spoken"}
    assert adapter._wording_policy.decide(_span()).final_text == _SPOKEN_TEXT
    adapter.set_adjudication_context(language="pt", register_policy="script")
    assert contexts[-1] == {"language": "pt", "register_policy": "script"}
    assert adapter._wording_policy.decide(_span()).final_text == _span().srt_text


def test_standalone_primary_and_review_prompts_default_to_spoken():
    span = _span()
    assert json.loads(llm_providers._adjudication_prompt([span], language="pt"))["register_policy"] == "spoken"
    review = llm_providers._adjudication_review_prompt(
        spans=[span], audio_snippets={span.case_id: AudioSnippet(case_id=span.case_id, path="synthetic.wav", start=0, end=3)},
        reasons={span.case_id: ["review"]}, primary_decisions={}, batch_spans=[span],
        episode_context=[], episode_words=[], confidence_gate=.7, language="pt",
    )
    assert json.loads(review)["register_policy"] == "spoken"


def test_native_normalizer_does_not_replace_spoken_register_with_script_by_default():
    span = _span()
    data = dict(case_id=span.case_id, verdict="use_audio", final_text=span.srt_text,
                heard_text=span.asr_text, evidence="heard_clearly", reason="Clear performed words")
    normalize = llm_providers._normalized_native_adjudication_decisions
    assert normalize([data], language="pt") == [{"case_id": span.case_id}]
    scripted = normalize([data], language="pt", register_policy="script")
    assert scripted[0]["final_text"] == span.srt_text
    assert scripted[0]["confidence"] == 1.0


def test_native_keep_of_script_register_still_requires_review_under_spoken_default():
    span = _span()
    decision = AdjudicationDecision(case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
                                   heard_text=span.asr_text, evidence="heard_clearly", confidence=1,
                                   reason="Keep authored form despite a different performed register")
    selected, flag = confidence_gated_decision(
        span, decision, 0, policy=DeterministicAdjudicationPolicy(language="pt"),
    )
    assert selected.final_text == span.srt_text
    assert flag is not None and flag.new_text == span.asr_text


def test_pipeline_cache_identity_distinguishes_new_default_from_explicit_script():
    span = _span()
    default_key = pipeline._adjudication_case_keys([span], {}, [], [], None)[span.case_id]
    spoken_key = pipeline._adjudication_case_keys(
        [span], {"adjudication": {"register_policy": "spoken"}}, [], [], None,
    )[span.case_id]
    script_key = pipeline._adjudication_case_keys(
        [span], {"adjudication": {"register_policy": "script"}}, [], [], None,
    )[span.case_id]
    assert default_key.digest == spoken_key.digest
    assert default_key.digest != script_key.digest


def test_no_llm_keeps_script_variant_when_public_default_is_spoken(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:03,000\nEu estou para sair agora.\n", encoding="utf-8")
    transcript = tmp_path / "words.json"
    transcript.write_text(json.dumps({"words": [
        dict(text=word, start=1 + index * .3, end=1.2 + index * .3)
        for index, word in enumerate("Eu tô pra sair agora.".split())
    ]}), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(transcript)}, "adjudication": {"register_policy": "spoken"},
    }), encoding="utf-8")
    output = tmp_path / "out.srt"
    pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=config, no_llm=True, language="pt")
    assert "Eu estou para sair agora." in output.read_text(encoding="utf-8")
