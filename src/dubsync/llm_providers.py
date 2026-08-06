from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .adjudication import LLMAdapter, StaticLLMAdapter
from .models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan
from .punctuation import PunctuationAdapter, StaticPunctuationAdapter
from .providers import ProviderError


class AdjudicationBatch(BaseModel):
    decisions: list[AdjudicationDecision]


class PunctuationCue(BaseModel):
    cue_id: int
    text: str
    speaker_id: str | None = None
    character: str | None = None


class PunctuationBatch(BaseModel):
    cues: list[PunctuationCue]


class SpeakerMappingItem(BaseModel):
    speaker_id: str
    character: str


class SpeakerMappingBatch(BaseModel):
    mappings: list[SpeakerMappingItem]


_LLM_PASS_NAMES = {"adjudication", "punctuation", "speaker_mapping"}
_LLM_PASS_CONFIG_KEYS = {
    "api_key",
    "audio_snippet_double_check",
    "cached_content",
    "confidence_gate",
    "input_per_million",
    "max_retries",
    "model",
    "output_per_million",
    "provider",
    "reasoning_effort",
    "responses",
    "scene_gap_seconds",
    "thinking_level",
    "timeout_seconds",
}

_GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_OPENAI_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
_ADJUDICATION_PROMPT_VERSION = "adjudication-v3-luna-evidence-boundary"
_PUNCTUATION_PROMPT_VERSION = "punctuation-v2-preserve-editorial-structure"
_SPEAKER_MAPPING_PROMPT_VERSION = "speaker-mapping-v2-context-only"


class GeminiLLMAdapter:  # pragma: no cover - live provider path
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        confidence_gate: float = 0.7,
        thinking_level: str | None = None,
        cached_content: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.confidence_gate = confidence_gate
        self.thinking_level = thinking_level
        self.cached_content = cached_content
        self.usage_events: list[object] = []

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is required for Gemini adjudication.")
        response = _gemini_generate_json(
            api_key=self.api_key,
            model=self.model,
            prompt=_adjudication_prompt(spans, confidence_gate=self.confidence_gate),
            response_schema=AdjudicationBatch,
            thinking_level=self.thinking_level,
            cached_content=self.cached_content,
        )
        self.usage_events.append(response)
        return AdjudicationBatch.model_validate_json(_response_text(response)).model_dump()["decisions"]

    def adjudicate_with_audio(
        self,
        spans: list[DivergenceSpan],
        audio_snippets: dict[str, AudioSnippet],
    ) -> list[dict[str, object]]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is required for Gemini adjudication.")
        response = _gemini_generate_json(
            api_key=self.api_key,
            model=self.model,
            prompt=_adjudication_prompt(
                spans,
                confidence_gate=self.confidence_gate,
                audio_snippets=audio_snippets,
            ),
            response_schema=AdjudicationBatch,
            thinking_level=self.thinking_level,
            cached_content=self.cached_content,
            audio_snippets=audio_snippets,
        )
        self.usage_events.append(response)
        return AdjudicationBatch.model_validate_json(_response_text(response)).model_dump()["decisions"]

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is required for Gemini punctuation.")
        response = _gemini_generate_json(
            api_key=self.api_key,
            model=self.model,
            prompt=_punctuation_prompt(cues),
            response_schema=PunctuationBatch,
            thinking_level=self.thinking_level,
            cached_content=self.cached_content,
        )
        self.usage_events.append(response)
        batch = PunctuationBatch.model_validate_json(_response_text(response))
        return {item.cue_id: item.text for item in batch.cues}

    def map_speakers(self, cues: list[Cue]) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is required for Gemini speaker mapping.")
        response = _gemini_generate_json(
            api_key=self.api_key,
            model=self.model,
            prompt=_speaker_mapping_prompt(cues),
            response_schema=SpeakerMappingBatch,
            thinking_level=self.thinking_level,
            cached_content=self.cached_content,
        )
        self.usage_events.append(response)
        return _speaker_mapping_dict(SpeakerMappingBatch.model_validate_json(_response_text(response)))


class OpenAILLMAdapter:  # pragma: no cover - live provider path
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.6-luna",
        confidence_gate: float = 0.7,
        reasoning_effort: str = "medium",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.confidence_gate = confidence_gate
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.usage_events: list[object] = []

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        batch = self._parse(
            task="adjudication",
            instructions=(
                "Resolve only the supplied subtitle divergence spans. Return the structured result; "
                "never emit timestamps or rewrite text outside a divergent span."
            ),
            prompt=_adjudication_prompt(spans, confidence_gate=self.confidence_gate),
            response_schema=AdjudicationBatch,
        )
        return batch.model_dump()["decisions"]

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        batch = self._parse(
            task="punctuation",
            instructions=(
                "Apply punctuation and casing only. Preserve words, cue IDs, cue boundaries, "
                "line-break positions, and the source quotation-mark convention."
            ),
            prompt=_punctuation_prompt(cues),
            response_schema=PunctuationBatch,
        )
        return {item.cue_id: item.text for item in batch.cues}

    def map_speakers(self, cues: list[Cue]) -> dict[str, str]:
        batch = self._parse(
            task="speaker mapping",
            instructions=(
                "Map diarization speaker IDs to character names from the supplied dialogue context only. "
                "Return unknown when evidence is insufficient; never change subtitle text or timestamps."
            ),
            prompt=_speaker_mapping_prompt(cues),
            response_schema=SpeakerMappingBatch,
        )
        return _speaker_mapping_dict(batch)

    def _parse(self, *, task: str, instructions: str, prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        if not self.api_key:
            raise ProviderError(f"OPENAI_API_KEY is required for OpenAI {task}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use OpenAI.") from exc

        client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        response = client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=prompt,
            text_format=response_schema,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        self.usage_events.append(response)
        return _openai_parsed_response(response, response_schema)


class AnthropicLLMAdapter:  # pragma: no cover - live provider path
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5", confidence_gate: float = 0.7):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.confidence_gate = confidence_gate
        self.usage_events: list[object] = []

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic adjudication.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use Anthropic.") from exc

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": _adjudication_prompt(spans, confidence_gate=self.confidence_gate)}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "adjudication_batch",
                    "schema": AdjudicationBatch.model_json_schema(),
                }
            },
        )
        self.usage_events.append(response)
        text = response.content[0].text
        return AdjudicationBatch.model_validate_json(text).model_dump()["decisions"]

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic punctuation.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use Anthropic.") from exc

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": _punctuation_prompt(cues)}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "punctuation_batch",
                    "schema": PunctuationBatch.model_json_schema(),
                }
            },
        )
        self.usage_events.append(response)
        batch = PunctuationBatch.model_validate_json(response.content[0].text)
        return {item.cue_id: item.text for item in batch.cues}

    def map_speakers(self, cues: list[Cue]) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic speaker mapping.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use Anthropic.") from exc

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": _speaker_mapping_prompt(cues)}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "speaker_mapping_batch",
                    "schema": SpeakerMappingBatch.model_json_schema(),
                }
            },
        )
        self.usage_events.append(response)
        return _speaker_mapping_dict(SpeakerMappingBatch.model_validate_json(response.content[0].text))


def drain_usage_events(adapter: object) -> list[object]:
    events = getattr(adapter, "usage_events", None)
    if not isinstance(events, list):
        return []
    drained = list(events)
    events.clear()
    return drained


def llm_config_for_pass(config: dict[str, Any], pass_name: str | None = None) -> dict[str, Any]:
    llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
    if not isinstance(llm_config, dict):
        raise ProviderError("providers.yaml llm section must be a mapping")
    base_config = {key: value for key, value in llm_config.items() if key not in _LLM_PASS_NAMES}
    if pass_name is None:
        return base_config
    pass_config = llm_config.get(pass_name)
    if pass_config is None:
        return base_config
    if not isinstance(pass_config, dict):
        raise ProviderError(f"llm.{pass_name} must be a mapping")
    if _is_fixture_punctuation_mapping(llm_config, pass_name, pass_config):
        return base_config
    merged = {**base_config, **pass_config}
    if _pass_changes_provider(base_config, pass_config) and "api_key" not in pass_config:
        merged.pop("api_key", None)
    return merged


def llm_adapter_from_config(config: dict[str, Any], pass_name: str | None = None) -> LLMAdapter:
    llm_config = llm_config_for_pass(config, pass_name)
    provider = str(llm_config.get("provider", "gemini")).lower()
    api_key = llm_config.get("api_key") if isinstance(llm_config.get("api_key"), str) else None
    model = llm_config.get("model")
    confidence_gate = _confidence_gate_from_config(llm_config)
    if provider == "fixture":
        responses = llm_config.get("responses", {})
        if not isinstance(responses, dict):
            raise ProviderError("llm.responses must be a mapping for fixture provider")
        return StaticLLMAdapter(responses)
    if provider == "gemini":
        return GeminiLLMAdapter(
            api_key=api_key,
            model=str(model or "gemini-3.5-flash"),
            confidence_gate=confidence_gate,
            thinking_level=_gemini_thinking_level_from_config(llm_config, pass_name),
            cached_content=_gemini_cached_content_from_config(llm_config),
        )
    if provider == "openai":
        if _audio_snippets_enabled(llm_config):
            raise ProviderError("The OpenAI LLM adapter does not support audio snippets; disable audio_snippet_double_check")
        return OpenAILLMAdapter(
            api_key=api_key,
            model=str(model or "gpt-5.6-luna"),
            confidence_gate=confidence_gate,
            reasoning_effort=_openai_reasoning_effort_from_config(llm_config),
            timeout_seconds=_positive_float_config(llm_config, "timeout_seconds", 90.0),
            max_retries=_nonnegative_int_config(llm_config, "max_retries", 2),
        )
    if provider == "anthropic":
        return AnthropicLLMAdapter(api_key=api_key, model=str(model or "claude-sonnet-5"), confidence_gate=confidence_gate)
    raise ProviderError(f"Unsupported LLM provider: {provider}")


def punctuation_adapter_from_config(config: dict[str, Any]) -> PunctuationAdapter | None:
    llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
    if not isinstance(llm_config, dict):
        raise ProviderError("providers.yaml llm section must be a mapping")
    provider = str(llm_config.get("provider", "gemini")).lower()
    punctuation = llm_config.get("punctuation")
    if provider == "fixture":
        if punctuation is None:
            return None
        if not isinstance(punctuation, dict):
            raise ProviderError("llm.punctuation must be a cue-id mapping for fixture provider")
        if _looks_like_pass_config(punctuation):
            pass_provider = str(punctuation.get("provider", provider)).lower()
            if pass_provider == "fixture":
                responses = punctuation.get("responses")
                return StaticPunctuationAdapter(responses) if isinstance(responses, dict) else None
            return llm_adapter_from_config(config, pass_name="punctuation")  # live LLM adapters also implement punctuate()
        return StaticPunctuationAdapter(punctuation)
    return llm_adapter_from_config(config, pass_name="punctuation")  # live LLM adapters also implement punctuate()


def _looks_like_pass_config(value: dict[str, object]) -> bool:
    return any(key in value for key in _LLM_PASS_CONFIG_KEYS)


def _is_fixture_punctuation_mapping(
    llm_config: dict[str, Any], pass_name: str, pass_config: dict[str, object]
) -> bool:
    return (
        pass_name == "punctuation"
        and str(llm_config.get("provider", "")).lower() == "fixture"
        and not _looks_like_pass_config(pass_config)
    )


def _pass_changes_provider(base_config: dict[str, Any], pass_config: dict[str, object]) -> bool:
    if "provider" not in pass_config:
        return False
    return str(pass_config["provider"]).lower() != str(base_config.get("provider", "gemini")).lower()


def _confidence_gate_from_config(llm_config: dict[str, Any]) -> float:
    value = llm_config.get("confidence_gate", 0.7)
    try:
        confidence_gate = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError("llm.adjudication.confidence_gate must be numeric") from exc
    if not 0 <= confidence_gate <= 1:
        raise ProviderError("llm.adjudication.confidence_gate must be between 0 and 1")
    return confidence_gate


def _gemini_thinking_level_from_config(llm_config: dict[str, Any], pass_name: str | None) -> str | None:
    value = llm_config.get("thinking_level")
    if value is None and pass_name == "punctuation":
        return "medium"
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderError("llm.thinking_level must be one of: minimal, low, medium, high")
    thinking_level = value.strip().lower()
    if thinking_level not in _GEMINI_THINKING_LEVELS:
        raise ProviderError("llm.thinking_level must be one of: minimal, low, medium, high")
    return thinking_level


def _gemini_cached_content_from_config(llm_config: dict[str, Any]) -> str | None:
    value = llm_config.get("cached_content")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProviderError("llm.cached_content must be a non-empty Gemini cached content resource name")
    return value.strip()


def _openai_reasoning_effort_from_config(llm_config: dict[str, Any]) -> str:
    value = llm_config.get("reasoning_effort", "medium")
    if not isinstance(value, str):
        raise ProviderError(
            "llm.reasoning_effort must be one of: none, low, medium, high, xhigh, max"
        )
    normalized = value.strip().lower()
    if normalized not in _OPENAI_REASONING_EFFORTS:
        raise ProviderError(
            "llm.reasoning_effort must be one of: none, low, medium, high, xhigh, max"
        )
    return normalized


def _positive_float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"llm.{key} must be a positive number") from exc
    if parsed <= 0:
        raise ProviderError(f"llm.{key} must be a positive number")
    return parsed


def _nonnegative_int_config(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ProviderError(f"llm.{key} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"llm.{key} must be a non-negative integer") from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ProviderError(f"llm.{key} must be a non-negative integer")
    return parsed


def _audio_snippets_enabled(llm_config: dict[str, Any]) -> bool:
    value = llm_config.get("audio_snippet_double_check")
    return isinstance(value, dict) and bool(value.get("enabled", False))


def _adjudication_prompt(
    spans: list[DivergenceSpan],
    confidence_gate: float = 0.7,
    audio_snippets: dict[str, AudioSnippet] | None = None,
) -> str:
    instructions = [
        "Listen to each attached audio snippet when one is provided; use it only to determine the literal words spoken in that case.",
        "Weigh the original SRT span, ASR hypothesis, and neighboring cue context. Audio or ASR word timing is acoustic evidence; matched neighboring SRT text is editorial context.",
        "final_text is the replacement for only the divergent span. Never include timestamps, neighboring cue text, explanations in final_text, or a full-cue rewrite.",
        "Do not drop matched cue words outside the divergent span. Example: for divergent 'Drachen Evolutionssystem' within 'Drachen- / Evolutionssystem besitze.', return 'Drachenevolutionssystem'; downstream text remains 'Drachenevolutionssystem besitze.'.",
        "Preserve source spellings of character and proper names unless acoustic evidence clearly supports a different spoken name; lower confidence for a near-homophone.",
        "For German compounds and rank labels, use subtitle-readable spelling supported by the evidence, for example 'SSS-Rangklasse' rather than 'SSS-Rang-Klasse'.",
        "Choose keep_srt for punctuation, casing, line-break, or likely ASR spelling-only differences; use_audio for a clear spoken-word difference; hybrid only when both sources contribute necessary words.",
        f"Return the best bounded answer for every case. If confidence is below {confidence_gate:.2f}, report that lower confidence so QC can hold it for review.",
    ]
    payload = {
        "task": "Adjudicate dubbed-dialogue text divergence spans without changing timing or cue structure.",
        "prompt_version": _ADJUDICATION_PROMPT_VERSION,
        "instructions": instructions,
        "allowed_verdicts": ["keep_srt", "use_audio", "hybrid"],
        "confidence_gate": confidence_gate,
        "spans": [span.model_dump() for span in spans],
        "audio_snippets": [
            {
                "case_id": snippet.case_id,
                "mime_type": snippet.mime_type,
                "duration_seconds": round(snippet.duration_seconds, 3),
            }
            for snippet in (audio_snippets or {}).values()
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _punctuation_prompt(cues: list[Cue]) -> str:
    payload = {
        "task": "Apply subtitle punctuation and casing while preserving the supplied editorial structure.",
        "prompt_version": _PUNCTUATION_PROMPT_VERSION,
        "instructions": [
            "Preserve every cue ID and cue boundary, and return exactly one result for every supplied cue.",
            "Do not add, remove, reorder, translate, or respell any alphanumeric word; never return timestamps.",
            "Preserve the source line-break positions. Do not flatten, add, or move line breaks.",
            "Do not add, remove, or restyle quotation marks. In particular, do not wrap ordinary German dialogue in new low-high quotes when the source has none.",
            "Use speaker and character labels only as context for sentence punctuation; never copy labels into subtitle text.",
            "Return only the structured cue results; no commentary.",
        ],
        "cues": [
            {
                "cue_id": cue.index,
                "text": cue.text,
                "speaker_id": cue.speaker_id,
                "character": cue.character,
            }
            for cue in cues
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _speaker_mapping_prompt(cues: list[Cue]) -> str:
    samples: dict[str, list[dict[str, object]]] = {}
    for cue in cues:
        if not cue.speaker_id:
            continue
        samples.setdefault(cue.speaker_id, []).append({"cue_id": cue.index, "text": cue.plain_text})
    payload = {
        "task": "Map diarization speaker clusters to character names from dialogue evidence only.",
        "prompt_version": _SPEAKER_MAPPING_PROMPT_VERSION,
        "instructions": [
            "Return one mapping per supplied speaker ID.",
            "Use only the supplied dialogue samples; do not alter subtitle text or produce timestamps.",
            "Return unknown rather than guessing when the character identity is not supported.",
        ],
        "speakers": [
            {"speaker_id": speaker_id, "samples": speaker_samples[:8]}
            for speaker_id, speaker_samples in sorted(samples.items())
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _speaker_mapping_dict(batch: SpeakerMappingBatch) -> dict[str, str]:
    return {
        item.speaker_id: item.character
        for item in batch.mappings
        if item.speaker_id and item.character and item.character.strip().lower() != "unknown"
    }


def _openai_parsed_response(response: object, response_schema: type[BaseModel]) -> BaseModel:
    status = str(_object_field(response, "status", "completed") or "completed").lower()
    if status == "incomplete":
        raise ProviderError("OpenAI response was incomplete; retry the episode or reduce the request size.")
    if status == "failed":
        raise ProviderError("OpenAI response failed before producing a structured result.")
    if _openai_response_has_refusal(response):
        raise ProviderError("OpenAI refused the structured subtitle request.")

    parsed = _object_field(response, "output_parsed", None)
    if isinstance(parsed, response_schema):
        return parsed
    if parsed is not None:
        try:
            return response_schema.model_validate(parsed)
        except (TypeError, ValueError) as exc:
            raise ProviderError("OpenAI returned an invalid structured response.") from exc

    output_text = _object_field(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ProviderError("OpenAI response did not include structured output.")
    try:
        return response_schema.model_validate_json(output_text)
    except (TypeError, ValueError) as exc:
        raise ProviderError("OpenAI returned invalid structured JSON.") from exc


def _openai_response_has_refusal(response: object) -> bool:
    output = _object_field(response, "output", [])
    if not isinstance(output, list):
        return False
    for item in output:
        content = _object_field(item, "content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if str(_object_field(part, "type", "")).lower() == "refusal":
                return True
    return False


def _object_field(source: object, name: str, default: object = None) -> object:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _gemini_generate_json(
    api_key: str,
    model: str,
    prompt: str,
    response_schema: type[BaseModel],
    thinking_level: str | None = None,
    cached_content: str | None = None,
    audio_snippets: dict[str, AudioSnippet] | None = None,
) -> object:
    try:
        from google import genai
    except ImportError as exc:
        raise ProviderError("Install dubsync[cloud] to use Gemini.") from exc

    client = genai.Client(api_key=api_key)
    config: dict[str, object] = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    }
    if thinking_level:
        config["thinking_config"] = {"thinking_level": thinking_level}
    if cached_content:
        config["cached_content"] = cached_content
    contents: object = prompt
    if audio_snippets:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use Gemini audio snippets.") from exc
        contents = [prompt]
        for snippet in audio_snippets.values():
            contents.append(
                types.Part.from_bytes(
                    data=Path(snippet.path).read_bytes(),
                    mime_type=snippet.mime_type,
                )
            )
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    raise ProviderError("Gemini response did not include text content.")
