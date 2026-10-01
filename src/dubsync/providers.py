from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from pydantic import ValidationError

from .cache import CacheKey, JsonDiskCache
from .cost import CostMeter, audio_seconds
from .models import QCFlag, Word


GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS = 30 * 60.0
GEMINI_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
MAI_TRANSCRIBE_MODEL = "microsoft/mai-transcribe-2"
SCRIBE_TRANSCRIBE_MODEL = "scribe_v2"
GEMINI_TRANSCRIBE_DISABLED_MESSAGE = (
    "Gemini 3.5 Transcribe ASR is disabled; use ElevenLabs Scribe v2."
)


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


class ASRAdapter(Protocol):
    def transcribe(self, audio_path: Path) -> list[Word]:
        raise NotImplementedError


class FixtureASRAdapter:
    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def transcribe(self, audio_path: Path) -> list[Word]:
        del audio_path
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        words = data.get("words", data)
        return [Word.model_validate(item) for item in words]


class CachedASRAdapter:
    def __init__(
        self,
        inner: ASRAdapter,
        cache: JsonDiskCache,
        model: str,
        params: dict[str, object],
        cost_meter: CostMeter | None = None,
        cost_provider: str | None = None,
        dollars_per_hour: float | None = None,
    ):
        self.inner = inner
        self.cache = cache
        self.model = model
        self.params = params
        self.cost_meter = cost_meter
        self.cost_provider = cost_provider or model
        self.dollars_per_hour = dollars_per_hour
        self.last_repair_flags: list[QCFlag] = []
        self.last_cache_key: CacheKey | None = None
        self.last_usage: dict[str, object] = {}
        self.last_evidence: dict[str, object] | None = None
        self.last_cache_hit = False

    def transcribe(self, audio_path: Path) -> list[Word]:
        self.last_repair_flags = []
        self.last_usage = {}
        self.last_evidence = None
        self.last_cache_hit = False
        params = self.params
        adapter_version = getattr(self.inner, "cache_version", None)
        if adapter_version is not None:
            # Word-stream post-processing inside an adapter changes cached
            # results, so entries written by an older adapter are not reused.
            params = {**params, "adapter_version": adapter_version}
        key = CacheKey.from_audio(audio_path, self.model, params)
        self.last_cache_key = key
        cached = self.cache.read(key)
        if cached is not None:
            self.last_cache_hit = True
            if isinstance(cached, dict):
                self.last_usage = _safe_asr_usage(cached.get("usage"))
                metadata = cached.get("metadata")
                if isinstance(metadata, dict):
                    self.last_evidence = _safe_provider_evidence(metadata.get("provider_evidence"))
            cached_words = cached.get("words", cached) if isinstance(cached, dict) else cached
            words, cache_repair_flags = repair_word_stream(cached_words, source="ASR cache")
            persisted_flags = _cached_repair_flags(cached)
            self.last_repair_flags = [*persisted_flags, *cache_repair_flags]
            if _is_raw_provider_cache(cached):
                self.cache.write(key, self._cache_payload(words, self.last_repair_flags))
            return words

        succeeded = False
        try:
            provider_words = self.inner.transcribe(audio_path)
            succeeded = True
        finally:
            self.last_usage = _safe_asr_usage(getattr(self.inner, "last_usage", None))
            self._record_cost(audio_path, succeeded=succeeded)
        provider_flags = list(getattr(self.inner, "last_repair_flags", []))
        self.last_evidence = _safe_provider_evidence(getattr(self.inner, "last_evidence", None))
        cacheable_words = _cacheable_word_items(provider_words)
        if cacheable_words is None:
            words, repair_flags = repair_word_stream(provider_words, source="ASR provider")
            self.last_repair_flags = [*provider_flags, *repair_flags]
            self.cache.write(key, self._cache_payload(words, self.last_repair_flags))
            return words
        raw_metadata: dict[str, object] = {
            "raw_provider_response": True, "repair_flags": [flag.model_dump() for flag in provider_flags],
        }
        if self.last_evidence is not None:
            raw_metadata["provider_evidence"] = self.last_evidence
        self.cache.write(key, {"words": cacheable_words, "metadata": raw_metadata, "usage": self.last_usage})
        words, repair_flags = repair_word_stream(cacheable_words, source="ASR provider")
        self.last_repair_flags = [*provider_flags, *repair_flags]
        self.cache.write(key, self._cache_payload(words, self.last_repair_flags))
        return words

    def _cache_payload(self, words: list[Word], flags: list[QCFlag]) -> dict[str, object]:
        payload = {**_validated_word_cache_payload(words, flags), "usage": self.last_usage}
        if self.last_evidence is not None:
            # Provider evidence that is not part of Word (Scribe logprob and
            # audio events, MAI chunk languages) stays with the saved result.
            payload["metadata"]["provider_evidence"] = self.last_evidence
        return payload

    def _record_cost(self, audio_path: Path, *, succeeded: bool) -> None:
        if self.cost_meter is None:
            return
        billed_cost = self.last_usage.get("cost")
        seconds = self.last_usage.get("seconds")
        uncertain_seconds = self.last_usage.get("uncertain_seconds")
        uncertain_seconds = float(uncertain_seconds) if isinstance(uncertain_seconds, (int, float)) else 0.0
        if isinstance(billed_cost, (int, float)):
            self.cost_meter.add_audio_billed(
                self.cost_provider,
                float(seconds) if isinstance(seconds, (int, float)) else audio_seconds(audio_path),
                float(billed_cost),
            )
        elif (not succeeded or uncertain_seconds > 0) and isinstance(self.last_usage.get("reported_cost"), (int, float)):
            # Known charges whose total is uncertain (a failed or retried request).
            self.cost_meter.add_audio_billed(
                self.cost_provider,
                float(self.last_usage.get("reported_seconds", 0)),
                float(self.last_usage["reported_cost"]),
                partial=True,
            )
        elif succeeded and self.dollars_per_hour is not None and self.dollars_per_hour > 0:
            self.cost_meter.add_audio(
                self.cost_provider,
                float(seconds) if isinstance(seconds, (int, float)) else audio_seconds(audio_path),
                self.dollars_per_hour,
            )
        if uncertain_seconds > 0 and self.dollars_per_hour is not None and self.dollars_per_hour > 0:
            # A retried request may have been billed twice; meter it as an
            # explicit estimate rather than failing the job or hiding it.
            self.cost_meter.add_audio_uncertain(self.cost_provider, uncertain_seconds, self.dollars_per_hour)


_MAX_PROVIDER_EVIDENCE_BYTES = 16 * 1024 * 1024


def _safe_provider_evidence(value: object) -> dict[str, object] | None:
    """Keep provider evidence only when it is a bounded, strict-JSON mapping."""
    if not isinstance(value, dict) or not value:
        return None
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > _MAX_PROVIDER_EVIDENCE_BYTES:
        return None
    return json.loads(encoded)


def _safe_asr_usage(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for name in (
        "seconds", "cost", "reported_seconds", "reported_cost", "request_count",
        "uncertain_request_count", "uncertain_seconds",
    ):
        number = value.get(name)
        if name in value and number is None:
            result[name] = None
        if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) and number >= 0:
            result[name] = number
    identifiers = value.get("generation_ids")
    if isinstance(identifiers, list):
        result["generation_ids"] = [item for item in identifiers if isinstance(item, str) and item.startswith("gen-")][:10000]
    return result


class ElevenLabsScribeAdapter:  # pragma: no cover - live provider path
    """Thin optional adapter for ElevenLabs Scribe.

    The import is delayed so the core CLI and tests run without cloud packages.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "scribe_v2",
        diarize: bool = True,
        keyterms: list[str] | None = None,
        language_code: str | None = None,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model_id = model_id
        self.diarize = diarize
        self.keyterms = list(keyterms or [])
        self.language_code = asr_language_code(language_code)
        self.last_usage: dict[str, object] = {}
        self.last_evidence: dict[str, object] = {}

    def transcribe(self, audio_path: Path) -> list[Word]:
        self.last_usage = {"request_count": 0}
        self.last_evidence = {}
        if not self.api_key:
            raise ProviderError("ELEVENLABS_API_KEY is required for ElevenLabs Scribe.", code="configuration")
        try:
            from elevenlabs import ElevenLabs
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use ElevenLabs Scribe.") from exc

        client = ElevenLabs(api_key=self.api_key)
        # temperature=0 with a fixed seed is deliberately not sent: a paid
        # probe on 2026-10-01 (3 runs each of a 70 s and a 169 s German clip)
        # still returned different words, timings and speaker counts per run.
        convert_kwargs = {
            "model_id": self.model_id,
            "timestamps_granularity": "word",
            "diarize": self.diarize,
        }
        if self.keyterms:
            convert_kwargs["keyterms"] = self.keyterms
        if self.language_code:
            convert_kwargs["language_code"] = self.language_code
        response = self._convert_with_retries(client, audio_path, convert_kwargs)
        raw_words = _field(response, "words", [])
        normalized = []
        # Scribe's only per-word certainty is logprob, and audio events mark
        # non-speech. Neither is a Word field, so both are kept as evidence;
        # Word.confidence is populated exactly as before.
        word_logprobs: list[dict[str, object]] = []
        audio_events: list[dict[str, object]] = []
        for item in raw_words:
            item_type = _field(item, "type", "word")
            text = _field(item, "text", _field(item, "word", ""))
            if item_type == "audio_event" and text:
                audio_events.append({
                    "text": str(text), "start": _finite_or_none(_field(item, "start")),
                    "end": _finite_or_none(_field(item, "end")), "speaker_id": _field(item, "speaker_id", None),
                })
            if item_type != "word" or not text:
                continue
            try:
                start = float(_field(item, "start"))
                end = float(_field(item, "end"))
            except (TypeError, ValueError, OverflowError):
                # Missing timestamps are provider failures, not zero-time
                # words. Never retry a successfully billed response here.
                raise ProviderError(
                    "ElevenLabs Scribe returned missing or invalid word timing.",
                    code="invalid_response",
                ) from None
            word = Word(
                text=str(text),
                start=start,
                end=end,
                confidence=float(_field(item, "confidence", 1.0)),
                speaker_id=_field(item, "speaker_id", None),
            )
            normalized.append(word)
            word_logprobs.append({
                "text": word.text, "start": word.start, "end": word.end,
                "logprob": _finite_or_none(_field(item, "logprob")),
            })
        language_code = _field(response, "language_code")
        self.last_evidence = {
            "provider": "elevenlabs",
            "language_code": language_code if isinstance(language_code, str) else None,
            "language_probability": _finite_or_none(_field(response, "language_probability")),
            "word_logprobs": word_logprobs,
            "audio_events": audio_events,
        }
        return normalized

    def _convert_with_retries(self, client, audio_path: Path, convert_kwargs: dict[str, object]):
        """Bounded retries for transient Scribe failures, mirroring the MAI policy.

        The SDK does not retry by default and never retries timeouts. The file is
        reopened for every attempt so a retry uploads the complete audio again.
        """
        rate_limit_retried = False
        failures = 0
        while True:
            self.last_usage["request_count"] += 1
            try:
                with audio_path.open("rb") as audio_file:
                    return client.speech_to_text.convert(file=audio_file, **convert_kwargs)
            except Exception as exc:
                failure = _scribe_failure(exc)
                if failure is None:
                    raise
            message, code, retryable, billing_unknown, retry_after = failure
            if billing_unknown:
                self.last_usage["uncertain_request_count"] = int(self.last_usage.get("uncertain_request_count", 0)) + 1
                self.last_usage["uncertain_seconds"] = (
                    float(self.last_usage.get("uncertain_seconds", 0.0)) + audio_seconds(audio_path)
                )
            if code == "rate_limit" and not rate_limit_retried:
                rate_limit_retried = True
                time.sleep(retry_after)
                continue
            failures += 1
            if not retryable or failures >= _SCRIBE_MAX_ATTEMPTS:
                raise ProviderError(message, code=code) from None
            time.sleep(_SCRIBE_RETRY_BACKOFF_SECONDS[min(failures, len(_SCRIBE_RETRY_BACKOFF_SECONDS)) - 1])


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


_SCRIBE_MAX_ATTEMPTS = 3
_SCRIBE_RETRY_BACKOFF_SECONDS = (2.0, 5.0)
_SCRIBE_MAX_RATE_LIMIT_DELAY_SECONDS = 5.0


def _scribe_failure(error: Exception) -> tuple[str, str | None, bool, bool, float] | None:
    """Classify a Scribe SDK failure as (message, code, retryable, billing_unknown, retry_after).

    Unknown exception types return None and propagate unchanged. Messages never
    include the upstream body, which may echo request details.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status in (401, 403):
            return ("ElevenLabs rejected Scribe authentication. Check ELEVENLABS_API_KEY and model access.",
                    "authentication", False, False, 0.0)
        if status == 402:
            return ("ElevenLabs credits are insufficient for Scribe. Add credits to the ElevenLabs account.",
                    "credits", False, False, 0.0)
        if status == 429:
            headers = getattr(error, "headers", None) or {}
            retry_after = headers.get("retry-after", headers.get("Retry-After", "1")) if isinstance(headers, dict) else "1"
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = 1.0
            delay = min(_SCRIBE_MAX_RATE_LIMIT_DELAY_SECONDS, max(0.0, delay)) if math.isfinite(delay) else 1.0
            return ("ElevenLabs rate limited Scribe. Try again later.", "rate_limit", False, False, delay)
        retryable = status == 408 or 500 <= status <= 599
        return (f"ElevenLabs Scribe request failed with HTTP {status}.", None, retryable, retryable, 0.0)
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the ElevenLabs SDK
        httpx = None
    if httpx is not None and isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        # The request never reached the provider, so nothing can have been billed.
        return ("ElevenLabs Scribe could not connect.", None, True, False, 0.0)
    if (httpx is not None and isinstance(error, httpx.TransportError)) or isinstance(error, (TimeoutError, ConnectionError)):
        return ("ElevenLabs Scribe did not complete the request within the connection timeout.", None, True, True, 0.0)
    return None


class OpenAIWhisperAdapter:  # pragma: no cover - live provider path
    def __init__(self, api_key: str | None = None, model: str = "whisper-1", language: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = asr_language_code(language)

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for OpenAI Whisper.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use OpenAI Whisper.") from exc

        client = OpenAI(api_key=self.api_key)
        language_kwargs = {"language": self.language} if self.language else {}
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                **language_kwargs,
            )
        raw_words = _field(response, "words", [])
        return [
            Word(
                text=str(_field(item, "word", _field(item, "text", ""))),
                start=float(_field(item, "start", 0.0)),
                end=float(_field(item, "end", 0.0)),
                confidence=float(_field(item, "confidence", 1.0)),
                speaker_id=None,
            )
            for item in raw_words
            if _field(item, "word", _field(item, "text", ""))
        ]


class AssemblyAIAdapter:  # pragma: no cover - live provider path
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "universal-3-pro",
        speaker_labels: bool = True,
        language_code: str | None = None,
    ):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.model = model
        self.speaker_labels = speaker_labels
        self.language_code = asr_language_code(language_code)

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("ASSEMBLYAI_API_KEY is required for AssemblyAI.")
        try:
            import assemblyai as aai
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use AssemblyAI.") from exc

        aai.settings.api_key = self.api_key
        speech_models = [self.model]
        # Universal-3 Pro does not cover Japanese. Preserve it as the first
        # choice while allowing Japanese (including auto-detected audio) to use
        # AssemblyAI's documented multilingual fallback.
        if self.model == "universal-3-pro" and self.language_code in {None, "ja"}:
            speech_models.append("universal-2")
        language_kwargs = {"language_code": self.language_code} if self.language_code else {}
        config = aai.TranscriptionConfig(
            speech_models=speech_models,
            language_detection=self.language_code is None,
            speaker_labels=self.speaker_labels,
            **language_kwargs,
        )
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
        error_status = _field(_field(aai, "TranscriptStatus", None), "error", "error")
        transcript_status = _field(transcript, "status", None)
        if _field(transcript, "error") or transcript_status == error_status or str(_field(transcript_status, "value", transcript_status)).lower() == "error":
            raise ProviderError("AssemblyAI transcription failed with a terminal error status.")
        raw_words = _field(transcript, "words", [])
        return [
            Word(
                text=str(_field(item, "text", "")),
                start=float(_field(item, "start", 0.0)) / 1000.0,
                end=float(_field(item, "end", 0.0)) / 1000.0,
                confidence=float(_field(item, "confidence", 1.0)),
                speaker_id=str(_field(item, "speaker", "")) or None,
            )
            for item in raw_words
            if _field(item, "text", "")
        ]


class GeminiTranscribeAdapter:
    """Retained import shim for the retired Gemini 3.5 Transcribe ASR adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_TRANSCRIBE_MODEL,
        language_codes: list[str] | None = None,
        custom_vocabulary: list[str] | None = None,
        diarize: bool = True,
        word_timestamps: bool = True,
        store: bool = False,
        max_audio_seconds: object = GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS,
    ):
        del (
            api_key,
            model,
            language_codes,
            custom_vocabulary,
            diarize,
            word_timestamps,
            store,
            max_audio_seconds,
        )
        raise ProviderError(GEMINI_TRANSCRIBE_DISABLED_MESSAGE)

    def transcribe(self, audio_path: Path) -> list[Word]:
        del audio_path
        raise ProviderError(GEMINI_TRANSCRIBE_DISABLED_MESSAGE)


class WhisperXAdapter:
    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 16,
        language: str | None = None,
        diarize: bool = False,
        hf_token: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = asr_language_code(language)
        self.diarize = diarize
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def transcribe(self, audio_path: Path) -> list[Word]:
        try:
            import whisperx
        except ImportError as exc:
            raise ProviderError("Install dubsync[local] to use WhisperX local mode.") from exc

        try:
            audio = whisperx.load_audio(str(audio_path))
            model = whisperx.load_model(self.model, self.device, compute_type=self.compute_type)
            language_kwargs = {"language": self.language} if self.language else {}
            result = model.transcribe(audio, batch_size=self.batch_size, **language_kwargs)
            language_code = self.language or result.get("language")
            if language_code:
                align_model, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
                result = whisperx.align(
                    result.get("segments", []),
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
            if self.diarize:
                result = self._assign_speakers(whisperx, audio, result)
            return _words_from_whisperx_result(result)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"WhisperX local mode failed: {exc}") from exc

    def _assign_speakers(self, whisperx, audio, result: dict[str, object]) -> dict[str, object]:
        if not self.hf_token:
            raise ProviderError("HUGGINGFACE_ACCESS_TOKEN, HUGGINGFACE_TOKEN, or HF_TOKEN is required for WhisperX diarization.")
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError as exc:
            raise ProviderError("Install dubsync[local] with diarization support to use WhisperX diarization.") from exc
        diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)
        kwargs = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        diarize_segments = diarize_model(audio, **kwargs)
        return whisperx.assign_word_speakers(diarize_segments, result)


def adapter_from_config(
    config: dict[str, object],
    *,
    local_mode: bool = False,
    allow_gemini_transcribe_web: bool = False,
) -> ASRAdapter:
    del local_mode, allow_gemini_transcribe_web
    asr_config = config.get("asr", {}) if isinstance(config, dict) else {}
    if not isinstance(asr_config, dict):
        raise ProviderError("providers.yaml asr section must be a mapping")
    fixture_path = asr_config.get("fixture_path")
    if fixture_path:
        return FixtureASRAdapter(Path(str(fixture_path)))
    # model_id is the legacy Scribe setting: retain that explicit selection
    # while making providerless shared settings use the product's MAI default.
    default_provider = "elevenlabs" if "model_id" in asr_config else "openrouter"
    provider = str(asr_config.get("provider", default_provider)).strip().lower()
    if provider in {"openrouter", MAI_TRANSCRIBE_MODEL}:
        from .mai_transcribe import MAITranscribeAdapter

        model = str(asr_config.get("model", MAI_TRANSCRIBE_MODEL))
        if model != MAI_TRANSCRIBE_MODEL:
            raise ProviderError("OpenRouter transcription model must be microsoft/mai-transcribe-2.")
        return MAITranscribeAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            diarize=bool(asr_config.get("diarize", True)),
            keyterms=_asr_keyterms(asr_config),
            language_code=asr_language_code(_configured_asr_language(asr_config)),
            timeout_seconds=float(asr_config.get("timeout_seconds", 90)),
            chunk_seconds=float(asr_config.get("chunk_seconds", 300)),
        )
    if provider == "elevenlabs":
        return ElevenLabsScribeAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model_id=str(asr_config.get("model_id", "scribe_v2")),
            diarize=bool(asr_config.get("diarize", True)),
            keyterms=_asr_keyterms(asr_config),
            language_code=asr_language_code(_configured_asr_language(asr_config)),
        )
    if provider == "openai":
        return OpenAIWhisperAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model=str(asr_config.get("model", "whisper-1")),
            language=_configured_asr_language(asr_config),
        )
    if provider == "assemblyai":
        return AssemblyAIAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model=str(asr_config.get("model", "universal-3-pro")),
            speaker_labels=bool(asr_config.get("speaker_labels", True)),
            language_code=_configured_asr_language(asr_config),
        )
    if _is_gemini_transcribe_provider(provider):
        raise ProviderError(GEMINI_TRANSCRIBE_DISABLED_MESSAGE)
    if provider == "whisperx":
        return WhisperXAdapter(
            model=str(asr_config.get("model", "large-v3")),
            device=str(asr_config.get("device", "cpu")),
            compute_type=str(asr_config.get("compute_type", "int8")),
            batch_size=int(asr_config.get("batch_size", 16)),
            language=str(asr_config["language"]) if asr_config.get("language") else None,
            diarize=bool(asr_config.get("diarize", False)),
            hf_token=asr_config.get("hf_token") if isinstance(asr_config.get("hf_token"), str) else None,
            min_speakers=int(asr_config["min_speakers"]) if asr_config.get("min_speakers") is not None else None,
            max_speakers=int(asr_config["max_speakers"]) if asr_config.get("max_speakers") is not None else None,
        )
    raise ProviderError(f"Unsupported ASR provider: {provider}")


def apply_asr_language(config: dict[str, object], language: str | None) -> dict[str, object]:
    next_config = dict(config)
    if not language or not language.strip():
        return next_config
    normalized = asr_language_code(language)
    existing = next_config.get("asr", {})
    if not isinstance(existing, dict):
        return next_config
    asr_config = dict(existing)
    provider = str(asr_config.get("provider", "elevenlabs")).lower()
    asr_config.pop("language", None)
    asr_config.pop("language_code", None)
    asr_config.pop("language_codes", None)
    if normalized is None:
        next_config["asr"] = asr_config
        return next_config
    if provider in {"whisperx", "openai"}:
        asr_config["language"] = normalized
    elif _is_gemini_transcribe_provider(provider):
        asr_config["language_codes"] = [normalized]
    else:
        asr_config["language_code"] = normalized
    next_config["asr"] = asr_config
    forced_alignment = next_config.get("forced_alignment")
    if normalized == "ja" and isinstance(forced_alignment, dict) and forced_alignment and forced_alignment.get("provider", "mms") == "mms":
        next_config["forced_alignment"] = {**forced_alignment, "language": "jpn"}
    return next_config


def normalize_language_code(language: str | None) -> str | None:
    """Normalize Japanese aliases without restricting other provider languages.

    Forced alignment relies on this keeping ISO-639-3 codes such as ``deu``;
    ASR requests use ``asr_language_code`` instead.
    """
    normalized = (language or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized.replace("_", "-").split("-", 1)[0] in {"ja", "jpn"}:
        return "ja"
    return normalized


# ISO-639-2/3 codes (bibliographic and terminology forms) of languages that have
# an ISO-639-1 code. Codes without one (``yue``, ``fil``) pass through.
_ISO_639_1_BY_639_3 = {
    "afr": "af", "ara": "ar", "arm": "hy", "aze": "az", "baq": "eu", "bel": "be", "ben": "bn", "bos": "bs",
    "bul": "bg", "bur": "my", "cat": "ca", "ces": "cs", "chi": "zh", "cym": "cy", "cze": "cs", "dan": "da",
    "deu": "de", "dut": "nl", "ell": "el", "eng": "en", "est": "et", "eus": "eu", "fas": "fa", "fin": "fi",
    "fra": "fr", "fre": "fr", "geo": "ka", "ger": "de", "gle": "ga", "glg": "gl", "gre": "el", "guj": "gu",
    "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu", "hye": "hy", "ice": "is", "ind": "id", "isl": "is",
    "ita": "it", "jpn": "ja", "kan": "kn", "kat": "ka", "kaz": "kk", "khm": "km", "kor": "ko", "lao": "lo",
    "lav": "lv", "lit": "lt", "mac": "mk", "mal": "ml", "mar": "mr", "may": "ms", "mkd": "mk", "mon": "mn",
    "msa": "ms", "mya": "my", "nep": "ne", "nld": "nl", "nor": "no", "pan": "pa", "per": "fa", "pol": "pl",
    "por": "pt", "ron": "ro", "rum": "ro", "rus": "ru", "sin": "si", "slk": "sk", "slo": "sk", "slv": "sl",
    "spa": "es", "srp": "sr", "swa": "sw", "swe": "sv", "tam": "ta", "tel": "te", "tgl": "tl", "tha": "th",
    "tur": "tr", "ukr": "uk", "urd": "ur", "uzb": "uz", "vie": "vi", "wel": "cy", "zho": "zh",
}


def asr_language_code(language: str | None) -> str | None:
    """Language hint in the form every supported ASR provider accepts.

    MAI (OpenRouter), OpenAI and WhisperX expect ISO-639-1; Scribe and
    AssemblyAI accept it too. Region and script subtags are dropped
    (``pt-BR`` -> ``pt``, ``ja-JP`` -> ``ja``) and ISO-639-3 codes are mapped
    (``deu`` -> ``de``, ``por`` -> ``pt``). ``auto`` or empty means detect.
    Codes without a known ISO-639-1 form pass through unchanged, so no
    language is gated by this table.
    """
    normalized = (language or "").strip().lower().replace("_", "-")
    if not normalized or normalized == "auto":
        return None
    primary = normalized.split("-", 1)[0]
    if primary in _ISO_639_1_BY_639_3:
        return _ISO_639_1_BY_639_3[primary]
    if len(primary) == 2 and primary.isalpha():
        return primary
    return normalized


def _configured_asr_language(asr_config: dict[str, object]) -> str | None:
    value = asr_config.get("language", asr_config.get("language_code"))
    return str(value) if value else None


def apply_transcription_provider_config(config: dict[str, object], provider: str) -> dict[str, object]:
    normalized = provider.strip().lower()
    if normalized in {"", "default"}:
        # Keep explicit provider/model choices, and resolve a providerless
        # section before cache identity and cost labels are read by callers.
        next_config = deepcopy(config)
        if not next_config.get("asr"):
            next_config["asr"] = {"provider": "openrouter", "model": MAI_TRANSCRIBE_MODEL}
        else:
            asr_config = next_config["asr"]
            if isinstance(asr_config, dict) and "provider" not in asr_config and not asr_config.get("fixture_path"):
                if "model_id" in asr_config:
                    asr_config["provider"] = "elevenlabs"
                else:
                    asr_config["provider"] = "openrouter"
                    asr_config.setdefault("model", MAI_TRANSCRIBE_MODEL)
        return next_config
    if normalized == GEMINI_TRANSCRIBE_MODEL:
        raise ProviderError(GEMINI_TRANSCRIBE_DISABLED_MESSAGE)
    if normalized not in {MAI_TRANSCRIBE_MODEL, SCRIBE_TRANSCRIBE_MODEL}:
        raise ProviderError("Invalid transcription provider.")
    next_config = deepcopy(config)
    existing = next_config.get("asr", {})
    if not isinstance(existing, dict):
        raise ProviderError("providers.yaml asr section must be a mapping")
    target = "openrouter" if normalized == MAI_TRANSCRIBE_MODEL else "elevenlabs"
    # Provider-specific credentials and prices must never cross providers.
    shared = {"diarize", "keyterms", "character_names", "language_code", "fixture_path", "local"}
    original_provider = str(existing.get("provider", "")).strip().lower()
    same_provider = original_provider == target or (target == "openrouter" and original_provider == MAI_TRANSCRIBE_MODEL)
    asr_config = existing if same_provider else {
        key: value for key, value in existing.items() if key in shared
    }
    asr_config.pop("model_id" if target == "openrouter" else "model", None)
    asr_config["provider"] = target
    asr_config["model" if target == "openrouter" else "model_id"] = normalized
    next_config["asr"] = asr_config
    return next_config


def apply_local_asr_config(config: dict[str, object], local: bool) -> dict[str, object]:
    if not local:
        return dict(config)
    next_config = dict(config)
    existing = next_config.get("asr", {})
    asr_config = dict(existing) if isinstance(existing, dict) else {}
    local_override = asr_config.get("local", {})
    if isinstance(local_override, dict) and local_override:
        preserved = {
            key: value
            for key, value in asr_config.items()
            if key not in {"fixture_path", "language_code", "local", "model", "model_id", "provider"}
        }
        preserved.update(local_override)
        asr_config = preserved
    elif not _is_gemini_transcribe_provider(str(asr_config.get("provider", "")).lower()):
        asr_config["provider"] = "whisperx"
    asr_config.pop("fixture_path", None)
    next_config["asr"] = asr_config
    return next_config


def _field(item: object, name: str, default: object = None) -> object:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


@dataclass(frozen=True)
class _RepairCounts:
    blank_dropped: int = 0
    invalid_dropped: int = 0
    timing_clamped: int = 0
    reordered: int = 0

    @property
    def total(self) -> int:
        return self.blank_dropped + self.invalid_dropped + self.timing_clamped + self.reordered


def _validated_word_stream(items: object, *, source: str) -> list[Word]:
    words, _flags = repair_word_stream(items, source=source)
    return words

def _cached_repair_flags(cached: object) -> list[QCFlag]:
    if not isinstance(cached, dict):
        return []
    metadata = cached.get("metadata", {})
    raw_flags = metadata.get("repair_flags", []) if isinstance(metadata, dict) else []
    if not isinstance(raw_flags, list):
        raw_flags = []
    if not raw_flags:
        raw_flags = cached.get("repair_flags", [])
    if not isinstance(raw_flags, list):
        return []
    flags: list[QCFlag] = []
    for item in raw_flags:
        try:
            flags.append(QCFlag.model_validate(item))
        except (TypeError, ValueError, ValidationError):
            continue
    return flags


def repair_word_stream(items: object, *, source: str) -> tuple[list[Word], list[QCFlag]]:
    if isinstance(items, (str, bytes, dict, Word)) or not isinstance(items, Iterable):
        raise ProviderError(f"{source} returned an invalid word stream.")

    repaired: list[tuple[int, Word]] = []
    blank_dropped = 0
    invalid_dropped = 0
    timing_clamped = 0
    total_items = 0
    for index, item in enumerate(items):
        total_items += 1
        try:
            word = Word.model_validate(item)
        except (TypeError, ValueError, ValidationError):
            invalid_dropped += 1
            continue

        if not word.text.strip():
            blank_dropped += 1
            continue
        if not math.isfinite(word.start) or not math.isfinite(word.end):
            invalid_dropped += 1
            continue

        start = max(0.0, float(word.start))
        end = float(word.end)
        if end <= start:
            end = start + 0.001
            timing_clamped += 1
        elif start != word.start:
            timing_clamped += 1
        next_word = word.model_copy(update={"text": word.text.strip(), "start": start, "end": end})
        repaired.append((index, next_word))

    if not repaired:
        raise ProviderError(f"{source} returned no usable words after validation.")

    malformed_dropped = blank_dropped + invalid_dropped
    malformed_limit = max(1, math.ceil(total_items * 0.05))
    if malformed_dropped > malformed_limit:
        raise ProviderError(
            f"{source} returned a malformed fraction too large to repair "
            f"({malformed_dropped}/{total_items} words; maximum {malformed_limit})."
        )

    sorted_repaired = sorted(repaired, key=lambda item: (item[1].start, item[0]))
    original_order = [original_index for original_index, _word in repaired]
    sorted_order = [original_index for original_index, _word in sorted_repaired]
    reordered = sum(1 for before, after in zip(original_order, sorted_order, strict=True) if before != after)
    words = [word for _index, word in sorted_repaired]
    counts = _RepairCounts(
        blank_dropped=blank_dropped,
        invalid_dropped=invalid_dropped,
        timing_clamped=timing_clamped,
        reordered=reordered,
    )
    flags = _word_stream_repair_flags(source, counts, len(words))
    return words, flags


def _repair_word_stream(items: object, *, source: str) -> tuple[list[Word], list[QCFlag]]:
    """Compatibility alias for callers outside the package that used the old private name."""

    return repair_word_stream(items, source=source)


def _cacheable_word_items(items: object) -> list[dict[str, object] | None] | None:
    if isinstance(items, (str, bytes, dict, Word)) or not isinstance(items, Iterable):
        return None
    cached: list[dict[str, object] | None] = []
    for item in items:
        try:
            cached.append(Word.model_validate(item).model_dump())
        except (TypeError, ValueError, ValidationError):
            cached.append(None)
    return cached


def _is_raw_provider_cache(cached: object) -> bool:
    if not isinstance(cached, dict):
        return False
    metadata = cached.get("metadata")
    return isinstance(metadata, dict) and metadata.get("raw_provider_response") is True


def _validated_word_cache_payload(words: list[Word], flags: list[QCFlag]) -> dict[str, object]:
    return {
        "words": [word.model_dump() for word in words],
        "metadata": {
            "repair_flags": [flag.model_dump() for flag in flags],
        },
    }


def _word_stream_repair_flags(source: str, counts: _RepairCounts, usable_words: int) -> list[QCFlag]:
    if counts.total == 0:
        return []
    parts = [
        f"{counts.blank_dropped} blank dropped",
        f"{counts.invalid_dropped} invalid dropped",
        f"{counts.timing_clamped} timing clamped",
        f"{counts.reordered} reordered",
    ]
    return [
        QCFlag(
            kind="word_stream_repaired",
            cue_ids=[],
            message=f"{source} word stream was repaired before alignment: {', '.join(parts)}; {usable_words} usable words remain.",
            severity="warning",
            confidence=None,
        )
    ]


def _asr_keyterms(asr_config: dict[str, object]) -> list[str]:
    terms: list[str] = []
    for key in ("keyterms", "character_names"):
        value = asr_config.get(key, [])
        if value is None:
            continue
        if not isinstance(value, list):
            raise ProviderError(f"asr.{key} must be a list of strings")
        for item in value:
            if not isinstance(item, str):
                raise ProviderError(f"asr.{key} must be a list of strings")
            term = item.strip()
            if term and term not in terms:
                terms.append(term)
    return terms


def _is_gemini_transcribe_provider(provider: str) -> bool:
    return provider.lower().replace("-", "_") in {
        "gemini",
        "gemini_transcribe",
        "gemini_3.5_transcribe",
        "gemini_3_5_transcribe",
    }


def _words_from_whisperx_result(result: dict[str, object]) -> list[Word]:
    raw_words = result.get("word_segments")
    if raw_words is None:
        raw_words = []
        for segment in result.get("segments", []):
            raw_words.extend(_field(segment, "words", []) or [])

    words: list[Word] = []
    for item in raw_words:
        text = _field(item, "word", _field(item, "text", ""))
        start = _field(item, "start", None)
        end = _field(item, "end", None)
        if not text or start is None or end is None:
            continue
        words.append(
            Word(
                text=str(text).strip(),
                start=float(start),
                end=float(end),
                confidence=float(_field(item, "score", _field(item, "confidence", 1.0))),
                speaker_id=_field(item, "speaker", _field(item, "speaker_id", None)),
            )
        )
    return words
