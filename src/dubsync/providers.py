from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from pydantic import ValidationError

from .cache import CacheKey, JsonDiskCache
from .cost import CostMeter, audio_seconds
from .models import QCFlag, Word


logger = logging.getLogger(__name__)
GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS = 30 * 60.0


class ProviderError(RuntimeError):
    pass


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

    def transcribe(self, audio_path: Path) -> list[Word]:
        self.last_repair_flags = []
        key = CacheKey.from_audio(audio_path, self.model, self.params)
        cached = self.cache.read(key)
        if cached is not None:
            cached_words = cached.get("words", cached) if isinstance(cached, dict) else cached
            words, cache_repair_flags = repair_word_stream(cached_words, source="ASR cache")
            persisted_flags = _cached_repair_flags(cached)
            self.last_repair_flags = [*persisted_flags, *cache_repair_flags]
            if _is_raw_provider_cache(cached):
                self.cache.write(key, _validated_word_cache_payload(words, self.last_repair_flags))
            return words

        provider_words = self.inner.transcribe(audio_path)
        if (
            self.cost_meter is not None
            and self.dollars_per_hour is not None
            and self.dollars_per_hour > 0
        ):
            self.cost_meter.add_audio(self.cost_provider, audio_seconds(audio_path), self.dollars_per_hour)
        cacheable_words = _cacheable_word_items(provider_words)
        if cacheable_words is None:
            words, repair_flags = repair_word_stream(provider_words, source="ASR provider")
            self.last_repair_flags = repair_flags
            self.cache.write(key, _validated_word_cache_payload(words, repair_flags))
            return words
        self.cache.write(
            key,
            {
                "words": cacheable_words,
                "metadata": {"raw_provider_response": True},
            },
        )
        words, repair_flags = repair_word_stream(cacheable_words, source="ASR provider")
        self.last_repair_flags = repair_flags
        self.cache.write(key, _validated_word_cache_payload(words, repair_flags))
        return words


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
        self.language_code = language_code

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("ELEVENLABS_API_KEY is required for ElevenLabs Scribe.")
        try:
            from elevenlabs import ElevenLabs
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use ElevenLabs Scribe.") from exc

        client = ElevenLabs(api_key=self.api_key)
        convert_kwargs = {
            "model_id": self.model_id,
            "timestamps_granularity": "word",
            "diarize": self.diarize,
        }
        if self.keyterms:
            convert_kwargs["keyterms"] = self.keyterms
        if self.language_code:
            convert_kwargs["language_code"] = self.language_code
        with audio_path.open("rb") as audio_file:
            response = client.speech_to_text.convert(
                file=audio_file,
                **convert_kwargs,
            )
        raw_words = _field(response, "words", [])
        normalized = []
        for item in raw_words:
            item_type = _field(item, "type", "word")
            text = _field(item, "text", _field(item, "word", ""))
            if item_type != "word" or not text:
                continue
            normalized.append(
                Word(
                    text=str(text),
                    start=float(_field(item, "start", 0.0)),
                    end=float(_field(item, "end", 0.0)),
                    confidence=float(_field(item, "confidence", 1.0)),
                    speaker_id=_field(item, "speaker_id", None),
                )
            )
        return normalized


class OpenAIWhisperAdapter:  # pragma: no cover - live provider path
    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for OpenAI Whisper.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use OpenAI Whisper.") from exc

        client = OpenAI(api_key=self.api_key)
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["word"],
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
    def __init__(self, api_key: str | None = None, model: str = "universal-3-pro", speaker_labels: bool = True):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.model = model
        self.speaker_labels = speaker_labels

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("ASSEMBLYAI_API_KEY is required for AssemblyAI.")
        try:
            import assemblyai as aai
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use AssemblyAI.") from exc

        aai.settings.api_key = self.api_key
        config = aai.TranscriptionConfig(
            speech_models=[self.model],
            language_detection=True,
            speaker_labels=self.speaker_labels,
        )
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
        error_status = _field(_field(aai, "TranscriptStatus", None), "error", "error")
        transcript_status = _field(transcript, "status", None)
        if transcript_status == error_status or str(_field(transcript_status, "value", transcript_status)).lower() == "error":
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


class GeminiTranscribeAdapter:  # pragma: no cover - live provider path
    """Optional Gemini 3.5 Transcribe ASR adapter for local comparison runs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-transcribe",
        language_codes: list[str] | None = None,
        custom_vocabulary: list[str] | None = None,
        diarize: bool = True,
        word_timestamps: bool = True,
        store: bool = False,
        max_audio_seconds: object = GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model.strip()
        self.language_codes = list(language_codes or [])
        self.custom_vocabulary = list(custom_vocabulary or [])
        self.diarize = diarize
        self.word_timestamps = word_timestamps
        self.store = bool(store)
        self.max_audio_seconds = _gemini_max_audio_seconds(max_audio_seconds)
        self.usage_events: list[object] = []
        if self.model != "gemini-3.5-transcribe":
            raise ProviderError(
                "Gemini Transcribe local tests require model: gemini-3.5-transcribe."
            )
        if not self.word_timestamps:
            raise ProviderError(
                "Gemini Transcribe requires asr.word_timestamps: true for DubSync's timing contract."
            )
        if self.store:
            raise ProviderError("Gemini Transcribe local tests require asr.store: false.")
        if len(self.custom_vocabulary) > 1000:
            raise ProviderError("Gemini Transcribe custom vocabulary supports at most 1000 terms.")

    def transcribe(self, audio_path: Path) -> list[Word]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is required for Gemini 3.5 Transcribe.")
        if self.word_timestamps or self.diarize:
            duration = _audio_seconds_for_limit(audio_path)
            if duration is None:
                raise ProviderError(
                    "Gemini 3.5 Transcribe audio duration could not be verified; "
                    "use the DubSync CLI to normalize the input to WAV first."
                )
            if duration > self.max_audio_seconds:
                raise ProviderError(
                    "Gemini 3.5 Transcribe is limited to 30 minutes per request when "
                    "word timestamps or speaker diarization are enabled; automatic "
                    "chunking is intentionally not enabled for this comparison path."
                )
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError("Install dubsync[cloud] to use Gemini 3.5 Transcribe.") from exc

        client = None
        uploaded_name = ""
        try:
            client = genai.Client(api_key=self.api_key)
            audio_file = client.files.upload(file=str(audio_path))
            uploaded_name = str(_field(audio_file, "name", "") or "")
            uploaded_uri = str(_field(audio_file, "uri", "") or "")
            if not uploaded_uri:
                raise ProviderError("Gemini file upload did not return an audio URI.")
            response = client.interactions.create(
                model=self.model,
                input=[
                    {
                        "type": "audio",
                        "uri": uploaded_uri,
                        "mime_type": _field(audio_file, "mime_type", _guess_audio_mime_type(audio_path)),
                    }
                ],
                generation_config={"transcription_config": self._transcription_config()},
                store=False,
            )
        except Exception as exc:
            raise ProviderError(
                f"Gemini 3.5 Transcribe failed: {_redact_provider_message(exc, [self.api_key])}"
            ) from None
        finally:
            if client is not None and uploaded_name:
                try:
                    client.files.delete(name=uploaded_name)
                except Exception as cleanup_exc:  # pragma: no cover - provider cleanup warning
                    logger.warning(
                        "Gemini Transcribe could not delete uploaded file %s: %s",
                        uploaded_name,
                        _redact_provider_message(cleanup_exc, [self.api_key]),
                    )

        usage = _field(response, "usage", None)
        if usage is not None:
            self.usage_events.append({"usage": usage})
        words = _words_from_gemini_transcribe_response(response)
        if not words:
            raise ProviderError(
                "Gemini 3.5 Transcribe returned no word annotations; keep verbatim word timestamps enabled."
            )
        return words

    def _transcription_config(self) -> dict[str, object]:
        config: dict[str, object] = {}
        if self.language_codes:
            config["language_codes"] = self.language_codes
        if self.custom_vocabulary:
            config["custom_vocabulary"] = self.custom_vocabulary
        mode: dict[str, object] = {"type": "verbatim"}
        if self.diarize:
            mode["diarization_mode"] = "speaker"
        if self.word_timestamps:
            mode["timestamp_granularities"] = ["word"]
        config["mode"] = mode
        return config


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
        self.language = language
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
            result = model.transcribe(audio, batch_size=self.batch_size)
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


def adapter_from_config(config: dict[str, object], *, local_mode: bool = False) -> ASRAdapter:
    asr_config = config.get("asr", {}) if isinstance(config, dict) else {}
    if not isinstance(asr_config, dict):
        raise ProviderError("providers.yaml asr section must be a mapping")
    fixture_path = asr_config.get("fixture_path")
    if fixture_path:
        return FixtureASRAdapter(Path(str(fixture_path)))
    provider = str(asr_config.get("provider", "elevenlabs")).lower()
    if provider == "elevenlabs":
        return ElevenLabsScribeAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model_id=str(asr_config.get("model_id", "scribe_v2")),
            diarize=bool(asr_config.get("diarize", True)),
            keyterms=_asr_keyterms(asr_config),
            language_code=str(asr_config["language_code"]) if asr_config.get("language_code") else None,
        )
    if provider == "openai":
        return OpenAIWhisperAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model=str(asr_config.get("model", "whisper-1")),
        )
    if provider == "assemblyai":
        return AssemblyAIAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model=str(asr_config.get("model", "universal-3-pro")),
            speaker_labels=bool(asr_config.get("speaker_labels", True)),
        )
    if _is_gemini_transcribe_provider(provider):
        if not local_mode:
            raise ProviderError("Gemini 3.5 Transcribe ASR is only available in --local test mode.")
        return GeminiTranscribeAdapter(
            api_key=asr_config.get("api_key") if isinstance(asr_config.get("api_key"), str) else None,
            model=str(asr_config.get("model", "gemini-3.5-transcribe")),
            language_codes=_asr_language_codes(asr_config),
            custom_vocabulary=_gemini_custom_vocabulary(asr_config),
            diarize=bool(asr_config.get("diarize", True)),
            word_timestamps=bool(asr_config.get("word_timestamps", True)),
            store=bool(asr_config.get("store", False)),
            max_audio_seconds=asr_config.get("max_audio_seconds", GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS),
        )
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
    normalized = (language or "").strip().lower()
    if not normalized or normalized == "auto":
        return next_config
    existing = next_config.get("asr", {})
    if not isinstance(existing, dict):
        return next_config
    asr_config = dict(existing)
    provider = str(asr_config.get("provider", "elevenlabs")).lower()
    if provider == "whisperx":
        asr_config["language"] = normalized
    elif _is_gemini_transcribe_provider(provider):
        asr_config["language_codes"] = [normalized]
    else:
        asr_config["language_code"] = normalized
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


def _asr_language_codes(asr_config: dict[str, object]) -> list[str]:
    configured = asr_config.get("language_codes")
    if isinstance(configured, list):
        codes: list[str] = []
        for item in configured:
            if not isinstance(item, str):
                raise ProviderError("asr.language_codes must be a list of strings")
            code = item.strip()
            if code and code not in codes:
                codes.append(code)
        return codes
    if configured is not None:
        raise ProviderError("asr.language_codes must be a list of strings")
    language_code = asr_config.get("language_code")
    if isinstance(language_code, str) and language_code.strip():
        return [language_code.strip()]
    return []


def _gemini_custom_vocabulary(asr_config: dict[str, object]) -> list[str]:
    configured = asr_config.get("custom_vocabulary", [])
    if configured is None:
        terms: list[str] = []
    elif isinstance(configured, list):
        terms = []
        for item in configured:
            if not isinstance(item, str):
                raise ProviderError("asr.custom_vocabulary must be a list of strings")
            term = item.strip()
            if term and term not in terms:
                terms.append(term)
    else:
        raise ProviderError("asr.custom_vocabulary must be a list of strings")
    for term in _asr_keyterms(asr_config):
        if term not in terms:
            terms.append(term)
    if len(terms) > 1000:
        raise ProviderError("Gemini Transcribe custom vocabulary supports at most 1000 terms.")
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


def _words_from_gemini_transcribe_response(response: object) -> list[Word]:
    words: list[Word] = []
    for step in _iter_items(_field(response, "steps", [])):
        for content in _iter_items(_field(step, "content", [])):
            for annotation in _iter_items(_field(content, "annotations", [])):
                payload = _gemini_annotation_payload(annotation)
                if _field(payload, "type", None) != "word_info":
                    continue
                text = _field(payload, "text", "")
                start = _seconds_offset(_field(payload, "start_offset", None))
                end = _seconds_offset(_field(payload, "end_offset", None))
                if not text or start is None or end is None:
                    continue
                words.append(
                    Word(
                        text=str(text),
                        start=start,
                        end=end,
                        confidence=_optional_float(_field(payload, "confidence", None)),
                        speaker_id=_field(payload, "speaker", None),
                    )
                )
    return words


def _gemini_annotation_payload(annotation: object) -> object:
    raw = _field(annotation, "raw", None)
    if isinstance(raw, dict):
        return raw
    return annotation


def _iter_items(items: object) -> Iterable[object]:
    if isinstance(items, Iterable) and not isinstance(items, (str, bytes, dict)):
        return items
    return []


def _seconds_offset(value: object) -> float | None:
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        try:
            seconds = float(total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", value)
    if match is None:
        return None
    seconds = float(match.group(1))
    return seconds if math.isfinite(seconds) else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _redact_provider_message(error: object, secrets: Iterable[str | None]) -> str:
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return re.sub(r"AIza[0-9A-Za-z_-]{35}", "[redacted]", message)


def _audio_seconds_for_limit(path: Path) -> float | None:
    seconds = audio_seconds(path)
    return seconds if seconds > 0 else None


def _gemini_max_audio_seconds(value: object) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError("asr.max_audio_seconds must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ProviderError("asr.max_audio_seconds must be finite and greater than zero")
    if seconds > GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS:
        raise ProviderError("asr.max_audio_seconds cannot exceed Gemini's 1800-second limit")
    return seconds


def _guess_audio_mime_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "audio/wav"
