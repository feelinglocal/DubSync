from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .cache import CacheKey, JsonDiskCache
from .cost import CostMeter, audio_seconds
from .models import Word


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

    def transcribe(self, audio_path: Path) -> list[Word]:
        key = CacheKey.from_audio(audio_path, self.model, self.params)
        cached = self.cache.read(key)
        if cached is not None:
            words = cached.get("words", cached) if isinstance(cached, dict) else cached
            return [Word.model_validate(item) for item in words]
        words = self.inner.transcribe(audio_path)
        if self.cost_meter is not None and self.dollars_per_hour is not None and self.dollars_per_hour > 0:
            self.cost_meter.add_audio(self.cost_provider, audio_seconds(audio_path), self.dollars_per_hour)
        self.cache.write(key, {"words": [word.model_dump() for word in words]})
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


def adapter_from_config(config: dict[str, object]) -> ASRAdapter:
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
    else:
        asr_config["language_code"] = normalized
    next_config["asr"] = asr_config
    return next_config


def _field(item: object, name: str, default: object = None) -> object:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


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
