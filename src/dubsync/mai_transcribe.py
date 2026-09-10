"""Bounded, word-timed MAI transcription through OpenRouter's audio API."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import QCFlag, Word


MAI_TRANSCRIBE_MODEL = "microsoft/mai-transcribe-2"
_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
_MAX_CHUNK_SECONDS = 300.0
_CONTEXT_SECONDS = 1.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_END_ROUNDING_SECONDS = 0.020


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer credential to a redirected endpoint.
        return None


def urlopen(request: Request, *, timeout: float):
    return build_opener(_NoRedirects()).open(request, timeout=timeout)


@dataclass(frozen=True)
class _Chunk:
    data: bytes
    offset: float
    duration: float
    owner_start: float
    owner_end: float


class MAITranscribeAdapter:
    """Transcribe normalized 16 kHz mono PCM WAV without loading the full file.

    Five-minute chunks include one second of context on both sides. Matching
    words in that overlap are joined once using original provider timestamps;
    otherwise a word's midpoint selects its owning chunk. Diarization labels
    are local to each API call, so namespacing prevents unrelated speakers from
    being merged between independent chunks.
    """

    def __init__(
        self,
        api_key: str | None = None,
        diarize: bool = True,
        language_code: str | None = None,
        keyterms: list[str] | None = None,
        timeout_seconds: float = 90.0,
        chunk_seconds: float = _MAX_CHUNK_SECONDS,
    ):
        for name, value in (("timeout_seconds", timeout_seconds), ("chunk_seconds", chunk_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")
        if chunk_seconds > _MAX_CHUNK_SECONDS:
            raise ValueError("chunk_seconds cannot exceed 300 seconds.")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = MAI_TRANSCRIBE_MODEL
        self.diarize = diarize
        self.language_code = language_code
        self.keyterms = list(keyterms or [])
        self.timeout_seconds = float(timeout_seconds)
        self.chunk_seconds = float(chunk_seconds)
        self.last_usage: dict[str, object] = self._empty_usage()
        self.last_repair_flags: list[QCFlag] = []

    @staticmethod
    def _empty_usage() -> dict[str, object]:
        return {
            "seconds": 0.0, "cost": 0.0, "generation_ids": [], "request_count": 0,
            "reported_seconds": 0.0, "reported_cost": 0.0,
        }

    def transcribe(self, audio_path: Path) -> list[Word]:
        from .providers import ProviderError

        self.last_usage = self._empty_usage()
        self.last_repair_flags = []
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is required for MAI-Transcribe 2.", code="configuration")
        words: list[Word] = []
        for index, chunk in enumerate(self._chunks(audio_path)):
            payload = self._request(chunk.data)
            chunk_words = self._words(payload, chunk, index)
            words = _join_words(words, chunk_words, chunk.owner_start) if index else chunk_words
        return sorted(words, key=lambda word: (word.start, word.end))

    def _chunks(self, audio_path: Path):
        from .providers import ProviderError

        try:
            with wave.open(str(audio_path), "rb") as source:
                rate = source.getframerate()
                if (rate, source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
                    raise ProviderError("MAI-Transcribe 2 requires normalized 16 kHz, 16-bit mono PCM WAV audio.")
                total = source.getnframes()
                if not total:
                    raise ProviderError("MAI-Transcribe 2 cannot transcribe empty audio.")
                step = max(1, int(self.chunk_seconds * rate))
                context = min(int(_CONTEXT_SECONDS * rate), step // 2)
                for owner_start in range(0, total, step):
                    owner_end = min(total, owner_start + step)
                    clip_start = max(0, owner_start - context)
                    clip_end = min(total, owner_end + context)
                    source.setpos(clip_start)
                    frames = source.readframes(clip_end - clip_start)
                    if len(frames) != (clip_end - clip_start) * 2:
                        raise ProviderError("MAI-Transcribe 2 received a truncated normalized WAV file.")
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as output:
                        output.setnchannels(1)
                        output.setsampwidth(2)
                        output.setframerate(rate)
                        output.writeframes(frames)
                    yield _Chunk(
                        data=buffer.getvalue(), offset=clip_start / rate,
                        duration=(clip_end - clip_start) / rate,
                        owner_start=owner_start / rate, owner_end=owner_end / rate,
                    )
        except (OSError, wave.Error, EOFError):
            raise ProviderError("MAI-Transcribe 2 could not read normalized WAV audio.") from None

    def _request(self, audio: bytes) -> dict:
        from .providers import ProviderError

        body = {
            "model": self.model,
            "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": "wav"},
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
            "provider": {"options": {"azure": {
                "diarization": {"enabled": self.diarize},
                "enhancedMode": {"modelOptions": {"transcribeStyle": "verbatim"}},
            }}},
        }
        if self.language_code:
            body["language"] = self.language_code
        if self.keyterms:
            body["provider"]["options"]["azure"]["phraseList"] = {"phrases": self.keyterms}
        request = Request(
            _ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(2):
            self.last_usage["request_count"] += 1
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    generation_id = response.headers.get("X-Generation-Id")
                    if generation_id:
                        self.last_usage["generation_ids"].append(str(generation_id))
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After", "1") if exc.headers else "1"
                exc.close()
                if status == 429 and attempt == 0:
                    try:
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        delay = 1.0
                    time.sleep(min(5.0, max(0.0, delay)) if math.isfinite(delay) else 1.0)
                    continue
                if status == 402:
                    message = "OpenRouter credits are insufficient for MAI-Transcribe 2. Add credits to the OpenRouter account."
                    code = "credits"
                elif status in (401, 403):
                    message = "OpenRouter rejected MAI-Transcribe 2 authentication. Check OPENROUTER_API_KEY and model access."
                    code = "authentication"
                elif status == 429:
                    message = "OpenRouter rate limited MAI-Transcribe 2. Try again later."
                    code = "rate_limit"
                else:
                    self._record_usage({})
                    message = f"MAI-Transcribe 2 request failed with HTTP {status}."
                    code = None
                raise ProviderError(message, code=code) from None
            except (URLError, OSError, ValueError):
                # Do not automatically repeat a request whose billing is unknown.
                self._record_usage({})
                raise ProviderError("MAI-Transcribe 2 could not complete the OpenRouter request within the connection timeout.") from None
            if len(raw) > _MAX_RESPONSE_BYTES:
                self._record_usage({})
                raise ProviderError("MAI-Transcribe 2 returned an oversized response.")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError):
                self._record_usage({})
                raise ProviderError("MAI-Transcribe 2 returned invalid JSON.") from None
            if not isinstance(payload, dict):
                self._record_usage({})
                raise ProviderError("MAI-Transcribe 2 returned an invalid transcription response.")
            # Record a paid response even if word validation subsequently rejects it.
            self._record_usage(payload)
            return payload
        raise AssertionError("Unreachable retry state")  # pragma: no cover

    def _record_usage(self, payload: dict) -> None:
        usage = payload.get("usage")
        for field in ("seconds", "cost"):
            value = usage.get(field) if isinstance(usage, dict) else None
            if not _nonnegative_number(value):
                self.last_usage[field] = None
            else:
                self.last_usage[f"reported_{field}"] += float(value)
                if self.last_usage[field] is not None:
                    self.last_usage[field] += float(value)

    def _words(self, payload: dict, chunk: _Chunk, index: int) -> list[Word]:
        from .providers import ProviderError

        raw_words = payload.get("words")
        if not isinstance(raw_words, list) or (not raw_words and str(payload.get("text", "")).strip()):
            raise ProviderError("MAI-Transcribe 2 did not return required word timestamps.")
        words = []
        for item in raw_words:
            if not isinstance(item, dict):
                raise ProviderError("MAI-Transcribe 2 returned an invalid word timing record.")
            text = item.get("word")
            start, end = item.get("start"), item.get("end")
            if (
                not isinstance(text, str) or not text.strip()
                or not _nonnegative_number(start) or not _nonnegative_number(end)
                or end <= start or start >= chunk.duration
                or end - chunk.duration > _MAX_END_ROUNDING_SECONDS + 1e-9
            ):
                raise ProviderError("MAI-Transcribe 2 returned missing or invalid word timing.")
            if end > chunk.duration:
                # Live MAI output can round the final endpoint 10 ms past the
                # supplied WAV. Bound that repair to 20 ms and retain its source
                # values for review; starts and larger errors remain untouched.
                self.last_repair_flags.append(QCFlag(
                    kind="asr_timestamp_rounding_clamped", severity="info",
                    message=(
                        f"MAI-Transcribe 2 chunk {index + 1} endpoint rounding for {text!r}: "
                        f"original local start={start!r}s, end={end!r}s, offset={chunk.offset!r}s; "
                        f"clamped local end={chunk.duration!r}s to the supplied audio boundary "
                        "(maximum allowed repair: 20 ms)."
                    ),
                    start=float(start) + chunk.offset, end=chunk.duration + chunk.offset,
                ))
                end = chunk.duration
            start, end = float(start) + chunk.offset, float(end) + chunk.offset
            speaker = item.get("speaker")
            speaker_id = None
            if self.diarize and speaker is not None and str(speaker).strip():
                speaker_id = f"chunk_{index + 1}:{speaker}"
            words.append(Word(text=text, start=start, end=end, confidence=None, speaker_id=speaker_id))
        return sorted(words, key=lambda word: (word.start, word.end))


def _join_words(left: list[Word], right: list[Word], boundary: float) -> list[Word]:
    """Match overlapping occurrences before ownership so timing jitter cannot split a pair.

    The monotonic one-to-one match keeps repeated words distinct. Requiring
    overlapping intervals avoids merging repetitions at clearly different times.
    A matched pair always contributes one real provider record, even when its
    two midpoints disagree about which chunk owns it.
    """
    from .providers import ProviderError

    left_overlap = [i for i, word in enumerate(left) if word.end > boundary - _CONTEXT_SECONDS and word.start < boundary + _CONTEXT_SECONDS]
    right_overlap = [i for i, word in enumerate(right) if word.end > boundary - _CONTEXT_SECONDS and word.start < boundary + _CONTEXT_SECONDS]
    if max(len(left_overlap), len(right_overlap)) > 200:
        raise ProviderError("MAI-Transcribe 2 returned too many overlapping word timestamps.")
    left_tokens = [_normalized_word(left[i]) for i in left_overlap]
    right_tokens = [_normalized_word(right[i]) for i in right_overlap]
    scores = [[(0, 0.0) for _ in range(len(right_overlap) + 1)] for _ in range(len(left_overlap) + 1)]
    matches = {}
    for row, left_index in enumerate(left_overlap, start=1):
        for col, right_index in enumerate(right_overlap, start=1):
            lword, rword = left[left_index], right[right_index]
            overlap = min(lword.end, rword.end) - max(lword.start, rword.start)
            score = max(scores[row - 1][col], scores[row][col - 1])
            if left_tokens[row - 1] == right_tokens[col - 1] and overlap > 0:
                similarity = overlap / min(lword.end - lword.start, rword.end - rword.start)
                diagonal = scores[row - 1][col - 1]
                candidate = (diagonal[0] + 1, diagonal[1] + similarity)
                if candidate >= score:
                    score = candidate
                    matches[(row, col)] = True
            scores[row][col] = score
    pairs = {}
    row, col = len(left_overlap), len(right_overlap)
    while row and col:
        if matches.get((row, col)):
            pairs[left_overlap[row - 1]] = right_overlap[col - 1]
            row -= 1
            col -= 1
        elif scores[row - 1][col] >= scores[row][col - 1]:
            row -= 1
        else:
            col -= 1
    paired_right = set(pairs.values())
    joined = []
    for index, word in enumerate(left):
        if index in pairs:
            joined.append(word if _midpoint(word) < boundary else right[pairs[index]])
        elif _midpoint(word) < boundary:
            joined.append(word)
    joined.extend(word for index, word in enumerate(right) if index not in paired_right and _midpoint(word) >= boundary)
    return sorted(joined, key=lambda word: (word.start, word.end))


def _normalized_word(word: Word) -> str:
    return "".join(character for character in word.text.casefold() if character.isalnum()) or word.text.casefold()


def _midpoint(word: Word) -> float:
    return (word.start + word.end) / 2.0


def _nonnegative_number(value) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )
