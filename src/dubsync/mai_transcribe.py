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
from difflib import SequenceMatcher
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
# MAI word timestamps sit on a 20 ms grid; a rewind is a start well before the
# previous end, never grid rounding.
_REWIND_EPSILON_SECONDS = 0.005
_MAX_REDECODED_RUN_WORDS = 12
# Adjacent identical tokens closer than this are candidate decoder doubling.
_MAX_DOUBLED_GAP_SECONDS = 0.15
_MAX_DOUBLED_RUN_LINK_SECONDS = 1.0
_MIN_DOUBLED_RUN_PAIRS = 3
_MAX_FLAG_EXAMPLES = 20


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

    # Bump when word post-processing changes so stale ASR cache entries are not reused.
    cache_version = "mai-words-2"

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
        self._dropped_runs: list[list[Word]] = []
        self._collapsed_pairs: list[tuple[Word, Word]] = []

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
        self._dropped_runs = []
        self._collapsed_pairs = []
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is required for MAI-Transcribe 2.", code="configuration")
        words: list[Word] = []
        for index, chunk in enumerate(self._chunks(audio_path)):
            payload = self._request(chunk.data)
            chunk_words = self._words(payload, chunk, index)
            words = _join_words(words, chunk_words, chunk.owner_start) if index else chunk_words
        self.last_repair_flags.extend(self._word_run_summary_flags())
        return sorted(words, key=lambda word: (word.start, word.end))

    def _word_run_summary_flags(self) -> list[QCFlag]:
        # One aggregated flag per repair kind keeps a long episode reviewable
        # without attaching provider housekeeping to individual cues.
        flags = []
        if self._dropped_runs:
            examples = "; ".join(
                f"{run[0].start:.2f}s {' '.join(word.text for word in run)!r}"
                for run in self._dropped_runs[:_MAX_FLAG_EXAMPLES]
            )
            flags.append(QCFlag(
                kind="asr_duplicate_words_dropped", severity="info",
                message=(
                    f"MAI-Transcribe 2 re-emitted {len(self._dropped_runs)} word run(s) over already transcribed "
                    f"audio; the earlier overlapping copy was dropped and the later copy kept: {examples}."
                ),
            ))
        if self._collapsed_pairs:
            examples = "; ".join(
                f"{dropped.start:.2f}s {dropped.text!r}" for dropped, _kept in self._collapsed_pairs[:_MAX_FLAG_EXAMPLES]
            )
            flags.append(QCFlag(
                kind="asr_doubled_words_collapsed", severity="info",
                message=(
                    f"MAI-Transcribe 2 doubled {len(self._collapsed_pairs)} adjacent number token(s) in a counted "
                    f"sequence; the shorter copy of each pair was dropped: {examples}."
                ),
            ))
        return flags

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
        if "words" not in payload and _explicit_empty_transcription(payload, chunk.duration):
            # MAI can omit words for a successful no-speech response. Accept
            # explicit empty text/segments only; never invent word timings.
            raw_words = []
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
        # The provider's own word order is the only evidence of a re-decoded
        # run, so both filters must run before sorting by time.
        words, dropped_runs = _drop_redecoded_runs(words)
        self._dropped_runs.extend(dropped_runs)
        words, collapsed_pairs, kept_runs = _collapse_doubled_number_runs(words)
        self._collapsed_pairs.extend(collapsed_pairs)
        for run in kept_runs:
            self.last_repair_flags.append(QCFlag(
                kind="asr_doubled_word_run_kept", severity="info",
                message=(
                    f"MAI-Transcribe 2 wrote {len(run)} consecutive doubled words "
                    f"({' '.join(word.text for word in run)!r}); this can be decoder doubling of sung or "
                    "chanted lines, but the words were kept because a genuine repetition cannot be ruled out."
                ),
                start=run[0].start, end=run[-1].end,
            ))
        return sorted(words, key=lambda word: (word.start, word.end))


def _explicit_empty_transcription(payload: dict, duration: float) -> bool:
    text = payload.get("text")
    segments = payload.get("segments")
    if "error" in payload or not isinstance(text, str) or text.strip() or not isinstance(segments, list):
        return False
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        text = segment.get("text")
        start, end = segment.get("start"), segment.get("end")
        if (
            not isinstance(text, str) or text.strip()
            or not _nonnegative_number(start) or not _nonnegative_number(end)
            or end < start or end > duration + _MAX_END_ROUNDING_SECONDS + 1e-9
        ):
            return False
    return True


def _drop_redecoded_runs(words: list[Word]) -> tuple[list[Word], list[list[Word]]]:
    """Drop the earlier copy when MAI rewinds and re-emits words it already wrote.

    In raw provider order, a word that starts well before the previous word's
    end begins a re-decoded run. The earlier run is the maximal suffix of kept
    words that end after that start. Every observed case (ep11/ep17) re-emits
    the same speech with a corrected continuation, so the later copy is kept.
    The run is only dropped when the later words re-cover at least half of it,
    it is short, and diarization does not attribute the two runs to different
    speakers (overlapping dialogue is not a re-decode).
    """
    kept: list[int] = []
    dropped: list[list[Word]] = []
    for position, word in enumerate(words):
        if kept and word.start < words[kept[-1]].end - _REWIND_EPSILON_SECONDS:
            cut = len(kept)
            while cut and words[kept[cut - 1]].end > word.start + _REWIND_EPSILON_SECONDS:
                cut -= 1
            earlier = [words[index] for index in kept[cut:]]
            earlier_end = max(item.end for item in earlier)
            later = []
            for item in words[position:]:
                if item.start >= earlier_end:
                    break
                later.append(item)
            if _later_run_recovers(earlier, later):
                dropped.append(earlier)
                del kept[cut:]
        kept.append(position)
    return [words[index] for index in kept], dropped


def _later_run_recovers(earlier: list[Word], later: list[Word]) -> bool:
    if not earlier or not later or len(earlier) > _MAX_REDECODED_RUN_WORDS:
        return False
    earlier_start = min(word.start for word in earlier)
    earlier_end = max(word.end for word in earlier)
    later_start = min(word.start for word in later)
    later_end = max(word.end for word in later)
    covered = min(earlier_end, later_end) - max(earlier_start, later_start)
    if covered < 0.5 * (earlier_end - earlier_start):
        return False
    earlier_speakers = {word.speaker_id for word in earlier}
    later_speakers = {word.speaker_id for word in later}
    if None not in earlier_speakers and None not in later_speakers and earlier_speakers.isdisjoint(later_speakers):
        return False
    return True


def _collapse_doubled_number_runs(
    words: list[Word],
) -> tuple[list[Word], list[tuple[Word, Word]], list[list[Word]]]:
    """Collapse MAI's AABBCC doubling only where it cannot be genuine speech.

    A doubled pair is two adjacent identical tokens (not three) with a short
    gap. Pairs linked by at most one other token form a run. A run of three or
    more pairs made only of numbers is a counted sequence (``10, 10, 9, 9, ...``)
    and keeps the longer copy of each pair. Any other run is left untouched and
    reported, because doubled words can be real (``nein, nein``; a chorus).
    """
    tokens = [_normalized_word(word) for word in words]
    pairs = []
    for index in range(len(words) - 1):
        if (
            tokens[index] == tokens[index + 1]
            and 0.0 <= words[index + 1].start - words[index].end + _REWIND_EPSILON_SECONDS
            and words[index + 1].start - words[index].end <= _MAX_DOUBLED_GAP_SECONDS
            and (index == 0 or tokens[index - 1] != tokens[index])
            and (index + 2 >= len(words) or tokens[index + 2] != tokens[index])
        ):
            pairs.append(index)
    runs: list[list[int]] = []
    for index in pairs:
        if runs:
            previous = runs[-1][-1]
            between = words[previous + 1:index + 1]
            if index - (previous + 2) <= 1 and all(
                right.start - left.end <= _MAX_DOUBLED_RUN_LINK_SECONDS for left, right in zip(between, between[1:])
            ):
                runs[-1].append(index)
                continue
        runs.append([index])
    drop: set[int] = set()
    collapsed: list[tuple[Word, Word]] = []
    kept_runs: list[list[Word]] = []
    for run in runs:
        if len(run) < _MIN_DOUBLED_RUN_PAIRS:
            continue
        if all(tokens[index].isdigit() for index in run):
            for index in run:
                first, second = words[index], words[index + 1]
                keep_first = first.end - first.start >= second.end - second.start
                drop.add(index + 1 if keep_first else index)
                collapsed.append((second, first) if keep_first else (first, second))
        else:
            kept_runs.append(words[run[0]:run[-1] + 2])
    return [word for index, word in enumerate(words) if index not in drop], collapsed, kept_runs


def _join_words(left: list[Word], right: list[Word], boundary: float) -> list[Word]:
    return _join_paired_words(left, right, _overlap_pairs(left, right, boundary), boundary)


def _overlap_pairs(left: list[Word], right: list[Word], boundary: float) -> dict[int, int]:
    """Match overlapping occurrences before ownership so timing jitter cannot split a pair.

    The monotonic one-to-one match keeps repeated words distinct. Requiring
    overlapping intervals avoids merging repetitions at clearly different times.
    The two chunks can spell one spoken word differently (``vamo``/``vamos``),
    so a strongly overlapping, similar word also pairs; exact text pairs are
    preferred over such spelling pairs.
    """
    from .providers import ProviderError

    left_overlap = [i for i, word in enumerate(left) if word.end > boundary - _CONTEXT_SECONDS and word.start < boundary + _CONTEXT_SECONDS]
    right_overlap = [i for i, word in enumerate(right) if word.end > boundary - _CONTEXT_SECONDS and word.start < boundary + _CONTEXT_SECONDS]
    if max(len(left_overlap), len(right_overlap)) > 200:
        raise ProviderError("MAI-Transcribe 2 returned too many overlapping word timestamps.")
    left_tokens = [_normalized_word(left[i]) for i in left_overlap]
    right_tokens = [_normalized_word(right[i]) for i in right_overlap]
    # Score: (exact text pairs, all pairs, summed overlap similarity).
    scores = [[(0, 0, 0.0) for _ in range(len(right_overlap) + 1)] for _ in range(len(left_overlap) + 1)]
    matches = {}
    for row, left_index in enumerate(left_overlap, start=1):
        for col, right_index in enumerate(right_overlap, start=1):
            lword, rword = left[left_index], right[right_index]
            overlap = min(lword.end, rword.end) - max(lword.start, rword.start)
            score = max(scores[row - 1][col], scores[row][col - 1])
            if overlap > 0:
                similarity = overlap / min(lword.end - lword.start, rword.end - rword.start)
                exact = left_tokens[row - 1] == right_tokens[col - 1]
                if exact or _same_spoken_word(lword, rword, left_tokens[row - 1], right_tokens[col - 1], overlap):
                    diagonal = scores[row - 1][col - 1]
                    candidate = (diagonal[0] + int(exact), diagonal[1] + 1, diagonal[2] + similarity)
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
    return pairs


def _same_spoken_word(left: Word, right: Word, left_token: str, right_token: str, overlap: float) -> bool:
    # MAI never lets two different words touch, so a substantial overlap
    # between the two chunks' words means they transcribe the same speech.
    if SequenceMatcher(None, left_token, right_token).ratio() >= 0.5:
        return overlap >= 0.4 * min(left.end - left.start, right.end - right.start)
    return overlap >= 0.6 * (max(left.end, right.end) - min(left.start, right.start))


def _join_paired_words(left: list[Word], right: list[Word], pairs: dict[int, int], boundary: float) -> list[Word]:
    """A matched pair always contributes one real provider record, even when
    its two midpoints disagree about which chunk owns it; unmatched words are
    owned by their own midpoint."""
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
