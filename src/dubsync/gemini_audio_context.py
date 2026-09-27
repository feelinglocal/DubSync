"""Job-owned Gemini audio upload/cache with bounded cost and explicit cleanup.

The compressed episode supplies linguistic context. Focused lossless clips and
ASR word times remain the local acoustic evidence; model timestamps are unused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import tempfile
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from .audio import probe_audio_duration
from .providers import ProviderError

logger = logging.getLogger(__name__)
_MIME_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
               ".ogg": "audio/ogg", ".aac": "audio/aac", ".m4a": "audio/m4a",
               ".aiff": "audio/aiff", ".aif": "audio/aiff"}
_OPTIONS = {"enabled", "compress_long_audio", "cache_enabled", "cache_ttl_seconds",
            "max_audio_bytes", "max_audio_duration_seconds", "max_uncached_audio_tokens",
            "max_lifetime_seconds", "upload_timeout_seconds", "compression_timeout_seconds",
            "request_timeout_seconds", "max_consecutive_request_failures"}


@dataclass(frozen=True)
class RequestAudioContext:
    cached_content: str | None = None
    file_uri: str | None = None
    mime_type: str | None = None
    label: str = ""
    lease_acquired: bool = False


def _locked(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return synchronized


def _new_client(**kwargs):
    try:
        from google import genai
    except ImportError as exc:
        raise ProviderError("Install dubsync[cloud] to use Gemini audio context.") from exc
    return genai.Client(**kwargs)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(obj: object, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _positive(options: dict, name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ProviderError(f"Gemini audio context {name} must be a positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"Gemini audio context {name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ProviderError(f"Gemini audio context {name} must be a positive number.")
    return number


def validate_audio_context_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, dict) or set(config) - _OPTIONS:
        raise ProviderError("Gemini audio context configuration contains unsupported options.")
    for name in ("enabled", "compress_long_audio", "cache_enabled"):
        if name in config and not isinstance(config[name], bool):
            raise ProviderError(f"Gemini audio context {name} must be true or false.")
    return dict(config)


class GeminiAudioContext:
    def __init__(self, *, api_key: str, model: str, path: str | Path,
                 duration_seconds: float, config: dict[str, Any] | None = None,
                 source_context: str = ""):
        self._lock = RLock()
        self.options = validate_audio_context_config(config)
        self.path = Path(path)
        self.api_key, self.model = api_key, model
        self.duration = _positive({"duration": duration_seconds}, "duration", 1)
        if self.duration > _positive(self.options, "max_audio_duration_seconds", 14400):
            raise ProviderError("Gemini full audio context exceeds the duration limit.")
        self.max_bytes = int(_positive(self.options, "max_audio_bytes", 512 * 1024 * 1024))
        try:
            self.original_stat = self.path.stat()
        except OSError as exc:
            raise ProviderError("Gemini full audio context could not be read.") from exc
        if not self.path.is_file() or not 0 < self.original_stat.st_size <= self.max_bytes:
            raise ProviderError("Gemini full audio context exceeds the size limit or is empty.")
        if self.path.suffix.lower() not in _MIME_TYPES:
            raise ProviderError("Gemini full audio context requires a supported audio format.")
        self.ttl = max(1, min(900, int(_positive(self.options, "cache_ttl_seconds", 900))))
        self.max_lifetime = min(7200, _positive(self.options, "max_lifetime_seconds", 3600))
        self.upload_timeout = min(300, _positive(self.options, "upload_timeout_seconds", 90))
        self.compression_timeout = min(600, _positive(self.options, "compression_timeout_seconds", 120))
        self.request_timeout = min(300, _positive(self.options, "request_timeout_seconds", 90))
        self.uncached_budget = int(_positive(self.options, "max_uncached_audio_tokens", 256000))
        self.failure_limit = max(1, min(3, int(_positive(self.options, "max_consecutive_request_failures", 2))))
        self.audio_tokens = math.ceil(self.duration * 32)
        self.source_context = source_context
        self.owner = f"dubsync-{uuid.uuid4().hex}"
        self.client = None
        self.file_name = self.file_uri = self.cache_name = None
        self.started = self.cache_started = self.cache_expires = None
        self.deadline_utc = None
        self.prepared = self.closed = False
        self._cleaned = False
        self.temporary = None
        self.upload_path = self.path
        self._report: dict[str, Any] = {
            "enabled": self.options.get("enabled", True), "duration_seconds": self.duration,
            "transport": "not_requested",
            "original_size_bytes": self.original_stat.st_size, "original_sha256": None,
            "uploaded_sha256": None, "uploaded_size_bytes": 0, "mime_type": None,
            "compression": "none", "preparation_seconds": 0.0, "upload_seconds": 0.0,
            "cache_create_seconds": 0.0, "cache_token_count": 0,
            "cache_token_count_is_estimate": False,
            "cache_create_input_tokens_reserved": 0, "cache_storage_token_seconds": 0.0,
            "cache_storage_token_seconds_reserved": 0.0, "cache_create_attempts": 0,
            "cache_ttl_seconds": self.ttl,
            "cached_requests": 0, "uncached_requests": 0, "degraded_requests": 0,
            "uncached_audio_tokens_reserved": 0, "cache_renewals": 0,
            "unreported_uncached_audio_tokens_reserved": 0,
            "unreported_cached_audio_tokens_reserved": 0, "failed_requests": 0,
            "consecutive_failed_requests": 0,
            "in_flight_requests": 0,
            "request_usage_events": [],
            "first_request_seconds": None, "total_request_seconds": 0.0,
            "total_generation_seconds": 0.0,
            "warnings": [], "cleanup_status": "not_started",
        }

    def _warn(self, reason: str) -> None:
        if reason not in self._report["warnings"]:
            self._report["warnings"].append(reason)

    @_locked
    def record_warning(self, reason: str) -> None:
        self._warn(reason)

    @_locked
    def record_unreported_request(self) -> None:
        self._report["unreported_uncached_audio_tokens_reserved"] += self.audio_tokens

    @_locked
    def record_generation_result(self, *, success: bool, cached: bool) -> None:
        if success:
            self._report["consecutive_failed_requests"] = 0
            return
        self._report["failed_requests"] += 1
        self._report["consecutive_failed_requests"] += 1
        if cached:
            self._report["unreported_cached_audio_tokens_reserved"] += self._report["cache_token_count"]
        else:
            self.record_unreported_request()
        if self._report["consecutive_failed_requests"] >= self.failure_limit:
            self._warn("context_request_circuit_open")
            self._report["transport"] = "unavailable"
            self.close()

    @_locked
    def record_request_metrics(self, *, elapsed_seconds: float, generation_seconds: float) -> None:
        if self._report["first_request_seconds"] is None:
            self._report["first_request_seconds"] = elapsed_seconds
        self._report["total_request_seconds"] += elapsed_seconds
        self._report["total_generation_seconds"] += generation_seconds

    @_locked
    def record_usage(self, event: dict[str, Any], *, cached: bool) -> None:
        self._report["request_usage_events"].append({
            "transport": "native_cache" if cached else "full_audio_uri", "usage": deepcopy(event),
        })

    def _lease_deadline(self) -> tuple[float, datetime]:
        self.ensure_available()
        now = time.monotonic()
        remaining = self.max_lifetime - (now - self.started)
        lease_seconds = min(self.ttl, remaining)
        return now + lease_seconds, min(_utc_now() + timedelta(seconds=lease_seconds), self.deadline_utc)

    @_locked
    def _open_client(self) -> None:
        if self.client is None:
            # Creation may have side effects. Do not automatically duplicate an
            # upload/cache after an ambiguous network error.
            self.client = _new_client(api_key=self.api_key, http_options={
                "timeout": int(min(self.upload_timeout, self.request_timeout) * 1000),
                "retry_options": {"attempts": 1},
            })

    def _prepare_media(self) -> None:
        before = time.monotonic()
        self._report["original_sha256"] = _sha256(self.path)
        compress = self.options.get("compress_long_audio", True)
        compact_mp3 = (self.path.suffix.lower() == ".mp3"
                       and self.original_stat.st_size * 8 / self.duration <= 66000)
        short_small = self.duration <= 180 and self.original_stat.st_size <= 10 * 1024 * 1024
        if compress and not compact_mp3 and not short_small:
            self.temporary = tempfile.TemporaryDirectory(prefix="dubsync-gemini-")
            self.upload_path = Path(self.temporary.name) / f"{self._report['original_sha256'][:24]}.mp3"
            output_cap = min(self.max_bytes, math.ceil((self.duration + 5) * 8000) + 131072)
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(self.path), "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "64k",
                "-map_metadata", "-1", "-fs", str(output_cap), str(self.upload_path),
            ], check=True, capture_output=True, timeout=self.compression_timeout)
            if not self.upload_path.exists() or not 0 < self.upload_path.stat().st_size < output_cap:
                raise ProviderError("Compressed Gemini context reached its size limit.")
            prepared_duration = probe_audio_duration(self.upload_path)
            if abs(prepared_duration - self.duration) > 0.25:
                raise ProviderError("Compressed Gemini audio context did not preserve episode duration.")
            self._report["compression"] = "mp3_mono_64kbps_24khz"
        elif compact_mp3:
            self._report["compression"] = "reused_compact_mp3"
        self._report["uploaded_sha256"] = (_sha256(self.upload_path) if self.upload_path != self.path
                                             else self._report["original_sha256"])
        self._report["uploaded_size_bytes"] = self.upload_path.stat().st_size
        self._report["mime_type"] = _MIME_TYPES[self.upload_path.suffix.lower()]
        self._report["preparation_seconds"] = time.monotonic() - before

    def _label(self) -> str:
        return json.dumps({
            "audio_role": "full_episode_read_only_context", "origin_seconds": 0,
            "duration_seconds": self.duration, "original_sha256": self._report["original_sha256"],
            "use": "Understand literal words, distinct speakers, overlap, and sentence boundaries within each scene. "
                   "Focused clips identify the actual decision cases. Never infer or return replacement timestamps; "
                   "ASR word timing remains timing evidence. Never move dialogue between scenes.",
        })

    def _prepare(self) -> None:
        self.prepared = True
        self.started = time.monotonic()
        self.deadline_utc = _utc_now() + timedelta(seconds=self.max_lifetime)
        upload_attempted = False
        try:
            self._prepare_media()
            self._open_client()
            before = time.monotonic()
            upload_attempted = True
            uploaded = self.client.files.upload(file=self.upload_path, config={
                "mime_type": self._report["mime_type"], "display_name": self.owner,
            })
            self.file_name = _value(uploaded, "name")
            self.file_uri = _value(uploaded, "uri")
            if not self.file_name or not self.file_uri:
                raise ProviderError("Gemini audio upload returned no usable file reference.")
            uploaded = wait_for_uploaded_audio(self.client, uploaded, timeout_seconds=self.upload_timeout,
                                               started=before)
            self._report["upload_seconds"] = time.monotonic() - before
            stat = self.path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (self.original_stat.st_size, self.original_stat.st_mtime_ns):
                raise ProviderError("Source audio changed during Gemini context preparation.")
        except BaseException as exc:
            if upload_attempted and not self.file_name:
                self._recover_unknown_resource(self.client.files, "file")
            self._warn("upload_unavailable")
            self._report["transport"] = "unavailable"
            self._report["degraded_requests"] += 1
            self.close()
            if not isinstance(exc, Exception):
                raise
            raise ProviderError("Full episode audio context could not be prepared; affected subtitles require review.") from exc
        if self.options.get("cache_enabled", True) and self.audio_tokens >= 4096:
            planned_expiry, expire_time = self._lease_deadline()
            before = time.monotonic()
            lease_seconds = max(0, planned_expiry - before)
            self._report["cache_create_attempts"] += 1
            prefix = self._label()
            if self.source_context:
                prefix += "\nRead-only ordered source subtitle context:\n" + self.source_context
            # Reserve byte-level text tokenization plus multimodal framing if
            # provider usage is absent or a creation response is lost.
            estimated_tokens = self.audio_tokens + len(prefix.encode("utf-8")) + 256
            try:
                # Dict Parts are accepted by google-genai. Files are URI references,
                # so neither a long MP3 nor a long WAV is loaded into Python bytes.
                created = self.client.caches.create(model=self.model, config={
                    "display_name": self.owner, "expire_time": expire_time,
                    "contents": [{"role": "user", "parts": [{"text": prefix}, {"file_data": {
                        "file_uri": self.file_uri, "mime_type": self._report["mime_type"],
                    }}]}],
                })
                self.cache_name = _value(created, "name")
                if not self.cache_name:
                    raise ProviderError("Gemini cache creation returned no reference.")
                # Ownership and lease state must not depend on optional usage
                # metadata: cleanup remains valid even if that metadata is bad.
                self.cache_started = before
                self.cache_expires = planned_expiry
                usage = _value(created, "usage_metadata", {})
                raw_tokens = _value(usage, "total_token_count")
                try:
                    if isinstance(raw_tokens, bool):
                        raise ValueError("Boolean token count")
                    measured_tokens = float(raw_tokens)
                    if not math.isfinite(measured_tokens) or measured_tokens <= 0:
                        raise ValueError("Nonpositive or nonfinite token count")
                    tokens = math.ceil(measured_tokens)
                except (TypeError, ValueError, OverflowError):
                    tokens = estimated_tokens
                    self._report["cache_token_count_is_estimate"] = True
                    self._warn("cache_usage_estimated")
                self._report["cache_token_count"] = tokens
                self._report["cache_create_input_tokens_reserved"] = self._report["cache_token_count"]
            except BaseException as exc:
                self._warn("cache_unavailable")
                # A response can be lost after server-side creation. Reserve the
                # possible input/TTL cost and expose the uncertainty in the report.
                self._report["cache_create_input_tokens_reserved"] = estimated_tokens
                self._report["cache_storage_token_seconds_reserved"] = estimated_tokens * lease_seconds
                if not self.cache_name and _value(exc, "code") not in {400, 401, 403, 404, 413, 422, 429}:
                    self._recover_unknown_resource(self.client.caches, "cache")
                if not isinstance(exc, Exception):
                    self.close()
                    raise
            self._report["cache_create_seconds"] = time.monotonic() - before

    def _recover_unknown_resource(self, service, kind: str) -> None:
        # A create response can be lost after the server accepted it. Match only
        # this unpredictable job-owned display name; never delete other jobs.
        recovered, complete = recover_owned_resources(service, self.owner)
        for resource in recovered:
            self._delete(service, _value(resource, "name"), f"{kind}_delete_failed")
        if not complete or not recovered:
            self._warn(f"{kind}_creation_unconfirmed")

    @_locked
    def ensure_available(self) -> None:
        if not self._report["enabled"]:
            return
        if self.closed:
            self._report["transport"] = "unavailable"
            self._report["degraded_requests"] += 1
            raise ProviderError("Full episode audio context is closed or unavailable; affected subtitles require review.")
        if self.started is not None and time.monotonic() - self.started >= self.max_lifetime:
            self._report["transport"] = "unavailable"
            self._warn("context_lifetime_exhausted")
            self._report["degraded_requests"] += 1
            self.close()
            raise ProviderError("Full episode audio context exceeded its job lifetime; affected subtitles require review.")

    @_locked
    def for_request(self) -> RequestAudioContext:
        if not self._report["enabled"]:
            return RequestAudioContext()
        self.ensure_available()
        if not self.prepared:
            self._prepare()
        self.ensure_available()
        now = time.monotonic()
        if self.cache_name and now >= self.cache_expires - min(60, self.ttl / 2):
            try:
                planned_expiry, expire_time = self._lease_deadline()
                self.client.caches.update(name=self.cache_name, config={"expire_time": expire_time})
                self.cache_expires = planned_expiry
                self._report["cache_renewals"] += 1
            except Exception:
                self._warn("cache_renewal_failed")
                if self._report["in_flight_requests"]:
                    # A failed renewal must not delete a cache that another
                    # worker already acquired. Stop new work and let current
                    # leases finish before the shared resources are removed.
                    self._report["transport"] = "unavailable"
                    self._report["degraded_requests"] += 1
                    self.close()
                    raise ProviderError("Full episode cache lease could not be renewed; affected subtitles require review.")
                self._delete_cache()
        self.ensure_available()
        if self.cache_name and time.monotonic() >= self.cache_expires:
            self._warn("cache_lease_expired")
            self._report["transport"] = "unavailable"
            self._report["degraded_requests"] += 1
            self.close()
            raise ProviderError("Full episode cache lease expired during renewal; affected subtitles require review.")
        if self.cache_name:
            self._report["transport"] = "native_cache"
            self._report["cached_requests"] += 1
            return RequestAudioContext(cached_content=self.cache_name, label=self._label())
        reserved = self._report["uncached_audio_tokens_reserved"]
        if reserved + self.audio_tokens > self.uncached_budget:
            self._report["transport"] = "unavailable"
            self._warn("uncached_budget_exhausted")
            self._report["degraded_requests"] += 1
            self.close()
            raise ProviderError("Full episode audio context exhausted its uncached input budget; affected subtitles require review.")
        self._report["uncached_audio_tokens_reserved"] += self.audio_tokens
        self._report["transport"] = "full_audio_uri"
        self._report["uncached_requests"] += 1
        return RequestAudioContext(file_uri=self.file_uri, mime_type=self._report["mime_type"], label=self._label())

    @_locked
    def acquire_request(self) -> RequestAudioContext:
        request = self.for_request()
        if not self._report["enabled"]:
            return request
        self._report["in_flight_requests"] += 1
        return replace(request, lease_acquired=True)

    @_locked
    def release_request(self) -> None:
        if self._report["in_flight_requests"] <= 0:
            raise RuntimeError("Gemini audio request lease was released without acquisition.")
        self._report["in_flight_requests"] -= 1
        if self.closed and self._report["in_flight_requests"] == 0:
            self.close()

    @_locked
    def set_source_context(self, source_context: str) -> None:
        if not self.prepared:
            self.source_context = source_context

    def _delete_cache(self) -> None:
        if self.cache_name:
            if self.cache_started is not None and self.cache_expires is not None:
                self._report["cache_storage_token_seconds"] = self._report["cache_token_count"] * max(
                    0, min(time.monotonic(), self.cache_expires) - self.cache_started)
            self._delete(self.client.caches, self.cache_name, "cache_delete_failed")
            if "cache_delete_failed" in self._report["warnings"]:
                duration = (max(0, self.cache_expires - self.cache_started)
                            if self.cache_expires is not None and self.cache_started is not None else self.ttl)
                tokens = max(self._report["cache_token_count"], self._report["cache_create_input_tokens_reserved"])
                self._report["cache_storage_token_seconds_reserved"] = tokens * duration
            self.cache_name = None

    def _delete(self, service, name: str, failure: str) -> None:
        for attempt in range(2):
            try:
                service.delete(name=name)
                return
            except Exception as exc:
                if _value(exc, "code") == 404:
                    return
                if attempt:
                    self._warn(failure)
                    logger.warning("Gemini audio resource cleanup incomplete: %s", failure)

    @_locked
    def close(self) -> None:
        self.closed = True
        if self._cleaned:
            return
        if self._report["in_flight_requests"]:
            self._report["cleanup_status"] = "pending"
            return
        self._cleaned = True
        self._delete_cache()
        if self.file_name and self.client:
            self._delete(self.client.files, self.file_name, "file_delete_failed")
            self.file_name = None
        if self.client:
            try:
                self.client.close()
            except Exception:
                self._warn("client_close_failed")
        if self.temporary:
            try:
                self.temporary.cleanup()
            except OSError:
                self._warn("local_cleanup_failed")
        failures = {"cache_delete_failed", "file_delete_failed", "local_cleanup_failed", "snippet_cleanup_failed",
                    "file_creation_unconfirmed", "cache_creation_unconfirmed"}
        self._report["cleanup_status"] = "incomplete" if failures.intersection(self._report["warnings"]) else "complete"

    @_locked
    def report(self) -> dict[str, Any]:
        result = deepcopy(self._report)
        if self.cache_name and self.cache_started is not None:
            result["cache_storage_token_seconds"] = result["cache_token_count"] * max(
                0, min(time.monotonic(), self.cache_expires) - self.cache_started)
        return result


def wait_for_uploaded_audio(client, uploaded, *, timeout_seconds: float, started: float | None = None):
    deadline = (time.monotonic() if started is None else started) + timeout_seconds
    while True:
        state = _value(uploaded, "state", "ACTIVE")
        state = str(_value(state, "name", state)).upper().split(".")[-1]
        if state == "ACTIVE":
            return uploaded
        if state != "PROCESSING":
            raise ProviderError("Gemini audio processing failed.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("Gemini audio processing timed out.")
        time.sleep(min(1.0, remaining))
        uploaded = client.files.get(name=_value(uploaded, "name"))


class GeminiSnippetUploads:
    """Request-scoped URI transport when encoded clips would exceed inline limits."""
    def __init__(self, *, api_key: str, timeout_seconds: float):
        self.timeout = min(90.0, timeout_seconds)
        self.client = _new_client(api_key=api_key, http_options={
            "timeout": max(1, int(self.timeout * 1000)), "retry_options": {"attempts": 1},
        })
        self.names: list[str] = []
        self.owner = f"dubsync-snippet-{uuid.uuid4().hex}"
        self.recovery_incomplete = False

    def upload(self, path: Path, mime_type: str) -> RequestAudioContext:
        if not 0 < path.stat().st_size <= 32 * 1024 * 1024:
            raise ProviderError("Gemini focused audio clip exceeds its size limit or is empty.")
        started = time.monotonic()
        try:
            uploaded = self.client.files.upload(file=path, config={"mime_type": mime_type, "display_name": self.owner})
        except BaseException:
            recovered, complete = recover_owned_resources(self.client.files, self.owner)
            self.names.extend(_value(item, "name") for item in recovered if _value(item, "name") not in self.names)
            self.recovery_incomplete = not complete or not recovered
            raise
        name, uri = _value(uploaded, "name"), _value(uploaded, "uri")
        if name:
            self.names.append(name)
        if not name or not uri:
            raise ProviderError("Gemini focused audio upload returned no usable file reference.")
        wait_for_uploaded_audio(self.client, uploaded, timeout_seconds=self.timeout, started=started)
        return RequestAudioContext(file_uri=uri, mime_type=mime_type)

    def close(self) -> bool:
        complete = not self.recovery_incomplete
        for name in self.names:
            for attempt in range(2):
                try:
                    self.client.files.delete(name=name)
                    break
                except Exception as exc:
                    if _value(exc, "code") == 404:
                        break
                    if attempt:
                        complete = False
                        logger.warning("Gemini focused audio resource cleanup incomplete.")
        self.names.clear()
        try:
            self.client.close()
        except Exception:
            logger.warning("Gemini focused audio client cleanup failed.")
        return complete


def recover_owned_resources(service, owner: str) -> tuple[list[object], bool]:
    list_resources = getattr(service, "list", None)
    if not callable(list_resources):
        return [], False
    matches = []
    try:
        for index, resource in enumerate(list_resources(config={"page_size": 100})):
            if index >= 300:
                return matches, False
            if _value(resource, "display_name") == owner and _value(resource, "name"):
                matches.append(resource)
    except Exception:
        return matches, False
    return matches, True
