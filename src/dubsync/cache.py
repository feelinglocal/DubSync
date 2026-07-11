from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SECRET_PARAM_NAMES = {
    "api_key",
    "access_token",
    "authorization",
    "auth_token",
    "hf_token",
    "password",
    "secret",
    "token",
}


class CacheKey(BaseModel):
    digest: str
    audio_sha256: str | None = None
    content_sha256: str | None = None
    model: str
    params: dict[str, Any]

    @classmethod
    def from_audio(cls, audio_path: Path, model: str, params: dict[str, Any]) -> "CacheKey":
        audio_sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        cache_params = _cache_safe_params(params)
        canonical = _canonical_json(
            {"kind": "audio", "audio_sha256": audio_sha, "model": model, "params": cache_params}
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(digest=digest, audio_sha256=audio_sha, model=model, params=cache_params)

    @classmethod
    def from_payload(cls, payload: Any, model: str, params: dict[str, Any]) -> "CacheKey":
        cache_params = _cache_safe_params(params)
        content = _canonical_json(payload)
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        canonical = _canonical_json(
            {"kind": "payload", "content_sha256": content_sha, "model": model, "params": cache_params}
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(digest=digest, content_sha256=content_sha, model=model, params=cache_params)


class JsonDiskCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: CacheKey) -> Path:
        return self.root / f"{key.digest}.json"

    def read(self, key: CacheKey) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "value" in payload:
            return payload["value"]
        return payload

    def write(self, key: CacheKey, value: dict[str, Any]) -> None:
        payload = {"cache_key": key.model_dump(), "value": value}
        self._path(key).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_safe_params(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _cache_safe_params(item)
            for key, item in value.items()
            if str(key).lower() not in SECRET_PARAM_NAMES
        }
    if isinstance(value, list):
        return [_cache_safe_params(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
