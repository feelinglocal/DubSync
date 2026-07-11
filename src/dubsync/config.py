from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .style_profile import StyleProfile


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        detail = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise ValueError(f"invalid YAML in {path.name}: {detail}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_style_profile(path: Path | None) -> StyleProfile | None:
    if path is None:
        return None
    try:
        return StyleProfile.model_validate(load_yaml(path))
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(part) for part in first_error.get("loc", ())) or "value"
        message = str(first_error.get("msg", "invalid value"))
        raise ValueError(f"invalid style profile in {path.name}: {field}: {message}") from exc


def write_style_profile(path: Path, profile: StyleProfile) -> None:
    path.write_text(yaml.safe_dump(profile.model_dump(exclude_none=True), sort_keys=False), encoding="utf-8")
