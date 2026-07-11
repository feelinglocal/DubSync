from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="Run opt-in live provider smoke tests.")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: opt-in test that may call external providers")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    live_items = [item for item in items if "live" in item.keywords]
    if not live_items:
        return
    for item in live_items:
        items.remove(item)
    config.hook.pytest_deselected(items=live_items)


@pytest.fixture()
def sample_srt_path() -> Path:
    return Path("Examples") / "srt test.srt"


@pytest.fixture()
def shifted_srt_text() -> str:
    return (
        "1\n"
        "00:00:10,000 --> 00:00:11,000\n"
        "hello there\n"
        "\n"
        "2\n"
        "00:00:11,000 --> 00:00:12,000\n"
        "general kenobi\n"
        "\n"
    )


@pytest.fixture()
def shifted_wordstream() -> list[dict[str, object]]:
    return [
        {"text": "hello", "start": 1.00, "end": 1.20, "confidence": 0.98, "speaker_id": "A"},
        {"text": "there", "start": 1.23, "end": 1.45, "confidence": 0.97, "speaker_id": "A"},
        {"text": "general", "start": 2.00, "end": 2.33, "confidence": 0.98, "speaker_id": "A"},
        {"text": "kenobi", "start": 2.36, "end": 2.80, "confidence": 0.99, "speaker_id": "A"},
    ]
