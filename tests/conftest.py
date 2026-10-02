from __future__ import annotations

import ipaddress
import os
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# Provider credentials, including the Hugging Face tokens local diarization reads.
_PROVIDER_KEY_SUFFIXES = ("_API_KEY", "_ACCESS_TOKEN")
_PROVIDER_KEY_NAMES = frozenset({"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
_NETWORK_EVENTS = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
# External connections attempted by the running offline test; None outside one.
_external_attempts: list[str] | None = None


def _is_provider_key(name: str) -> bool:
    name = name.upper()
    return name.endswith(_PROVIDER_KEY_SUFFIXES) or name in _PROVIDER_KEY_NAMES


def _is_local_address(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # Unix sockets, named pipes, or a send on a connected socket
    host = address[0]
    host = (host.decode("ascii", "replace") if isinstance(host, bytes) else str(host)).split("%", 1)[0].lower()
    if host in {"", "localhost"} or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or ip.is_unspecified or (mapped is not None and mapped.is_loopback)


def _refuse_external_network(event: str, args: tuple) -> None:
    # An audit hook sees every Python socket, including those opened inside
    # provider SDKs, and runs before the operating system sends anything.
    if event not in _NETWORK_EVENTS:
        return
    attempts = _external_attempts
    if attempts is None or len(args) < 2 or _is_local_address(args[1]):
        return
    attempts.append(f"{event} {args[1]!r}")
    raise OSError(f"Offline test refused a connection to the external host {args[1]!r}; "
                  "only tests marked 'live' and run with --live may use the network.")


sys.addaudithook(_refuse_external_network)


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


@pytest.fixture(autouse=True)
def _offline_provider_isolation(request):
    """Offline tests see no provider credential and cannot reach an external host.

    Credentials are restored afterwards, so a CLI test that loads a .env cannot
    leak one into later tests. Live tests run with --live keep both.
    """
    global _external_attempts
    live = request.config.getoption("--live") and request.node.get_closest_marker("live") is not None
    saved = {name: value for name, value in os.environ.items() if _is_provider_key(name)}
    if not live:
        for name in saved:
            del os.environ[name]
        _external_attempts = []
    try:
        yield
    finally:
        attempts, _external_attempts = _external_attempts, None
        for name in [name for name in os.environ if _is_provider_key(name)]:
            del os.environ[name]
        os.environ.update(saved)
    if attempts:
        pytest.fail(f"Offline test tried to reach an external host {len(attempts)} time(s): "
                    + "; ".join(dict.fromkeys(attempts))
                    + ". Configure a fixture provider, or mark the test 'live' and run it with --live.",
                    pytrace=False)


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
