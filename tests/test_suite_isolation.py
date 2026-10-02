from __future__ import annotations

import textwrap
from pathlib import Path


def _inner_suite(pytester, source: str) -> None:
    # Run a separate session with this suite's own conftest, so the guard is
    # exercised exactly as every backend test sees it.
    pytester.makeini("[pytest]\n")
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(test_inner=textwrap.dedent(source))


def test_provider_keys_loaded_from_dotenv_do_not_reach_later_tests(pytester):
    _inner_suite(pytester, """
        import os

        from dubsync.cli import _load_dotenv


        def test_cli_loads_dotenv(tmp_path, monkeypatch):
            monkeypatch.chdir(tmp_path)
            (tmp_path / ".env").write_text("GEMINI_API_KEY=dotenv-key\\nHF_TOKEN=dotenv-token\\n", encoding="utf-8")
            _load_dotenv()
            assert os.environ["GEMINI_API_KEY"] == "dotenv-key"


        def test_later_test_sees_no_provider_key():
            assert os.environ.get("GEMINI_API_KEY") is None
            assert os.environ.get("HF_TOKEN") is None
    """)

    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=2)


def test_ambient_provider_keys_are_hidden_from_offline_tests_and_kept_for_live_tests(pytester, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-placeholder")
    _inner_suite(pytester, """
        import os
        import sys

        import pytest


        def test_offline_test_sees_no_provider_key():
            assert os.environ.get("OPENROUTER_API_KEY") is None


        @pytest.mark.live
        def test_live_test_keeps_key_and_may_connect():
            assert os.environ.get("OPENROUTER_API_KEY") == "ambient-placeholder"
            # The event a socket connect raises, without opening a socket.
            sys.audit("socket.connect", None, ("192.0.2.1", 443))
    """)

    # The synthetic inner suite has no provider code; --live only selects its live-marked test.
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "--live").assert_outcomes(passed=2)


def test_offline_test_fails_when_it_reaches_an_external_host_even_if_the_error_is_swallowed(pytester):
    _inner_suite(pytester, """
        import socket


        def test_swallowed_external_connect():
            # TEST-NET-1 over UDP: connect() only records the peer and sends nothing.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                try:
                    sock.connect(("192.0.2.1", 9))
                except OSError:
                    pass  # provider adapters turn transport errors into soft QC flags


        def test_loopback_connect_is_allowed():
            with socket.create_server(("127.0.0.1", 0)) as server:
                with socket.create_connection(server.getsockname(), timeout=5):
                    pass
    """)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines([
        "*ERROR at teardown of test_swallowed_external_connect*",
        "*tried to reach an external host*192.0.2.1*",
    ])
