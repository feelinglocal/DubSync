from __future__ import annotations

import importlib.util
from pathlib import Path


def test_local_launcher_uses_project_env_even_when_parent_has_stale_key(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_local", Path("scripts/run_local.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=project-key\nDUBSYNC_DATA_DIR=local-data\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-parent-key")
    monkeypatch.setenv("DUBSYNC_DATA_DIR", "parent-data")
    monkeypatch.chdir(tmp_path.parent)
    module.load_local_environment(tmp_path)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "project-key"
    assert os.environ["DUBSYNC_DATA_DIR"] == "local-data"
    assert Path.cwd() == tmp_path


def test_regular_server_settings_preserve_deployment_environment(tmp_path, monkeypatch):
    from dubsync.web.settings import WebSettings
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=project-key\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "deployment-key")
    WebSettings.from_env()
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "deployment-key"
