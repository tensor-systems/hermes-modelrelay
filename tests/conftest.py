"""Fixtures: an isolated HERMES_HOME with the plugin installed as a directory plugin, and a fake HTTP layer.

Run inside a Python environment where hermes-agent is importable (e.g. its own venv).
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hermes_modelrelay"

CATALOG = {
    "object": "list",
    "data": [
        {
            "id": "glm-5.3-flash",
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens", "tools", "tool_choice", "reasoning"],
        },
        {
            "id": "muse-spark-1.3",
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens", "tools", "tool_choice", "reasoning"],
        },
        {"id": "whisper-large-v3-turbo", "architecture": {"input_modalities": ["audio"]},
         "supported_parameters": ["language"]},
        {"id": "bare-model"},
    ],
}


class FakeHTTP:
    """Stands in for ``hermes_cli.urllib_security.open_credentialed_url``."""

    def __init__(self) -> None:
        self.requests: list = []
        self.payload: object = CATALOG
        self.error: Exception | None = None

    def __call__(self, request, *, timeout, **_kw):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return io.BytesIO(json.dumps(self.payload).encode())


@pytest.fixture
def http(monkeypatch):
    import hermes_cli.urllib_security as urllib_security

    fake = FakeHTTP()
    monkeypatch.setattr(urllib_security, "open_credentialed_url", fake)
    return fake


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A fresh HERMES_HOME with the plugin copied to plugins/model-providers/modelrelay/."""
    home = tmp_path / "hermes-home"
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "model-providers" / "modelrelay",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (home / "config.yaml").write_text("model:\n  provider: modelrelay\n  default: glm-5.3-flash\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("MODELRELAY_API_KEY", "MODELRELAY_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def plugin(hermes_home, http):
    """The plugin module as Hermes imported it from the home's plugin directory."""
    from providers import get_provider_profile

    profile = get_provider_profile("modelrelay")
    assert profile is not None, "Hermes did not discover the modelrelay plugin"
    module = sys.modules[type(profile).__module__]
    # The module was imported from the copy in this test's HERMES_HOME, not from the repo.
    assert Path(module.__file__).resolve().is_relative_to(hermes_home.resolve())
    return module
