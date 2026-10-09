"""The ModelRelay plugin against stock Hermes: discovery, capability seam, caching, failure behaviour."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time

import pytest

from conftest import CATALOG, PLUGIN_DIR


# ── catalog parsing ─────────────────────────────────────────────────────────────────────────


def test_parse_catalog_declares_only_published_fields(plugin):
    caps = plugin.parse_catalog(CATALOG["data"])
    assert caps["glm-5.3-flash"] == {
        "supports_vision": True, "supports_tools": True, "supports_reasoning": True, "context_window": 1048576,
    }
    assert caps["muse-spark-1.3"]["supports_vision"] is False
    assert caps["whisper-large-v3-turbo"] == {
        "supports_vision": False, "supports_tools": False, "supports_reasoning": False,
    }
    # A row that publishes nothing declares nothing: no guessed capabilities.
    assert caps["bare-model"] == {}


def test_parse_catalog_rejects_unusable_payloads(plugin):
    assert plugin.parse_catalog(None) is None
    assert plugin.parse_catalog([]) is None
    assert plugin.parse_catalog([{"no": "id"}, "junk"]) is None


# ── import / discovery ──────────────────────────────────────────────────────────────────────


def test_import_makes_no_network_call(plugin, http):
    assert http.requests == []
    assert repr(plugin.modelrelay.model_capabilities) == "<LiveModelCapabilities unloaded>"


def test_profile_shape(plugin):
    profile = plugin.modelrelay
    assert profile.base_url == "https://api.modelrelay.ai/v1"
    assert profile.api_mode == "chat_completions"
    assert "MODELRELAY_API_KEY" in profile.env_vars
    # Vision is per model, never claimed provider-wide.
    assert profile.supports_vision is False


def test_auth_registry_knows_the_plugin(plugin):
    from hermes_cli.auth import resolve_provider

    assert resolve_provider("modelrelay") == "modelrelay"


# ── the capability seam Hermes reads ────────────────────────────────────────────────────────


def test_models_dev_capabilities_come_from_catalog(plugin, http):
    from agent.models_dev import get_model_capabilities

    glm = get_model_capabilities("modelrelay", "glm-5.3-flash")
    muse = get_model_capabilities("modelrelay", "muse-spark-1.3")
    assert glm.supports_vision is True and glm.supports_tools is True and glm.context_window == 1048576
    assert muse.supports_vision is False
    assert len(http.requests) == 1  # one fetch serves every lookup


def test_image_routing_verdicts(plugin, http):
    from agent.image_routing import _lookup_supports_vision, decide_image_input_mode

    assert _lookup_supports_vision("modelrelay", "glm-5.3-flash", {}) is True
    assert _lookup_supports_vision("modelrelay", "muse-spark-1.3", {}) is False
    assert decide_image_input_mode("modelrelay", "glm-5.3-flash", {}) == "native"
    assert decide_image_input_mode("modelrelay", "muse-spark-1.3", {}) == "text"


def test_explicit_user_override_still_wins(plugin, http):
    from agent.image_routing import _lookup_supports_vision

    cfg = {"providers": {"modelrelay": {"models": {"muse-spark-1.3": {"supports_vision": True}}}}}
    assert _lookup_supports_vision("modelrelay", "muse-spark-1.3", cfg) is True


def test_unlisted_model_is_unknown(plugin, http):
    from agent.models_dev import get_model_capabilities

    assert plugin.modelrelay.model_capabilities.get("not-a-model", {}) == {}
    assert get_model_capabilities("modelrelay", "not-a-model") is None


def test_request_sends_key_and_non_default_user_agent(plugin, http, monkeypatch):
    monkeypatch.setenv("MODELRELAY_API_KEY", "mr-test-key")
    plugin.modelrelay.model_capabilities.get("glm-5.3-flash")
    req = http.requests[0]
    assert req.full_url == "https://api.modelrelay.ai/v1/models"
    assert req.get_header("Authorization") == "Bearer mr-test-key"
    assert not req.get_header("User-agent", "").startswith("Python-urllib")


def test_base_url_env_override(plugin, http, monkeypatch):
    monkeypatch.setenv("MODELRELAY_BASE_URL", "https://relay.example.test/v1/")
    plugin.modelrelay.model_capabilities.get("glm-5.3-flash")
    assert http.requests[0].full_url == "https://relay.example.test/v1/models"


# ── failure: no guessing, say what happens, back off ────────────────────────────────────────


def test_fetch_failure_declares_nothing_and_warns(plugin, http, caplog):
    from agent.image_routing import decide_image_input_mode
    from agent.models_dev import get_model_capabilities

    http.error = OSError("connection refused")
    with caplog.at_level(logging.WARNING):
        assert plugin.modelrelay.model_capabilities.get("glm-5.3-flash", {}) == {}
    assert "could not load model capabilities" in caplog.text
    assert "vision_analyze" in caplog.text
    assert get_model_capabilities("modelrelay", "glm-5.3-flash") is None
    assert decide_image_input_mode("modelrelay", "glm-5.3-flash", {}) == "text"


def test_fetch_failure_backs_off_then_retries(plugin, http, monkeypatch):
    http.error = OSError("down")
    caps = plugin.modelrelay.model_capabilities
    caps.get("glm-5.3-flash")
    caps.get("glm-5.3-flash")
    assert len(http.requests) == 1  # no refetch inside the back-off window

    http.error = None
    monkeypatch.setattr(plugin, "RETRY_AFTER_FAILURE_SECONDS", 0.0)
    assert caps["glm-5.3-flash"]["supports_vision"] is True
    assert len(http.requests) == 2


def test_malformed_payload_is_a_failure(plugin, http):
    http.payload = {"error": "nope"}
    assert plugin.modelrelay.model_capabilities.get("glm-5.3-flash", {}) == {}


# ── caching ─────────────────────────────────────────────────────────────────────────────────


def test_disk_mirror_serves_a_new_process_without_network(plugin, http, hermes_home):
    plugin.modelrelay.model_capabilities.get("glm-5.3-flash")
    mirror = hermes_home / "cache" / "modelrelay_models.json"
    assert json.loads(mirror.read_text())["models"]["glm-5.3-flash"]["supports_vision"] is True

    plugin._states.clear()  # what a restart looks like to the plugin
    http.error = OSError("offline")
    assert plugin.modelrelay.model_capabilities["glm-5.3-flash"]["supports_vision"] is True
    assert len(http.requests) == 1


def test_mirror_for_another_base_url_is_ignored(plugin, http, hermes_home, monkeypatch):
    plugin.modelrelay.model_capabilities.get("glm-5.3-flash")
    plugin._states.clear()
    monkeypatch.setenv("MODELRELAY_BASE_URL", "https://other.example.test/v1")
    http.error = OSError("offline")
    assert plugin.modelrelay.model_capabilities.get("glm-5.3-flash", {}) == {}


def test_stale_catalog_is_served_while_refreshing(plugin, http, monkeypatch):
    spawned = []
    monkeypatch.setattr(plugin, "_spawn_refresh", lambda state: spawned.append(state))
    caps = plugin.modelrelay.model_capabilities
    caps.get("glm-5.3-flash")
    plugin._state().loaded_at = time.time() - plugin.CATALOG_TTL_SECONDS - 1
    plugin._state().last_attempt = None
    assert caps["glm-5.3-flash"]["supports_vision"] is True  # stale data still served, not blanked
    assert len(spawned) == 1 and len(http.requests) == 1


def test_background_refresh_updates_the_map(plugin, http):
    caps = plugin.modelrelay.model_capabilities
    caps.get("glm-5.3-flash")
    http.payload = {"data": [{"id": "new-model", "architecture": {"input_modalities": ["text", "image"]}}]}
    state = plugin._state()
    plugin._spawn_refresh(state)
    deadline = time.time() + 5
    while state.refreshing and time.time() < deadline:
        time.sleep(0.01)
    assert caps["new-model"]["supports_vision"] is True


def test_fetch_models_lists_ids_and_seeds_capabilities(plugin, http):
    ids = plugin.modelrelay.fetch_models(api_key="k")
    assert "glm-5.3-flash" in ids and "muse-spark-1.3" in ids
    assert plugin.modelrelay.model_capabilities["glm-5.3-flash"]["supports_vision"] is True
    assert len(http.requests) == 1


def test_fetch_models_failure_returns_none(plugin, http):
    http.error = OSError("down")
    assert plugin.modelrelay.fetch_models() is None


def test_profile_copy_shares_live_capabilities(plugin):
    import copy
    import dataclasses

    profile = plugin.modelrelay
    assert copy.deepcopy(profile).model_capabilities is profile.model_capabilities
    assert dataclasses.replace(profile).model_capabilities is profile.model_capabilities


# ── runtime resolution and the existing ``providers.modelrelay`` custom entry ───────────────


def test_runtime_resolves_to_plugin_with_env_key(plugin, hermes_home, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    (hermes_home / ".env").write_text("MODELRELAY_API_KEY=mr-from-dotenv\n")
    rt = resolve_runtime_provider(requested="modelrelay")
    assert rt["provider"] == "modelrelay"
    assert rt["base_url"] == "https://api.modelrelay.ai/v1"
    assert rt["api_key"] == "mr-from-dotenv"


def test_plugin_shadows_a_same_named_custom_entry(plugin, hermes_home, monkeypatch):
    """``model.provider: modelrelay`` goes to the plugin even when ``providers.modelrelay`` exists;
    ``custom:modelrelay`` still targets the custom entry (and so bypasses the plugin)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: modelrelay\n  default: glm-5.3-flash\n"
        "providers:\n  modelrelay:\n    api: https://custom.example.test/v1\n"
        "    api_key: custom-entry-key\n    transport: chat_completions\n"
    )
    monkeypatch.setenv("MODELRELAY_API_KEY", "mr-env-key")
    plugin_rt = resolve_runtime_provider(requested="modelrelay")
    assert (plugin_rt["provider"], plugin_rt["api_key"]) == ("modelrelay", "mr-env-key")

    custom_rt = resolve_runtime_provider(requested="custom:modelrelay")
    assert custom_rt["provider"] == "custom"
    assert custom_rt["base_url"] == "https://custom.example.test/v1"


# ── pip entry-point form ────────────────────────────────────────────────────────────────────


def test_package_module_registers_on_import(hermes_home, http, monkeypatch):
    """The entry point target is the bare package: importing it registers the profile."""
    import providers

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_SOURCES", dict(providers._SOURCES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    spec = importlib.util.spec_from_file_location("_hermes_modelrelay_ep_test", PLUGIN_DIR / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert providers._REGISTRY["modelrelay"] is module.modelrelay
    assert http.requests == []


def test_source_is_classified_as_model_provider():
    """PluginManager must route the entry point to provider discovery, not call it as a general plugin."""
    from hermes_cli.plugins_manifest import _detect_kind_from_source

    source = (PLUGIN_DIR / "__init__.py").read_text()[:8192]
    assert _detect_kind_from_source(source) == "model-provider"
