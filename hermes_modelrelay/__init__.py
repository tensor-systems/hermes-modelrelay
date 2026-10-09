"""ModelRelay (modelrelay.ai) provider profile: OpenAI-compatible relay with a published catalog.

ModelRelay's ``GET /v1/models`` lists every model with ``architecture.input_modalities``,
``context_length`` and ``supported_parameters``. models.dev does not know the relay, so without
this plugin Hermes treats every ModelRelay model as capability-unknown and swaps attached images
for a ``vision_analyze`` description. This profile serves that catalog through
``ProviderProfile.model_capabilities`` (the plugin seam ``agent.models_dev`` reads), translated to
the canonical ``model_overrides`` schema: ``supports_vision``, ``supports_tools``,
``supports_reasoning``, ``context_window``. A field the catalog does not publish is left out, never
guessed.

The capability map is lazy: nothing touches the network at import. The first lookup in a process
loads the disk mirror (``$HERMES_HOME/cache/modelrelay_models.json``) or, when there is none,
fetches the catalog once (short timeout; a failed fetch is retried at most once a minute). A stale
mirror is served while a background refresh runs. If the catalog cannot be loaded, lookups return
nothing and a warning says so: Hermes then treats the model as unknown, which for an attached image
means its text path (a ``vision_analyze`` description) — the plugin never claims a capability it
has not read from ModelRelay.

Installs as ``$HERMES_HOME/plugins/model-providers/modelrelay/`` (copy this directory) or as a pip
package exposing the ``hermes_agent.plugins`` entry point ``modelrelay``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

MODELRELAY_DEFAULT_BASE_URL = "https://api.modelrelay.ai/v1"
API_KEY_ENV_VARS = ("MODELRELAY_API_KEY",)
BASE_URL_ENV_VAR = "MODELRELAY_BASE_URL"

#: Age after which a loaded catalog is refreshed in the background (it keeps being served meanwhile).
CATALOG_TTL_SECONDS = 6 * 3600
#: Minimum gap between two catalog fetches after a failure, so a down relay is not hammered per turn.
RETRY_AFTER_FAILURE_SECONDS = 60.0
FETCH_TIMEOUT_SECONDS = 5.0
_CACHE_FILENAME = "modelrelay_models.json"

_lock = threading.Lock()


class _CatalogState:
    """Capability map + bookkeeping for one Hermes home (the catalog is key-scoped, the mirror per home)."""

    __slots__ = ("caps", "loaded_at", "last_attempt", "disk_checked", "refreshing")

    def __init__(self) -> None:
        self.caps: Optional[dict[str, dict[str, Any]]] = None
        self.loaded_at = 0.0
        self.last_attempt: Optional[float] = None
        self.disk_checked = False
        self.refreshing = False


_states: dict[str, _CatalogState] = {}


def _home() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return None


def _state() -> _CatalogState:
    home = _home()
    key = str(home) if home is not None else ""
    with _lock:
        return _states.setdefault(key, _CatalogState())


def _env(var: str) -> str:
    """Profile ``.env`` first (scope-aware), plain ``os.environ`` as the fallback."""
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv as prefer_dotenv
    except Exception:
        prefer_dotenv = None
    for resolve in filter(None, (prefer_dotenv, os.environ.get)):
        try:
            value = str(resolve(var) or "").strip()
        except Exception:
            value = ""
        if value:
            return value
    return ""


def _base_url() -> str:
    return (_env(BASE_URL_ENV_VAR) or MODELRELAY_DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    return next((v for v in map(_env, API_KEY_ENV_VARS) if v), "")


def _user_agent() -> str:
    # ModelRelay's edge rejects urllib's default ``Python-urllib/x.y`` UA with 403.
    try:
        from providers.base import _profile_user_agent

        return _profile_user_agent()
    except Exception:
        return "hermes-cli"


# ── catalog → capabilities ──────────────────────────────────────────────────────────────────


def capabilities_from_item(item: Any) -> Optional[tuple[str, dict[str, Any]]]:
    """One ``/v1/models`` row -> ``(model id, canonical capability dict)``; None when it has no id.

    Only what the row publishes is declared: a row without ``architecture.input_modalities``
    declares no vision verdict, one without ``supported_parameters`` no tools/reasoning verdict.
    """
    if not isinstance(item, dict):
        return None
    mid = str(item.get("id") or "").strip()
    if not mid:
        return None
    caps: dict[str, Any] = {}
    arch = item.get("architecture")
    inputs = arch.get("input_modalities") if isinstance(arch, dict) else None
    if isinstance(inputs, list):
        caps["supports_vision"] = "image" in inputs
    params = item.get("supported_parameters")
    if isinstance(params, list):
        caps["supports_tools"] = "tools" in params
        caps["supports_reasoning"] = "reasoning" in params
    ctx = item.get("context_length")
    if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
        caps["context_window"] = ctx
    return mid, caps


def parse_catalog(items: Any) -> Optional[dict[str, dict[str, Any]]]:
    """A ``/v1/models`` ``data`` array -> ``{model id: capabilities}``; None when unusable."""
    if not isinstance(items, list):
        return None
    parsed = dict(filter(None, map(capabilities_from_item, items)))
    return parsed or None


def _fetch_catalog_items(*, api_key: str = "", base_url: str = "", timeout: float = FETCH_TIMEOUT_SECONDS) -> list:
    """GET ``{base}/models`` and return its ``data`` array. Raises on any failure."""
    import urllib.request

    from hermes_cli.urllib_security import open_credentialed_url

    req = urllib.request.Request((base_url or _base_url()).rstrip("/") + "/models")
    key = api_key or _api_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _user_agent())
    with open_credentialed_url(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    items = data if isinstance(data, list) else data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("response has no model list")
    return items


# ── disk mirror ─────────────────────────────────────────────────────────────────────────────


def _disk_path() -> Optional[Path]:
    home = _home()
    return home / "cache" / _CACHE_FILENAME if home is not None else None


def _save_disk(caps: dict[str, dict[str, Any]], ts: float) -> None:
    path = _disk_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")  # write-then-rename: readers never see a torn file
        tmp.write_text(json.dumps({"ts": ts, "base_url": _base_url(), "models": caps}), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("modelrelay: catalog mirror write failed: %s", exc)


def _load_disk() -> tuple[Optional[dict[str, dict[str, Any]]], float]:
    """Disk mirror -> (capability map or None, its fetch time). A mirror of another base URL is ignored."""
    path = _disk_path()
    if path is None or not path.is_file():
        return None, 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("base_url") != _base_url():
            return None, 0.0
        models = data.get("models")
        if not isinstance(models, dict) or not models:
            return None, 0.0
        caps = {str(mid): dict(c) for mid, c in models.items() if isinstance(c, dict)}
        return (caps or None), float(data.get("ts") or 0.0)
    except Exception as exc:
        logger.debug("modelrelay: catalog mirror unreadable: %s", exc)
        return None, 0.0


# ── loading ─────────────────────────────────────────────────────────────────────────────────


def _store(state: _CatalogState, caps: dict[str, dict[str, Any]], ts: float, *, persist: bool) -> None:
    with _lock:
        state.caps, state.loaded_at = caps, ts
    if persist:
        _save_disk(caps, ts)


def seed_from_items(items: Any) -> Optional[dict[str, dict[str, Any]]]:
    """Seed the current home's capability map (memory + mirror) from a ``/v1/models`` payload."""
    caps = parse_catalog(items)
    if caps is not None:
        _store(_state(), caps, time.time(), persist=True)
    return caps


def _fetch_into(state: _CatalogState) -> bool:
    """One catalog fetch into *state*; logs a warning saying what Hermes does without it."""
    with _lock:
        state.last_attempt = time.monotonic()
    url = _base_url() + "/models"
    try:
        caps = parse_catalog(_fetch_catalog_items())
        if caps is None:
            raise ValueError("catalog lists no models")
    except Exception as exc:
        logger.warning(
            "modelrelay: could not load model capabilities from %s (%s). Until it loads, Hermes treats "
            "ModelRelay models as capability-unknown: an attached image goes to its text path "
            "(vision_analyze description), not to the model. Retrying in %ds.",
            url, exc, int(RETRY_AFTER_FAILURE_SECONDS),
        )
        return False
    _store(state, caps, time.time(), persist=True)
    return True


def _spawn_refresh(state: _CatalogState) -> None:
    """Refresh a stale map in the background, one refresh at a time per home."""
    with _lock:
        if state.refreshing:
            return
        state.refreshing = True

    def run() -> None:
        try:
            _fetch_into(state)
        finally:
            with _lock:
                state.refreshing = False

    try:
        # copy_context: the home override / secret scope are ContextVars; a bare thread would
        # resolve the launch profile's key and mirror into its cache dir.
        threading.Thread(target=contextvars.copy_context().run, args=(run,),
                         name="modelrelay-catalog-refresh", daemon=True).start()
    except Exception as exc:
        with _lock:
            state.refreshing = False
        logger.debug("modelrelay: background refresh failed to start: %s", exc)


def current_capabilities() -> Optional[dict[str, dict[str, Any]]]:
    """The capability map for the bound home, loading it on first use; None when unavailable."""
    state = _state()
    if state.caps is None and not state.disk_checked:
        state.disk_checked = True
        caps, ts = _load_disk()
        if caps is not None:
            _store(state, caps, ts, persist=False)
    if state.caps is None:
        last = state.last_attempt
        if last is None or time.monotonic() - last >= RETRY_AFTER_FAILURE_SECONDS:
            _fetch_into(state)
        return state.caps
    if time.time() - state.loaded_at >= CATALOG_TTL_SECONDS:
        last = state.last_attempt
        if last is None or time.monotonic() - last >= RETRY_AFTER_FAILURE_SECONDS:
            _spawn_refresh(state)
    return state.caps


class LiveModelCapabilities(Mapping):
    """``ProviderProfile.model_capabilities`` backed by ModelRelay's live catalog (see module docstring).

    Hermes reads it as ``profile.model_capabilities.get(model, {})``; a model the catalog does not
    list — or any model while the catalog is unavailable — yields the default (no declaration).
    """

    def __getitem__(self, model: str) -> dict[str, Any]:
        caps = current_capabilities()
        if caps is None or model not in caps:
            raise KeyError(model)
        return dict(caps[model])

    def __iter__(self) -> Iterator[str]:
        return iter(list(current_capabilities() or ()))

    def __len__(self) -> int:
        return len(current_capabilities() or ())

    def __repr__(self) -> str:
        state = _state()
        return f"<LiveModelCapabilities {'unloaded' if state.caps is None else f'{len(state.caps)} models'}>"

    # Profiles are dataclasses; ``copy``/``deepcopy``/``asdict`` must not clone (or choke on) live state.
    def __copy__(self) -> "LiveModelCapabilities":
        return self

    def __deepcopy__(self, memo: dict) -> "LiveModelCapabilities":
        return self


class ModelRelayProfile(ProviderProfile):
    """ModelRelay — OpenAI-compatible relay; per-model capabilities from its ``/v1/models``."""

    def fetch_models(
        self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 8.0
    ) -> Optional[list[str]]:
        """Live catalog for the picker; the same payload seeds the capability map."""
        try:
            items = _fetch_catalog_items(api_key=api_key or "", base_url=base_url or "", timeout=timeout)
        except Exception as exc:
            logger.debug("modelrelay: fetch_models failed: %s", exc)
            return None
        if not base_url or base_url.rstrip("/") == _base_url():
            seed_from_items(items)
        try:
            from hermes_cli.chat_catalog import chat_catalog_ids

            ids = chat_catalog_ids(items)
        except Exception:
            ids = [str(i["id"]) for i in items if isinstance(i, dict) and i.get("id")]
        return list(dict.fromkeys(ids)) or None


modelrelay = ModelRelayProfile(
    name="modelrelay", aliases=("model-relay", "modelrelay.ai"), display_name="ModelRelay",
    description="ModelRelay — one OpenAI-compatible API for many models",
    signup_url="https://modelrelay.ai/",
    env_vars=(*API_KEY_ENV_VARS, BASE_URL_ENV_VAR), base_url=_base_url(), auth_type="api_key",
    fallback_models=(),  # the picker uses fetch_models()
    model_capabilities=LiveModelCapabilities(),
)

register_provider(modelrelay)
