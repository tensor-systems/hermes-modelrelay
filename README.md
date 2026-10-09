# hermes-modelrelay

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin for
[ModelRelay](https://modelrelay.ai). It tells Hermes what each ModelRelay model can do:
whether it takes images, whether it supports tools and reasoning, and its context window.
It reads all of this from ModelRelay's own `GET /v1/models` catalog. Hermes source is not changed.

## Why

models.dev has no entry for ModelRelay. When ModelRelay is set up as a custom provider
(`provider: custom:modelrelay`), Hermes has no capability data for its models. When an image
arrives, Hermes then replaces it with a `vision_analyze` text description, even for a model
like `glm-5.3-flash` that takes images natively. This plugin fills
`ProviderProfile.model_capabilities`, which is the seam `agent/models_dev.py` reads. Image routing,
context-length lookup, picker badges and `/api/model/info` all read through that seam.

## How capabilities are sourced

They are fetched live, not kept in a static table. `model_capabilities` is a lazy mapping:

- Importing the plugin makes no network call.
- On first lookup in a process, the plugin reads the mirror at `$HERMES_HOME/cache/modelrelay_models.json`.
  If there is no mirror, it fetches `/v1/models` once, with a 5 s timeout.
- A catalog older than 6 h is still served while a background thread refreshes it.
- The model picker's `fetch_models()` also seeds the cache.
- Each catalog row is mapped to Hermes' canonical schema:
  - `supports_vision`: `"image"` is in `architecture.input_modalities`
  - `supports_tools`: `"tools"` is in `supported_parameters`
  - `supports_reasoning`: `"reasoning"` is in `supported_parameters`
  - `context_window`: `context_length`

  A field the row does not publish is left out, never guessed.

If the catalog can't be loaded, the plugin declares nothing and logs a warning. It retries at most
once a minute. Hermes then treats the model's capabilities as unknown. For an attached image that
means the text path (a `vision_analyze` description), exactly as without the plugin. An explicit
`providers.modelrelay.models.<model>.supports_vision` in `config.yaml` still overrides the catalog.

## Install

Pick one of these two methods.

**Directory plugin** (per Hermes profile, no pip):

```sh
mkdir -p "$HERMES_HOME/plugins/model-providers"
cp -R hermes_modelrelay "$HERMES_HOME/plugins/model-providers/modelrelay"
```

**pip** (into the Python environment Hermes runs from):

```sh
pip install /path/to/hermes-modelrelay      # or: uv pip install --python <hermes venv python> ...
```

Pip-installed provider plugins load only when they are opted in. Add this to `config.yaml`:

```yaml
plugins:
  enabled: [modelrelay]
```

If both methods are installed, the directory plugin wins.

## Configure

Put the key in the profile's `.env` (or in the environment):

```sh
MODELRELAY_API_KEY=mr_...
# optional: MODELRELAY_BASE_URL=https://api.modelrelay.ai/v1
```

Then select the provider by its plugin name in `config.yaml`:

```yaml
model:
  provider: modelrelay
  default: glm-5.3-flash
```

Do **not** use `custom:modelrelay`. Do not set `model.base_url` either: the plugin supplies it.

### Coming from a `providers.modelrelay` custom entry

- `model.provider: modelrelay` resolves to the plugin, even when `config.yaml` still has a
  `providers.modelrelay:` custom entry. Hermes doesn't map a name onto a custom entry once a
  provider with that canonical name is registered. The custom entry's `api`, `key_cmd` and
  `transport` are then not used.
- `model.provider: custom:modelrelay` keeps using the custom entry. Hermes resolves the request to
  its generic `custom` provider, so the plugin is never consulted and images get the text path.
- Credentials come from `MODELRELAY_API_KEY`. The plugin path
  does not run `key_cmd`. Move the key into the profile `.env`.
- You can remove the `providers.modelrelay` block. If you keep it, its
  `models.<id>.supports_vision` entries still work as explicit overrides.

## Tests

Run them with the Python environment that has hermes-agent installed:

```sh
<hermes venv>/bin/python -m pytest
```

The tests mock HTTP and run against stock Hermes. Each one loads the plugin through Hermes' own
directory discovery in an isolated `HERMES_HOME`.
