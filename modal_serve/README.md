# Modal Deployment

Deploys Gazet to Modal as three independently scaled containers:

| Cls | Hardware | Role |
|---|---|---|
| `LlamaServer` | GPU T4 | `llama-server` from the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image |
| `Api` | CPU | `gazet.api` FastAPI app served as native ASGI |
| `Demo` | CPU | Streamlit UI from `gazet_demo.py` |

All three scale to zero independently. The Demo URL is the user-facing entrypoint.

## Prerequisites

- Modal account + CLI: `uv tool install modal` then `modal token new`
- Modal volumes already exist:
  - `gazet` (model checkpoints)
  - `gazet-data` (normalized parquet datasets)

## 1. Upload the Q4 GGUF model

```bash
modal volume put gazet \
    finetune/models/ckpt-q4_k_m.gguf \
    checkpoints/qwen35-fientune-v3/ckpt-q4_k_m.gguf
```

Verify:

```bash
modal volume ls gazet checkpoints/qwen35-fientune-v3 | grep ckpt-q4_k_m
```

## 2. Deploy

From the repo root:

```bash
modal deploy modal_serve/serve.py
```

The output prints three URLs:

- `LlamaServer.serve` -> internal, called by `Api`
- `Api.fastapi_app` -> public FastAPI (`/search`, `/search/stream`)
- `Demo.serve` -> public Streamlit UI

## 3. Smoke test

Streamlit:

```bash
open https://<workspace>--gazet-demo-serve.modal.run
```

API directly:

```bash
curl "https://<workspace>--gazet-api-fastapi-app.modal.run/search?q=Odisha"
```

Expected timings:

- First request after idle: ~25-30s (cold start cascade across all three)
- Warm requests within scaledown windows: ~1-3s

## 4. Set the budget cap

Modal dashboard -> Settings -> Billing:

- Workspace spending limit: **$50/mo**
- Email alerts at $25 / $40 / $50

Modal pauses new container starts when the limit is hit.

## Updating

Code or dependency changes:

```bash
modal deploy modal_serve/serve.py
```

Model updates: re-upload to the `gazet` volume; running containers pick up the new file at next cold start.

## Architecture notes

- **No supervisord.** Each Cls runs one logical service.
- **`@modal.asgi_app`** serves FastAPI natively, no uvicorn subprocess.
- **`@modal.web_server`** wraps non-ASGI processes (`llama-server` binary, Streamlit).
- **Cross-Cls URLs** resolved at runtime via `modal.Cls.from_name(...)`.
- **`scaledown_window`** tuned per tier: 120s GPU, 300s API, 600s Demo (UI sessions are sticky).

## Cost reference

| Traffic | Monthly estimate |
|---|---|
| 200 queries/day | ~$5-7 |
| 1000 queries/day | ~$18-25 |
| Idle | $0 |

T4 GPU at ~$0.59/hr is only billed during active inference + brief warmup. CPU containers are negligible.

## Troubleshooting

**Cold start fails on `LlamaServer`**: check the binary path in `modal_serve/serve.py`. The official image's binary is at `/app/llama-server`; if upstream changes, run `modal shell gazet::LlamaServer` and `which llama-server`.

**`Api` cannot reach `LlamaServer`**: confirm `modal.Cls.from_name(...).serve.web_url` returns a non-empty string. The first deploy registers URLs; redeploys keep them stable.

**Streamlit websocket errors**: `@modal.web_server` supports websockets natively; if a proxy issue appears, raise `startup_timeout` and check `modal logs gazet`.

**Model not found**: the path in `serve.py` is `/models/checkpoints/qwen35-fientune-v3/ckpt-q4_k_m.gguf`. Verify the volume layout matches.
