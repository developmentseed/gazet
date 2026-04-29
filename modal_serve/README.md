<img src="../assets/gazet-logo.svg" alt="Gazet logo" width="64" />

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

## Updating

```bash
modal deploy modal_serve/serve.py
```

Model updates: re-upload to the `gazet` volume; running containers pick up the new file at next cold start.
