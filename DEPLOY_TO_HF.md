# Deploy to Hugging Face

This guide covers pushing the model and geodata to Hugging Face Hub and deploying to HF Spaces using the new `hf` CLI.

## Prerequisites

- Hugging Face account with write access to `developmentseed` organization
- HF CLI installed: `uvx hf` or `pip install hf` or `brew install hf`
- HF access token: `hf auth login`

```bash
# Install HF CLI (choose one)
pip install hf
brew install hf
uvx hf  # run without installing
```

---

## 1. Push Model to Hugging Face

The finetuned model should be converted to GGUF format before uploading.

### Convert to GGUF (if not already done)

```bash
# Download merged model from Modal
modal volume get gazet checkpoints/qwen35-v1/merged ./finetune/models/qwen35-v1-merged

# Convert to GGUF (requires llama.cpp repo)
uv run \
    --no-project \
    --with transformers \
    --with sentencepiece \
    --with protobuf \
    --with torch \
    python convert_hf_to_gguf.py \
    ./finetune/models/qwen35-v1-merged \
    --outtype bf16 \
    --outfile ./finetune/models/ckpt-bf16.gguf
```

# Quantize to 8-bits
```bash
llama-quantize ckpt-bf16.gguf ckpt-q8_0.gguf Q8_0
```

### Upload GGUF model

```bash
# Upload GGUF file (repo is created automatically if it doesn't exist)
hf upload developmentseed/gazet-model \
    ./finetune/models/ckpt-q8_0.gguf \
    /models/ckpt-q8_0.gguf \
    --commit-message <message>
```

```bash
hf upload developmentseed/gazet-model \
    finetune/models/qwen-finetune-v1/merged \
    merged \
    --commit-message <message>
```

### Upload Dataset (if needed)

```bash
# Upload dataset (repo is created automatically if it doesn't exist)
hf upload developmentseed/gazet-dataset \
    ./dataset/output/runs/v1 \
    . \
    --repo-type dataset \
    --commit-message <message>
```

The HF Spaces Dockerfile expects the model at `models/ckpt-q8_0.gguf`.

---

## 2. Push Geodata to Hugging Face

Upload only the normalized geodata copies. The app uses `gazet.config` which automatically prefers normalized data when present.

> **Important**: Only upload normalized data to HF. Don't upload the original `overture/` or `natural_earth_geoparquet/` directories. This keeps the HF repo small and reduces download time during Space builds.

### Prepare normalized data

Ensure normalized data exists locally:

```bash
data/overture_normalized/divisions_area/*.parquet
data/natural_earth_normalized/ne_geography.parquet
```

### Upload dataset

```bash
# Upload Overture normalized data (repo is created automatically if it doesn't exist)
hf upload developmentseed/gazet-geodata \
    ./data/overture_normalized \
    /overture_normalized \
    --repo-type dataset \
    --commit-message <message>

# Upload Natural Earth normalized data
hf upload developmentseed/gazet-geodata \
    ./data/natural_earth_normalized \
    /natural_earth_normalized \
    --repo-type dataset \
    --commit-message <message>
```

The HF repo will only contain:

```bash
gazet-geodata/
├── overture_normalized/
│   └── divisions_area/
│       └── *.parquet
└── natural_earth_normalized/
    └── ne_geography.parquet
```

This keeps the repo size minimal - only what's needed for deployment.

---

## 3. Deploy to Hugging Face Spaces

### Create Space

Create via the HF web UI: https://huggingface.co/new-space?sdk=docker

The Space will use the Dockerfile at the repo root.

### Push code to Space

```bash
# Add remote for the Space
git remote add space https://huggingface.co/spaces/developmentseed/gazet-space

# Push main branch
git push space main
```

### What the Dockerfile does

1. Copies llama-server binary and backend libraries
2. Installs Python dependencies via `uv`
3. Downloads model from `developmentseed/gazet-model` to `models/ckpt-q8_0.gguf`
4. Downloads geodata from `developmentseed/gazet-geodata` to `data/`
5. Runs supervisord to manage three processes:
   - `llama-server` on port 9000 (GGUF inference)
   - FastAPI `gazet.api:app` on port 8000 (SQL/geospatial API)
   - Streamlit demo on port 7860 (web UI)

### Environment variables

The Space is configured with:

```dockerfile
ENV GAZET_DATA_DIR=$HOME/app/data \
    LLAMA_SERVER_URL=http://localhost:9000 \
    GAZET_API_URL=http://localhost:8000
```

`GAZET_USE_NORMALIZED_DATA=1` by default, so the app will use the normalized copies.

---

## 4. Verify Deployment

Once the Space builds and starts:

1. Visit: https://huggingface.co/spaces/developmentseed/gazet-space
2. The Streamlit demo should load on port 7860
3. Try a query like "mountain ranges in Ecuador"

### Check logs

View logs in the HF Space UI.

---

## 5. Update Model or Data

### Update model

```bash
# Upload new GGUF
hf upload developmentseed/gazet-model \
    ./finetune/models/ckpt-q8_0.gguf \
    /models/ckpt-q8_0.gguf \
    --commit-message <message>
```

Then rebuild the Space (via HF UI or by pushing a commit).

### Update geodata

```bash
# Upload new normalized data
hf upload developmentseed/gazet-geodata \
    ./data/overture_normalized \
    /overture_normalized \
    --repo-type dataset \
    --commit-message <message>

hf upload developmentseed/gazet-geodata \
    ./data/natural_earth_normalized \
    /natural_earth_normalized \
    --repo-type dataset \
    --commit-message <message>
```

The Space will use the updated data on next rebuild.

---

## Local Testing with HF Artifacts

Test locally before deploying to ensure model and data work together:

```bash
# Download model from HF
hf download developmentseed/gazet-model \
    /models/ckpt-q8_0.gguf \
    --local-dir models

# Download geodata from HF (only normalized data is downloaded)
hf download developmentseed/gazet-geodata \
    --repo-type dataset \
    --local-dir data

# Run llama-server
llama-server -m models/ckpt-q8_0.gguf -ngl 99 --port 9000 --ctx-size 2048

# Run API
uv run uvicorn gazet.api:app --reload

# Run demo
uv run streamlit run gazet_demo.py
```

---

## Resources

- Model: https://huggingface.co/developmentseed/gazet-model
- Dataset: https://huggingface.co/datasets/developmentseed/gazet-geodata
- Space: https://huggingface.co/spaces/developmentseed/gazet-space
- HF CLI docs: https://huggingface.co/docs/huggingface_hub/en/guides/cli
