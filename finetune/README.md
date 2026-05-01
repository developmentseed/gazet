# Fine-tuning and Inference

LoRA fine-tuning of Qwen3.5-0.8B (via Unsloth) to perform two geospatial
tasks (text-to-SQL and place extraction), then serving locally via
llama-server.

---

## End-to-end workflow

```
1. Generate dataset       →  dataset/  (see dataset/README.md)
2. Check token lengths    →  check_token_lengths.py
3. Train on Modal         →  train_modal_qwen35.py
4. Convert to GGUF        →  llama.cpp
5. Serve locally          →  llama-server
6. Eval locally           →  eval_cli.py (interactive or batch) + eval_demo.py
```

---

## Step 1 — Check token lengths

Before training, verify that your `max_length` setting covers the data.
SQL samples are long (schema + candidates + SQL), places samples are short.

```bash
modal run finetune/check_token_lengths.py
modal run finetune/check_token_lengths.py --run-dir /mnt/gazet/data/smalltest-v1
```

This prints per-split statistics (min, max, P95, P99) and recommends a
`max_length` value. Adjust `--max-seq-length` in `train_modal_qwen35.py` accordingly.

---

## Step 2 — Train (Qwen3.5 + Unsloth)

Training runs on Modal with an A100-80GB GPU. The script loads both SQL and
places JSONL files from the run directory, applies the Qwen3.5 ChatML
template, and trains a LoRA adapter using Unsloth's
`train_on_responses_only` to mask non-assistant tokens.

```bash
# Default settings (Qwen3.5-0.8B, r=16, 1 epoch)
modal run finetune/train_modal_qwen35.py --experiment-name qwen35-v1

# Override any config field from CLI
modal run finetune/train_modal_qwen35.py \
    --experiment-name qwen35-v1 \
    --base-model unsloth/Qwen3.5-0.8B \
    --num-train-epochs 3 \
    --lora-r 32 \
    --max-seq-length 2048

# Quick smoke test
modal run finetune/train_modal_qwen35.py --experiment-name qwen35-v1 --max-train-samples 100
```

All CLI overrides: `--base-model`, `--experiment-name`, `--run-dir`,
`--num-train-epochs`, `--per-device-train-batch-size`, `--max-train-samples`,
`--max-eval-samples`, `--lora-r`, `--max-seq-length`. When `--lora-r` is
overridden, `lora_alpha` is automatically set to `2 * r`.

### Training config defaults (`Qwen35Config`)

```
base_model:       unsloth/Qwen3.5-0.8B
run_dir:          /mnt/gazet/data/v1   # override to your exported run, e.g. /mnt/gazet/data/smalltest-v1
lora_r:           16
lora_alpha:       32       (2 * r, Unsloth recommendation for Qwen)
lora_dropout:     0.0
num_train_epochs: 1
batch_size:       32 (x 1 gradient accumulation = 32 effective)
learning_rate:    1e-4
lr_scheduler:     linear
optim:            adamw_8bit
max_seq_length:   2048
```

### Output

Checkpoints and the merged model are saved to the Modal volume:

```
/mnt/gazet/checkpoints/{experiment_name}/
    adapter_config.json       # LoRA adapter
    adapter_model.safetensors
    checkpoint-*/             # intermediate checkpoints
    merged/                   # full merged 16-bit model
        model.safetensors
        tokenizer.json
```

Pass `--experiment-name` to set a human-readable name (e.g. `qwen35-v1`).
If omitted, it is auto-generated as `{model}-r{lora_r}-{timestamp}`.

Training metrics are logged to [trackio](https://huggingface.co/spaces/srmsoumya/gazet-trackio).

---

## Step 3 — Convert merged model to GGUF

After training, download the merged model from Modal and convert to GGUF
for local inference with llama-server.

```bash
# Download from Modal volume
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
```
llama-quantize ckpt-bf16.gguf ckpt-q8_0.gguf Q8_0
```

---

## Step 4 — Serve with llama-server

### Local

```bash
llama-server \
    -m finetune/models/ckpt-q8_0.gguf \
    -ngl 99 \
    --port 9000 \
    --ctx-size 2048
```

`--ctx-size` is the total KV cache shared across all parallel slots. SQL
prompts can be ~600 tokens; with `--parallel 4` and up to 2048 output
tokens, use at least `8192`. Match `--parallel` to `--workers` in
`eval_cli.py`.

### Docker (CPU-only)

Useful for testing inference in a constrained environment. Adjust `--cpus`
and `--memory` to simulate deployment targets. Set `-t` to match `--cpus`.

```bash
docker run \
    --cpus="2" --memory="4g" \
    -v $(pwd)/finetune/models:/models \
    -p 9000:9000 \
    ghcr.io/ggml-org/llama.cpp:server \
        -m /models/ckpt-q8_0.gguf \
        --port 9000 --host 0.0.0.0 \
        --ctx-size 2048 -t 2 -v
```

Notes:
- `--host 0.0.0.0` is required so the port forward from Docker works
- `-v` (verbose) enables per-request timing logs (prompt eval t/s, generation t/s)
- `-ngl` is omitted since the default Docker image is CPU-only; for GPU use
  the CUDA image (`ghcr.io/ggml-org/llama.cpp:server-cuda`) with `--gpus`
- The model is memory-mapped by default (`mmap = true`), so containers with
  less RAM than the model size may still start but will be slow due to page
  thrashing

The server exposes `/v1/chat/completions` (chat API) on
`http://localhost:9000`. All eval scripts use this endpoint.

---

## Step 5 — Evaluate

Two evaluation tools, both using a locally running llama-server.

### Interactive or batch eval (`eval_cli.py`)

Requires llama-server running on port 9000 (see Step 4).

**Interactive** — spot-check individual samples:

```bash
uv run finetune/eval_cli.py              # prompts for sample index
uv run finetune/eval_cli.py 0 5 12       # run specific samples
uv run finetune/eval_cli.py --task places 0 5
uv run finetune/eval_cli.py -v 0         # print full prompt
```

**Batch** — run the full split and save a JSON results file:

```bash
# Full val set, SQL task
uv run finetune/eval_cli.py --all --label finetuned-qwen35

# Places task
uv run finetune/eval_cli.py --all --task places --label finetuned-places

# Limit samples, custom output path
uv run finetune/eval_cli.py --all --max-samples 100 --output results/eval-v5.json

# Evaluate test split instead of val
uv run finetune/eval_cli.py --all --split test --label finetuned-qwen35
```

All batch CLI args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--all` | off | Enable batch mode |
| `--label` | `local-gguf` | Label used in the output filename |
| `--task` | `sql` | `sql` or `places` |
| `--split` | `val` | Data split to evaluate (`val`, `test`) |
| `--run-dir` | `dataset/output/runs/v1` | Directory with `{task}/{split}.jsonl`; override to your exported run, e.g. `dataset/output/runs/smalltest-v1` |
| `--max-samples` | all | Cap the number of samples |
| `--output` | `eval-{label}-{task}.json` | Output JSON path |
| `--workers` | `4` | Concurrent requests; match llama-server `--parallel` |

Results are saved to `results/eval-{label}-{task}.json` with this structure:

```json
{
  "summary": {"label": "...", "task": "sql", "exact_match_rate": 0.85, ...},
  "results": [
    {"index": 0, "question": "...", "expected": "...", "predicted": "...", "exact_match": true},
    ...
  ]
}
```

Config constants at the top of `eval_cli.py`: `SERVER_URL` (default
`http://localhost:9000`), `MAX_TOKENS` (2048), `TEMPERATURE` (0.6).

### Visual eval (`eval_demo.py`)

Streamlit app that loads JSON results from `eval_cli.py --all` and displays
them interactively. For SQL results, it shows formatted SQL side-by-side,
a diff view for mismatches, and executes both queries against DuckDB to
render the geometry on a map. For places results, it shows expected vs
predicted JSON.

```bash
streamlit run finetune/eval_demo.py
```

Reads result files from `results/eval-*.json` by default. Override with:

```bash
GAZET_EVAL_DIR=/path/to/results streamlit run finetune/eval_demo.py
```

Set `GAZET_DATA_DIR` if your parquet data is not in the default `data/` directory.
This only affects the visual SQL viewer (`eval_demo.py`), which executes SQL
against DuckDB; `eval_cli.py` does not read parquet files directly.

The eval viewer resolves parquet paths through `gazet.config`, which now
prefers normalized copies automatically when present:

- `data/overture_normalized/divisions_area/*.parquet`
- `data/natural_earth_normalized/ne_geography.parquet`

Disable that fallback only if needed with:

```bash
GAZET_USE_NORMALIZED_DATA=0 streamlit run finetune/eval_demo.py
```

---

## File reference

| File | What it does |
|---|---|
| `train_modal_qwen35.py` | Modal training script — Qwen3.5 LoRA fine-tuning with Unsloth |
| `check_token_lengths.py` | Modal script to analyze token length distribution before training |
| `eval_cli.py` | Local eval — interactive spot-check or full batch mode via llama-server |
| `eval_demo.py` | Streamlit app — visual diff + map rendering of `eval_cli.py --all` results |
| `models/` | GGUF model files for local llama-server inference |

---

## Data format

The Qwen3.5 training pipeline (`train_modal_qwen35.py`) expects data in
**messages format**:

```json
{
  "messages": [
    {"role": "system", "content": "You are a text to SQL query translator..."},
    {"role": "user",   "content": "GIVEN the <SCHEMA_DETAILS>..."},
    {"role": "assistant", "content": "SELECT ST_AsGeoJSON(geometry) ..."}
  ]
}
```

The Qwen3.5 chat template (ChatML) is applied by the tokenizer. Unsloth's
`train_on_responses_only` then masks everything before the assistant
response marker (`<|im_start|>assistant\n<think>\n\n</think>\n\n`), so
loss is computed only on the completion tokens.

SQL in the training data uses symbolic path placeholders
(`read_parquet('divisions_area')`) instead of real file paths. At inference
and eval time, `src/gazet/sql.py` / `finetune/eval_demo.py` replace these with
actual runtime paths before executing against DuckDB. When normalized parquet
copies are present, `gazet.config` prefers:

- `data/overture_normalized/divisions_area/*.parquet`
- `data/natural_earth_normalized/ne_geography.parquet`
