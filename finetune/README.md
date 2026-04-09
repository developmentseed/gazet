# Fine-tuning and Inference

LoRA fine-tuning of a small language model (Gemma-3-270m-it) to perform
two geospatial tasks, then serving it locally via llama-server.

---

## End-to-end workflow

```
1. Generate dataset       →  dataset/  (see dataset/README.md)
2. Check token lengths    →  check_token_lengths.py
3. Train on Modal         →  train_modal.py
4. Convert to GGUF        →  llama.cpp
5. Serve + eval locally   →  llama-server + eval_cli.py
```

---

## Step 1 — Check token lengths

Before training, verify that your `max_length` setting covers the data.
SQL samples are long (schema + candidates + SQL), places samples are short.

```bash
modal run finetune/check_token_lengths.py \
    --run-dir /mnt/gazet/data/output/runs/v2-all-families
```

This prints per-split statistics (min, max, P95, P99) and recommends a
`max_length` value. Adjust `config.py` accordingly before training.

---

## Step 2 — Train

Training runs on Modal with an A100-80GB GPU. The script loads both SQL and
places JSONL files from the run directory, combines them into one dataset,
and trains a LoRA adapter with `completion_only_loss=True`.

```bash
# Default settings (gemma-3-270m-it, r=16, 3 epochs)
modal run finetune/train_modal.py

# Override any config field from CLI
modal run finetune/train_modal.py \
    --base-model google/gemma-3-1b-it \
    --run-dir /mnt/gazet/data/output/runs/v2-all-families \
    --num-train-epochs 5 \
    --lora-r 32 \
    --max-length 2048

# Quick smoke test
modal run finetune/train_modal.py --max-train-samples 100
```

Output on the Modal volume:

```
/mnt/gazet/checkpoints/{experiment_name}/
    adapter_config.json      # LoRA adapter
    adapter_model.safetensors
    merged/                  # full merged model (if merge_after_training=True)
        model.safetensors
        tokenizer.json
```

Training metrics are logged to [trackio](https://huggingface.co/spaces/srmsoumya/gazet-trackio).

---

## Step 3 — Convert merged model to GGUF

After training, download the merged model from Modal and convert to GGUF
for local inference with llama-server.

```bash
# Download from Modal volume
modal volume get gazet checkpoints/{experiment_name}/merged ./finetune/models/merged

# Convert to GGUF (requires llama.cpp repo)
python llama.cpp/convert_hf_to_gguf.py \
    finetune/models/merged \
    --outtype bf16 \
    --outfile finetune/models/gemma-270m-bf16.gguf

# Optional: quantize to Q8
llama.cpp/build/bin/llama-quantize \
    finetune/models/gemma-270m-bf16.gguf \
    finetune/models/gemma-270m-q8.gguf Q8_0
```

---

## Step 4 — Serve locally with llama-server

```bash
llama-server \
    -m finetune/models/gemma-270m-q8.gguf \
    -ngl 99 \
    --port 8080 \
    --log-disable
```

The server exposes `/v1/chat/completions` (chat API) on
`http://localhost:8080`. All eval scripts use this endpoint.

---

## Step 5 — Evaluate

### Interactive eval (local llama-server)

Run individual test samples through the local server:

```bash
# SQL test samples (default)
uv run finetune/eval_cli.py              # prompts for sample index
uv run finetune/eval_cli.py 0 5 12       # run specific samples

# Place extraction test samples
uv run finetune/eval_cli.py --task places 0 5

# Use a different run directory
uv run finetune/eval_cli.py --run-dir dataset/output/runs/my-run 0
```

### Batch eval (Modal GPU)

Run the full test set and compute exact match rate:

```bash
# Eval fine-tuned model on SQL
modal run finetune/eval_batch.py --label finetuned

# Eval fine-tuned model on place extraction
modal run finetune/eval_batch.py --task places --label finetuned-places

# Eval base model (no fine-tuning)
modal run finetune/eval_batch.py \
    --model-path google/gemma-3-270m-it \
    --label base

# Limit samples for quick check
modal run finetune/eval_batch.py --max-samples 50
```

Results are saved to `/mnt/gazet/eval_results/eval-{label}-{task}.json`.

### Visual eval (Streamlit)

Compare expected vs predicted SQL side-by-side on a map:

```bash
streamlit run finetune/eval_demo.py
```

### Inspect a test prompt

Print the full chat messages for a sample — useful for pasting into the
llama-server Chat UI:

```bash
uv run finetune/make_test_prompt.py             # SQL sample 0
uv run finetune/make_test_prompt.py 5           # SQL sample 5
uv run finetune/make_test_prompt.py --task places 3
```

---

## File reference

| File | What it does |
|---|---|
| `config.py` | `TrainingConfig` dataclass — model, LoRA, hyperparameters, run directory |
| `data.py` | Loads pre-formatted SQL + places JSONL from a run directory into a HF Dataset |
| `prompts.py` | System/user prompt templates and schema text — used by inference code in `src/gazet/` |
| `train_modal.py` | Modal training script — loads data, trains LoRA adapter with SFTTrainer |
| `check_token_lengths.py` | Modal script to analyze token length distribution before training |
| `eval_batch.py` | Modal script — batched inference on test set, computes exact match rate |
| `eval_cli.py` | Local interactive eval — sends test samples to llama-server via chat API |
| `eval_demo.py` | Streamlit app — visual diff of expected vs predicted SQL on a map |
| `make_test_prompt.py` | Print chat messages for a test sample — paste into llama-server UI |
| `nlg.py` | Standalone workbench (train/generate/execute) for local experimentation |
| `models/` | GGUF model files for local llama-server inference |

---

## Configuration defaults (`config.py`)

```
base_model:           google/gemma-3-270m-it
run_dir:              /mnt/gazet/data/output/runs/v2-all-families
lora_r:               16
lora_alpha:           16
num_train_epochs:     3
batch_size:           12 (x 2 gradient accumulation = 24 effective)
learning_rate:        1e-4
max_length:           2048
completion_only_loss: True
```

---

## Data format

The training data uses conversational prompt-completion format. SFTTrainer
applies the Gemma chat template internally and computes loss only on the
completion (assistant) tokens.

```json
{
  "prompt": [
    {"role": "system", "content": "You are a text to SQL query translator..."},
    {"role": "user",   "content": "GIVEN the <SCHEMA_DETAILS>..."}
  ],
  "completion": [
    {"role": "assistant", "content": "SELECT ST_AsGeoJSON(geometry) ..."}
  ]
}
```

SQL in the training data uses symbolic path placeholders
(`read_parquet('divisions_area')`) instead of real file paths. At inference
time, `src/gazet/sql.py` replaces these with actual runtime paths before
executing against DuckDB.
