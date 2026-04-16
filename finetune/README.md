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
5. Serve + eval locally   →  llama-server + eval_cli.py
6. Batch eval on Modal    →  eval_batch.py + eval_demo.py
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

## Step 2 — Train (Qwen3.5 + Unsloth)

Training runs on Modal with an A100-80GB GPU. The script loads both SQL and
places JSONL files from the run directory, applies the Qwen3.5 ChatML
template, and trains a LoRA adapter using Unsloth's
`train_on_responses_only` to mask non-assistant tokens.

```bash
# Default settings (Qwen3.5-0.8B, r=16, 1 epoch)
modal run finetune/train_modal_qwen35.py

# Override any config field from CLI
modal run finetune/train_modal_qwen35.py \
    --base-model unsloth/Qwen3.5-0.8B \
    --num-train-epochs 3 \
    --lora-r 32 \
    --max-seq-length 2048

# Quick smoke test
modal run finetune/train_modal_qwen35.py --max-train-samples 100
```

All CLI overrides: `--base-model`, `--experiment-name`, `--run-dir`,
`--num-train-epochs`, `--per-device-train-batch-size`, `--max-train-samples`,
`--max-eval-samples`, `--lora-r`, `--max-seq-length`. When `--lora-r` is
overridden, `lora_alpha` is automatically set to `2 * r`.

### Training config defaults (`Qwen35Config`)

```
base_model:       unsloth/Qwen3.5-0.8B
run_dir:          /mnt/gazet/data/v4-conversation-format
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

The experiment name is auto-generated as `{model}-r{lora_r}-{timestamp}`,
e.g. `Qwen3.5-0.8B-r16-20260415-161156`.

Training metrics are logged to [trackio](https://huggingface.co/spaces/srmsoumya/gazet-trackio).

### Merging a checkpoint manually

If you need to merge a specific intermediate checkpoint (not the final one):

```bash
modal run finetune/merge_checkpoint.py \
    --checkpoint /mnt/gazet/checkpoints/Qwen3.5-0.8B-r16-20260415-161156/checkpoint-1800
```

Output goes to `<checkpoint>-merged` by default, or use `--output` to override.

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

## Step 4 — Serve with llama-server

### Local

```bash
llama-server \
    -m finetune/models/qwen-base-v2/ckpt-001.gguf \
    -ngl 99 \
    --port 9000 \
    --ctx-size 2048
```

### Docker (CPU-only)

Useful for testing inference in a constrained environment. Adjust `--cpus`
and `--memory` to simulate deployment targets. Set `-t` to match `--cpus`.

```bash
docker run \
    --cpus="2" --memory="4g" \
    -v $(pwd)/finetune/models:/models \
    -p 9000:9000 \
    ghcr.io/ggml-org/llama.cpp:server \
        -m /models/qwen-base-v2/ckpt-001.gguf \
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

Three evaluation tools, each for a different stage of the workflow.

### Interactive eval via llama-server (`eval_cli.py`)

Sends individual test samples to a running llama-server and prints
expected vs generated output with an exact-match check. Useful for
quick spot-checking during development.

Requires llama-server running on port 9000 (see Step 4).

```bash
# SQL test samples (default task)
uv run finetune/eval_cli.py              # prompts for sample index
uv run finetune/eval_cli.py 0 5 12       # run specific samples

# Place extraction test samples
uv run finetune/eval_cli.py --task places 0 5

# Show the full prompt sent to the model
uv run finetune/eval_cli.py -v 0

# Use a different run directory
uv run finetune/eval_cli.py --run-dir dataset/output/runs/my-run 0
```

Config constants at the top of `eval_cli.py`: `SERVER_URL` (default
`http://localhost:9000`), `MAX_TOKENS` (2048), `TEMPERATURE` (0.6).

### Batch eval on Modal GPU (`eval_batch.py`)

Runs the full evaluation split through a model on Modal using Unsloth,
computes exact match rate, and saves detailed results to the Modal volume.
Uses the Qwen3.5 chat template with thinking disabled.

```bash
# Eval fine-tuned Qwen3.5 on SQL (default)
modal run finetune/eval_batch.py --label finetuned-qwen35

# Eval on place extraction
modal run finetune/eval_batch.py --task places --label finetuned-places

# Eval base model (no fine-tuning)
modal run finetune/eval_batch.py \
    --model-path unsloth/Qwen3.5-0.8B \
    --label base-qwen35

# Point to a specific merged checkpoint
modal run finetune/eval_batch.py \
    --model-path /mnt/gazet/checkpoints/Qwen3.5-0.8B-r16-20260415-161156/merged \
    --label my-checkpoint

# Limit samples for quick check
modal run finetune/eval_batch.py --max-samples 50

# Use a different split or run directory
modal run finetune/eval_batch.py --split test --run-dir /mnt/gazet/data/v4-conversation-format
```

All CLI args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--model-path` | Latest Qwen3.5 merged checkpoint | HF model ID or path on Modal volume |
| `--label` | `finetuned` | Label for the output file name |
| `--task` | `sql` | `sql` or `places` |
| `--split` | `val` | Which data split to evaluate (`val`, `test`) |
| `--run-dir` | `v4-conversation-format` | Run directory with `{task}/{split}.jsonl` files |
| `--max-samples` | all | Cap the number of samples |
| `--max-new-tokens` | 512 | Max tokens to generate per sample |
| `--batch-size` | 16 | Inference batch size |
| `--output-dir` | `/mnt/gazet/eval_results` | Where to write the JSON results file |

Results are saved to `/mnt/gazet/eval_results/eval-{label}-{task}.json` with
this structure:

```json
{
  "summary": {"label": "...", "task": "sql", "exact_match_rate": 0.85, ...},
  "results": [
    {"index": 0, "question": "...", "expected": "...", "predicted": "...", "exact_match": true},
    ...
  ]
}
```

### Visual eval (`eval_demo.py`)

Streamlit app that loads the JSON results from `eval_batch.py` and displays
them interactively. For SQL results, it shows formatted SQL side-by-side,
a diff view for mismatches, and executes both queries against DuckDB to
render the geometry on a map. For places results, it shows expected vs
predicted JSON.

```bash
streamlit run finetune/eval_demo.py
```

Set `GAZET_DATA_DIR` env var if your parquet data is not in the default
`data/` directory.

### Inspect a test prompt

Print the full chat messages for a sample, useful for pasting into the
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
| `train_modal_qwen35.py` | Modal training script — Qwen3.5 LoRA fine-tuning with Unsloth |
| `train_modal.py` | Modal training script — Gemma LoRA fine-tuning with SFTTrainer (legacy) |
| `config.py` | `TrainingConfig` dataclass for Gemma training (legacy) |
| `data.py` | Loads SQL + places JSONL from a run directory into a HF Dataset (Gemma format) |
| `prompts.py` | System/user prompt templates and schema text — used by `src/gazet/` |
| `check_token_lengths.py` | Modal script to analyze token length distribution before training |
| `merge_checkpoint.py` | Modal script — merge a LoRA checkpoint into the base model |
| `eval_cli.py` | Local interactive eval — sends samples to llama-server `/v1/chat/completions` |
| `eval_batch.py` | Modal batch eval — runs full split through Unsloth, computes exact match rate |
| `eval_demo.py` | Streamlit app — visual diff + map rendering of `eval_batch.py` results |
| `make_test_prompt.py` | Print chat messages for a test sample — paste into llama-server UI |
| `test_raw_completion.py` | Test with raw `/completion` endpoint (no chat template) |
| `nlg.py` | Standalone workbench (train/generate/execute) for local experimentation |
| `models/` | GGUF model files for local llama-server inference |

---

## Data format

The Qwen3.5 training pipeline (`train_modal_qwen35.py`) expects data in
**messages format** (v4-conversation-format):

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
time, `src/gazet/sql.py` replaces these with actual runtime paths before
executing against DuckDB.
