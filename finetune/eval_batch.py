"""Modal eval script: run a model on the test set and save results.

Usage
-----
# Eval fine-tuned model (sql task):
modal run finetune/eval_batch.py --label finetuned

# Eval base model:
modal run finetune/eval_batch.py \
    --model-path google/gemma-3-270m-it \
    --label base

# Eval places task:
modal run finetune/eval_batch.py --task places --label finetuned-places

# Limit samples:
modal run finetune/eval_batch.py --max-samples 50
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Optional

import modal

app = modal.App("gazet-nlg-eval")

infer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0",
        "torch>=2.4",
        "transformers>=4.46",
    )
    .add_local_python_source("finetune", copy=True)
    .env({"HF_HOME": "/mnt/gazet/model_cache"})
)

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}

DEFAULT_MODEL_PATH = "/mnt/gazet/checkpoints/gemma-3-270m-it-r16-20260331-134642/merged"
DEFAULT_RUN_DIR = "/mnt/gazet/data/output/runs/v3-symbolic-paths"


def postprocess_sql(text: str) -> str:
    cleaned = text.strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()


@app.function(
    image=infer_image,
    gpu="L40S",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)
def run_eval(
    model_path: str,
    label: str,
    task: str,
    samples: list[dict],
    output_path: str,
    max_new_tokens: int = 512,
    batch_size: int = 16,
):
    """Run batched inference on all samples, save results to volume.

    Uses raw prompt strings (no chat template) — matching training format.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model [{label}]: {model_path}")
    print(f"Task: {task}, Samples: {len(samples)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    # Build raw prompt strings — same format as training (no chat template)
    prompts = []
    for sample in samples:
        prompt_str = sample["prompt"][0]["content"] + "\n\n" + sample["prompt"][1]["content"]
        prompts.append(prompt_str)

    # Batched inference
    all_predictions = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch_prompts)):
            generated = tokenizer.decode(
                outputs[j][input_len:], skip_special_tokens=True
            )
            if task == "sql":
                generated = postprocess_sql(generated)
            else:
                generated = generated.strip()
            all_predictions.append(generated)

        print(f"Batch {batch_idx+1}/{num_batches} done ({end}/{len(prompts)} samples)")

    # Build results
    results = []
    matches = 0
    for i, sample in enumerate(samples):
        expected = sample["completion"][0]["content"]
        predicted = all_predictions[i]
        is_match = predicted.strip() == expected.strip()
        if is_match:
            matches += 1

        results.append({
            "index": i,
            "expected": expected,
            "predicted": predicted,
            "exact_match": is_match,
            "metadata": sample.get("metadata", {}),
        })

    total = len(results)
    exact_match_rate = matches / total if total else 0

    output = {
        "summary": {
            "label": label,
            "task": task,
            "model_path": model_path,
            "num_samples": total,
            "exact_matches": matches,
            "exact_match_rate": exact_match_rate,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    gazet_vol.commit()

    print(f"\n{'='*60}")
    print(f"[{label}] {matches}/{total} exact matches ({100*exact_match_rate:.1f}%)")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")


@app.function(
    image=infer_image,
    volumes=VOLUMES,
)
def read_test_data(run_dir: str, task: str) -> list[dict]:
    """Read test JSONL from the volume."""
    path = pathlib.Path(run_dir) / task / "test.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    label: str = "finetuned",
    task: str = "sql",
    run_dir: str = DEFAULT_RUN_DIR,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    output_dir: str = "/mnt/gazet/eval_results",
):
    print(f"Model:   {model_path}")
    print(f"Label:   {label}")
    print(f"Task:    {task}")
    print(f"Run dir: {run_dir}")

    print("Loading test data...")
    samples = read_test_data.remote(run_dir, task)
    if max_samples:
        samples = samples[:max_samples]
    print(f"Eval samples: {len(samples)}")

    output_file = f"{output_dir}/eval-{label}-{task}.json"
    run_eval.remote(
        model_path, label, task, samples, output_file,
        max_new_tokens, batch_size,
    )
