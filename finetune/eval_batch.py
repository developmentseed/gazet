"""Modal eval script: run a model on the test set and save results.

Usage
-----
# Eval finetuned model (uses raw prompt-completion format):
modal run finetune/eval_batch.py --label finetuned

# Eval base model (uses chat template so the model understands the instruction):
modal run finetune/eval_batch.py \
    --model-path google/gemma-3-270m-it \
    --label base \
    --use-chat-template

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
        "pandas>=2.2",
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
    samples: list[dict],
    output_path: str,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    use_chat_template: bool = False,
):
    """Run batched inference on all samples, save results to volume."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.prompts import SYSTEM_PROMPT, build_user_prompt, DEFAULT_SCHEMA_DETAILS

    print(f"Loading model [{label}]: {model_path}")
    print(f"Chat template: {use_chat_template}")
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

    # Build all prompts upfront
    prompts = []
    for sample in samples:
        user_content = build_user_prompt(
            question=sample["question"],
            candidates=sample["candidates"],
            schema_details=DEFAULT_SCHEMA_DETAILS,
        )
        if use_chat_template:
            messages = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = SYSTEM_PROMPT + "\n\n" + user_content
        prompts.append(prompt)

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
            all_predictions.append(postprocess_sql(generated))

        print(f"Batch {batch_idx+1}/{num_batches} done ({end}/{len(prompts)} samples)")

    # Build results
    results = []
    matches = 0
    for i, sample in enumerate(samples):
        expected = sample.get("target", {}).get("sql", "")
        predicted = all_predictions[i]
        is_match = predicted.strip() == expected.strip()
        if is_match:
            matches += 1

        results.append({
            "index": i,
            "question": sample["question"],
            "candidates": sample["candidates"],
            "expected_sql": expected,
            "predicted_sql": predicted,
            "exact_match": is_match,
        })

    total = len(results)
    exact_match_rate = matches / total if total else 0

    output = {
        "summary": {
            "label": label,
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
def read_test_data(test_jsonl: str) -> list[dict]:
    """Read test JSONL from the volume."""
    lines = []
    with open(test_jsonl) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    label: str = "finetuned",
    test_jsonl: str = "/mnt/gazet/data/output/test.jsonl",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    use_chat_template: bool = False,
    output_dir: str = "/mnt/gazet/eval_results",
):
    print(f"Model: {model_path}")
    print(f"Label: {label}")
    print(f"Chat template: {use_chat_template}")

    print("Loading test data...")
    samples = read_test_data.remote(test_jsonl)
    if max_samples:
        samples = samples[:max_samples]
    print(f"Eval samples: {len(samples)}")

    output_file = f"{output_dir}/eval-{label}.json"
    run_eval.remote(
        model_path, label, samples, output_file,
        max_new_tokens, batch_size, use_chat_template,
    )
