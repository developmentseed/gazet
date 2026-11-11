# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "unsloth==2026.11.4",
#     "unsloth_zoo",
#     "torch==2.14.1",
#     "transformers==5.2.0",
#     "flash-linear-attention",
#     "datasets",
#     "huggingface_hub",
# ]
# ///
"""Hugging Face Jobs training script for the generative SQL student: Qwen3.5 LoRA with Unsloth.

Same recipe as train_modal_qwen35.py, but the data comes from the private dataset
developmentseed/gazet-generative-sql (built by dataset/generative/export.py) and everything the run
produces goes to a private model repo:

    <run>/adapter/               LoRA adapter + tokenizer
    <run>/merged/                merged 16-bit model (for GGUF conversion)
    <run>/predictions/*.jsonl    answers to the gold and dev prompts, plain and hinted
    <run>/log_history.json       training and eval loss
    <run>/config.json

The job cannot score the answers (that needs the DuckDB data lake); score them locally with
    python dataset/generative/evaluate.py run <run> --preds <predictions file> --data gold|dev [--hints] --system minimal

Usage
-----
hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN finetune/train_hf_jobs_qwen35.py --run-name sql-v1-r16
hf jobs uv run --flavor a100-large --timeout 30m --secrets HF_TOKEN finetune/train_hf_jobs_qwen35.py --run-name smoke \
    --max-train-samples 256 --max-eval-prompts 16 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

DATASET = "developmentseed/gazet-generative-sql"
MODEL_REPO = "developmentseed/gazet-sql-qwen35"

# flash-linear-attention gives the fast kernels for Qwen3.5's linear-attention layers; without it transformers
# falls back to a slow torch implementation (about 2 s per generated answer on an A100).

# Qwen3.5 renders an empty think block before non-thinking answers; train only on tokens after it.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def generate(model, tokenizer, rows, batch_size, max_new_tokens):
    """Greedy answers to {"id", "messages"} prompts, batched by length."""
    import torch

    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    out = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        secs = (time.time() - t0) / len(idx)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, ids in zip(idx, new):
            n = int((ids != tokenizer.pad_token_id).sum())
            out[i] = {"id": rows[i]["id"], "text": tokenizer.decode(ids, skip_special_tokens=True),
                      "out_tokens": n, "secs": round(secs, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default="chatml/v1")
    ap.add_argument("--base-model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-eval-prompts", type=int)
    ap.add_argument("--seed", type=int, default=108)
    a = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    from unsloth import FastLanguageModel  # before transformers/trl, so Unsloth can patch them
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(a.seed)
    data = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=[f"{a.data}/*", f"{a.data}/**/*"])) / a.data
    out = pathlib.Path("/tmp/run") / a.run_name
    out.mkdir(parents=True, exist_ok=True)

    model, processor = FastLanguageModel.from_pretrained(
        a.base_model, max_seq_length=a.max_seq_length, load_in_4bit=False,
        use_gradient_checkpointing="unsloth", fast_inference=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastLanguageModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        bias="none", random_state=a.seed, use_gradient_checkpointing=False,
    )

    def as_text(rows):
        return Dataset.from_list([{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)} for r in rows])

    train_rows = load_jsonl(data / "sql" / "train.jsonl")
    val_rows = load_jsonl(data / "sql" / "val.jsonl")
    train = as_text(train_rows).shuffle(seed=a.seed)
    if a.max_train_samples:
        train = train.select(range(min(a.max_train_samples, len(train))))
        val_rows = val_rows[:max(16, a.max_train_samples // 8)]
    val = as_text(val_rows)
    print(f"train {len(train):,}  val {len(val):,}  example:\n{train[0]['text'][-400:]}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train, eval_dataset=val,
        args=SFTConfig(
            output_dir=str(out / "checkpoints"), dataset_text_field="text", max_seq_length=a.max_seq_length,
            num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
            learning_rate=a.lr, lr_scheduler_type="linear", warmup_ratio=0.03, weight_decay=0.01,
            optim="adamw_8bit", bf16=True, max_grad_norm=1.0, logging_steps=10,
            eval_strategy="steps", eval_steps=100, save_strategy="no", report_to="none",
            dataset_num_proc=8, seed=a.seed,
        ),
    )
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    print(f"training took {train_minutes:.1f} min")

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    FastLanguageModel.for_inference(model)
    (out / "predictions").mkdir(exist_ok=True)
    for f in sorted((data / "eval").glob("*.jsonl")):
        rows = load_jsonl(f)[:a.max_eval_prompts]
        t0 = time.time()
        preds = generate(model, tokenizer, rows, batch_size=64, max_new_tokens=512)
        (out / "predictions" / f.name).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
        print(f"{f.name}: {len(preds)} answers in {time.time() - t0:.0f} s; first: {preds[0]['text'][:200]!r}")

    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    (out / "config.json").write_text(json.dumps(dict(vars(a), train_rows=len(train), train_minutes=round(train_minutes, 1)), indent=1))

    api = HfApi()
    api.create_repo(MODEL_REPO, private=True, exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, folder_path=str(out), path_in_repo=a.run_name,
                      ignore_patterns=["checkpoints/*"], commit_message=f"Add run {a.run_name}")
    print(f"uploaded to https://huggingface.co/{MODEL_REPO}/tree/main/{a.run_name}")


if __name__ == "__main__":
    main()
