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
#     "duckdb==2.0.0",
#     "sqlglot==30.25.0",
#     "requests",
# ]
# ///
"""GRPO on top of an SFT run: reward = the generated SQL returns the reference result (execution match).

The model writes --num-generations answers per question; each is run on the data lake next to its reference and
scored 1 (lenient match, the metric of evaluate.py) or 0. GRPO raises the answers that beat their group's
average. Questions come from dataset/generative/rl_data.py (training tables only, reference confirmed by a
second model family); the lake and the harness code are downloaded from the dataset repo, and every query runs
in its own capped subprocess (harness.run_capped), so a crash or memory blow-up only fails that answer.

Writes <run>/adapter, <run>/merged, <run>/predictions/*.jsonl (greedy answers to the test prompts, scored
locally with score_run.sh) and <run>/log_history.json to the model repo.

hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN finetune/train_grpo_hf_jobs.py \
    --run-name rl-q3-06b --init-run sql-v2-q3-06b
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("GAZET_QUERY_RSS_MB", "2000")  # the job machine has far more RAM than the laptop VM

DATASET = "developmentseed/gazet-generative-sql"
MODEL_REPO = "developmentseed/gazet-sql-qwen35"
_ANS = "\x00ANSWER\x00"


def prompt_text(tokenizer, messages):
    """Training-format text up to the answer (same as the SFT script), so RL sees the SFT prompt exactly."""
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": _ANS}], tokenize=False)
    return full[:full.index(_ANS)]


def load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def setup_lake(rl_data):
    """Download the lake and the harness code into one folder (harness reads HERE/lake) and import it."""
    from huggingface_hub import snapshot_download
    snap = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["lake/*", f"{rl_data}/*", f"{rl_data}/**/*"]))
    work = pathlib.Path("/tmp/gen")
    shutil.copytree(snap / rl_data / "code", work, dirs_exist_ok=True)
    if not (work / "lake").exists():
        (work / "lake").symlink_to(snap / "lake")
    sys.path.insert(0, str(work))
    import duckdb
    duckdb.connect().execute("INSTALL spatial")
    import consistency
    import harness
    c = harness.con()
    print(f"lake: {len(harness.TABLE_DOCS)} documented tables, views ok: {c.execute('SELECT count(*) FROM countries').fetchone()}")
    return snap / rl_data, consistency


def generate(model, tokenizer, rows, batch_size, max_new_tokens):
    import torch
    tokenizer.padding_side = "left"
    texts = [prompt_text(tokenizer, r["messages"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    out = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        secs = (time.time() - t0) / len(idx)
        for i, ids in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = {"id": rows[i]["id"], "text": tokenizer.decode(ids, skip_special_tokens=True),
                      "out_tokens": int((ids != tokenizer.pad_token_id).sum()), "secs": round(secs, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--init-run", required=True, help="SFT run whose merged model RL starts from")
    ap.add_argument("--rl-data", default="rl/v1")
    ap.add_argument("--eval-data", default="chatml/v2")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one pass over the questions")
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--reward-workers", type=int, default=16)
    ap.add_argument("--max-eval-prompts", type=int)
    ap.add_argument("--seed", type=int, default=108)
    a = ap.parse_args()

    rl_dir, consistency = setup_lake(a.rl_data)
    from huggingface_hub import HfApi, snapshot_download
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    import evaluate  # shipped with the harness code: extract_sql, the same answer parsing as scoring

    set_seed(a.seed)
    init = pathlib.Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{a.init_run}/merged/*"])) / a.init_run / "merged"
    model, tokenizer = FastLanguageModel.from_pretrained(str(init), max_seq_length=a.max_seq_length, load_in_4bit=False,
                                                         fast_inference=False)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    import torch
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)} - {"lm_head"})
    model = FastLanguageModel.get_peft_model(model, r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0, target_modules=names,
                                             bias="none", random_state=a.seed, use_gradient_checkpointing="unsloth")

    rows = load_jsonl(rl_dir / "train.jsonl")
    ds = Dataset.from_list([{"prompt": prompt_text(tokenizer, r["messages"]), "ref_sql": r["sql"]} for r in rows])
    print(f"{len(ds):,} RL questions; example prompt tail:\n{ds[0]['prompt'][-300:]}")

    stats = {"calls": 0, "ok": 0}

    def execution_reward(prompts, completions, ref_sql, **kw):
        """1.0 when the answer's result matches the reference (lenient), else 0.0. Duplicates run once."""
        sqls = [evaluate.extract_sql(c if isinstance(c, str) else c[0]["content"]) for c in completions]
        uniq = sorted({(g, s) for g, s in zip(ref_sql, sqls)})
        with ThreadPoolExecutor(a.reward_workers) as ex:
            res = dict(zip(uniq, ex.map(lambda gs: consistency._compare_isolated({"sql": gs[0], "solver_sql": gs[1]}), uniq)))
        rewards = [1.0 if res[(g, s)].get("lenient") else 0.0 for g, s in zip(ref_sql, sqls)]
        stats["calls"] += len(rewards)
        stats["ok"] += sum(rewards)
        return rewards

    steps = a.max_steps or len(ds) // a.prompts_per_step
    want = dict(
        output_dir="/tmp/grpo", learning_rate=a.lr, beta=a.beta, temperature=a.temperature,
        num_generations=a.num_generations, per_device_train_batch_size=a.prompts_per_step * a.num_generations,
        generation_batch_size=a.prompts_per_step * a.num_generations, gradient_accumulation_steps=1,
        max_completion_length=a.max_completion_length, max_prompt_length=a.max_seq_length - a.max_completion_length,
        max_steps=steps, lr_scheduler_type="constant_with_warmup", warmup_steps=5, weight_decay=0.0,
        optim="adamw_8bit", bf16=True, max_grad_norm=0.2, logging_steps=1, save_strategy="no", report_to="none",
        mask_truncated_completions=True, loss_type="dapo", seed=a.seed, remove_unused_columns=False,
    )
    fields = set(GRPOConfig.__dataclass_fields__)
    print(f"GRPOConfig: dropped unsupported {sorted(set(want) - fields)}")
    args = GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[execution_reward],
                          args=args, train_dataset=ds)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    print(f"training took {minutes:.1f} min; mean reward {stats['ok'] / max(stats['calls'], 1):.3f} over {stats['calls']} answers")

    out = pathlib.Path("/tmp/run") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    FastLanguageModel.for_inference(model)
    ev = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=[f"{a.eval_data}/eval/*"])) / a.eval_data / "eval"
    (out / "predictions").mkdir(exist_ok=True)
    for f in sorted(ev.glob("*.jsonl")):
        rows = load_jsonl(f)[:a.max_eval_prompts]
        t1 = time.time()
        preds = generate(model, tokenizer, rows, batch_size=32, max_new_tokens=512)
        (out / "predictions" / f.name).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
        print(f"{f.name}: {len(preds)} answers in {time.time() - t1:.0f} s; first: {preds[0]['text'][:150]!r}")
    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    (out / "config.json").write_text(json.dumps(dict(vars(a), steps=steps, train_minutes=round(minutes, 1)), indent=1))
    HfApi().upload_folder(repo_id=MODEL_REPO, folder_path=str(out), path_in_repo=a.run_name,
                          commit_message=f"Add RL run {a.run_name}")
    print(f"uploaded to https://huggingface.co/{MODEL_REPO}/tree/main/{a.run_name}")


if __name__ == "__main__":
    main()
