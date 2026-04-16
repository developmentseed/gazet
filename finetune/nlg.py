#!/usr/bin/env python3
"""
Text-to-SQL workbench for training, inference, and SQL execution.

Features
- Train an SFT/LoRA model on a JSONL dataset.
- Generate SQL with a base model, adapter model, or merged model.
- Execute generated SQL against a SQLite database.
- Keep prompt creation and dataset formatting modular so base models are easy to swap.

Expected dataset record shape
{
  "question": "...",
  "candidates": [{...}, ...],
  "target": {"sql": "SELECT ..."}
}

Usage examples
--------------
Train:
  python text2sql_workbench.py train \
    --base-model google/gemma-3-270m-it \
    --train-jsonl /path/train.jsonl \
    --val-jsonl /path/val.jsonl \
    --output-dir runs/gemma-exp1

Generate from base model:
  python text2sql_workbench.py generate \
    --model-path google/gemma-3-270m-it \
    --question "Show me the geometry for Berlin" \
    --candidates-json /path/candidates.json

Generate from adapter checkpoint:
  python text2sql_workbench.py generate \
    --base-model google/gemma-3-270m-it \
    --adapter-path runs/gemma-exp1/checkpoint-100 \
    --question "..." \
    --candidates-json /path/candidates.json

Generate and execute SQL:
  python text2sql_workbench.py generate \
    --model-path runs/gemma-exp1/merged \
    --question "..." \
    --candidates-json /path/candidates.json \
    --sqlite-db /path/geocode.db \
    --execute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("nlg")

SYSTEM_PROMPT = """You are a text to SQL query translator that helps in natural language geocoding.

You have access to two DuckDB parquet tables. Given a set of candidate entities and a user query, generate the SQL to retrieve the desired geometry.

<SCHEMA>
1. divisions_area  -- Overture polygon/multipolygon admin boundaries
   query: read_parquet('divisions_area')
   columns:
     id VARCHAR              -- unique feature id
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | dependency | county | localadmin |
                               locality | macrohood | neighborhood | microhood
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     division_id VARCHAR
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)

2. natural_earth  -- Natural Earth geography polygons (oceans, seas, rivers, terrain)
   query: read_parquet('natural_earth')
   columns:
     id VARCHAR              -- unique feature id prefixed 'ne_'
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'Terrain area', 'Island group'
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)
</SCHEMA>

The candidates table has a 'source' column: 'divisions_area' or 'natural_earth'.
Use read_parquet('divisions_area') or read_parquet('natural_earth') accordingly.
Use ST_AsGeoJSON(geometry) for all geometry outputs."""

USER_PROMPT_TEMPLATE = """<CANDIDATES>
{candidates_csv}
</CANDIDATES>

<USER_QUERY>
{question}
</USER_QUERY>
"""


@dataclass
class GenerationResult:
    prompt: str
    sql: str
    raw_text: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def candidates_to_csv(candidates: Sequence[Dict[str, Any]]) -> str:
    df = pd.DataFrame(list(candidates))
    if "candidate_id" in df.columns:
        df = df.drop(columns=["candidate_id"])
    return df.to_csv(index=False)


def build_user_prompt(question: str, candidates: Sequence[Dict[str, Any]]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        candidates_csv=candidates_to_csv(candidates).strip(),
        question=question.strip(),
    )


def make_messages(sample: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                question=sample["question"],
                candidates=sample["candidates"],
            ),
        },
    ]
    if sample.get("target", {}).get("sql"):
        messages.append({"role": "assistant", "content": sample["target"]["sql"]})
    return {"messages": messages}


def load_jsonl_splits(
    train_jsonl: str,
    val_jsonl: Optional[str] = None,
    test_jsonl: Optional[str] = None,
) -> DatasetDict:
    data_files: Dict[str, str] = {"train": train_jsonl}
    if val_jsonl:
        data_files["val"] = val_jsonl
    if test_jsonl:
        data_files["test"] = test_jsonl
    LOGGER.info("Loading dataset splits: %s", data_files)
    return load_dataset("json", data_files=data_files)


def format_dataset_for_sft(dataset: DatasetDict) -> DatasetDict:
    formatted = DatasetDict()
    for split, ds in dataset.items():
        formatted[split] = ds.map(make_messages)
    return formatted


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "auto": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name_or_path: str,
    *,
    dtype: str = "bf16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
):
    kwargs: Dict[str, Any] = {"device_map": device_map}
    if dtype != "auto":
        kwargs["torch_dtype"] = get_torch_dtype(dtype)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    LOGGER.info("Loading model from %s", model_name_or_path)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def load_model_for_inference(
    *,
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    dtype: str = "bf16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
):
    if model_path:
        tokenizer = load_tokenizer(model_path)
        model = load_model(
            model_path,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        return model, tokenizer

    if not (base_model and adapter_path):
        raise ValueError("Either --model-path or both --base-model and --adapter-path are required.")

    tokenizer = load_tokenizer(base_model)
    base = load_model(
        base_model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    LOGGER.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    raw_ds = load_jsonl_splits(args.train_jsonl, args.val_jsonl, args.test_jsonl)
    ds = format_dataset_for_sft(raw_ds)

    if args.max_train_samples is not None:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_eval_samples is not None and "val" in ds:
        ds["val"] = ds["val"].select(range(min(args.max_eval_samples, len(ds["val"]))))

    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.base_model, dtype=args.dtype, device_map=args.device_map)

    peft_config = build_lora_config(args)

    trainer_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        packing=args.packing,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        bf16=(args.dtype.lower() in {"bf16", "bfloat16"}),
        fp16=(args.dtype.lower() in {"fp16", "float16"}),
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        assistant_only_loss=args.assistant_only_loss,
        dataset_num_proc=args.dataset_num_proc,
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    LOGGER.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info("Saving adapter/checkpoint to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_after_training:
        merged_out = str(Path(args.output_dir) / "merged")
        merge_adapter(base_model=args.base_model, adapter_path=args.output_dir, output_path=merged_out, dtype=args.dtype)


def merge_adapter(*, base_model: str, adapter_path: str, output_path: str, dtype: str = "bf16") -> None:
    tokenizer = load_tokenizer(base_model)
    base = load_model(base_model, dtype=dtype, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    LOGGER.info("Merging adapter into base model")
    merged = peft_model.merge_and_unload()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_path)
    LOGGER.info("Merged model saved to %s", output_path)


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def postprocess_generated_sql(text: str) -> str:
    cleaned = text.strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()


def generate_sql(
    *,
    model,
    tokenizer,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 50,
) -> GenerationResult:
    messages = make_messages(
        {"question": question, "candidates": list(candidates), "target": {}},
    )["messages"]
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    sql = postprocess_generated_sql(generated)
    return GenerationResult(prompt=prompt, sql=sql, raw_text=generated)


def read_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.candidates_json:
        with open(args.candidates_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "candidates" in data:
                return data["candidates"]
            raise ValueError("Candidates JSON must be a list or an object with a 'candidates' field.")

    if args.sample_jsonl:
        ds = load_dataset("json", data_files={"sample": args.sample_jsonl})["sample"]
        row = ds[int(args.sample_index)]
        return row["candidates"]

    raise ValueError("Provide --candidates-json or --sample-jsonl.")


def read_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    if args.sample_jsonl:
        ds = load_dataset("json", data_files={"sample": args.sample_jsonl})["sample"]
        row = ds[int(args.sample_index)]
        return row["question"]
    raise ValueError("Provide --question or --sample-jsonl.")


def execute_sqlite(sql: str, sqlite_db: str, limit: Optional[int] = None) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    if not Path(sqlite_db).exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_db}")

    statement = sql.strip().rstrip(";")
    if limit and statement.lower().startswith("select") and " limit " not in statement.lower():
        statement = f"{statement} LIMIT {limit}"

    LOGGER.info("Executing SQL against %s", sqlite_db)
    with sqlite3.connect(sqlite_db) as conn:
        cursor = conn.cursor()
        cursor.execute(statement)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
    return columns, rows


def cmd_generate(args: argparse.Namespace) -> None:
    question = read_question(args)
    candidates = read_candidates(args)
    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    result = generate_sql(
        model=model,
        tokenizer=tokenizer,
        question=question,
        candidates=candidates,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\n=== QUESTION ===\n")
    print(question)
    print("\n=== GENERATED SQL ===\n")
    print(result.sql)

    if args.show_prompt:
        print("\n=== PROMPT ===\n")
        print(result.prompt)

    if args.execute:
        if not args.sqlite_db:
            raise ValueError("--sqlite-db is required with --execute")
        columns, rows = execute_sqlite(result.sql, args.sqlite_db, limit=args.limit)
        print("\n=== SQL RESULT ===\n")
        if columns:
            print(pd.DataFrame(rows, columns=columns).to_string(index=False))
        else:
            print("Statement executed. No row output.")


def cmd_merge(args: argparse.Namespace) -> None:
    merge_adapter(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        dtype=args.dtype,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test text-to-SQL models.")
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a LoRA SFT model.")
    train_p.add_argument("--base-model", required=True)
    train_p.add_argument("--train-jsonl", required=True)
    train_p.add_argument("--val-jsonl")
    train_p.add_argument("--test-jsonl")
    train_p.add_argument("--output-dir", required=True)
    train_p.add_argument("--max-train-samples", type=int)
    train_p.add_argument("--max-eval-samples", type=int)
    train_p.add_argument("--dtype", default="bf16")
    train_p.add_argument("--device-map", default="auto")
    train_p.add_argument("--max-length", type=int, default=1024)
    train_p.add_argument("--packing", action="store_true")
    train_p.add_argument("--num-train-epochs", type=int, default=3)
    train_p.add_argument("--per-device-train-batch-size", type=int, default=1)
    train_p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_p.add_argument("--gradient-checkpointing", action="store_true")
    train_p.add_argument("--optim", default="adamw_torch")
    train_p.add_argument("--logging-steps", type=int, default=10)
    train_p.add_argument("--save-strategy", default="epoch")
    train_p.add_argument("--eval-strategy", default="epoch")
    train_p.add_argument("--learning-rate", type=float, default=1e-4)
    train_p.add_argument("--max-grad-norm", type=float, default=0.3)
    train_p.add_argument("--warmup-ratio", type=float, default=0.03)
    train_p.add_argument("--lr-scheduler-type", default="constant")
    train_p.add_argument("--report-to", default="none")
    train_p.add_argument("--assistant-only-loss", action="store_true")
    train_p.add_argument("--dataset-num-proc", type=int)
    train_p.add_argument("--resume-from-checkpoint")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--merge-after-training", action="store_true")
    train_p.add_argument("--lora-r", type=int, default=16)
    train_p.add_argument("--lora-alpha", type=int, default=16)
    train_p.add_argument("--lora-dropout", type=float, default=0.05)
    train_p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_p.set_defaults(func=train)

    gen_p = subparsers.add_parser("generate", help="Generate SQL from a model.")
    gen_p.add_argument("--model-path")
    gen_p.add_argument("--base-model")
    gen_p.add_argument("--adapter-path")
    gen_p.add_argument("--question")
    gen_p.add_argument("--candidates-json")
    gen_p.add_argument("--sample-jsonl")
    gen_p.add_argument("--sample-index", type=int, default=0)
    gen_p.add_argument("--dtype", default="bf16")
    gen_p.add_argument("--device-map", default="auto")
    gen_p.add_argument("--attn-implementation")
    gen_p.add_argument("--max-new-tokens", type=int, default=256)
    gen_p.add_argument("--do-sample", action="store_true")
    gen_p.add_argument("--temperature", type=float, default=0.1)
    gen_p.add_argument("--top-p", type=float, default=0.95)
    gen_p.add_argument("--top-k", type=int, default=50)
    gen_p.add_argument("--show-prompt", action="store_true")
    gen_p.add_argument("--sqlite-db")
    gen_p.add_argument("--execute", action="store_true")
    gen_p.add_argument("--limit", type=int, default=20)
    gen_p.set_defaults(func=cmd_generate)

    merge_p = subparsers.add_parser("merge", help="Merge an adapter into the base model.")
    merge_p.add_argument("--base-model", required=True)
    merge_p.add_argument("--adapter-path", required=True)
    merge_p.add_argument("--output-path", required=True)
    merge_p.add_argument("--dtype", default="bf16")
    merge_p.set_defaults(func=cmd_merge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
