"""Modal training script for text-to-SQL LoRA finetuning.

Usage
-----
modal run finetune/train_modal.py \
    --train-jsonl /data/train.jsonl \
    --val-jsonl /data/val.jsonl \
    --base-model google/gemma-3-1b-it

All CLI arguments map to TrainingConfig fields. Run with --help for details.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import modal

app = modal.App("gazet-nlg-finetune")

GPU_TYPE = "A100-80GB"  # "L40S"
TIMEOUT_HOURS = 6
MAX_RETRIES = 1

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0",
        "datasets>=3.0",
        "hf-transfer>=0.1",
        "huggingface_hub>=0.25",
        "jinja2>=3.0",
        "pandas>=2.2",
        "peft>=0.13",
        "torch>=2.4",
        "trackio[gpu]",
        "transformers>=4.46",
        "trl>=0.12",
    )
    .add_local_python_source("finetune", copy=True)
    .env({"HF_HOME": "/mnt/gazet/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with train_image.imports():
    import torch
    from datasets import DatasetDict
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )


def _build_lora_config(config) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )


def _load_and_format_dataset(config) -> DatasetDict:
    """Load JSONL splits and apply prompt-completion formatting."""
    from finetune.data import (
        format_dataset_for_sft,
        load_jsonl_splits,
        read_text,
    )
    from finetune.prompts import DEFAULT_SCHEMA_DETAILS

    schema_details = read_text(config.schema_file, DEFAULT_SCHEMA_DETAILS)
    raw_ds = load_jsonl_splits(config.train_jsonl, config.val_jsonl, config.test_jsonl)
    ds = format_dataset_for_sft(raw_ds, schema_details)

    if config.max_train_samples is not None:
        ds["train"] = ds["train"].select(
            range(min(config.max_train_samples, len(ds["train"])))
        )
    if config.max_eval_samples is not None and "val" in ds:
        ds["val"] = ds["val"].select(
            range(min(config.max_eval_samples, len(ds["val"])))
        )
    return ds


def _find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    print(f"Found existing checkpoint: {latest}")
    return str(latest)


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune(config_dict: dict):
    """Run LoRA SFT training inside a Modal container."""
    from finetune.config import TrainingConfig

    config = TrainingConfig(**config_dict)
    set_seed(config.seed)

    experiment_dir = pathlib.Path("/mnt/gazet/checkpoints") / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.base_model}")

    # Model and tokenizer
    tokenizer = _load_tokenizer(config.base_model)
    model = _load_model(config.base_model)

    # Dataset
    ds = _load_and_format_dataset(config)
    print(f"Train samples: {len(ds['train']):,}")
    if "val" in ds:
        print(f"Val samples: {len(ds['val']):,}")

    # LoRA
    peft_config = _build_lora_config(config)

    # SFT config
    sft_args = SFTConfig(
        output_dir=str(experiment_dir),
        max_length=config.max_length,
        packing=config.packing,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        report_to=config.report_to,
        trackio_space_id=config.trackio_space_id,
        project=config.project,
        completion_only_loss=config.completion_only_loss,
        dataset_num_proc=config.dataset_num_proc,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Resume from checkpoint if available (handles preemption)
    resume_from = _find_latest_checkpoint(experiment_dir)
    if resume_from:
        print(f"Resuming from {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final adapter + tokenizer
    print(f"Saving adapter to {experiment_dir}")
    trainer.save_model(str(experiment_dir))
    tokenizer.save_pretrained(str(experiment_dir))
    gazet_vol.commit()

    # Optionally merge adapter into base model
    if config.merge_after_training:
        _merge_and_save(config, experiment_dir)

    print(f"Training complete: {config.experiment_name}")
    return config.experiment_name


def _merge_and_save(config, experiment_dir: pathlib.Path):
    from peft import PeftModel

    merged_dir = experiment_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, str(experiment_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")

    tokenizer = _load_tokenizer(config.base_model)
    tokenizer.save_pretrained(str(merged_dir))
    gazet_vol.commit()
    print(f"Merged model saved to {merged_dir}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    base_model: Optional[str] = None,
    experiment_name: Optional[str] = None,
    per_device_train_batch_size: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
    lora_r: Optional[int] = None,
    max_length: Optional[int] = None,
):
    from finetune.config import TrainingConfig

    overrides = {
        k: v for k, v in dict(
            base_model=base_model,
            experiment_name=experiment_name,
            per_device_train_batch_size=per_device_train_batch_size,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            num_train_epochs=num_train_epochs,
            lora_r=lora_r,
            max_length=max_length,
        ).items() if v is not None
    }

    config = TrainingConfig(**overrides)

    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model: {config.base_model}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch}")

    result = finetune.remote(config.__dict__)
    print(f"Training complete: {result}")
