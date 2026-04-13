"""Modal training script for gazet LoRA fine-tuning.

Usage
-----
modal run finetune/train_modal.py
modal run finetune/train_modal.py --base-model google/gemma-3-1b-it
modal run finetune/train_modal.py --run-dir /mnt/gazet/data/output/runs/v3-symbolic-paths

All CLI arguments map to TrainingConfig fields. Run with --help for details.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import modal

app = modal.App("gazet-nlg-finetune")

GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 6
MAX_RETRIES = 1

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.13.0",
        "datasets==4.8.4",
        "hf-transfer==0.1.9",
        "huggingface_hub==1.9.2",
        "jinja2==3.1.6",
        "pandas==2.3.3",
        "peft==0.18.1",
        "torch==2.11.0",
        "trackio[gpu]==0.21.1",
        "transformers==5.5.1",
        "trl==1.0.0",
    )
    .add_local_python_source("finetune", copy=True)
    .env({"HF_HOME": "/mnt/gazet/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with train_image.imports():
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_name: str):
    return AutoModelForImageTextToText.from_pretrained(
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
        modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
        ensure_weight_tying=True,
    )


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
    from finetune.data import load_and_prepare

    config = TrainingConfig(**config_dict)
    set_seed(config.seed)

    experiment_dir = pathlib.Path("/mnt/gazet/checkpoints") / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment:       {config.experiment_name}")
    print(f"Model:            {config.base_model}")
    print(f"Run dir:          {config.run_dir}")

    tokenizer = _load_tokenizer(config.base_model)
    model = _load_model(config.base_model)

    ds = load_and_prepare(
        config.run_dir,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
    )
    print(f"Train samples:    {len(ds['train']):,}")
    if "val" in ds:
        print(f"Val samples:      {len(ds['val']):,}")

    peft_config = _build_lora_config(config)

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
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    resume_from = _find_latest_checkpoint(experiment_dir)
    if resume_from:
        print(f"Resuming from {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    print(f"Saving adapter to {experiment_dir}")
    trainer.save_model(str(experiment_dir))
    tokenizer.save_pretrained(str(experiment_dir))
    gazet_vol.commit()

    if config.merge_after_training:
        _merge_and_save(config, experiment_dir)

    print(f"Training complete: {config.experiment_name}")
    return config.experiment_name


def _merge_and_save(config, experiment_dir: pathlib.Path):
    import shutil
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    merged_dir = experiment_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForImageTextToText.from_pretrained(
        config.base_model,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, str(experiment_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")

    tokenizer = _load_tokenizer("google/gemma-4-E2B-it")
    tokenizer.save_pretrained(str(merged_dir))

    # Copy tokenizer.model from base model cache (fast tokenizer doesn't save it)
    sp_dest = merged_dir / "tokenizer.model"
    if not sp_dest.exists():
        base_dir = pathlib.Path(snapshot_download("google/gemma-4-E2B-it"))
        sp_src = base_dir / "tokenizer.model"
        if sp_src.exists():
            shutil.copy2(sp_src, sp_dest)
            print(f"Copied tokenizer.model from {sp_src}")
        else:
            print("WARNING: tokenizer.model not found in base model cache")

    gazet_vol.commit()
    print(f"Merged model saved to {merged_dir}")


@app.local_entrypoint()
def main(
    base_model: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_dir: Optional[str] = None,
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
            run_dir=run_dir,
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
    print(f"Model:               {config.base_model}")
    print(f"Run dir:             {config.run_dir}")
    print(f"LoRA:                r={config.lora_r}, alpha={config.lora_alpha}")
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch}")

    result = finetune.remote(config.__dict__)
    print(f"Training complete: {result}")