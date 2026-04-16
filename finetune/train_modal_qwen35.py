"""Modal training script for gazet Qwen3.5 LoRA fine-tuning with Unsloth.

Key differences from train_modal.py (Gemma):
- Uses Unsloth's FastLanguageModel for memory-efficient training
- Applies Qwen3.5 chat template to format data (not plain prompt+completion strings)
- Uses train_on_responses_only with ChatML markers to mask non-assistant tokens
- Saves merged 16-bit model via unsloth's save_pretrained_merged

Usage
-----
modal run finetune/train_modal_qwen35.py
modal run finetune/train_modal_qwen35.py --base-model unsloth/Qwen3.5-0.8B
modal run finetune/train_modal_qwen35.py --run-dir /mnt/gazet/data/v3-symbolic-paths
modal run finetune/train_modal_qwen35.py --num-train-epochs 5 --lora-r 32
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

app = modal.App("gazet-nlg-qwen35-finetune")

GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 24
MAX_RETRIES = 1

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Use unsloth's bundled CUDA+torch extra so bitsandbytes, xformers,
        # and trl are all resolved together against the same CUDA/torch build.
        # Mirrors the approach in https://modal.com/docs/examples/unsloth_finetune
        "unsloth[cu129-torch280]",
        "unsloth_zoo",
        "transformers~=5.2.0",
        "hf-transfer==0.1.9",
        "trackio[gpu]==0.21.1",
        "datasets",
        "pandas",
    )
    .add_local_python_source("finetune", copy=True)
    .env({
        "HF_HOME": "/mnt/gazet/model_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

with train_image.imports():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer
    from transformers import set_seed

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}

# ChatML response markers for Qwen3.5 — the empty <think> block is how Qwen3.5
# formats non-thinking responses. We train only on tokens after this prefix.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class Qwen35Config:
    # Model
    base_model: str = "unsloth/Qwen3.5-0.8B"

    # Dataset — path to run dir with {task}/{split}.jsonl files
    run_dir: str = "/mnt/gazet/data/v3-symbolic-paths"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Sequence
    max_seq_length: int = 2048

    # LoRA — alpha=2*r follows unsloth recommendation for Qwen models
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1  # effective batch = 48
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"

    # Logging / saving
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 400
    eval_strategy: str = "steps"
    eval_steps: int = 200
    report_to: str = "trackio"
    trackio_space_id: Optional[str] = "srmsoumya/gazet-trackio"
    project: str = "gazet-nlg-qwen35"

    # Experiment
    seed: int = 42
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.base_model.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"


def _load_data(run_dir: str, tokenizer, max_train_samples=None, max_eval_samples=None):
    """Load JSONL data and apply Qwen3.5 chat template.

    Each sample must have:
      messages: list of {role, content} dicts (system + user + assistant)

    The chat template produces the full ChatML string including the assistant turn.
    train_on_responses_only then masks everything except the assistant response.
    """
    import json
    from datasets import Dataset, DatasetDict

    def load_jsonl(path: pathlib.Path) -> list[dict]:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def to_message(sample: dict) -> dict:
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"messages": text}

    run_dir = pathlib.Path(run_dir)
    tasks = ("sql", "places")
    splits = ("train", "val")
    ds_dict: dict = {}

    for split in splits:
        combined: list[dict] = []
        for task in tasks:
            path = run_dir / task / f"{split}.jsonl"
            if not path.exists():
                print(f"Missing {path} — skipping")
                continue
            rows = load_jsonl(path)
            flattened = [to_message(r) for r in rows]
            combined.extend(flattened)
            print(f"Loaded {len(rows):,} {task}/{split} rows")

        if combined:
            ds_dict[split] = Dataset.from_list(combined)
            print(f"{split} split: {len(combined):,} total rows")

    ds = DatasetDict(ds_dict).shuffle(seed=42)

    if max_train_samples is not None and "train" in ds:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    if max_eval_samples is not None and "val" in ds:
        ds["val"] = ds["val"].select(range(min(max_eval_samples, len(ds["val"]))))

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
    """Run Qwen3.5 LoRA SFT training with Unsloth inside a Modal container."""
    config = Qwen35Config(**config_dict)
    set_seed(config.seed)

    experiment_dir = pathlib.Path("/mnt/gazet/checkpoints") / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment:       {config.experiment_name}")
    print(f"Model:            {config.base_model}")
    print(f"Run dir:          {config.run_dir}")

    # Load base model with unsloth — gradient checkpointing is handled internally
    model, processor = FastLanguageModel.from_pretrained(
        config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        fast_inference=False,
    )
    tokenizer = processor.tokenizer

    # Apply LoRA adapters — let unsloth select target modules via finetune_* flags
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        bias="none",
        random_state=config.seed,
        use_gradient_checkpointing=False,  # already set in from_pretrained
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    ds = _load_data(
        config.run_dir,
        tokenizer,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
    )
    print(f"Train samples:    {len(ds['train']):,}")
    if "val" in ds:
        print(f"Val samples:      {len(ds['val']):,}")
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(f"Effective batch:  {effective_batch}")

    sft_args = SFTConfig(
        output_dir=str(experiment_dir),
        dataset_text_field="messages",
        max_seq_length=config.max_seq_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        optim=config.optim,
        bf16=True,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        report_to=config.report_to,
        trackio_space_id=config.trackio_space_id,
        project=config.project,
        dataset_num_proc=8,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val"),
        args=sft_args,
    )

    # Mask all tokens except the assistant response — train on completions only
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    resume_from = _find_latest_checkpoint(experiment_dir)
    if resume_from:
        print(f"Resuming from {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save LoRA adapter + tokenizer (lightweight, for future merging)
    print(f"Saving LoRA adapter to {experiment_dir}")
    model.save_pretrained(str(experiment_dir))
    tokenizer.save_pretrained(str(experiment_dir))

    # Save merged 16-bit model (full weights, ready for inference / GGUF conversion)
    merged_dir = experiment_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged 16-bit model to {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    gazet_vol.commit()
    print(f"Training complete: {config.experiment_name}")
    return config.experiment_name


@app.local_entrypoint()
def main(
    base_model: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_dir: Optional[str] = None,
    num_train_epochs: Optional[int] = None,
    per_device_train_batch_size: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    lora_r: Optional[int] = None,
    max_seq_length: Optional[int] = None,
):
    overrides = {
        k: v for k, v in dict(
            base_model=base_model,
            experiment_name=experiment_name,
            run_dir=run_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            lora_r=lora_r,
            max_seq_length=max_seq_length,
        ).items() if v is not None
    }

    config = Qwen35Config(**overrides)
    # lora_alpha follows r if r was overridden and alpha wasn't
    if lora_r is not None:
        config.lora_alpha = 2 * config.lora_r

    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model:               {config.base_model}")
    print(f"Run dir:             {config.run_dir}")
    print(f"LoRA:                r={config.lora_r}, alpha={config.lora_alpha}")
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(f"Effective batch:     {effective_batch}")

    result = finetune.remote(config.__dict__)
    print(f"Training complete: {result}")
