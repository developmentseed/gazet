"""Training configuration for text-to-SQL LoRA finetuning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainingConfig:
    # Model
    base_model: str = "google/gemma-3-270m-it"

    # Dataset (paths on the Modal volume)
    train_jsonl: str = "/mnt/gazet/data/output/train.jsonl"
    val_jsonl: Optional[str] = "/mnt/gazet/data/output/val.jsonl"
    test_jsonl: Optional[str] = "/mnt/gazet/data/output/test.jsonl"
    schema_file: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 12
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.7
    warmup_steps: int = 50
    lr_scheduler_type: str = "constant"
    weight_decay: float = 0.0
    packing: bool = False
    max_length: int = 2048

    # Logging / saving
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 300
    eval_strategy: str = "steps"
    eval_steps: int = 100
    report_to: str = "trackio"
    trackio_space_id: Optional[str] = "srmsoumya/gazet-trackio"
    project: str = "gazet-nlg"

    # SFT-specific
    completion_only_loss: bool = True
    dataset_num_proc: Optional[int] = 8

    # Experiment
    seed: int = 42
    experiment_name: Optional[str] = None
    merge_after_training: bool = True

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.base_model.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"
