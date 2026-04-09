"""Merge a LoRA checkpoint into the base model on Modal.

The checkpoint and output both live on the gazet Modal volume.

Usage
-----
modal run finetune/merge_checkpoint.py \
    --checkpoint /mnt/gazet/checkpoints/my-run/checkpoint-1800

Output defaults to <checkpoint>-merged alongside the checkpoint.
Override with --output:
    modal run finetune/merge_checkpoint.py \
        --checkpoint /mnt/gazet/checkpoints/my-run/checkpoint-1800 \
        --output /mnt/gazet/checkpoints/my-run/checkpoint-1800-merged
"""

from __future__ import annotations

import pathlib
from typing import Optional

import modal

app = modal.App("gazet-merge-checkpoint")

merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0",
        "hf-transfer>=0.1",
        "huggingface_hub>=0.25",
        "peft>=0.13",
        "torch>=2.4",
        "transformers>=4.46",
    )
    .env({"HF_HOME": "/mnt/gazet/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}


@app.function(
    image=merge_image,
    cpu=4,
    memory=16384,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=30 * 60,
)
def merge(checkpoint_path: str, output_path: str) -> None:
    import json
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = pathlib.Path(checkpoint_path)
    output_dir = pathlib.Path(output_path)

    adapter_cfg = checkpoint_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(f"No adapter_config.json in {checkpoint_dir}")

    base_model_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
    print(f"Base model:  {base_model_name}")
    print(f"Checkpoint:  {checkpoint_dir}")
    print(f"Output:      {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Applying LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base, str(checkpoint_dir))

    print("Merging and unloading adapter...")
    merged = peft_model.merge_and_unload()

    print("Saving merged model...")
    merged.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="2GB")

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(output_dir))

    # SentencePiece vocab file — fast tokenizer doesn't always save it
    sp_dest = output_dir / "tokenizer.model"
    if not sp_dest.exists():
        base_dir = pathlib.Path(snapshot_download(base_model_name))
        sp_src = base_dir / "tokenizer.model"
        if sp_src.exists():
            shutil.copy2(sp_src, sp_dest)
            print(f"Copied tokenizer.model from {sp_src}")
        else:
            print("WARNING: tokenizer.model not found in base model cache")

    gazet_vol.commit()
    print(f"\nDone — merged model saved to {output_dir}")


@app.local_entrypoint()
def main(
    checkpoint: str,
    output: Optional[str] = None,
):
    checkpoint_path = checkpoint
    output_path = output or checkpoint_path.rstrip("/") + "-merged"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_path}")

    merge.remote(checkpoint_path, output_path)
