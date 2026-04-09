"""Dataset loading for gazet fine-tuning.

Loads prompt+completion data from run directories where each sample has:
  - ``prompt``: list of role-content dicts  [{role:system,...}, {role:user,...}]
  - ``completion``: list of role-content dicts [{role:assistant, content:...}]
  - ``metadata``: task info

Flattens these to plain strings so SFTTrainer can train with completion-only
loss without any chat template applied — matching how llama-server's
/completion endpoint is used at inference time.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict

LOGGER = logging.getLogger("gazet.finetune.data")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_prompt_completion(sample: dict) -> dict:
    """Flatten message-list format to plain strings.

    prompt[0] is the system message, prompt[1] is the user message.
    completion[0] is the assistant message.
    Joins system + user with a blank line separator — exactly what was
    passed to the /completion endpoint during the original working eval.
    """
    prompt_str = sample["prompt"][0]["content"] + "\n\n" + sample["prompt"][1]["content"]
    completion_str = sample["completion"][0]["content"]
    return {"prompt": prompt_str, "completion": completion_str}


def load_run_splits(
    run_dir: str | Path,
    tasks: tuple[str, ...] = ("sql", "places"),
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> DatasetDict:
    """Load and combine SQL + places JSONL files from a run directory.

    Args:
        run_dir: Directory with ``{task}/{split}.jsonl`` files.
        tasks: Which task subdirectories to include.
        splits: Which splits to load.

    Returns:
        DatasetDict where each split contains rows with ``prompt`` (str)
        and ``completion`` (str) columns ready for SFTTrainer with
        completion_only_loss=True.
    """
    run_dir = Path(run_dir)
    ds_dict = {}

    for split in splits:
        combined: list[dict] = []
        for task in tasks:
            path = run_dir / task / f"{split}.jsonl"
            if not path.exists():
                LOGGER.warning("Missing %s — skipping", path)
                continue
            rows = _load_jsonl(path)
            flattened = [_to_prompt_completion(r) for r in rows]
            combined.extend(flattened)
            LOGGER.info("Loaded %d %s/%s rows", len(rows), task, split)

        if combined:
            ds_dict[split] = Dataset.from_list(combined)
            LOGGER.info("%s split: %d total rows", split, len(combined))

    return DatasetDict(ds_dict)


def load_and_prepare(
    run_dir: str | Path,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    """Main entry point used by the training script.

    Loads both sql and places splits, shuffles, and optionally truncates.

    Returns:
        DatasetDict with ``prompt`` and ``completion`` string columns.
    """
    ds = load_run_splits(run_dir)
    ds = ds.shuffle(seed=42)

    if max_train_samples is not None and "train" in ds:
        ds["train"] = ds["train"].select(
            range(min(max_train_samples, len(ds["train"])))
        )
    if max_eval_samples is not None and "val" in ds:
        ds["val"] = ds["val"].select(
            range(min(max_eval_samples, len(ds["val"])))
        )

    return ds
