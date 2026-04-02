"""Dataset loading and SFT formatting for text-to-SQL finetuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from datasets import DatasetDict, load_dataset

from finetune.prompts import DEFAULT_SCHEMA_DETAILS, make_prompt_completion

LOGGER = logging.getLogger("nlg.data")


def read_text(path: Optional[str], default: str) -> str:
    if not path:
        return default
    return Path(path).read_text(encoding="utf-8")


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


def format_dataset_for_sft(
    dataset: DatasetDict,
    schema_details: str = DEFAULT_SCHEMA_DETAILS,
) -> DatasetDict:
    formatted = DatasetDict()
    for split, ds in dataset.items():
        formatted[split] = ds.map(
            lambda row: make_prompt_completion(row, schema_details)
        )
    return formatted
