#!/usr/bin/env python3
"""Convert prompt-completion format to conversation format.

Reads SQL and places JSONL from a run directory and converts to a single
"messages" list format suitable for various downstream uses.

Input format (current):
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}

Output format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Saves to JSONL files:
  - train_conversation_sql.jsonl
  - val_conversation_sql.jsonl
  - test_conversation_sql.jsonl
  - train_conversation_places.jsonl
  - val_conversation_places.jsonl
  - test_conversation_places.jsonl

Usage with datasets library:
    from datasets import load_dataset

    train_sql = load_dataset(
        "json",
        data_files="dataset/output/conversations/train_conversation_sql.jsonl",
        split="train"
    )

    # Access messages:
    print(train_sql[0]["messages"])
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_conversation_format(sample: dict) -> dict:
    """Convert prompt+completion format to messages format."""
    return {
        "messages": sample["prompt"] + sample["completion"],
    }


def process_task(run_dir: Path, task: str, output_dir: Path):
    """Process all splits for a single task (sql or places)."""
    task_dir = run_dir / task

    for split in ["train", "val", "test"]:
        input_path = task_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"  Skipping {task}/{split}: {input_path} not found")
            continue

        samples = load_jsonl(input_path)
        conversations = [to_conversation_format(s) for s in samples]

        output_path = output_dir / f"{split}_conversation_{task}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        print(f"  {task}/{split}: {len(conversations)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompt-completion format to conversation format"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("dataset/output/runs/v3-symbolic-paths"),
        help="Path to run directory containing sql/ and places/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/output/conversations"),
        help="Output directory for JSONL files",
    )

    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    print(f"Converting from: {run_dir}")
    print(f"Output directory: {output_dir}")
    print()

    for task in ["sql", "places"]:
        print(f"Processing {task}:")
        process_task(run_dir, task, output_dir)

    print()
    print("Conversion complete!")
    print(f"Output files in: {output_dir}/")


if __name__ == "__main__":
    main()
