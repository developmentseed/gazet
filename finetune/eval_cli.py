"""Interactive eval: run a test.jsonl sample through the local GGUF model.

Requires llama-server running on port 8080:
  llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable

Usage
-----
uv run finetune/eval_cli.py          # prompts for index
uv run finetune/eval_cli.py 5        # run sample at index 5
uv run finetune/eval_cli.py 5 12 20  # run multiple samples
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
from finetune.prompts import DEFAULT_SCHEMA_DETAILS, SYSTEM_PROMPT, build_user_prompt

# ── config ────────────────────────────────────────────────────────────────────
TEST_JSONL = Path("dataset/output/test.jsonl")
SERVER_URL = "http://localhost:8080"
MAX_TOKENS = 350
TEMPERATURE = 0
# ─────────────────────────────────────────────────────────────────────────────


def postprocess_sql(text: str) -> str:
    cleaned = text.strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()


def check_server() -> bool:
    try:
        urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def complete(prompt: str) -> str:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stop": ["<eos>", "</s>"],
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["content"]


def load_samples() -> list[dict]:
    with TEST_JSONL.open() as f:
        return [json.loads(line) for line in f]


def run_sample(sample: dict, samples: list[dict], index: int) -> None:
    question = sample["question"]
    expected_sql = sample["target"]["sql"]
    task = sample["metadata"]["task_family"]
    difficulty = sample["metadata"]["sql_difficulty"]

    user_prompt = build_user_prompt(
        question=question,
        candidates=sample["candidates"],
        schema_details=DEFAULT_SCHEMA_DETAILS,
    )

    # Raw prompt-completion format — matches training (use_chat_template=False)
    raw_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    header = f"  Sample {index}/{len(samples)-1} · {task} · {difficulty}  "
    print(f"\n{'━' * 60}")
    print(f"{'━' * ((60 - len(header)) // 2)}{header}{'━' * ((60 - len(header)) // 2)}")
    print(f"{'━' * 60}")
    print(f"\nQuestion: {question}\n")
    print(f"{'─' * 60}")
    print("Expected SQL:")
    print(f"{'─' * 60}")
    print(expected_sql)
    print(f"\n{'─' * 60}")
    print("Generated SQL:")
    print(f"{'─' * 60}")

    generated = postprocess_sql(complete(raw_prompt))
    print(generated)


def main() -> None:
    if not check_server():
        print("llama-server not running. Start it with:")
        print(f"  llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable")
        sys.exit(1)

    samples = load_samples()
    total = len(samples)
    args = sys.argv[1:]

    if not args:
        print(f"Test set has {total} samples (0–{total-1})")
        raw = input("Enter index (or press Enter for 0): ").strip()
        indices = [int(raw) if raw else 0]
    else:
        indices = [int(a) for a in args]

    for idx in indices:
        if not (0 <= idx < total):
            print(f"Index {idx} out of range (0–{total-1}), skipping")
            continue
        run_sample(samples[idx], samples, idx)

    print(f"\n{'━' * 60}\n")


if __name__ == "__main__":
    main()
