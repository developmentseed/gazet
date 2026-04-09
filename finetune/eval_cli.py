"""Interactive eval: run test samples through the local GGUF model.

Requires llama-server running on port 8080:
  llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable

Uses the /completion endpoint with a raw prompt string — no chat template —
matching how the model was fine-tuned (completion_only_loss on plain text).

Usage
-----
uv run finetune/eval_cli.py          # prompts for index
uv run finetune/eval_cli.py 5        # run sample at index 5
uv run finetune/eval_cli.py 5 12 20  # run multiple samples

Use --task places for place extraction:
  uv run finetune/eval_cli.py --task places 0 5

Override run directory:
  uv run finetune/eval_cli.py --run-dir dataset/output/runs/v3-symbolic-paths 0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

SERVER_URL = "http://localhost:8080"
MAX_TOKENS = 512
TEMPERATURE = 0

DEFAULT_RUN_DIR = Path("dataset/output/runs/v3-symbolic-paths")


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
    """Call llama-server /completion endpoint with a raw prompt string."""
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["content"]


def load_samples(run_dir: Path, task: str) -> list[dict]:
    path = run_dir / task / "val.jsonl"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    print(f"Loading {task} samples from: {path}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_raw_prompt(sample: dict) -> str:
    """Reconstruct the plain prompt string from message-list format.

    sample["prompt"] is [{role:system, content:...}, {role:user, content:...}].
    Joins them with a blank line — same format used during training.
    """
    return sample["prompt"][0]["content"] + "\n\n" + sample["prompt"][1]["content"]


def run_sample(sample: dict, task: str, total: int, index: int, verbose: bool = False) -> None:
    expected = sample["completion"][0]["content"]
    prompt = build_raw_prompt(sample)

    user_content = sample["prompt"][1]["content"]
    if "<USER_QUERY>" in user_content:
        question = user_content.split("<USER_QUERY>")[-1].split("</USER_QUERY>")[0].strip()
    else:
        question = user_content[:120]

    header = f"  Sample {index}/{total-1} | {task}  "
    print(f"\n{'━' * 60}")
    print(f"{'━' * ((60 - len(header)) // 2)}{header}{'━' * ((60 - len(header)) // 2)}")
    print(f"{'━' * 60}")
    print(f"\nQuestion: {question}\n")

    if verbose:
        print(f"{'─' * 60}")
        print(f"Full prompt ({len(prompt)} chars, ~{len(prompt.split()) } words):")
        print(f"{'─' * 60}")
        print(prompt)

    print(f"{'─' * 60}")
    print("Expected:")
    print(f"{'─' * 60}")
    print(expected)

    print(f"\n{'─' * 60}")
    print("Generated:")
    print(f"{'─' * 60}")

    raw = complete(prompt)
    generated = postprocess_sql(raw) if task == "sql" else raw.strip()
    print(generated)

    match = generated.strip() == expected.strip()
    print(f"\n{'─' * 60}")
    print(f"Match: {'YES' if match else 'NO'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive eval against llama-server")
    parser.add_argument("indices", nargs="*", type=int, help="Sample indices to evaluate")
    parser.add_argument("--task", default="sql", choices=["sql", "places"])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory containing {task}/{split}.jsonl files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full prompt sent to the model")
    args = parser.parse_args()

    if not check_server():
        print("llama-server not running. Start it with:")
        print("  llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable")
        sys.exit(1)

    samples = load_samples(args.run_dir, args.task)
    total = len(samples)

    if not args.indices:
        print(f"Test set has {total} {args.task} samples (0-{total-1})")
        raw = input("Enter index (or press Enter for 0): ").strip()
        indices = [int(raw) if raw else 0]
    else:
        indices = args.indices

    for idx in indices:
        if not (0 <= idx < total):
            print(f"Index {idx} out of range (0-{total-1}), skipping")
            continue
        run_sample(samples[idx], args.task, total, idx, verbose=args.verbose)

    print(f"\n{'━' * 60}\n")


if __name__ == "__main__":
    main()
