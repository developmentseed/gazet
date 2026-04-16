"""Interactive eval: run test samples through the local GGUF model.

Requires llama-server running on port 8080:
  llama-server -m finetune/models/<model>.gguf -ngl 99 --port 8080 --ctx-size 4096 --log-disable

Uses the /v1/chat/completions endpoint with a messages list. The Qwen3 GGUF
embeds its chat template in metadata, so llama-server applies it automatically.

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

SERVER_URL = "http://localhost:9000"
MAX_TOKENS = 2048
TEMPERATURE = 0.6

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


def chat_complete(messages: list[dict]) -> str:
    """Call llama-server /v1/chat/completions with a messages list."""
    payload = json.dumps({
        "messages": messages,
        "n_predict": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def load_samples(run_dir: Path, task: str) -> list[dict]:
    path = run_dir / task / "val.jsonl"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    print(f"Loading {task} samples from: {path}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_raw_prompt(sample: dict) -> str:
    """Reconstruct the plain prompt string from messages format (all turns except assistant)."""
    return "\n\n".join(m["content"] for m in sample["messages"][:-1])


def run_sample(sample: dict, task: str, total: int, index: int, verbose: bool = False) -> None:
    expected = sample["messages"][-1]["content"]
    messages = sample["messages"][:-1]

    user_content = sample["messages"][-2]["content"]
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
        prompt = build_raw_prompt(sample)
        print(f"{'─' * 60}")
        print(f"Full prompt ({len(prompt)} chars, ~{len(prompt.split())} words):")
        print(f"{'─' * 60}")
        print(prompt)

    print(f"{'─' * 60}")
    print("Expected:")
    print(f"{'─' * 60}")
    print(expected)

    print(f"\n{'─' * 60}")
    print("Generated:")
    print(f"{'─' * 60}")

    raw = chat_complete(messages)
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
        print("llama-server -m finetune/models/<model>.gguf -ngl 99 --port 8080 --ctx-size 2048 --log-disable")
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
