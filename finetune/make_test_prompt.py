"""Print the full chat messages for a test sample — paste into llama-server Chat UI.

Usage
-----
uv run finetune/make_test_prompt.py             # sample 0 from sql test set
uv run finetune/make_test_prompt.py 5           # sample 5
uv run finetune/make_test_prompt.py --task places 3
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RUN_DIR = Path("dataset/output/runs/v2-all-families")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument("--task", default="sql", choices=["sql", "places"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    path = args.run_dir / args.task / "test.jsonl"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    with open(path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.index >= len(samples):
        print(f"Index {args.index} out of range (0-{len(samples)-1})")
        sys.exit(1)

    sample = samples[args.index]
    prompt_msgs = sample["prompt"]
    expected = sample["completion"][0]["content"]

    out_path = "/tmp/gazet_prompt.json"
    with open(out_path, "w") as f:
        json.dump(prompt_msgs, f, indent=2)

    print(f"Task     : {args.task}")
    print(f"Sample   : {args.index}")
    print(f"Expected : {expected[:120]}{'...' if len(expected) > 120 else ''}")
    print(f"Messages : {out_path}")

    print(f"\n{'─' * 60}")
    for msg in prompt_msgs:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]")
        print(content[:500])
        if len(content) > 500:
            print(f"  ... ({len(content)} chars total)")

    print(f"\n{'─' * 60}")
    print(f"\n[EXPECTED COMPLETION]")
    print(expected)


if __name__ == "__main__":
    main()
