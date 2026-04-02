"""Print the full raw prompt for a test sample — paste into llama-server Completion UI.

Usage
-----
uv run finetune/make_test_prompt.py        # sample 0
uv run finetune/make_test_prompt.py 5      # sample 5
"""
import json
import sys
sys.path.insert(0, ".")

from finetune.prompts import DEFAULT_SCHEMA_DETAILS, SYSTEM_PROMPT, build_user_prompt

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

with open("dataset/output/test.jsonl") as f:
    samples = [json.loads(line) for line in f]

sample = samples[index]

raw_prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(
    question=sample["question"],
    candidates=sample["candidates"],
    schema_details=DEFAULT_SCHEMA_DETAILS,
)

out_path = "/tmp/gazet_prompt.txt"
with open(out_path, "w") as f:
    f.write(raw_prompt)

print(f"Question : {sample['question']}")
print(f"Expected : {sample['target']['sql']}")
print(f"Prompt   : {out_path}")
print(f"\n{'─' * 60}")
print("Paste into llama-server → Completion tab:")
print(f"{'─' * 60}\n")
print(raw_prompt)
