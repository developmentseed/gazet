"""Build a formatted prompt from a test sample and write to stdout."""
import json
import sys
sys.path.insert(0, ".")

from finetune.prompts import DEFAULT_SCHEMA_DETAILS, build_user_prompt

# Load first test sample
with open("dataset/output/test.jsonl") as f:
    sample = json.loads(f.readline())

user_prompt = build_user_prompt(
    question=sample["question"],
    candidates=sample["candidates"],
    schema_details=DEFAULT_SCHEMA_DETAILS,
)

print(f"Question : {sample['question']}")
print(f"Expected : {sample['target']['sql']}")
print("---")
print(user_prompt)

# Write prompt to file for llama-cli
with open("/tmp/gazet_prompt.txt", "w") as f:
    f.write(user_prompt)

print("\n[Prompt written to /tmp/gazet_prompt.txt]")
