"""Test with raw /completion endpoint (no chat template)."""
import json
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

SERVER_URL = "http://localhost:8080"
MAX_TOKENS = 350
TEMPERATURE = 0

# Load a sample
run_dir = Path("dataset/output/runs/v3-symbolic-paths")
with open(run_dir / "sql" / "test.jsonl") as f:
    sample = json.loads(f.readline())

# Apply chat template manually to get the formatted prompt
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
prompt_text = tokenizer.apply_chat_template(
    sample['prompt'], tokenize=False, add_generation_prompt=True
)

print("=== Applying chat template and calling /completion (bypassing server template) ===\n")
print(f"Prompt length: {len(prompt_text)} chars\n")

# Call raw /completion endpoint
payload = json.dumps({
    "prompt": prompt_text,
    "n_predict": MAX_TOKENS,
    "temperature": TEMPERATURE,
}).encode()

req = urllib.request.Request(
    f"{SERVER_URL}/completion",
    data=payload,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read())

generated = data["content"]

print("=== GENERATED ===")
print(generated)

# Check for markers
if '<start_of_turn>' in generated or '</start_of_turn>' in generated:
    print("\n⚠️  CONTAINS MARKERS")
else:
    print("\n✓ No markers")
