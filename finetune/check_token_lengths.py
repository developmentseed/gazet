"""Check token lengths of training samples to validate max_length setting.

Usage
-----
modal run finetune/check_token_lengths.py
modal run finetune/check_token_lengths.py --run-dir /mnt/gazet/data/v1
"""

from __future__ import annotations

import modal

app = modal.App("gazet-check-token-lengths")

check_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=3.0",
        "transformers>=4.46",
        "jinja2>=3.1",
    )
    .add_local_python_source("finetune", copy=True)
    .env({"HF_HOME": "/mnt/gazet/model_cache"})
)

gazet_vol = modal.Volume.from_name("gazet", create_if_missing=True)

VOLUMES = {
    "/mnt/gazet": gazet_vol,
}


@app.function(
    image=check_image,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def analyze_token_lengths(run_dir: str, base_model: str):
    import json
    import pathlib

    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    def load_jsonl(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    root = pathlib.Path(run_dir)
    ds_dict = {}
    for split in ("train", "val", "test"):
        combined = []
        for task in ("sql", "places"):
            path = root / task / f"{split}.jsonl"
            if path.exists():
                combined.extend(load_jsonl(path))
        if combined:
            ds_dict[split] = Dataset.from_list(combined)
    ds = DatasetDict(ds_dict)

    def token_lengths(dataset):
        lengths = []
        for row in dataset:
            msgs = row["messages"]
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            lengths.append(len(tokenizer.encode(text)))
        return lengths

    def report(split_name: str, lengths: list[int]):
        lengths.sort()
        n = len(lengths)
        if not n:
            print(f"\n{split_name}: empty")
            return

        print(f"\n{'='*60}")
        print(f"{split_name} ({n:,} samples)")
        print(f"{'='*60}")
        print(f"  Min:    {min(lengths):,}")
        print(f"  Max:    {max(lengths):,}")
        print(f"  Mean:   {sum(lengths)/n:.0f}")
        print(f"  Median: {lengths[n//2]:,}")
        print(f"  P90:    {lengths[int(n*0.90)]:,}")
        print(f"  P95:    {lengths[int(n*0.95)]:,}")
        print(f"  P99:    {lengths[int(n*0.99)]:,}")

        buckets = [512, 1024, 2048, 4096, 8192]
        print(f"\n  Distribution:")
        prev = 0
        for limit in buckets:
            count = sum(1 for l in lengths if prev < l <= limit)
            pct = 100 * count / n
            bar = "#" * int(pct / 2)
            print(f"    {prev+1:>5}-{limit:<5}: {count:5,} ({pct:5.1f}%) {bar}")
            prev = limit
        over = sum(1 for l in lengths if l > buckets[-1])
        if over:
            print(f"    {buckets[-1]+1:>5}+     : {over:5,} ({100*over/n:5.1f}%)")

        return lengths

    all_lengths = []
    for split in ("train", "val", "test"):
        if split not in ds:
            continue
        lengths = token_lengths(ds[split])
        report(split, lengths)
        all_lengths.extend(lengths)

    if all_lengths:
        all_lengths.sort()
        n = len(all_lengths)
        max_len = max(all_lengths)
        p99 = all_lengths[int(n * 0.99)]

        print(f"\n{'='*60}")
        print(f"RECOMMENDATION")
        print(f"{'='*60}")
        print(f"  Total samples: {n:,}")
        print(f"  Max length:    {max_len:,}")
        print(f"  P99:           {p99:,}")

        for threshold in [1024, 2048, 4096]:
            over = sum(1 for l in all_lengths if l > threshold)
            pct = 100 * over / n
            print(f"  > {threshold:5,}: {over:5,} ({pct:5.1f}%)")

        if max_len <= 1024:
            print(f"\n  All samples fit in 1024 tokens. Use --max-length 1024.")
        elif max_len <= 2048:
            print(f"\n  All samples fit in 2048 tokens. Use --max-length 2048.")
        else:
            over_2048 = sum(1 for l in all_lengths if l > 2048)
            print(f"\n  {over_2048} samples exceed 2048. Consider --max-length {max_len}")
            print(f"  or reduce candidate count to keep samples shorter.")


@app.local_entrypoint()
def main(
    run_dir: str = "/mnt/gazet/data/v1",
    base_model: str = "unsloth/Qwen3.5-0.8B",
):
    print(f"Checking token lengths:")
    print(f"  Model:   {base_model}")
    print(f"  Run dir: {run_dir}")
    analyze_token_lengths.remote(run_dir, base_model)
    print("Analysis complete!")
