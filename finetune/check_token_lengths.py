"""Check token lengths of training samples to validate max_length setting.

Usage
-----
modal run finetune/check_token_lengths.py \
    --train-jsonl /data/train.jsonl \
    --val-jsonl /data/val.jsonl \
    --base-model google/gemma-3-270m-it
"""

from __future__ import annotations

import modal

app = modal.App("gazet-check-token-lengths")

check_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=3.0",
        "pandas>=2.2",
        "transformers>=4.46",
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
def analyze_token_lengths(
    train_jsonl: str,
    val_jsonl: str | None,
    base_model: str,
    schema_file: str | None = None,
):
    from transformers import AutoTokenizer
    from finetune.data import format_dataset_for_sft, load_jsonl_splits, read_text
    from finetune.prompts import DEFAULT_SCHEMA_DETAILS

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading dataset from {train_jsonl}")
    schema_details = read_text(schema_file, DEFAULT_SCHEMA_DETAILS)
    ds = load_jsonl_splits(train_jsonl, val_jsonl)
    formatted = format_dataset_for_sft(ds, schema_details)

    def compute_lengths(split_name: str, dataset):
        print(f"\n{'='*60}")
        print(f"Analyzing {split_name} split ({len(dataset)} samples)")
        print(f"{'='*60}")

        lengths = []
        for row in dataset:
            text = row["prompt"] + row["completion"]
            tokens = tokenizer.encode(text)
            lengths.append(len(tokens))

        lengths.sort()
        n = len(lengths)

        print(f"\nToken length statistics:")
        print(f"  Samples:  {n:,}")
        print(f"  Min:      {min(lengths):,}")
        print(f"  Max:      {max(lengths):,}")
        print(f"  Mean:     {sum(lengths)/n:.0f}")
        print(f"  Median:   {lengths[n//2]:,}")
        print(f"  P90:      {lengths[int(n*0.90)]:,}")
        print(f"  P95:      {lengths[int(n*0.95)]:,}")
        print(f"  P99:      {lengths[int(n*0.99)]:,}")

        buckets = [
            (512, "0-512"),
            (1024, "513-1024"),
            (2048, "1025-2048"),
            (4096, "2049-4096"),
            (8192, "4097-8192"),
            (float("inf"), "8193+"),
        ]

        print(f"\nFrequency distribution:")
        prev_limit = 0
        for limit, label in buckets:
            count = sum(1 for l in lengths if prev_limit < l <= limit)
            pct = 100 * count / n
            bar = "█" * int(pct / 2)
            print(f"  {label:>12}: {count:5,} ({pct:5.1f}%) {bar}")
            prev_limit = limit

        thresholds = [1024, 2048, 4096, 8192]
        print(f"\nSamples exceeding thresholds:")
        for threshold in thresholds:
            count = sum(1 for l in lengths if l > threshold)
            pct = 100 * count / n
            print(f"  > {threshold:5,}: {count:5,} ({pct:5.1f}%)")

        return lengths

    train_lengths = compute_lengths("train", formatted["train"])

    if "val" in formatted:
        val_lengths = compute_lengths("val", formatted["val"])
    else:
        val_lengths = []

    all_lengths = train_lengths + val_lengths
    if all_lengths:
        print(f"\n{'='*60}")
        print(f"COMBINED STATISTICS")
        print(f"{'='*60}")
        all_lengths.sort()
        n = len(all_lengths)
        print(f"  Total samples: {n:,}")
        print(f"  Max length:    {max(all_lengths):,}")
        print(f"  P99:           {all_lengths[int(n*0.99)]:,}")

        for threshold in [1024, 2048, 4096]:
            count = sum(1 for l in all_lengths if l > threshold)
            pct = 100 * count / n
            status = "⚠️  WARNING" if count > 0 and threshold == 2048 else "✓ OK"
            print(f"  > {threshold:5,}: {count:5,} ({pct:5.1f}%) {status}")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    max_len = max(all_lengths) if all_lengths else 0
    over_2048 = sum(1 for l in all_lengths if l > 2048) if all_lengths else 0

    if max_len <= 1024:
        print("✓ All samples fit within 1024 tokens")
        print("  Recommended max_length: 1024")
    elif max_len <= 2048:
        print("✓ All samples fit within 2048 tokens")
        print("  Recommended max_length: 2048")
    elif over_2048 < n * 0.01:
        print(f"⚠️  {over_2048} samples ({100*over_2048/n:.1f}%) exceed 2048 tokens")
        print("  Options:")
        print("    1. Keep max_length=2048 (truncates <1% of samples)")
        print("    2. Increase to max_length=4096 (uses more GPU memory)")
        print("    3. Reduce candidate rows in preprocessing")
    else:
        print(f"⚠️  {over_2048} samples ({100*over_2048/n:.1f}%) exceed 2048 tokens")
        print(f"  Recommended max_length: {max_len} (or reduce candidate rows)")

    print()


@app.local_entrypoint()
def main(
    train_jsonl: str = "/mnt/gazet/data/output/train.jsonl",
    val_jsonl: str | None = "/mnt/gazet/data/output/val.jsonl",
    base_model: str = "google/gemma-3-270m-it",
    schema_file: str | None = None,
):
    print(f"Checking token lengths for:")
    print(f"  Model: {base_model}")
    print(f"  Train: {train_jsonl}")
    if val_jsonl:
        print(f"  Val:   {val_jsonl}")

    analyze_token_lengths.remote(train_jsonl, val_jsonl, base_model, schema_file)
    print("Analysis complete!")
