"""
Export validated dataset to train/val/test splits.

This script:
1. Loads validated samples
2. Splits into train (80%), val (10%), test (10%)
3. Ensures balanced splits across task families
4. Exports to JSONL format

Output:
- output/train.jsonl (80% of samples)
- output/val.jsonl (10% of samples)
- output/test.jsonl (10% of samples)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def stratified_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples by task family to ensure balanced distribution."""
    
    random.seed(random_seed)
    
    # Group by task family
    by_family = defaultdict(list)
    for sample in samples:
        family = sample['metadata']['task_family']
        by_family[family].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Split each family
    for family, family_samples in by_family.items():
        # Shuffle
        random.shuffle(family_samples)
        
        n = len(family_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_samples.extend(family_samples[:n_train])
        val_samples.extend(family_samples[n_train:n_train + n_val])
        test_samples.extend(family_samples[n_train + n_val:])
    
    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def save_split(samples: List[Dict[str, Any]], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def print_split_stats(split_name: str, samples: List[Dict[str, Any]]):
    """Print statistics for a split."""
    families = defaultdict(int)
    for sample in samples:
        family = sample['metadata']['task_family']
        families[family] += 1
    
    print(f"\n{split_name}:")
    print(f"  Total: {len(samples)}")
    for family, count in sorted(families.items()):
        print(f"    {family:20s}: {count:3d}")


def print_country_stats(samples: List[Dict[str, Any]]):
    """Print country distribution statistics."""
    country_counts = defaultdict(int)
    
    # Extract countries from selected/answer candidates only
    for sample in samples:
        selected_ids = set(sample.get('target', {}).get('selected_candidates', []))
        countries_in_sample = set()
        for candidate in sample.get('candidates', []):
            if candidate.get('candidate_id') in selected_ids:
                country = candidate.get('country')
                if country:
                    countries_in_sample.add(country)
        
        # Count each unique country once per sample
        for country in countries_in_sample:
            country_counts[country] += 1
    
    if not country_counts:
        print("\nNo country information found in samples")
        return
    
    print(f"\nCOUNTRY DISTRIBUTION:")
    print(f"  Total unique countries: {len(country_counts)}")
    print(f"\n  Top countries by sample count:")
    
    # Sort by count descending
    sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Show top 20
    for country, count in sorted_countries[:20]:
        percentage = (count / len(samples)) * 100
        print(f"    {country:3s}: {count:4d} samples ({percentage:5.1f}%)")
    
    if len(sorted_countries) > 20:
        remaining = len(sorted_countries) - 20
        remaining_count = sum(c for _, c in sorted_countries[20:])
        print(f"    ... and {remaining} more countries ({remaining_count} samples)")


def main():
    """Export dataset splits."""
    
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "output"
    
    validated_file = output_dir / "dataset_validated.jsonl"
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    
    if not validated_file.exists():
        print(f"Error: {validated_file} not found. Run 05_validate_dataset.py first.")
        return
    
    # Load validated samples
    print("Loading validated samples...")
    samples = load_samples(validated_file)
    print(f"Loaded {len(samples)} samples")
    
    # Split
    print("\nSplitting dataset (80/10/10)...")
    train_samples, val_samples, test_samples = stratified_split(samples)
    
    # Save splits
    print("\nSaving splits...")
    save_split(train_samples, train_file)
    save_split(val_samples, val_file)
    save_split(test_samples, test_file)
    
    print(f"  Train: {train_file} ({len(train_samples)} samples)")
    print(f"  Val:   {val_file} ({len(val_samples)} samples)")
    print(f"  Test:  {test_file} ({len(test_samples)} samples)")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    
    print_split_stats("TRAIN", train_samples)
    print_split_stats("VAL", val_samples)
    print_split_stats("TEST", test_samples)
    
    # Print country distribution
    print("\n" + "=" * 60)
    print("GEOGRAPHIC DISTRIBUTION")
    print("=" * 60)
    print_country_stats(samples)
    
    print("\n✓ Export complete")
    print(f"\nReady for training!")
    print(f"  Training data: {train_file}")
    print(f"  Validation data: {val_file}")
    print(f"  Test data: {test_file}")


if __name__ == "__main__":
    main()
