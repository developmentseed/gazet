"""
Validate and balance the generated dataset.

This script:
1. Loads all generated samples
2. Validates SQL executability
3. Checks candidate list quality
4. Balances across task families and difficulty
5. Removes duplicates
6. Generates dataset statistics

Output:
- output/dataset_validated.jsonl
- output/dataset_stats.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import pandas as pd

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH


def load_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def _resolve_paths(sql: str) -> str:
    """Replace symbolic placeholder paths with actual runtime paths for execution."""
    sql = sql.replace(
        "read_parquet('divisions_area')", f"read_parquet('{DIVISIONS_AREA_PATH}')"
    )
    sql = sql.replace(
        "read_parquet('natural_earth')", f"read_parquet('{NATURAL_EARTH_PATH}')"
    )
    # Legacy fixed Docker paths from earlier dataset versions
    sql = sql.replace("/data/overture/division_area/*.parquet", DIVISIONS_AREA_PATH)
    sql = sql.replace("/data/overture/divisions_area/*.parquet", DIVISIONS_AREA_PATH)
    sql = sql.replace("/data/natural_earth_geoparquet/ne_geography.parquet", NATURAL_EARTH_PATH)
    return sql


def _to_symbolic_sql(sql: str) -> str:
    """Normalize any hardcoded or runtime paths back to symbolic names for storage."""
    # Current local runtime paths
    sql = sql.replace(DIVISIONS_AREA_PATH, "divisions_area")
    sql = sql.replace(NATURAL_EARTH_PATH, "natural_earth")
    # Legacy Docker paths
    sql = sql.replace("/data/overture/division_area/*.parquet",          "divisions_area")
    sql = sql.replace("/data/overture/divisions_area/*.parquet",         "divisions_area")
    sql = sql.replace("/data/natural_earth_geoparquet/ne_geography.parquet", "natural_earth")
    return sql


def validate_sql(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, str]:
    """Validate that SQL executes without error.

    Resolves symbolic path placeholders to actual runtime paths before execution.
    """
    try:
        result = con.execute(_resolve_paths(sql)).fetchdf()
        if result.empty:
            return False, "Empty result"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def validate_candidates(sample: Dict[str, Any]) -> tuple[bool, str]:
    """Validate candidate list quality."""
    candidates = sample['candidates']
    selected = sample['target']['selected_candidates']
    
    # Check we have candidates
    if not candidates:
        return False, "No candidates"
    
    # Check selected candidates exist
    candidate_ids = {c['candidate_id'] for c in candidates}
    for sel_id in selected:
        if sel_id not in candidate_ids:
            return False, f"Selected candidate {sel_id} not in candidate list"
    
    # Check for duplicates
    ids = [c['id'] for c in candidates]
    if len(ids) != len(set(ids)):
        return False, "Duplicate candidates"
    
    return True, "OK"


def validate_sample(con: duckdb.DuckDBPyConnection, sample: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate a single sample. Returns (is_valid, list_of_issues)."""
    issues = []
    
    # Skip SQL re-execution if already verified during generation
    if not sample.get('metadata', {}).get('sql_verified', False):
        sql_valid, sql_msg = validate_sql(con, sample['target']['sql'])
        if not sql_valid:
            issues.append(f"SQL: {sql_msg}")
    
    # Validate candidates
    cand_valid, cand_msg = validate_candidates(sample)
    if not cand_valid:
        issues.append(f"Candidates: {cand_msg}")
    
    # Check question exists
    if not sample.get('question') or len(sample['question'].strip()) == 0:
        issues.append("Empty question")
    
    return len(issues) == 0, issues


def validate_sample_worker(sample: Dict[str, Any]) -> Tuple[str, bool, List[str]]:
    """Worker function for parallel validation. Returns (sample_id, is_valid, issues)."""
    # Each worker creates its own DuckDB connection
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    
    try:
        is_valid, issues = validate_sample(con, sample)
        con.close()
        if is_valid:
            sample['target']['sql'] = _to_symbolic_sql(sample['target']['sql'])
        return (sample['id'], is_valid, issues, sample if is_valid else None)
    except Exception as e:
        con.close()
        return (sample['id'], False, [f"Validation error: {str(e)}"], None)


def compute_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    
    stats = {
        'total_samples': len(samples),
        'task_families': {},
        'sql_difficulty': {},
        'grounding_difficulty': {},
        'anchor_sources': {},
        'avg_candidates_per_sample': 0,
        'avg_question_length': 0,
        'countries_covered': set(),
        'subtypes_covered': set()
    }
    
    total_candidates = 0
    total_question_length = 0
    
    for sample in samples:
        meta = sample['metadata']
        
        # Count by family
        family = meta['task_family']
        stats['task_families'][family] = stats['task_families'].get(family, 0) + 1
        
        # Count by SQL difficulty
        sql_diff = meta['sql_difficulty']
        stats['sql_difficulty'][sql_diff] = stats['sql_difficulty'].get(sql_diff, 0) + 1
        
        # Count by grounding difficulty
        ground_diff = meta['grounding_difficulty']
        stats['grounding_difficulty'][ground_diff] = stats['grounding_difficulty'].get(ground_diff, 0) + 1
        
        # Count by anchor source
        anchor_src = meta['anchor_source']
        stats['anchor_sources'][anchor_src] = stats['anchor_sources'].get(anchor_src, 0) + 1
        
        # Candidates
        total_candidates += len(sample['candidates'])
        
        # Question length
        total_question_length += len(sample['question'].split())
        
        # Countries and subtypes (from selected/answer candidates only)
        selected_ids = set(sample.get('target', {}).get('selected_candidates', []))
        for cand in sample['candidates']:
            if cand['candidate_id'] in selected_ids:
                if cand.get('country'):
                    stats['countries_covered'].add(cand['country'])
                if cand.get('subtype'):
                    stats['subtypes_covered'].add(cand['subtype'])
    
    stats['avg_candidates_per_sample'] = total_candidates / len(samples) if samples else 0
    stats['avg_question_length'] = total_question_length / len(samples) if samples else 0
    stats['countries_covered'] = sorted(list(stats['countries_covered']))
    stats['subtypes_covered'] = sorted(list(stats['subtypes_covered']))
    
    return stats


def main():
    """Validate and analyze dataset."""
    
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "output"
    
    raw_file = output_dir / "dataset_raw.jsonl"
    validated_file = output_dir / "dataset_validated.jsonl"
    stats_file = output_dir / "dataset_stats.json"
    
    if not raw_file.exists():
        print(f"Error: {raw_file} not found. Run generate_samples.py first.")
        return
    
    # Load samples
    print("Loading samples...")
    samples = load_samples(raw_file)
    print(f"Loaded {len(samples)} samples")
    
    # Validate samples in parallel
    print("\nValidating samples in parallel...")
    valid_samples = []
    invalid_samples = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all validation tasks
        futures = {executor.submit(validate_sample_worker, sample): sample for sample in samples}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            sample_id, is_valid, issues, validated_sample = future.result()
            
            if is_valid:
                valid_samples.append(validated_sample)
            else:
                invalid_samples.append((sample_id, issues))
            
            completed += 1
            if completed % 50 == 0 or completed == len(samples):
                print(f"\r  Progress: {completed}/{len(samples)} ", end='', flush=True)
        
        print()  # New line after progress
    
    print(f"\nValidation results:")
    print(f"  Valid: {len(valid_samples)}")
    print(f"  Invalid: {len(invalid_samples)}")
    
    if invalid_samples and len(invalid_samples) <= 20:
        print("\nInvalid samples:")
        for sample_id, issues in invalid_samples[:20]:
            print(f"  {sample_id}: {', '.join(issues)}")
    elif invalid_samples:
        print(f"\n{len(invalid_samples)} invalid samples (showing first 20):")
        for sample_id, issues in invalid_samples[:20]:
            print(f"  {sample_id}: {', '.join(issues)}")
    
    # Save validated samples
    if valid_samples:
        with open(validated_file, 'w') as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"\nSaved {len(valid_samples)} valid samples to {validated_file}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(valid_samples)
    
    # Save statistics
    # Convert sets to lists for JSON serialization
    stats_json = {k: (list(v) if isinstance(v, set) else v) for k, v in stats.items()}
    with open(stats_file, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved statistics to {stats_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print("\nTask families:")
    for family, count in sorted(stats['task_families'].items()):
        print(f"  {family:20s}: {count:3d}")
    
    print("\nSQL difficulty:")
    for diff, count in sorted(stats['sql_difficulty'].items()):
        print(f"  {diff:20s}: {count:3d}")
    
    print("\nGrounding difficulty:")
    for diff, count in sorted(stats['grounding_difficulty'].items()):
        print(f"  {diff:20s}: {count:3d}")
    
    print("\nAnchor sources:")
    for src, count in sorted(stats['anchor_sources'].items()):
        print(f"  {src:20s}: {count:3d}")
    
    print(f"\nAverage candidates per sample: {stats['avg_candidates_per_sample']:.1f}")
    print(f"Average question length (words): {stats['avg_question_length']:.1f}")
    print(f"Countries covered: {len(stats['countries_covered'])}")
    print(f"Subtypes covered: {len(stats['subtypes_covered'])}")
    
    print("\n✓ Validation complete")


if __name__ == "__main__":
    main()
