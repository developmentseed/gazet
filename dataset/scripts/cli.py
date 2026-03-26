#!/usr/bin/env python3
"""
CLI for synthetic dataset generation.

Usage:
    python cli.py build-relations --config ../config.yaml
    python cli.py generate-samples --config ../config.yaml
    python cli.py generate-samples --config ../config.yaml --append
    python cli.py full-pipeline --config ../config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import subprocess
from typing import Dict, Set
import pandas as pd


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def should_rebuild_relations(config: dict, intermediate_dir: Path, append: bool) -> bool:
    """Check if relation tables need to be rebuilt.
    
    Returns True if:
    - Not in append mode (always rebuild)
    - Relation tables don't exist
    - Countries in config differ from countries in existing relation tables
    """
    if not append:
        return True
    
    # Check if relation tables exist
    adjacency_file = intermediate_dir / "adjacency_pairs.parquet"
    if not adjacency_file.exists():
        print("WARNING: Relation tables not found, will rebuild despite append mode")
        return True
    
    # Check if countries have changed
    try:
        df = pd.read_parquet(adjacency_file)
        if 'anchor_country' in df.columns:
            existing_countries = set(df['anchor_country'].unique())
            config_countries = set(config['countries'])
            
            if existing_countries != config_countries:
                print(f"WARNING: Countries changed:")
                print(f"    Previous: {sorted(existing_countries)}")
                print(f"    New: {sorted(config_countries)}")
                print(f"    Will rebuild relation tables to include new countries")
                return True
            else:
                print(f"Countries unchanged: {sorted(config_countries)}")
                return False
        else:
            # Can't determine countries, rebuild to be safe
            print("WARNING: Cannot determine countries from existing tables, will rebuild")
            return True
    except Exception as e:
        print(f"WARNING: Error reading existing relation tables: {e}")
        print("    Will rebuild to be safe")
        return True


def calculate_relation_limits(config: dict) -> Dict[str, int]:
    """Auto-calculate relation limits based on sample targets."""
    sample_targets = config['sample_targets']
    retry_mult = config['generation']['retry_multiplier']
    safety = config.get('auto_scaling', {}).get('safety_factor', 1.5)
    
    # Map task families to relation types they need
    family_to_relation = {
        'adjacency': 'adjacency',
        'containment': 'containment',
        'intersection': 'intersection',
        'buffer': 'adjacency',  # Buffer uses adjacency pairs
        'set_operations': 'intersection',  # Set ops use intersection pairs
        'partial_selection': 'containment',  # Partial uses containment
        'aggregation': 'containment',  # Aggregation uses containment
        'direct_lookup': None,  # Uses inventory only
    }
    
    # Calculate required limits by summing needs per relation type
    relation_needs = {}
    for family, target in sample_targets.items():
        relation_type = family_to_relation.get(family)
        if relation_type:
            needed = int(target * retry_mult * safety)
            relation_needs[relation_type] = relation_needs.get(relation_type, 0) + needed
    
    # Add cross-source (used by mixed-source partial selection)
    partial_target = sample_targets.get('partial_selection', 0)
    relation_needs['cross_source'] = int(partial_target * retry_mult * safety * 0.3)
    
    # Apply manual overrides if specified
    manual = config.get('auto_scaling', {}).get('manual_limits', {})
    relation_needs.update(manual)
    
    return relation_needs


def build_relations(config_path: Path):
    """Run relation building with config."""
    config = load_config(config_path)
    
    # Auto-calculate relation limits
    relation_limits = calculate_relation_limits(config)
    
    print("=" * 60)
    print("STEP 1: Building Relation Tables")
    print("=" * 60)
    print(f"Countries: {', '.join(config['countries'])}")
    print(f"\nAuto-calculated relation limits:")
    for rel_type, limit in relation_limits.items():
        print(f"  {rel_type:20s}: {limit:,}")
    print()
    
    # Import and run the relation builder
    from dataset.scripts import build_relations
    
    # Run with config parameters
    build_relations.main(
        countries=config['countries'],
        relation_limits=relation_limits
    )
    
    print("\n✓ Relation tables built successfully")


def generate_samples(config_path: Path, append: bool = False):
    """Run sample generation with config."""
    config = load_config(config_path)
    
    print("=" * 60)
    print("STEP 2: Generating Samples")
    print("=" * 60)
    print(f"Targets: {config['sample_targets']}")
    print(f"Workers: {config['generation']['max_workers']}")
    print(f"Append mode: {append or config['generation']['append_mode']}")
    print()
    
    # Simple import - no number prefixes needed
    from dataset.scripts import generate_samples as gs_module
    
    # Override config values
    gs_module.TARGET_COUNTS = config['sample_targets']
    gs_module.MAX_WORKERS = config['generation']['max_workers']
    gs_module.RETRY_MULTIPLIER = config['generation']['retry_multiplier']
    gs_module.APPEND_MODE = append or config['generation']['append_mode']
    
    # Run the main function
    gs_module.main()
    
    print("\n✓ Samples generated successfully")


def validate_dataset(config_path: Path):
    """Run dataset validation."""
    print("=" * 60)
    print("STEP 3: Validating Dataset")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, str(script_dir / "validate_dataset.py")],
        check=True
    )
    
    print("\n✓ Dataset validated successfully")


def export_dataset(config_path: Path):
    """Run dataset export."""
    print("=" * 60)
    print("STEP 4: Exporting Dataset")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, str(script_dir / "export_training_data.py")],
        check=True
    )
    
    print("\n✓ Dataset exported successfully")


def full_pipeline(config_path: Path, append: bool = False):
    """Run the full pipeline."""
    print("\n" + "=" * 60)
    print("RUNNING FULL DATASET GENERATION PIPELINE")
    print("=" * 60 + "\n")
    
    config = load_config(config_path)
    
    # Check if inventory exists, create if not
    script_dir = Path(__file__).parent
    intermediate_dir = script_dir.parent / "intermediate"
    inventory_files = [
        intermediate_dir / "divisions_area_inventory.parquet",
        intermediate_dir / "natural_earth_inventory.parquet"
    ]
    
    inventory_missing = any(not f.exists() for f in inventory_files)
    
    if inventory_missing:
        print("=" * 60)
        print("STEP 0: Building Entity Inventory")
        print("=" * 60)
        print("Inventory files not found. Building inventory...\n")
        
        from dataset.scripts import build_inventory
        build_inventory.main()
        
        print("\n✓ Inventory built successfully\n")
    
    # Check if we need to rebuild relations
    need_rebuild = should_rebuild_relations(config, intermediate_dir, append)
    
    if need_rebuild:
        build_relations(config_path)
    else:
        print("Using existing relation tables (append mode, same countries)")
    
    generate_samples(config_path, append=append)
    validate_dataset(config_path)
    export_dataset(config_path)
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build relation tables only
  python cli.py build-relations --config ../config.yaml
  
  # Generate samples only
  python cli.py generate-samples --config ../config.yaml
  
  # Generate and append to existing dataset
  python cli.py generate-samples --config ../config.yaml --append
  
  # Run full pipeline
  python cli.py full-pipeline --config ../config.yaml
  
  # Run full pipeline in append mode (skip relation building)
  python cli.py full-pipeline --config ../config.yaml --append
        """
    )
    
    parser.add_argument(
        'command',
        choices=['build-relations', 'generate-samples', 'validate', 'export', 'full-pipeline'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing dataset instead of overwriting'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run the appropriate command
    try:
        if args.command == 'build-relations':
            build_relations(args.config)
        elif args.command == 'generate-samples':
            generate_samples(args.config, args.append)
        elif args.command == 'validate':
            validate_dataset(args.config)
        elif args.command == 'export':
            export_dataset(args.config)
        elif args.command == 'full-pipeline':
            full_pipeline(args.config, args.append)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
