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
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


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
    
    # Map each task family to the relation tables it draws anchors from.
    # A family can need multiple relation types.
    family_to_relations = {
        'direct_lookup':      [],
        'adjacency':          ['adjacency'],
        'multi_adjacency':    ['adjacency', 'common_neighbor'],
        'containment':        ['containment'],
        'intersection':       ['intersection', 'cross_source'],
        'buffer':             ['adjacency'],
        'chained':            ['coastal_containment', 'landlocked_containment', 'containment'],
        'difference':         ['containment', 'cross_source'],
        'border_corridor':    ['adjacency'],
        'set_operations':     ['containment', 'cross_source'],
        'partial_selection':  ['containment', 'cross_source'],
        'aggregation':        ['containment'],
        'window_function':    [],
        'attribute_filter':   [],
    }

    relation_needs: Dict[str, int] = {}
    for family, target in sample_targets.items():
        for rel_type in family_to_relations.get(family, []):
            needed = int(target * retry_mult * safety)
            relation_needs[rel_type] = relation_needs.get(rel_type, 0) + needed

    # common_neighbor is derived from adjacency — keep its limit proportional
    if 'common_neighbor' not in relation_needs and 'adjacency' in relation_needs:
        relation_needs['common_neighbor'] = relation_needs['adjacency'] * 3
    
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
    
    print("\nRelation tables built successfully")


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
    
    print("\nSamples generated successfully")


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
    
    print("\nDataset validated successfully")


def export_dataset(config_path: Path):
    """Run dataset export for both SQL generation and place extraction tasks."""
    print("=" * 60)
    print("STEP 4: Exporting Dataset")
    print("=" * 60)

    from dataset.scripts.export_training_data import main as export_main
    export_main(config_path=config_path)

    print("\nDataset exported successfully")


def modal_upload(config_path: Path):
    """Upload local data to Modal volume."""
    subprocess.run(
        [sys.executable, "-m", "modal", "run",
         "dataset/modal_app.py::upload_data"],
        check=True
    )


def modal_generate(config_path: Path, num_containers: int = 0,
                   skip_inventory: bool = False, skip_relations: bool = False,
                   fresh: bool = False):
    """Run distributed generation on Modal (appends by default)."""
    cmd = [
        sys.executable, "-m", "modal", "run",
        "dataset/modal_app.py::run_pipeline",
        "--config-path", str(config_path),
    ]
    if num_containers > 0:
        cmd.extend(["--num-containers", str(num_containers)])
    if skip_inventory:
        cmd.append("--skip-inventory")
    if skip_relations:
        cmd.append("--skip-relations")
    if fresh:
        cmd.append("--fresh")
    
    subprocess.run(cmd, check=True)
    validate_dataset(config_path)
    export_dataset(config_path)


def full_pipeline(config_path: Path, append: bool = False):
    """Run the full pipeline."""
    print("Running full dataset generation pipeline")
    
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
        print("Inventory files not found, building...")
        from dataset.scripts import build_inventory
        build_inventory.main()
    
    # Check if we need to rebuild relations
    need_rebuild = should_rebuild_relations(config, intermediate_dir, append)
    
    if need_rebuild:
        build_relations(config_path)
    else:
        print("Using existing relation tables (append mode, same countries)")
    
    generate_samples(config_path, append=append)
    validate_dataset(config_path)
    export_dataset(config_path)
    
    print("\nPipeline complete")


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
  
  # Upload data to Modal volume (one-time)
  python cli.py modal-upload --config ../config.yaml
  
  # Run distributed generation on Modal
  python cli.py modal-generate --config ../config.yaml
  python cli.py modal-generate --config ../config.yaml --num-containers 100
  python cli.py modal-generate --config ../config.yaml --skip-inventory --skip-relations
        """
    )
    
    parser.add_argument(
        'command',
        choices=['build-relations', 'generate-samples', 'validate', 'export',
                 'full-pipeline', 'modal-upload', 'modal-generate'],
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
    
    parser.add_argument(
        '--num-containers',
        type=int,
        default=0,
        help='Number of Modal containers (0 = use config default)'
    )
    
    parser.add_argument(
        '--skip-inventory',
        action='store_true',
        help='Skip inventory building on Modal'
    )
    
    parser.add_argument(
        '--skip-relations',
        action='store_true',
        help='Skip relation building on Modal'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Overwrite existing dataset instead of appending (Modal only)'
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
        elif args.command == 'modal-upload':
            modal_upload(args.config)
        elif args.command == 'modal-generate':
            modal_generate(
                args.config,
                num_containers=args.num_containers,
                skip_inventory=args.skip_inventory,
                skip_relations=args.skip_relations,
                fresh=args.fresh,
            )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
