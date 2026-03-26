# Dataset Generation CLI

Generate synthetic text-to-SQL training datasets.

## Quick Start

```bash
# Install
uv sync

# Generate dataset
gazet-dataset full-pipeline --config dataset/config.yaml
```

## Configuration

Edit `dataset/config.yaml`:

```yaml
countries:
  - EC  # Ecuador
  - BE  # Belgium
  - KE  # Kenya
  - AE  # UAE
  - SG  # Singapore
  - CH  # Switzerland

sample_targets:
  direct_lookup: 100
  adjacency: 100
  containment: 100
  intersection: 100
  buffer: 100
  set_operations: 100
  partial_selection: 100
  aggregation: 100

generation:
  max_workers: 8
  retry_multiplier: 2
  append_mode: true

auto_scaling:
  safety_factor: 1.5  # Auto-calculates relation limits
```

## Growing Your Dataset

### Start Small
```bash
# Generate initial dataset (e.g., 100 samples)
gazet-dataset full-pipeline --config dataset/config.yaml
```

### Add More Samples (Same Countries)
```bash
# Increase sample_targets in config.yaml, then:
gazet-dataset full-pipeline --config dataset/config.yaml --append
```

### Add New Countries
```bash
# Add countries to config.yaml, then:
gazet-dataset full-pipeline --config dataset/config.yaml --append
# Auto-rebuilds relations if countries changed
```

### Scale to 10K+ with Modal

For large datasets, use Modal to distribute generation across cloud containers:

```bash
# One-time setup: install modal and authenticate
uv sync --group dataset
modal setup

# Upload parquet data to Modal volume (one-time)
gazet-dataset modal-upload --config dataset/config.yaml

# Run distributed generation (50 containers by default)
gazet-dataset modal-generate --config dataset/config.yaml

# Or override container count
gazet-dataset modal-generate --config dataset/config.yaml --num-containers 100

# Skip inventory/relations if already built
gazet-dataset modal-generate --config dataset/config.yaml --skip-inventory --skip-relations
```

Or run Modal directly:
```bash
modal run dataset/modal_app.py::upload_data --data-dir data
modal run dataset/modal_app.py::run_pipeline --config-path dataset/config.yaml
```

Configure in `config.yaml`:
```yaml
countries:
  - all          # Use all countries for maximum diversity

sample_targets:
  adjacency: 1250
  containment: 1250
  # ... 8 families x 1250 = 10K samples

modal:
  num_containers: 50
  container_cpu: 2
  container_memory: 4096
```

## Commands

```bash
gazet-dataset full-pipeline --config <path>     # Run everything
gazet-dataset build-relations --config <path>   # Build spatial relations
gazet-dataset generate-samples --config <path>  # Generate samples
gazet-dataset validate --config <path>          # Validate dataset
gazet-dataset export --config <path>            # Export train/val/test
gazet-dataset modal-upload --config <path>      # Upload data to Modal volume
gazet-dataset modal-generate --config <path>    # Distributed generation via Modal
```

**Options:**
- `--append`: Add to existing dataset instead of overwriting

## Output

- `dataset/output/dataset_raw.jsonl` - Generated samples
- `dataset/output/dataset_validated.jsonl` - Validated samples
- `dataset/output/train.jsonl` - Training split
- `dataset/output/val.jsonl` - Validation split
- `dataset/output/test.jsonl` - Test split

## Tips

- Start with 2-3 countries and small sample targets
- Use `--append` to grow dataset incrementally
- Relation limits auto-calculate from sample targets
- Check success rates in output summary
