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

### Scale to 10K+
```yaml
# config.yaml - increase targets
sample_targets:
  adjacency: 1000
  containment: 1000
  intersection: 1000
  # ... etc
```

```bash
gazet-dataset full-pipeline --config dataset/config.yaml --append
```

## Commands

```bash
gazet-dataset full-pipeline --config <path>     # Run everything
gazet-dataset build-relations --config <path>   # Build spatial relations
gazet-dataset generate-samples --config <path>  # Generate samples
gazet-dataset validate --config <path>          # Validate dataset
gazet-dataset export --config <path>            # Export train/val/test
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
