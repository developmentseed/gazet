# Dataset Generation Pipeline Flow

This document explains how the optimized pipeline processes data with concrete examples.

**Example Configuration:**
- Countries: `['EC', 'BE', 'KE', 'AE', 'SG']` (5 countries)
- Sample targets: 100 per family × 8 families = 800 samples
- Retry multiplier: 2 (generate 1,600 attempts to get 800 valid samples)
- Max workers: 8

---

## Step 0: Build Entity Inventory (One-Time Setup)

**Script:** `build_inventory.py`

**What it does:** Extracts compact metadata from the full parquet files for fast sampling.

**Input:**
- `divisions_area.parquet` (~500K entities globally)
- `natural_earth.parquet` (~50K entities globally)

**Process:**
```sql
-- For each parquet, extract:
SELECT 
    id, 
    names."primary" AS name,
    subtype,
    country,
    region,
    admin_level,
    ST_Area(geometry) AS area_sq_deg,
    -- bounding box for spatial filtering
FROM read_parquet(...)
WHERE names."primary" IS NOT NULL
```

**Output:**
- `intermediate/divisions_area_inventory.parquet` (~50K rows for 5 countries)
- `intermediate/natural_earth_inventory.parquet` (~50K rows)

**Parallelization:** None (runs once, fast enough)

---

## Step 1: Build Relation Tables (Parallelized)

**Script:** `build_relations.py`

**What it does:** Pre-computes spatial relationships between entities so sample generation doesn't need to run expensive spatial joins.

### Before Optimization (Sequential)
```
Total time: adjacency (60s) + containment (15s) + intersection (10s) + cross_source (8s) = 93s
```

### After Optimization (Parallel)
```
Total time: max(60s, 15s, 10s, 8s) = 60s
```

**4 concurrent tasks, each with its own DuckDB connection:**

#### Task 1: Adjacency Pairs (60s)
```sql
-- Find all touching boundaries within 5 countries
WITH features AS (
    SELECT id, name, subtype, country, geometry, ST_Envelope(geometry) AS bbox
    FROM divisions_area
    WHERE country IN ('EC', 'BE', 'KE', 'AE', 'SG')
)
SELECT 
    a.id AS anchor_id,
    a.name AS anchor_name,
    b.id AS target_id,
    b.subtype AS target_subtype
FROM features a
JOIN features b ON (
    a.id < b.id
    AND ST_Intersects(a.bbox, b.bbox)  -- Fast bbox pre-filter
    AND ST_Touches(a.geometry, b.geometry)  -- Expensive but necessary
)
LIMIT 50000
```
**Output:** `adjacency_pairs.parquet` (50,000 rows)

#### Task 2: Containment Pairs (15s)
```sql
-- Find all parent-child relationships
-- Example: Ecuador contains Quito
SELECT 
    container.id AS container_id,
    container.name AS container_name,
    contained.id AS contained_id,
    contained.subtype AS contained_subtype
FROM features container
JOIN features contained ON (
    container.admin_level < contained.admin_level  -- Parent has lower level
    AND ST_Within(contained.geometry, container.geometry)
)
LIMIT 1000
```
**Output:** `containment_pairs.parquet` (1,000 rows)

#### Task 3: Intersection Pairs (10s)
```sql
-- Find overlapping regions (not touching, not containing)
-- Example: Two administrative regions that overlap
SELECT a.id, a.name, b.id, b.subtype
FROM features a
JOIN features b ON (
    ST_Intersects(a.geometry, b.geometry)
    AND NOT ST_Touches(a.geometry, b.geometry)
    AND NOT ST_Within(a.geometry, b.geometry)
)
LIMIT 500
```
**Output:** `intersection_pairs.parquet` (500 rows)

#### Task 4: Cross-Source Relations (8s)
```sql
-- Find relationships between divisions and natural features
-- Example: Ecuador intersects Pacific Ocean
SELECT 
    d.id AS division_id,
    d.name AS division_name,
    n.id AS natural_id,
    n.name AS natural_name,
    CASE 
        WHEN ST_Touches(...) THEN 'touches'
        WHEN ST_Intersects(...) THEN 'intersects'
    END AS relation_type
FROM divisions d
JOIN natural_features n ON ST_Intersects(d.geometry, n.geometry)
WHERE d.country IN ('EC', 'BE', 'KE', 'AE', 'SG')
  AND n.subtype IN ('sea', 'ocean', 'Lake', 'River')
LIMIT 500
```
**Output:** `cross_source_relations.parquet` (500 rows)

**ThreadPoolExecutor with 4 workers runs all tasks concurrently.**

---

## Step 2: Generate Samples (Batch-Parallelized)

**Script:** `generate_samples.py`

**What it does:** Creates training samples by:
1. Sampling anchors from relation tables
2. Rendering SQL templates
3. Executing SQL to verify it works
4. Building candidate lists with distractors
5. Generating questions

### Work Item Preparation

**Total work items:** 8 families × 100 targets × 2 retry_multiplier = **1,600 items**

```python
work_items = [
    ('adjacency', template_dict_1, 'sample_001', '/path/to/intermediate'),
    ('containment', template_dict_2, 'sample_002', '/path/to/intermediate'),
    ('adjacency', template_dict_3, 'sample_003', '/path/to/intermediate'),
    # ... 1,597 more items
]

# Shuffle for balanced batches
random.shuffle(work_items)

# Partition into 8 batches (one per worker)
batch_size = 1600 / 8 = 200 items per batch
batches = [
    batch_1: items[0:200],    # ~25 of each family (mixed)
    batch_2: items[200:400],
    batch_3: items[400:600],
    # ... 8 batches total
]
```

### Before Optimization (Per-Sample Workers)

```
For each of 1,600 samples:
    - Fork new process
    - Create DuckDB connection
    - INSTALL spatial (5-10ms)
    - LOAD spatial (5-10ms)
    - Import sql_templates module
    - Load 4 relation parquet files (50-100ms)
    - Generate 1 sample (20-50ms)
    - Close connection

Total overhead per sample: ~100ms
Total overhead: 1,600 × 100ms = 160 seconds of pure overhead
```

### After Optimization (Batch Workers)

```
8 workers run in parallel, each processes 200 samples:

Worker 1 (batch of 200 items):
    - Create DuckDB connection (once)
    - INSTALL + LOAD spatial (once, 10ms)
    - Import sql_templates (once)
    - Load 4 relation tables (once, 100ms)
    
    FOR EACH of 200 items:
        - Sample anchor from pre-loaded table (instant)
        - Render SQL template
        - Execute SQL to verify (20-50ms)
        - Build candidate list with Jaro-Winkler (10-30ms)
        - Generate question
    
    - Close connection

Total overhead per worker: ~110ms (one-time)
Total overhead across 8 workers: ~110ms (parallel)
```

**Speedup:** 160s → 0.11s overhead = **~1,450x faster on initialization overhead**

### Sample Generation Example

**Adjacency sample generation:**

```python
# 1. Sample anchor from pre-loaded adjacency_pairs DataFrame
row = adjacency_df.sample(n=1).iloc[0]
# Result: anchor_id='EC-123', anchor_name='Quito', target_subtype='locality'

# 2. Render SQL template
sql = f"""
WITH a AS (
  SELECT geometry FROM divisions_area WHERE id = 'EC-123'
)
SELECT b.id, b.names."primary" AS name, b.geometry
FROM divisions_area AS b, a
WHERE b.id != 'EC-123'
  AND b.subtype = 'locality'
  AND ST_Touches(a.geometry, b.geometry)
"""

# 3. Execute to verify (returns 5 neighboring localities)
result = con.execute(sql).fetchdf()  # 30ms
# ✓ Not empty, sample is valid

# 4. Build candidate list (10 candidates: 1 true + 9 distractors)
candidates = build_candidate_list(
    con, 'EC-123', 'Quito', 'divisions_area', num_candidates=10
)
# Uses Jaro-Winkler to find similar names:
SELECT id, name, subtype, country,
       jaro_winkler_similarity(lower(name), lower('Quito')) AS similarity
FROM divisions_area
WHERE id != 'EC-123'
ORDER BY similarity DESC
LIMIT 9

# Results: ['Quito', 'Cuito', 'Quijos', 'Quinindé', ...]
# Shuffle and reassign IDs: c1, c2, ..., c10

# 5. Generate question
question = "Which localities border Quito?"

# 6. Return TrainingSample with sql_verified=True
```

### Batch Progress Tracking

```
Console output:

Generating 1600 samples across 8 families...
  Split into 8 batches of ~200 items (1 DuckDB init per batch)

  Progress: 200/1600 samples (1/8 batches)   # Worker 1 done
  Progress: 400/1600 samples (2/8 batches)   # Worker 2 done
  Progress: 600/1600 samples (3/8 batches)   # Worker 3 done
  ...
  Progress: 1600/1600 samples (8/8 batches)  # All done

Results by family:
  adjacency           : 185 success /  15 failed (92.5% success rate, target: 100)
  aggregation         : 178 success /  22 failed (89.0% success rate, target: 100)
  buffer              : 192 success /   8 failed (96.0% success rate, target: 100)
  containment         : 188 success /  12 failed (94.0% success rate, target: 100)
  direct_lookup       : 200 success /   0 failed (100% success rate, target: 100)
  intersection        : 181 success /  19 failed (90.5% success rate, target: 100)
  partial_selection   : 175 success /  25 failed (87.5% success rate, target: 100)
  set_operations      : 190 success /  10 failed (95.0% success rate, target: 100)

Total: 1,489 valid samples from 1,600 attempts
```

---

## Step 3: Validate Dataset (Optimized)

**Script:** `validate_dataset.py`

**What it does:** Validates samples in parallel, but **skips SQL re-execution** for samples with `sql_verified: True`.

### Before Optimization

```
For each of 1,489 samples:
    - Execute SQL to verify (30ms)
    - Validate candidates (1ms)
    - Check question (1ms)

Total: 1,489 × 32ms = 47.6 seconds
```

### After Optimization

```
For each of 1,489 samples:
    - Check metadata.sql_verified flag
    - IF True: skip SQL execution (saved 30ms)
    - Validate candidates (1ms)
    - Check question (1ms)

Total: 1,489 × 2ms = 3.0 seconds
```

**Speedup:** 47.6s → 3.0s = **~16x faster**

**Parallelization:** 8 workers process samples in parallel batches

---

## Step 4: Export Splits

**Script:** `export_training_data.py`

**What it does:** Stratified split into train/val/test (80/10/10) by task family.

**Input:** `dataset_validated.jsonl` (1,489 samples)

**Process:**
```python
# Group by family
adjacency_samples: 185
aggregation_samples: 178
buffer_samples: 192
# ... etc

# Split each family 80/10/10
adjacency_train: 148, val: 19, test: 18
aggregation_train: 142, val: 18, test: 18
# ... etc

# Combine and shuffle
train: 1,191 samples
val: 149 samples  
test: 149 samples
```

**Output:**
- `output/train.jsonl` (1,191 samples)
- `output/val.jsonl` (149 samples)
- `output/test.jsonl` (149 samples)

**Parallelization:** None needed (fast enough)

---

## Overall Pipeline Timing

### Before Optimizations
```
Step 0: Build Inventory        :    5s (one-time)
Step 1: Build Relations         :   93s (sequential)
Step 2: Generate Samples        :  320s (160s overhead + 160s generation)
Step 3: Validate Dataset        :   48s (re-executing all SQL)
Step 4: Export Splits           :    2s
                                 ------
Total                           :  468s (~7.8 minutes)
```

### After Optimizations
```
Step 0: Build Inventory        :    5s (one-time)
Step 1: Build Relations         :   60s (parallel, limited by slowest task)
Step 2: Generate Samples        :  165s (0.11s overhead + 165s generation)
Step 3: Validate Dataset        :    3s (skips SQL re-execution)
Step 4: Export Splits           :    2s
                                 ------
Total                           :  235s (~3.9 minutes)
```

**Overall speedup:** 468s → 235s = **~2x faster**

**At 10K scale (100x more samples):**
- Before: ~780 minutes (13 hours)
- After: ~390 minutes (6.5 hours)
- With further optimizations (sampling without replacement, better caching): **<2 hours**

---

## Key Optimizations Summary

| Optimization | Impact | Where |
|-------------|--------|-------|
| **Batch workers** | 1,450x on init overhead | `generate_samples.py` |
| **Parallel relations** | 1.5x on relation building | `build_relations.py` |
| **Jaro-Winkler** | 2-3x on distractor search | `generate_samples.py` |
| **Skip SQL re-validation** | 16x on validation | `validate_dataset.py` |
| **Drop individual JSON files** | 1.2x on I/O | `generate_samples.py` |

**Combined:** Enables scaling from hundreds to tens of thousands of samples efficiently.
