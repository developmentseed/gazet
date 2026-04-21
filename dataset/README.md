# Gazet Dataset Generation

Generates synthetic training data for fine-tuning the geocoding model.
Two datasets come out of one pipeline run:

- **SQL generation** — `(question + candidates) -> DuckDB SQL`
- **Place extraction** — `question -> place names JSON`

Both tasks export in **conversation format** (`messages` list of
system/user/assistant turns), ready for chat-template fine-tuning.

---

## Prerequisites

```bash
uv sync
```

You need the Overture and Natural Earth parquet files under `data/` locally,
or on a Modal volume if running in the cloud.

---

## Option A — Run locally (small datasets, development)

Use this when you want to iterate quickly on a laptop with a subset of countries.

**Step 1 — Pick a run name and countries in `config.yaml`**

```yaml
run_name: "v1"   # change this every time you generate fresh data

countries:
  - IN   # India
  - BR   # Brazil
  - US   # United States
  # add more, or use "- all" for every country (slow locally)
```

**Step 2 — Run the full pipeline**

```bash
gazet-dataset full-pipeline --config dataset/config.yaml
```

That's it. It runs all four steps in order and puts the results in
`dataset/output/runs/my-run-001/`.

If you want to run steps individually (e.g. to re-export without regenerating):

```bash
gazet-dataset build-relations  --config dataset/config.yaml  # ~5 min
gazet-dataset generate-samples --config dataset/config.yaml  # ~15 min
gazet-dataset validate         --config dataset/config.yaml  # ~5 min
gazet-dataset export           --config dataset/config.yaml  # <1 min
```

---

## Option B — Run on Modal (large datasets, production)

Use this when you need 10 K+ samples or want to use all countries. Modal
distributes generation across many containers in parallel.

Modal uses two volumes:

- `gazet-data` — read-only source parquets (Overture + Natural Earth). Populated
  once by `modal-upload`.
- `gazet-intermediate` — entity inventories and relation tables built by the
  pipeline. Regenerated on each run.

**Step 1 — One-time setup (only first time, or when source parquets change)**

```bash
modal setup                                                # authenticate
gazet-dataset modal-upload --config dataset/config.yaml    # ~15 min, uploads data/ to gazet-data volume
```

Verify:

```bash
modal volume ls gazet-data
# should show: overture/, natural_earth/, natural_earth_geoparquet/
```

Skip this step on subsequent runs — the volume persists across runs.

**Step 2 — Set run name and targets in `config.yaml`**

```yaml
run_name: "v2"   # bump this every time you regenerate from scratch

countries:
  - all

sample_targets:
  adjacency:     1500
  containment:   1200
  # ... see config.yaml for all families
```

**Step 3 — Run on Modal**

```bash
gazet-dataset modal-generate --config dataset/config.yaml --fresh
```

This builds inventories + relations, generates samples across ~100 containers,
validates, and exports. Output lands in `dataset/output/runs/{run_name}/`.

Flags:

- `--fresh` overwrites `dataset/output/dataset_raw.jsonl` instead of appending.
- `--skip-inventory` reuses `{divisions_area,natural_earth}_inventory.parquet`
  on the intermediate volume.
- `--skip-relations` reuses the seven `*_pairs.parquet` / `*_relations.parquet`
  files on the intermediate volume. Only safe when countries and template
  families are unchanged.

### Fresh-start recipe (after template / SQL / prompt changes)

Always clear stale state so nothing from the previous run leaks in:

```bash
# 1. Bump run_name in config.yaml (e.g. v1 -> v2)

# 2. Wipe the intermediate volume so inventories and relations are rebuilt
modal volume ls gazet-intermediate
# for each file shown:
modal volume rm gazet-intermediate <filename>

# 3. Remove local raw/validated files so nothing gets appended to
rm -f dataset/output/dataset_raw.jsonl dataset/output/dataset_validated.jsonl

# 4. Run the full pipeline
gazet-dataset modal-generate --config dataset/config.yaml --fresh
```

You do NOT need to re-run `modal-upload` — source parquets on `gazet-data`
don't change.

### Faster iteration (same templates, just more samples)

If relations + inventories are still valid from a previous run:

```bash
gazet-dataset modal-generate --config dataset/config.yaml \
  --skip-inventory --skip-relations --fresh
```

---

## Output

After running, your training files are at:

```
dataset/output/runs/{run_name}/
  sql/
    train.jsonl    <- fine-tune the SQL generation model
    val.jsonl
    test.jsonl
  places/
    train.jsonl    <- fine-tune the place extraction model
    val.jsonl
    test.jsonl
  stats.json       <- sample counts by family
```

Each JSONL row is a conversation-format dict:

```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**SQL task**: the system prompt includes the full two-table schema inside
`<SCHEMA>` tags. The user prompt contains only `<CANDIDATES>` CSV and
`<USER_QUERY>`. The assistant response is pretty-printed SQL (via `sqlparse`).
All parquet paths are symbolic (`divisions_area` / `natural_earth`), never
runtime-specific.

**Places task**: the system prompt includes output format, extraction rules,
and the full list of Overture subtypes. The assistant response is a JSON
object with a `places` array.

---

## When to regenerate from scratch

Change `run_name` and regenerate from scratch whenever you:

- Change any SQL templates (`sql_templates.py`)
- Add new template families
- Change the candidate format or count
- Change the system/user prompt structure or content
- Change the export format

For local runs, the default is a clean run. For Modal, `modal-generate` appends
by default; pass `--fresh` to overwrite existing samples.

---

## Troubleshooting

**Very few samples generated for a family**
The generation loop tries `retry_multiplier × target` and discards SQL that
returns empty results. Some families (e.g. `multi_adjacency`, `chained`) have
a lower success rate. Increase `sample_targets` for those families or increase
`retry_multiplier` in `config.yaml`.

**Relations step is slow**
Normal for `countries: [all]` — it's a spatial self-join over millions of
features. Use a country subset for development. Relations only need to be
rebuilt when you add countries or change template families.

**Validate step drops many samples**
The validate step re-executes every SQL query and discards ones that return
empty results. This is expected — check `output/runs/{run_name}/stats.json`
for per-family counts after export.
