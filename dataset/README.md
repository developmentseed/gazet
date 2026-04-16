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
run_name: "my-run-001"   # change this every time you generate fresh data

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

**Step 1 — One-time setup**

```bash
modal setup   # authenticate with Modal (one time)
gazet-dataset modal-upload --config dataset/config.yaml   # upload parquet data to Modal volume
```

**Step 2 — Set run name and targets in `config.yaml`**

```yaml
run_name: "v2-full-10k"

countries:
  - all

sample_targets:
  adjacency:     1250
  containment:   1250
  # ... see config.yaml for all families
```

**Step 3 — Run on Modal**

```bash
gazet-dataset modal-generate --config dataset/config.yaml
```

This builds relations, generates samples, validates, and exports — same as
`full-pipeline` but distributed across 100 cloud containers.

If relations are already built from a previous run (same countries, same
template version), skip rebuilding them:

```bash
gazet-dataset modal-generate --config dataset/config.yaml --skip-relations
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

Change `run_name` and run without `--append` whenever you:

- Change any SQL templates (`sql_templates.py`)
- Add new template families
- Change the candidate format or count
- Change the system/user prompt structure or content
- Change the export format (e.g. prompt/completion to messages)

Use `--append` only when you're adding more samples of the same type
(e.g. adding more countries to an existing run with identical templates).

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

**`dataset_validated.jsonl` has far fewer samples than `dataset_raw.jsonl`**
The validate step re-executes every SQL query and drops ones that return empty
results. This is expected — check `output/dataset_stats.json` for per-family
pass rates.
