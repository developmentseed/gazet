# Generative text-to-SQL data

Experimental pipeline for generating `question + schema -> DuckDB spatial SQL` training pairs
with open-weight LLMs, validated by executing them against real data. Unlike `dataset/scripts`
(fixed templates over `divisions_area` / `natural_earth`), the generator writes free-form pairs
over arbitrary schemas, so the small model learns DuckDB spatial rather than two tables.

Goal: find the smallest model that writes good spatial SQL on a laptop CPU (llama.cpp / Ollama),
for use as a tool in agentic geospatial workflows.

See [FINDINGS.md](FINDINGS.md) for the decisions, experiments and results so far.

## Pipeline

1. **Data lake** — `build_lake.py` downloads Natural Earth, NYC taxi zones/trips and Overture
   Lisbon places/buildings/roads into `lake/*.parquet` (~34 MB). `build_lake_world.py` adds 20
   tables from UK, European, South American, Asian, African and global publishers (~100 MB);
   `harness.HELD_OUT` lists the tables reserved for the gold eval set (`harness.py prompts N SEED DIR eval`).
2. **Schema cards + scenario prompts** — `harness.py prompts N SEED DIR` derives a compact card per
   table (types, struct fields, top values, ranges, geometry type, extent) and writes stratified
   scenarios (table bundle x skill x difficulty x question style).
3. **Generation** — `or_generate.py MODEL --prompts 'runs/DIR/*.txt' --tag T` calls OpenRouter,
   restricted to fp8/bf16 providers, and records provider, tokens and cost per call.
4. **Execution check** — `collect.py runs/out_X runs/DIR` then `harness.py validate runs/out_X.jsonl`
   runs every SQL (read-only, 30 s timeout) and keeps pairs that return non-empty results.
5. **Semantic grading** — `blind_sample.py RUN_DIR` builds a blind sheet for hand grading;
   `summarize.py RUN_DIR` prints the per-model table.
6. **Small-model baseline** — `run_evals.sh MODEL...` serves GGUFs with llama-server and runs
   `eval_small.py` (execution match against the hand-checked gold pairs).

## Setup

```bash
uv venv .venv && uv pip install --python .venv duckdb pandas pyarrow sqlglot tiktoken requests
.venv/bin/python build_lake.py
mkdir -p ~/.config/openrouter && read -rs K && printf '%s' "$K" > ~/.config/openrouter/key && chmod 600 ~/.config/openrouter/key
```

DuckDB 2.0 notes: spheroid functions need `SET geometry_always_xy = true` (set in `harness.con()`),
and the generator prompt carries a short "DuckDB spatial is not PostGIS" section, which removed
most dialect errors. For llama.cpp on a laptop, use fewer threads than cores (`-t 6` of 8 here);
using all cores collapsed generation speed about tenfold.

## Pilot results (`pilot/`)

Same 12 scenarios (72 pairs) per model; "usable" = executes x hand-graded correct.

| Generator | Executes | Correct | Usable | $ / 1k usable |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 90% | 83% | 75% | ~0.95 |
| GLM-5.3-Flash | 82% | 92% | 75% | ~1.1 |
| Muse Spark Open | 83% | 83% | 69% | ~10 |
| MiniMax-M3 | 72% | 91% | 66% | ~2.3 |
| Muse Glimmer 30B | 72% | 80% | 58% | ~1.9 |
| gpt-oss-120b | 69% | 73% | 51% | ~0.22 |
| Qwen3-Coder-30B-A3B | 57% | 70% | 40% | ~0.11 |
| Qwen3-Coder-480B | 68% | 58% | 40% | ~0.67 |
| DeepSeek-V4-Pro | 81% | 50% | 40% | ~17 |

10-12 graded pairs per model, so differences of one or two items are noise. Zero-shot small models
(Qwen2.5-Coder 0.5B/1.5B, Qwen3-0.6B, Qwen3.5-0.8B, gazet-model) all scored 0% execution match on
the gold pairs (`pilot/small_model_baseline/`), so fine-tuning is required.

`pilot/out.validated.jsonl` is a Haiku reference run; its hand labels live in `eval_small.py`
(`WRONG`, `MINOR`), and `pilot/grading/` holds the blind bake-off grades (C = correct, M = minor
issue, W = wrong).
