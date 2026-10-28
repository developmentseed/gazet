# Findings: generative spatial text-to-SQL data

Working log of the investigation started 2026-10-27, carried out together with Claude Code.
It records what we decided, what we measured and what is still open. Numbers come from the files in
`pilot/` unless stated otherwise.

## Goal and decisions

- **Goal:** train the smallest LLM that still writes good DuckDB spatial SQL, fast enough on a laptop
  CPU (llama.cpp / Ollama) to serve as a tool in agentic geospatial workflows. Inspired by gazet and by
  the fast completion models in coding tools. "Type-ahead" is a speed analogy only; the task stays
  `question + schema -> full SQL`.
- **Scope:** spatial-first DuckDB SQL over *any* schema, plus normal analytics SQL, not just gazet's two tables.
- **Callers:** both LLM agents (precise requests, schema passed in) and humans (vague, messy questions).
- **Data form:** free-form question/SQL pairs only, no LLM-authored templates, even if that costs more.
- **Generators:** open-weight models only, called through OpenRouter. Claude Haiku was used once as a quality reference.

## Starting point: gazet's template pipeline

`dataset/scripts/sql_templates.py` has 74 hand-written templates in 15 families (adjacency, buffer,
containment, window functions, ...). They are fixed to `divisions_area` and `natural_earth`. Anchors are
sampled from precomputed spatial relation tables, SQL is filled with `.format()`, and questions come from
fixed hint lists plus light regex rewording. A query is kept if it executes and returns rows.
Variety therefore comes only from the templates, and the schema never changes, so the model learns two
tables rather than DuckDB spatial.

## Validation data lake

`build_lake.py` builds an 11-table GeoParquet lake (~34 MB) with real, varied schemas:

| Source | Tables | What it exercises |
|---|---|---|
| Natural Earth 1:50m / 1:10m | countries, populated_places, rivers, lakes, airports, ports | points, lines, polygons, attribute joins |
| NYC TLC | nyc_taxi_zones (reprojected from EPSG:2263), nyc_taxi_trips (200k sample) | polygon + non-spatial fact table joins, time |
| Overture 2026-10-21, Lisbon bbox | lisbon_places, lisbon_buildings, lisbon_roads | nested structs and lists, large tables |

Each table gets an automatic schema card (types, struct fields, top values, ranges, geometry type, extent).
The cards are what the small model will see.

Candidate sources for growing the lake: Overture (all themes, global), Natural Earth (all layers),
geoBoundaries (CC-BY; avoid GADM, which is non-commercial), OSM extracts from Geofabrik, GHSL urban
centres (already in gazet), city open data (Chicago, SF), US Census TIGER, Source Cooperative collections.

Held-out benchmarks to port from PostGIS: GeoSQL-Bench (14k instances; we found no download yet),
SpatialQueryQA, FloodSQL-Bench, AtlasSQL-BR. General DuckDB coverage: `motherduckdb/duckdb-text2sql-25k`
(CC-BY-SA, ~4% spatial), from DuckDB-NSQL, which used a similar generate-and-validate recipe. OmniSQL and
its critics show that execution checks alone miss about 12% semantic errors.

## Experiment 1: Haiku reference pilot

12 stratified scenarios (table bundle x skill x difficulty x question style), 6 pairs each, 72 pairs total.

- 65/72 (90%) executed with non-empty results, once DuckDB was configured correctly.
- Blind hand review: 74% correct, 10% minor issues, 7% executed but answered the wrong question, 10% empty results.
- Main failures: categorical values it guessed (`'pharmacy'` instead of `pharmacy_and_drug_store`),
  bounding boxes made up for neighbourhoods ("Alfama"), and distance buffers drawn in EPSG:6933 / 3857.
- These pairs plus their hand labels are the first gold set (`pilot/out.validated.jsonl`; labels in `eval_small.py`).

## Experiment 2: zero-shot small models on the laptop CPU

Five GGUF models served with llama.cpp (6 threads) on the 53 gold pairs:

| Model | Executes | Execution match | Decode tok/s | Median s/query |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B Q4 | 25% | 0% | 51 | 1.8 |
| Qwen3-0.6B Q8 | 38% | 0% | 36 | 2.4 |
| Qwen3.5-0.8B Q4 (gazet base) | 8% | 0% | 29 | 16 |
| gazet-model Q8 (fine-tuned) | 17% | 0% | 33 | 9.2 |
| Qwen2.5-Coder-1.5B Q4 | 47% | 0% | 19 | 4.9 |

- Fine-tuning is required. Even the 1.5B produces wrong-in-substance SQL (missing GROUP BY columns,
  degree-based measurements, made-up joins). gazet-model does not transfer to new schemas.
- **Prompt processing dominates latency.** It runs at only 40-100 tok/s on this machine, so an
  uncached 1,000-token schema costs 10-25 s while generating the SQL takes 2-5 s. Design implications:
  put the schema first so llama.cpp can reuse its cached prefix, keep cards compact, and let agents
  pass only the tables they need.
- Qwen3.5-0.8B often fails to stop (33/53 answers hit the 320-token cap).

## Experiment 3: open-model bake-off on OpenRouter

About 160 paid open-weight models are on OpenRouter. The same model is served at very different
precisions: the cheapest providers are often fp4/int4. Generation is pinned to fp8/bf16 providers,
and each pair records which provider produced it.

New releases checked on 2026-10-28:

- **Meta Muse Spark Open** (816B / 42B active, Oct 15): its license allows distillation, but
  any model trained on its outputs must display "Built with Muse Spark Open".
- **Meta Muse Glimmer 30B** (Apache 2.0, GGUF available).
- **Meta MobileMoE**: right size for a student model, but it has a non-commercial research
  license and needs custom code, so it can't be shipped.
- **Microsoft:** no new general-purpose open model. FrogNano-4B (MIT) is notable for its recipe:
  reinforcement learning on ~1,500 synthetic, automatically validated tasks. That maps directly onto
  reinforcement learning with an execution-match reward after supervised fine-tuning.

Same 12 scenarios for all models. v1 is the original prompt; v2 adds a "DuckDB spatial is NOT PostGIS"
section (verified rule by rule against DuckDB 2.0). Results below are for v2. "Usable" means executes
x hand-graded correct (blind, 10-12 graded pairs per model).

| Generator | Executes | Correct | Usable | $ per 1k usable | Reasoning tok/call |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 90% | 83% | 75% | ~0.95 | 7.2k |
| GLM-5.3-Flash | 82% | 92% | 75% | ~1.1 | 13.6k |
| Muse Spark Open | 83% | 83% | 69% | ~10 | 8.8k |
| MiniMax-M3 | 72% | 91% | 66% | ~2.3 | 6.1k |
| Muse Glimmer 30B | 72% | 80% | 58% | ~1.9 | 4.5k |
| gpt-oss-120b | 69% | 73% | 51% | ~0.22 | 2.0k |
| Qwen3-Coder-30B-A3B | 57% | 70% | 40% | ~0.11 | 0 |
| Qwen3-Coder-480B | 68% | 58% | 40% | ~0.67 | 0 |
| DeepSeek-V4-Pro | 81% | 50% | 40% | ~17 | 11.3k |

- **Recommendation:** DeepSeek-V4.1-Flash and GLM-5.3-Flash. They match Haiku's usable yield at about
  $1 per 1k correct pairs. Using both gives two model families for diversity and a fallback. Larger
  models did not buy quality here.
- **The PostGIS note works.** PostGIS-style errors across all models went from 69 to 16, and to zero
  for the strongest four. Typical PostGIS habits: SRID arguments in `ST_MakeEnvelope`/`ST_GeomFromText`,
  integer EPSG codes in `ST_Transform`, `ST_Collect` used as an aggregate, `ST_Distance(a, b, 'spheroid')`.
- **Wrong queries that still execute** (13 of 102 graded) need a semantic filter:
  - distance/buffer in Web Mercator (4);
  - join fan-out double counting (2);
  - unit errors (1);
  - wrong proxies such as centroid distance used as "nearest" (3);
  - logic bugs (2);
  - geography the model made up (1).
- **Reasoning models use 5k-15k reasoning tokens per call.** An 8k output cap truncated most of their
  answers; 32k is needed. Cost estimates must use measured tokens.
- Grading caveat: with ~12 graded pairs per model, the top four are statistically indistinguishable.

Total OpenRouter spend for the pilot: about $2.8.

## Experiment 4 (in progress): grounded generation (v3)

Prompt changes based on the bake-off failures:

- **Grounding:** each scenario gets random real rows per table (with centroid lon/lat) and full
  value lists for low-cardinality columns. The small model still sees only the compact card.
- **New rules:**
  - no places that aren't in the data (give coordinates instead);
  - never measure in EPSG:3857/6933;
  - deterministic results (no `current_date`; unique `ORDER BY` with `LIMIT`);
  - watch join fan-out.
- **Scale:** 48 scenarios x 6 pairs with DeepSeek-V4.1-Flash and GLM-5.3-Flash (~576 pairs);
  results go to `runs/` and will be snapshotted once graded.

## Environment gotchas

- DuckDB 2.0 spheroid functions need `SET geometry_always_xy = true`, or they refuse to run.
- `/tmp` on the ChromeOS Linux container is RAM (tmpfs): model files there starved llama.cpp of memory.
- llama.cpp with all 8 threads collapsed decode about tenfold (71 -> 7 tok/s) because of VM thread
  contention; use fewer threads than cores.
- The validator needs a per-query timeout (30 s, enforced by interrupting the connection); some
  generated spatial joins run for many minutes.

## Next steps

1. Grade the v3 round and compare it with v2. If grounding works, add rules as new failure modes show up.
2. **Judge step.** Build an LLM judge from a different model family (MiniMax-M3 is a candidate) and
   check it against the ~170 hand labels. Add a self-consistency check: a second model writes SQL from
   the question alone, and the pair is kept only when the result sets match.
3. **Gold evaluation set.** ~300-500 pairs on held-out datasets (never used in training), with
   deterministic ordering, reviewed by hand.
4. **First training set.** ~20k pairs (mixed agent-style and human-style questions, plus repair
   examples where failed SQL + error leads to fixed SQL). Fine-tune a size sweep
   (Qwen2.5-Coder-0.5B, Qwen3-0.6B, Qwen2.5-Coder-1.5B) with `finetune/`. Plot execution match against
   laptop latency.
5. Later: reinforcement learning with an execution-match reward (FrogNano-style), and a larger, more diverse data lake.
