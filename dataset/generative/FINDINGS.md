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

## Experiment 4: grounded generation (v3, `round_v3/`)

Prompt changes based on the bake-off failures:

- **Grounding:** each scenario gets random real rows per table (with centroid lon/lat) and full
  value lists for low-cardinality columns. The small model still sees only the compact card.
- **New rules:**
  - no places that aren't in the data (give coordinates instead);
  - never measure in EPSG:3857/6933;
  - deterministic results (no `current_date`; unique `ORDER BY` with `LIMIT`);
  - watch join fan-out.

48 new scenarios x 6 pairs per model (576 pairs, $0.47 total). Blind grading of 24 pairs per model:

| Generator | Correct (pilot -> v3) | Executes | Executes, excluding the spheroid CRS quirk | Usable | Usable with the cast rule |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 83% -> 96% | 86% | 94% | 83% | ~90% |
| GLM-5.3-Flash | 92% -> 92% | 75% | 91% | 68% | ~84% |

- **Grounding works.** Questions now use real names (Avenida das Tulipas, Maiduguri, Teatro Nacional de
  São Carlos) and real categories. No EPSG:3857/6933 measurement and no `current_date` appear in any of
  the 576 queries, and there are zero PostGIS-dialect errors. The queries are also richer:
  per-group ranking, cumulative shares, land/water splits, local UTM zones (EPSG:32718 for Cusco).
- **The "use spheroid functions" rule exposed a DuckDB 2.0 quirk.** `ST_Distance_Spheroid` and
  `ST_DWithin_Spheroid` reject CRS-typed columns (`GEOMETRY('OGC:CRS84')`), even when both sides share
  the same CRS; `geometry::GEOMETRY` fixes it. This caused 25 DeepSeek and 55 GLM failures.
  Real GeoParquet carries a CRS too, so the cast is now a prompt rule and the small model should learn it.
- **The two remaining wrong pairs:**
  - a UTM zone applied to global data ("countries near the Dnipro" puts Brazil at 0 km); now a prompt rule;
  - a nonsense question that builds points from taxi-zone IDs.
- **DeepSeek-V4.1-Flash is the primary generator:** higher yield and fewer reasoning tokens.
  GLM-5.3-Flash stays as the second family for diversity.

## Experiment 5: LLM judge calibration (`judge_calibration/`)

`judge.py` gives the judge the schema cards, question, SQL and an 8-row result preview, and asks for
correct / minor / wrong using the same criteria as the hand grading. Calibrated on all 215 hand-labelled
pairs (175 correct, 20 minor, 20 wrong):

| Judge | Wrong caught | Correct kept | Kept set hand-labelled correct | $ per 1k pairs judged |
|---|---|---|---|---|
| MiniMax-M3 | 19/20 | 158/175 (90%) | 93% (vs 81% before judging) | ~1.6 |
| gpt-oss-120b | 17/20 | 133/175 (76%) | 92% | ~0.13 |

- **MiniMax-M3 is the judge.** It catches nearly every wrong pair and loses few good ones.
- **The judge also found mistakes in the hand labels.** Of the 17 "correct" pairs it rejected, about 6
  are real errors that I missed:
  - a reference airport name matching two rows, which doubled every count;
  - an area threshold off by 1000x (5e9 m² is 5,000 km², not 5 million);
  - a missing January filter on data that has outlier dates back to 2009;
  - `&&` (bounding-box overlap) used where the question asked about the actual geometry;
  - a category filter not reapplied in an outer query.

  The rest are judgement calls (arbitrary HAVING thresholds, bounding boxes approximating named areas,
  centroid distances). So the kept set is likely cleaner than 93%.
- **The one wrong pair it missed:** a "southern hemisphere" envelope that covers only the western half.
- **Cost:** generation plus judging is about $3 per 1k accepted pairs, so ~$60 for a 20k-pair training set.

## Experiment 6: self-consistency check (`consistency.py`, `judge_calibration/consistency_*`)

A second model writes SQL from the schema cards and question alone (the student's input), always from a
different family than the generator. Both queries run in isolated processes and the full result sets are
compared, ignoring row and column order ("lenient": numbers to 2 significant figures, extra columns allowed).
Calibrated on the same 215 labelled pairs ($0.23):

| Keep rule | Kept | Correct kept | Wrong kept | Kept set correct |
|---|---|---|---|---|
| judge correct | 169 | 158/175 | 1 | 93% |
| both models agree | 73 | 69/175 | 1 | 95% |
| judge correct AND agree | 68 | 65/175 | 0 | 96% |
| judge correct OR agree | 174 | 162/175 | 2 | 93% |

- **The two checks are complementary.** The judge missed the half-globe "southern hemisphere"
  envelope, and the solver disagreed with it. The solver agreed on a buffer drawn in a distorted
  projection, and the judge caught that.
- **Agreement is too strict for training data:** only 39% of correct pairs are reproduced. Valid
  alternative readings, extra columns, rounding and `LIMIT` choices all count as disagreement, and the
  solver's SQL failed to run in 34/215 cases.
- **Policy:**
  - training data keeps pairs the judge calls correct;
  - the gold evaluation set requires judge correct AND agreement, then a hand review. Agreement
    selects unambiguous questions with reproducible answers, which is what execution-match scoring needs.

## Environment gotchas

- DuckDB 2.0 spheroid functions need `SET geometry_always_xy = true`, or they refuse to run.
- `/tmp` on the ChromeOS Linux container is RAM (tmpfs): model files there starved llama.cpp of memory.
- llama.cpp with all 8 threads collapsed decode about tenfold (71 -> 7 tok/s) because of VM thread
  contention; use fewer threads than cores.
- **Each query is validated in its own process** (`harness.py exec1`, 4 in parallel, 30 s timeout,
  2 GB memory limit). A shared long-lived connection was silently killed after ~180 heavy spatial
  queries. In-process interrupts can't stop crashes. Isolated, 288 queries validate in ~30 s.
- **DuckDB 2.0 segfault:** an ordered aggregate together with `ST_Extent_Agg` in one query, e.g.
  `SELECT string_agg(name, '; ' ORDER BY name), ST_AsGeoJSON(ST_Extent_Agg(geometry)) FROM countries
  WHERE subregion = 'Caribbean'`, crashes the process in most runs. Either aggregate alone is fine.
  Worth reporting upstream.

## Next steps

1. Done: v3 grounding improved correctness. The spheroid-cast and UTM-scope rules are added for the
   next round.
2. Done: MiniMax-M3 judge (Experiment 5) and the self-consistency check (Experiment 6). Add a self-consistency check: a second model writes SQL from
   the question alone, and the pair is kept only when the result sets match.
3. **Gold evaluation set.** ~300-500 pairs on held-out datasets (never used in training), with
   deterministic ordering, reviewed by hand.
4. **First training set.** ~20k pairs (mixed agent-style and human-style questions, plus repair
   examples where failed SQL + error leads to fixed SQL). Fine-tune a size sweep
   (Qwen2.5-Coder-0.5B, Qwen3-0.6B, Qwen2.5-Coder-1.5B) with `finetune/`. Plot execution match against
   laptop latency.
5. Later: reinforcement learning with an execution-match reward (FrogNano-style), and a larger, more diverse data lake.
