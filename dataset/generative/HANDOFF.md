# Handoff: generative spatial text-to-SQL data

State as of 2026-10-29. Read this first, then `FINDINGS.md` for the full evidence.
Branch `feat/generative-sql-data` on developmentseed/gazet is pushed. No PR yet, on purpose.

## Goal and decisions (made by Daniel)

- Train the **smallest LLM** that writes good **DuckDB spatial SQL** fast on a **laptop CPU** (llama.cpp / Ollama).
  It will be used as a tool in agentic geospatial workflows. "Type-ahead" was only a speed analogy;
  the task is `question + schema -> full SQL`.
- Spatial-first, over **any schema**, not gazet's two tables. Callers are both LLM agents and humans.
- **Pairs only** (free-form question/SQL pairs), no LLM-authored templates, even if more costly.
- **Open-weight generators only**, via OpenRouter. Haiku was used once as a quality reference.

## Where things stand

| Step | Status | Result |
|---|---|---|
| Data lake | done | 31 tables: pilot set + 20 regionally diverse ones (`build_lake.py`, `build_lake_world.py`) |
| Generator choice | done | DeepSeek-V4.1-Flash primary, GLM-5.3-Flash secondary (9-model bake-off) |
| Prompt | done | grounding with real rows and values + DuckDB-not-PostGIS rules: 96% correct (DeepSeek, blind) |
| Execution check | done | each query in its own process, 30 s timeout (`harness.py validate`) |
| LLM judge | done | MiniMax-M3: catches 19/20 wrong pairs, keeps 90% of good ones (`judge.py`) |
| Self-consistency | done | cross-family solver + result comparison (`consistency.py`) |
| Held-out split | done | 8 eval-only tables in `harness.HELD_OUT`; training bundles assert they are excluded |
| Gold eval set | done | 342 pairs in `gold/gold_v1.jsonl` (held-out tables only, hand-reviewed) |
| Training set v1 | done | 21,218 judge-correct pairs in `runs/train/` (local only, see FINDINGS Exp. 8) |
| **Storage + export** | **next** | archive on Modal, cleaned Parquet on HF (private), ChatML export to `/mnt/gazet/data/<run>` |
| Eval harness on gold set | next | point `eval_small.py` at the gold set |
| Fine-tune sweep | after that | |

**Keep policies (from calibration on 215 hand labels):**
- **training data:** keep pairs the judge calls `correct` (~93% clean);
- **gold eval set:** judge `correct` AND solver agreement (96% clean, 0 wrong in calibration), then hand review.

**Spend:** $57.83 of the $90 OpenRouter key (gold set $1.92, training set v1 ~$52).
**Cost estimate:** ~$3 per 1k accepted training pairs (generation + judge), so ~$60 for 20k pairs.

## Gold set: how it was built (done 2026-10-29)

`harness.py prompts` (eval split) -> `or_generate.py` -> `collect.py` -> `harness.py validate` ->
`judge.py run` -> `consistency.py run` (solver agreement + `LIMIT` tie check) -> `gold.py sheet` ->
blind review -> `gold.py write`. Numbers are in FINDINGS Experiment 7. The review was done by Claude,
blind to judge and solver. Grades are in `gold/grading/grades.json` and reasons for every rejection
in `notes.json`, so Daniel can audit or override and then rerun `gold.py write`.

Gold record: `id, question, tables, sql, difficulty, skill, n_rows, cols, source`. Score a model by
executing `sql` and its prediction on the lake and comparing with `consistency.compare_results`
(row/column-order-insensitive, numbers to 3 / 2 significant figures, and now nested lists too).

## Next step: eval harness on the gold set

- `eval_small.py` still reads the pilot pairs. Switch it to `gold/gold_v1.jsonl`, reuse
  `consistency.compare_results` (or `_compare_isolated`, for crash safety) instead of its own `match`, and
  report exact / lenient execution match by difficulty and skill. Give the model the same prompt as
  the solver (`consistency.SYSTEM` + schema cards of `tables` + question).
- Run a strong open model through it first (DeepSeek or GLM) to get a reference ceiling before the small models.

## After that

1. Training set: ~20k pairs from `BUNDLES` (training tables only). Generation now has the CRS
   relabelling and the lambda rule, which pushed execution to 99% in the last gold round. Mix agent-style and human-style
   questions, and add repair examples (failed SQL + error -> fixed SQL) from the retry step.
2. Fine-tune a size sweep with `finetune/` (Qwen2.5-Coder-0.5B, Qwen3-0.6B, Qwen2.5-Coder-1.5B). Plot
   execution match against laptop latency with `run_evals.sh` + `eval_small.py`. All are at 0% before
   fine-tuning.
3. Later: reinforcement learning with an execution-match reward (Microsoft FrogNano-style), and a full-width
   Dutch table as a schema-selection test.

## Open questions for Daniel

- Confirm or adjust the held-out tables: uk_police_crimes, nl_municipalities, br_fire_hotspots,
  br_sp_municipalities, sg_parks, zm_health_facilities, power_plants, tokyo_places.
- Licences to check before publishing derived data: INPE fire data, GBFS feeds, DataMeet India
  districts. Paris data is ODbL. If Muse Spark Open were ever used, derived models must say
  "Built with Muse Spark Open".
- Whether to report the DuckDB 2.0 segfault upstream (ordered aggregate + `ST_Extent_Agg` in one query).
- CRS labels: the lake views now relabel all lon/lat to `EPSG:4326`. Real user files will mix
  `OGC:CRS84` and `EPSG:4326`. Decide whether training should also include some mixed-label tables, so the
  model learns the `::GEOMETRY` cast for cross-source joins.

## Local environment (Daniel's ChromeOS Linux container)

- Work folder: `~/repos/gazet/dataset/generative`. Gitignored there: `lake/` (137 MB), `raw/`, `.venv/`,
  `llama/` (llama.cpp ARM build), `runs/` (working output), and `models`, a symlink to
  `~/.cache/gazet-exp/models` (3.4 GB of GGUFs; delete when done).
- OpenRouter key: `~/.config/openrouter/key` (chmod 600). Never put it in files or commits.
- Rebuild from scratch:
  ```bash
  uv venv .venv && uv pip install --python .venv duckdb pandas pyarrow sqlglot tiktoken requests
  .venv/bin/python build_lake.py
  .venv/bin/python build_lake_world.py
  ```
- Git has no global identity on this machine. Commit with
  `git -c user.name="Daniel Wiesmann" -c user.email="yellowcap@users.noreply.github.com" commit ...`.

### Gotchas that cost time

- `/tmp` is RAM (tmpfs). Never put models or data there; it starved llama.cpp.
- llama.cpp: use `-t 6`, not all 8 threads; with 8, decode collapses about tenfold.
- DuckDB 2.0 needs `SET geometry_always_xy = true` (done in `harness.con()`).
- DuckDB 2.0: `ST_Distance_Spheroid` / `ST_DWithin_Spheroid` reject CRS-typed columns; cast with `::GEOMETRY`.
- DuckDB 2.0: an ordered aggregate plus `ST_Extent_Agg` in one query segfaults. The per-process validator contains it.
- DuckDB 2.0 rejects `x -> x + 1` lambdas; mixed CRS labels block joins (both handled now).
- Generated pairs have no `source` field; `consistency.py run` takes the generator from the filename
  (`out_<model>__<tag>...`) to pick a cross-family solver, so keep that naming.
- Reasoning models use 5-15k reasoning tokens per call; keep `max_tokens` at 32k.
- Background shell loops: don't `pgrep -f` a pattern that appears in your own command line (it loops forever).

## Repo map (`dataset/generative/`)

| File / folder | What it is |
|---|---|
| `harness.py` | schema cards, grounding, scenario prompts (`prompts N SEED DIR [train|eval]`), isolated validation |
| `or_generate.py` | OpenRouter generation, fp8/bf16 providers only; records provider, tokens, cost |
| `collect.py`, `summarize.py`, `blind_sample.py` | parse outputs, per-model table, blind grading sheets |
| `judge.py` | MiniMax-M3 judge (`calibrate`, `run`) |
| `consistency.py` | cross-family solver + result comparison + `LIMIT` tie check (`calibrate`, `run`) |
| `gold.py`, `gold/` | blind review sheet + export; `gold_v1.jsonl` and its grading files |
| `eval_small.py`, `run_evals.sh` | small-model CPU eval; also holds the Haiku pilot hand labels |
| `build_lake.py`, `build_lake_world.py` | data lake builders (licences in `LICENSES`) |
| `pilot/`, `round_v3/`, `judge_calibration/` | committed snapshots, including all hand grades |
| `FINDINGS.md` | full experiment log with numbers |
