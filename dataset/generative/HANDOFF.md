# Handoff: generative spatial text-to-SQL data

State as of 2026-10-28. Read this first, then `FINDINGS.md` for the full evidence.
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
| **Gold eval set** | **next** | proposed, awaiting go-ahead |
| Training set + fine-tune sweep | later | |

**Keep policies (from calibration on 215 hand labels):**
- **training data:** keep pairs the judge calls `correct` (~93% clean);
- **gold eval set:** judge `correct` AND solver agreement (96% clean, 0 wrong in calibration), then hand review.

**Spend:** about $3.90 of the $50 OpenRouter key.
**Cost estimate:** ~$3 per 1k accepted training pairs (generation + judge), so ~$60 for 20k pairs.

## Next step: build the gold eval set

Generate candidates from held-out bundles, filter, then hand-review down to ~300 pairs. About $2-4.

```bash
cd ~/repos/gazet/dataset/generative
# ~70 scenarios x 6 = ~420 candidates from EVAL_BUNDLES (held-out tables only)
.venv/bin/python -I harness.py prompts 70 31 runs/prompts_gold eval
.venv/bin/python -I or_generate.py deepseek/deepseek-v4.1-flash --prompts "runs/prompts_gold/*.txt" --tag gold
.venv/bin/python -I collect.py runs/out_deepseek__deepseek-v4.1-flash__gold runs/prompts_gold
.venv/bin/python -I harness.py validate runs/out_deepseek__deepseek-v4.1-flash__gold.jsonl
.venv/bin/python -I judge.py run minimax/minimax-m3 runs/out_deepseek__deepseek-v4.1-flash__gold.validated.jsonl
```

Still to write:
- `consistency.py` only has a `calibrate` mode. Add a `run` mode, mirroring `judge.py run`, that takes a
  judged file and adds `solver_sql` / `lenient`. Note that `solver_for()` picks GLM for DeepSeek-generated pairs.
- Keep `judge == "correct" and lenient`, then do a blind hand review (`blind_sample.py`) and record
  grades the same way as in `round_v3/grading/`.
- Before writing the gold set, add the not-yet-tested rules from round v3 (`::GEOMETRY` cast for
  `ST_Distance_Spheroid` / `ST_DWithin_Spheroid`, UTM-zone scope); they are already in `GEN_PROMPT`.
  Verify they work.
- Gold queries need deterministic results (unique `ORDER BY` with `LIMIT`), because the small model is
  scored by execution match.

## After the gold set

1. Training set: ~20k pairs from `BUNDLES` (training tables only). Mix agent-style and human-style
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
- Reasoning models use 5-15k reasoning tokens per call; keep `max_tokens` at 32k.
- Background shell loops: don't `pgrep -f` a pattern that appears in your own command line (it loops forever).

## Repo map (`dataset/generative/`)

| File / folder | What it is |
|---|---|
| `harness.py` | schema cards, grounding, scenario prompts (`prompts N SEED DIR [train|eval]`), isolated validation |
| `or_generate.py` | OpenRouter generation, fp8/bf16 providers only; records provider, tokens, cost |
| `collect.py`, `summarize.py`, `blind_sample.py` | parse outputs, per-model table, blind grading sheets |
| `judge.py` | MiniMax-M3 judge (`calibrate`, `run`) |
| `consistency.py` | cross-family solver + result-set comparison (`calibrate`) |
| `eval_small.py`, `run_evals.sh` | small-model CPU eval; also holds the Haiku pilot hand labels |
| `build_lake.py`, `build_lake_world.py` | data lake builders (licences in `LICENSES`) |
| `pilot/`, `round_v3/`, `judge_calibration/` | committed snapshots, including all hand grades |
| `FINDINGS.md` | full experiment log with numbers |
