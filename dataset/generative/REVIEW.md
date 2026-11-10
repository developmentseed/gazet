# Review of the generative SQL data work (2026-11-10)

A critical read of the pipeline, the 21,218-pair training set v1 and the gold set, done with Claude Code
before any fine-tune. Numbers come from profiling `runs/train/` and `gold/gold_v1.jsonl`.
The data is now on Hugging Face (private): `developmentseed/gazet-generative-sql`.

## What holds up

- Per-process execution, a calibrated judge and a cross-family solver; held-out tables enforced in code.
- Surface diversity: 20,419 distinct SQL skeletons in 21,218 pairs; difficulty, skill and style are balanced.
- The DuckDB rules work: no measurement in EPSG:3857/6933 anywhere in the set.

## Criticism

1. **Nothing has been measured.** No fine-tune, no gold baseline, no ChatML export; `eval_small.py` still
   reads the 53 pilot pairs. We do not know whether 21k pairs is too many, too few or the wrong kind.
2. **23 tables is the real dataset size.** 43 bundles, so each identical schema card appears about 500 times
   (`countries` is in 5,046 pairs). A 0.5B model can memorise table name -> columns instead of reading the card.
3. **The input format never varies.** One card layout with stats and a hand-written description that carries
   join hints. The geometry column is always `geometry` in EPSG:4326. No tables with only lon/lat columns,
   WKT/WKB columns or a native projected CRS. `read_parquet` / `ST_Read` appear 0 times, `ILIKE` 11 times,
   Overture `bbox.` pruning 88 times.
4. **About 1,000 accepted pairs (5%) need information the student never sees.**
   - 64 read a table that is not in the prompt (validation runs against the whole lake, so they execute);
   - about 750 filter on a value that is in neither the question nor the card (`'Aichi Ken'` for
     "Aichi Prefecture", `'EL434'` for "Chania");
   - about 200 have coordinates in the SQL that are not in the question (some are computed box corners);
   - 54 build points from taxi zone IDs.
5. **Few real joins.** 19% of pairs read two or more tables; 72% of two-table prompts use one.
6. **The gold set leans easy and towards the generator.** It keeps only pairs GLM reproduced, so GLM scores
   100% by construction and DeepSeek wrote the pairs; neither can be the reference ceiling. 39 of 342 pairs
   touch no held-out table. 36% return one row. The hand review was done by Claude, and no question was
   written by a human or taken from a real agent trace.
7. **The judge numbers are soft.** "Catches 19/20" has a 95% interval of roughly 76-99%, it was calibrated
   before the agent styles and the 20 new tables existed, and the final set was not audited. A 36-pair
   spot check found no wrong SQL but 3 pairs with the hidden-information problem.
8. **One generator.** GLM was chosen as the second family and then not used for training.
9. **Latency is not designed in.** The median prompt is 590 tokens: 6-15 s of prefill at the measured
   40-100 tok/s. No target hardware or budget. 82% of SQL ends in `;`, 19% is multi-line.

## Plan

| # | Step | Cost |
|---|---|---|
| 1 | Back up the data (done: private HF dataset, including the lake snapshot) | free |
| 2 | Eval harness on gold; ceiling from a third model family; a 500-pair dev split on training tables, so the dev-gold gap measures schema generalisation | <$1 |
| 3 | Fine-tune on the data as it is at 1k / 5k / 21k pairs, full fine-tune and LoRA, to get a learning curve | GPU time |
| 4 | Clean for free: drop the pairs in point 4, normalise SQL formatting, re-judge the 123 API errors | free |
| 5 | Execution-verified schema perturbation: rename tables, columns and the geometry column, drop unused columns, add distractor tables, vary the card format, drop descriptions | free |
| 6 | Spend the remaining ~$32 on new tables rather than more pairs per table: GLM as second generator, real joins, the missing patterns in point 3, repair examples from the 761 failures | ~$30 |
| 7 | Gold v2: tag or drop the 39 pairs, add hand-verified pairs the solver got wrong, add 50-100 human or agent-trace questions, report bootstrap intervals | ~$2 |
| 8 | Set a latency budget and target laptop; test a stats-free card and a minimal system prompt | free |

Steps 2-4 need no new generation and come before any further spend.

## Open question

Should the student input carry value hints (matched rows from a lookup, as gazet's candidates do)?
If yes, the ~750 hidden-value pairs become trainable instead of being dropped.
