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
| 2 | (done, see below) Eval harness on gold; ceiling from a third model family; a 500-pair dev split on training tables, so the dev-gold gap measures schema generalisation | <$1 |
| 3 | Fine-tune on the data as it is at 1k / 5k / 21k pairs, full fine-tune and LoRA, to get a learning curve | GPU time |
| 4 | Clean for free: drop the pairs in point 4, normalise SQL formatting, re-judge the 123 API errors | free |
| 5 | Execution-verified schema perturbation: rename tables, columns and the geometry column, drop unused columns, add distractor tables, vary the card format, drop descriptions | free |
| 6 | Spend the remaining ~$32 on new tables rather than more pairs per table: GLM as second generator, real joins, the missing patterns in point 3, repair examples from the 761 failures | ~$30 |
| 7 | Gold v2: tag or drop the 39 pairs, add hand-verified pairs the solver got wrong, add 50-100 human or agent-trace questions, report bootstrap intervals | ~$2 |
| 8 | Set a latency budget and target laptop; test a stats-free card and a minimal system prompt | free |

Steps 2-4 need no new generation and come before any further spend.

## Step 2 results (2026-11-10)

Code: `student.py` (the one prompt format), `hints.py` (value hints), `evaluate.py` (execution match with bootstrap
intervals, OpenRouter or local llama-server), `dev_split.py`. Reference runs are in `eval/reference_v1/`.

**Dev split.** 215 whole scenarios, 5 per training bundle (`dev/dev_scenarios.json`), are held back from training.
They pass the gold filter (judge correct, GLM solver agrees, no `LIMIT` tie): 755 of 1,187 judge-correct pairs remain
in `dev/dev_v1.jsonl`. No hand review. The training export must exclude every pair of these scenarios.

**Reference models on gold** (342 pairs, lenient match, 95% bootstrap interval; rules system prompt, temperature 0):

| Model | executes | exact | lenient | lenient with hints |
|---|---|---|---|---|
| Mistral Large 4.0 (reasoning) | 92.4 | 58.8 | 83.3 [79-87] | 83.0 [79-87] |
| gpt-oss-120b | 71.9 | 40.9 | 56.4 [51-61] | 55.6 [51-61] |
| Qwen3 Coder Next | 66.7 | 37.4 | 47.1 [42-52] | 44.2 [39-49] |

Qwen3 Coder Next scores 47.2 [43-51] on dev, the same as on gold: before training, the two sets are equally hard,
so a dev-gold gap after fine-tuning can be read as schema generalisation.

What this says:

- The ceiling for a model that did not help build gold is about 83% lenient, not 100%. Of Mistral's 57 misses, 26 do
  not execute and 29 return the right number of rows with other values or columns. The ones read by hand were
  choices the question leaves open: an extra count column, sphere or spheroid distance, another tie-break column.
  The metric under-credits these. Gold v2 questions should name the expected output columns.
- Hints show no gain on gold, and gold cannot show one: it kept only pairs the solver answered without hints, so 7 of
  342 need a value that is in neither the question nor the card. Run-to-run noise is also large (about 25 answers
  flip between two Mistral runs on identical prompts). Measuring hints needs questions that name things loosely
  (gold v2). The Mistral hinted run used a slightly noisier version of the lookup than the committed one.
- In the training data, values that are in neither the question nor the card are mostly codes (`'TR51'`, `'PRT'`,
  airport idents) where the question gives a name. Hints supply the name, not the code. Drop these pairs in step 4.
- Cost: $5.4, mostly Mistral reasoning tokens ($4.4). Use it sparingly; the two cheap models cost $0.06 per run.

## Open question

Should the student input carry value hints (matched rows from a lookup, as gazet's candidates do)?
If yes, the ~750 hidden-value pairs become trainable instead of being dropped.
