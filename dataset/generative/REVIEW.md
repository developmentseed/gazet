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
| 3 | (first run done, see below) Fine-tune on the data as it is at 1k / 5k / 21k pairs, full fine-tune and LoRA, to get a learning curve | GPU time |
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

On dev, Qwen3 Coder Next scores 47.2 [43-51] and gpt-oss-120b 59.1 [56-63], against 47.1 and 56.4 on gold. Before
training the two sets are about equally hard, so a dev-gold gap after fine-tuning can be read as schema generalisation.

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

## First fine-tune (2026-11-11)

Qwen3.5-0.8B, LoRA r=16, 2 epochs over the 19,419 training pairs of `chatml/v1` (export.py), on one A100 with
HF Jobs (`finetune/train_hf_jobs_qwen35.py`): 20 minutes of training, about $2 in total. Weights, GGUF and answers
are in the private model repo `developmentseed/gazet-sql-qwen35` under `sql-v1-r16/`.

| Set | executes | exact | lenient | lenient with hints |
|---|---|---|---|---|
| dev (755, known tables) | 91.3 | 62.0 | 69.3 [66-73] | 70.3 [67-74] |
| gold (342, new tables) | 83.3 | 36.3 | 44.2 [39-49] | 48.5 [43-54] |

For comparison on gold: Mistral Large 4.0 83.3, gpt-oss-120b 56.4, Qwen3 Coder Next 47.1 (lenient, no hints).

- On known tables the 0.8B student beats both mid-size reference models (dev: gpt-oss-120b 59.1, Qwen3 Coder
  Next 47.2). On new tables it falls to about Qwen3 Coder Next's level. The 25-point dev-gold gap is the schema
  memorisation that criticism point 2 predicted. Of the 57 gold answers that fail to run, most refer to columns
  or tables that do not exist.
- Hints help on new tables: 22 gold answers gained, 7 lost (paired; p about 0.01). On dev they are close to neutral.
- Eval loss was still falling slowly at the end (0.175 after one epoch, 0.160 after two). More epochs on the same
  23 tables would most likely widen the gap; the next gains should come from schema variety (steps 5 and 6).
- The q8_0 GGUF on CPU (llama.cpp, 6 threads on this 8-core VM) scores the same as the GPU model on 60 gold pairs
  (25 vs 25) at a median of 6 s per question: prompt about 200 tok/s, answer 25-40 tok/s. llama-server turns
  Qwen3.5 thinking on by default, so requests must pass `enable_thinking: false`.
- Not yet measured: the untrained base model on gold, so the gain from fine-tuning is not isolated.

## Open question

Should the student input carry value hints (matched rows from a lookup, as gazet's candidates do)?
If yes, the ~750 hidden-value pairs become trainable instead of being dropped.

## Size sweep v2 (2026-11-14)

Training set v2 (`chatml/v2`, 30,660 pairs, 15 new tables) on nine base models, HF Jobs A100, 2 epochs, seq 3072,
LoRA (r16 for Qwen3.5, r64 below 0.8B; full fine-tuning was 6-15x slower on the tiny models). Scores are lenient
execution match, plain prompts, all pairs; laptop columns are q8_0 GGUF via llama-server, 6 threads, on 60 gold v1
questions (wide intervals, about +-12 points). Full tables: `eval/sweep_v2/report.txt`; per-pair scores in the HF dataset under `eval/sweep_v2/`.

| Model | Params | Dev v1 | Dev v2 | Gold v1 | Gold v2 | Laptop s/query | RAM |
|---|---|---|---|---|---|---|---|
| v1 0.8B (reference) | 0.8B | 69 | - | 44 | 13 | - | - |
| Qwen3.5-2B | 2B | 79 | 73 | 59 | 63 | 12.4 | 4.2 GB |
| Qwen3.5-0.8B | 0.8B | 72 | 63 | 48 | 41 | 6.8 | 1.8 GB |
| Qwen3.5-0.8B, half data | 0.8B | 65 | 55 | 47 | 37 | 7.1 | 1.8 GB |
| Qwen3.5-0.8B, cap 250 | 0.8B | 66 | 59 | 47 | 40 | 7.2 | 1.8 GB |
| Qwen3-0.6B | 0.6B | 75 | 68 | 48 | 46 | 4.1 | 1.8 GB |
| Qwen2.5-Coder-0.5B | 0.5B | 72 | 64 | 48 | 35 | 2.5 | 1.1 GB |
| LFM2.5-350M | 350M | 67 | 59 | 33 | 32 | 2.8 | 0.8 GB |
| Gemma 3 270M | 270M | 54 | 41 | 19 | 18 | 4.1 | 0.7 GB |
| SmolLM2-135M | 135M | 36 | 26 | 12 | 11 | 1.3 | 0.4 GB |

Findings:
- The new tables close most of the gold v2 gap (0.8B: 13% -> 41%). Half the data is worse on every test, so more
  data still helps; capping at 250 per bundle does not help.
- Qwen3-0.6B matches the 0.8B; Qwen2.5-Coder-0.5B matches it on gold v1 and is 2.7x faster on the laptop, the best
  "system 1" candidate. Below 0.5B, unseen-table accuracy falls off a cliff (LFM2.5 keeps dev but not gold).
- Parameter count misleads on speed: Qwen3.5's linear-attention layers are slow on this ARM CPU, Gemma's 262k
  vocabulary costs time. q4_k_m is not faster than q8_0 here and loses accuracy; ship q8_0.

Next: pass@8 on Coder-0.5B and Qwen3-0.6B to size the RL headroom; distillation from the 2B as the cheap alternative;
2B as the fallback ("system 2") when the small model's SQL fails; diffusion side test on Qwen3-0.6B.

## RL headroom: pass@8 (2026-11-15)

8 samples per question (temperature 0.8, top-p 0.95; `finetune/sample_hf_jobs.py`), scored with `passk.py`
(lenient execution match, identical SQL run once). "some" = questions right in 1-7 of 8 tries, the RL signal.
Table: `eval/passk_v2/report.txt`.

| Model | Test | Greedy | pass@8 | never | some | always | Laptop s/query |
|---|---|---|---|---|---|---|---|
| Coder-0.5B | gold v1 | 48 | 67 | 33 | 49 | 19 | 2.5 |
| Coder-0.5B | gold v2 | 35 | 59 | 41 | 39 | 20 | |
| Qwen3-0.6B | gold v1 | 48 | 69 | 31 | 49 | 20 | 4.1 |
| Qwen3-0.6B | gold v2 | 46 | 72 | 29 | 54 | 17 | |
| Qwen3.5-2B | gold v1 | 59 | 80 | 20 | 54 | 26 | 12.4 |
| Qwen3.5-2B | gold v2 | 63 | 85 | 15 | 58 | 27 | |

Both small models reach the 2B's greedy score within 8 tries, so most of their gap is selection, not capacity:
RL is the right next step. Qwen3-0.6B has more headroom on the new storage styles (gold v2 pass@8 72 vs 59);
Coder-0.5B is 1.6x faster on the laptop.

## RL run 1: GRPO on Qwen3-0.6B (2026-11-15)

`finetune/train_grpo_hf_jobs.py` from `sql-v2-q3-06b`: 3,046 solver-confirmed training-table questions (`rl_data.py`,
`rl/v1` in the dataset), 8 answers each, reward 1 for a lenient result match, LoRA r64, lr 5e-6, beta 0.02,
190 steps, 208 min on an A100. Training reward stayed flat (about 0.70) and KL ended near 0.004, yet the test
scores rose. Paired change per question against the SFT model (plain prompts, lenient, bootstrap 95%):

| Test | SFT | RL | fixed / broken | change |
|---|---|---|---|---|
| gold v1 | 48.2 | 55.0 | 30 / 7 | +6.7 [+3.5, +10.2] |
| gold v2 | 45.6 | 45.6 | 6 / 6 | 0.0 [-4.4, +3.8] |
| dev v1 | 74.7 | 78.4 | 53 / 25 | +3.7 [+1.5, +6.0] |
| dev v2 | 68.1 | 72.8 | 29 / 14 | +4.8 [+1.0, +8.6] |

Execution rate on gold v1 rose from 89% to 93%: fewer broken queries. Gold v1 at 55% is close to the 2B's 59%,
at a third of its laptop latency. Gold v2 (the new storage styles) did not move, although 41% of the RL questions use
new-style bundles. Why is open: gold v2 is small (158 questions on 2 tables) and its tables test styles (plain
lat/lon columns, EPSG:2154 under `geom`) that RL may not reach without more training. A second run with lr 2e-5 is
running.

## RL run 2: higher learning rate (2026-11-15)

Same as run 1 with lr 2e-5 (`rl-q3-06b-lr2e5`, 197 min). Training reward rose from 0.69 to about 0.75 and KL
reached about 0.018 (run 1: flat, 0.004). Plain prompts, lenient, paired bootstrap 95%:

| Test | SFT | RL 1 | RL 2 | RL 2 vs SFT | RL 2 vs RL 1 |
|---|---|---|---|---|---|
| gold v1 | 48.2 | 55.0 | **57.0** | +8.8 [+5.0, +12.9] | +2.0 [-1.5, +5.6] |
| gold v2 | 45.6 | 45.6 | 48.7 | +3.2 [-3.2, +9.5] | +3.2 [-3.2, +9.5] |
| dev v1 | 74.7 | 78.4 | 79.6 | +4.9 [+2.4, +7.3] | +1.2 [-0.9, +3.3] |
| dev v2 | 68.1 | 72.8 | 73.5 | +5.4 [+1.3, +9.6] | +0.6 [-2.6, +3.8] |

RL 2 is the best small model: gold v1 57% against the 2B's 59%, at about 4 s per query on the laptop (2B: 12 s).
Its gains over RL 1 point the same way on every test but are each within noise. GGUF:
`developmentseed/gazet-sql-qwen35/rl-q3-06b-lr2e5/gguf`.

## RL on Coder-0.5B (2026-11-15)

Same recipe as RL run 2 (lr 2e-5, `rl/v1`, 190 steps at about 30 s each, 2.5x faster than Qwen3-0.6B) from
`sql-v2-coder05` -> `rl-coder05-lr2e5`. Plain prompts, lenient, paired bootstrap 95% against its SFT model:

| Test | SFT | RL | change |
|---|---|---|---|
| gold v1 | 48.0 | 54.7 | +6.7 [+2.0, +11.7] |
| gold v2 | 34.8 | 39.9 | +5.1 [+0.0, +10.1] |
| dev v1 | 71.5 | 77.5 | +6.0 [+3.3, +8.5] |
| dev v2 | 63.9 | 70.9 | +7.0 [+3.2, +11.2] |

Small-model standings after RL (gold v1 / gold v2, laptop s/query): Qwen3-0.6B RL 57.0 / 48.7 at about 4.1-4.6 s;
Coder-0.5B RL 54.7 / 39.9 at about 2.5 s. Coder is the faster choice with similar gold v1 accuracy; Qwen3 is
clearly better on the new storage styles. GGUF: `developmentseed/gazet-sql-qwen35/rl-coder05-lr2e5/gguf`.

## RL round 2 (2026-11-16)

From `rl-q3-06b-lr2e5`, questions `rl/v2` = round 1 (3,046) + 1,698 new solver-checked questions from the new-style
bundles. Prefilter (8 samples with the start model): 1,515 sometimes right, 2,742 always right (58%), 487 never;
only the 1,515 were trained on (95 steps, lr 2e-5; prefilter 201 min, training 115 min) -> `rl2-q3-06b`.

| Test | SFT | RL run 2 | RL round 2 | vs RL run 2 | vs SFT |
|---|---|---|---|---|---|
| gold v1 | 48.2 | 57.0 | 57.6 | +0.6 [-2.6, +3.5] | +9.4 [+5.3, +13.5] |
| gold v2 | 45.6 | 48.7 | **57.6** | **+8.9 [+3.2, +14.6]** | +12.0 [+5.7, +19.0] |
| dev v1 | 74.7 | 79.6 | 80.1 | +0.5 [-1.5, +2.5] | +5.4 [+2.8, +7.8] |
| dev v2 | 68.1 | 73.5 | 74.1 | +0.6 [-1.9, +3.2] | +6.1 [+2.2, +10.2] |

The targeted new-style questions closed most of the gold v2 gap without losing anything elsewhere. The 0.6B model
now scores 58% / 58% on the two unseen-table sets against the 2B SFT model's 59% / 63%. Laptop check (q8_0 GGUF,
llama-server, 60 gold v2 questions): 53% vs 50% for the training job's answers on the same questions, 2.9 s/query.
The demo now serves this model. 58% of the training questions are always right for it: further RL needs harder
or newer questions, not more of the same.

## Can the new model replace gazet's model? (2026-11-16)

`gazet_compat.py`: the old gazet model (`developmentseed/gazet-model`, Qwen3.5-0.8B LoRA) and `rl2-q3-06b`, both
through gazet's own production calls (`gazet.lm.generate_places` / `generate_sql`, the llama-server path behind the
API; prompts captured from gazet.lm, answered on a GPU with each model's chat template, greedy, thinking off, then
replayed through gazet.lm's parsing). Old test set (`developmentseed/gazet-dataset`), 25 questions per family x 14
families per task; SQL questions limited to those whose reference returns a geometry on the current gazet-geodata
(82 of 432 drawn did not: test IDs come from an older Overture release). SQL correct = same returned geometry
(union IoU >= 0.99, or Hausdorff <= 1e-4 deg).

| | old gazet model | rl2-q3-06b | rl2-q3-06b, plain table names allowed |
|---|---|---|---|
| places: same place set | **99.7** | 61.1 | - |
| sql: executes | 72.3 | 11.1 | 28.6 |
| sql: same geometry | **59.1** | 4.9 | 7.7 |

Old model per family (same geometry): buffer, disambiguation, window_function 96; attribute_filter 84; direct_lookup
80; set_operations 68; aggregation, containment, difference 60; partial_selection 44; adjacency 32; intersection 24;
chained 20; border_corridor 8.

Verdict: not a drop-in replacement. The new model never learned gazet's contract: it writes `FROM divisions_area`
instead of `read_parquet('divisions_area')` (213/350), uses a `name` column instead of `names."primary"`, queries a
`candidates` table that is only text in the prompt, often returns no geometry, and translates or swaps place names
in extraction. Replacing the model behind the unchanged API needs training on gazet's own data (34,780 SQL and the
places split of `gazet-dataset`), mixed with the generative data so the general skills stay.

## Plan-language coverage (2026-11-17)

Could a structured plan (operations compiled to SQL) replace free SQL? `dsl_coverage.py` parses every query
(sqlglot) and maps its constructs to a draft plan language: tier 1 = core (filters, joins, nested steps, grouping,
order/limit, arithmetic, struct access, common spatial predicates and operations); tier 2 = extensions (rank within
group, conditional labels and aggregates, text and date functions, set operations, scalar subqueries, expanding a list
column, list membership); tier 3 = free SQL (list processing with lambdas, JSON, regex, inline VALUES, series).
Full table: `eval/dsl_coverage/report.txt`.

| Corpus | queries | tier 1 | tier 1+2 | tier 3 |
|---|---|---|---|---|
| gazet-dataset train | 34,780 | 95.1 | **100.0** | 0.0 |
| generative train v2 | 30,660 | 64.6 | 87.8 | 12.2 |
| dev v1 | 755 | 71.3 | 91.9 | 8.1 |
| gold v1 | 342 | 74.3 | 92.1 | 7.9 |
| gold v2 | 158 | 76.6 | 96.8 | 3.2 |

Gazet's whole task fits the plan language (its 34,780 queries use about 20 functions; 4.9% need rank-within-group).
For general spatial questions the extensions bring coverage to 88-97%; the largest single gain is ranking within
groups (the only blocker for 10-12% of queries; 46% of window uses are top-N per group). The remaining 3-12% (mostly
list processing and JSON) would go to free SQL. Caveat: synthetic questions, so this measures what our generators
ask; a real-use test set is the better measure, and expressible does not yet mean learnable by a small model.
