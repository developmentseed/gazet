# Browser demo: gazet with the SmolLM2-135M plan model running client-side (plan, 2026-11-18)

**Status 2026-11-18: built, in `web/` (see `web/README.md`).** Changes from this plan: wllama v3 takes the plan JSON
schema directly (no GBNF port); geometry is read with hyparquet, not DuckDB-WASM, which downloaded whole files from
S3; the index stores each feature's file and row instead of relying on bbox pruning.

Goal: a static web page that does what the gazet demo does (natural-language query -> map geometry), with the model
running in the browser and the data read from Overture's public S3 bucket. No gazet API, no llama-server.

## Verdict: feasible, with two changes to how the data is reached

| Piece | Status | Evidence |
| --- | --- | --- |
| Model in browser | OK | `plan-v1-smol135` (SmolLM2-135M, `LlamaForCausalLM`, 30 layers, 576 hidden, 8k context) is 269 MB as merged bf16 safetensors in `developmentseed/gazet-sql-qwen35` (private). Not yet converted to GGUF; expect about 145 MB at q8_0 and 105 MB at q4_k_m (sizes of its SQL sibling `sql-v2-smol135`). Llama is the best-supported architecture in every browser runtime: wllama (GGUF), WebLLM (prebuilt SmolLM2-135M model library, WebGPU) and Transformers.js. |
| Accuracy | Lower than LFM2.5, still well above the old model | gazet_compat, GPU, unconstrained: places 97.7%, executes 89.4%, same geometry 78.9% (LFM2.5-350M 82.9%, old 0.8B 59.1%). Weakest families against LFM2.5: disambiguation 76 vs 92, direct lookup 64 vs 76, buffer 80 vs 96. Not yet run quantized or constrained. |
| Prompt size | OK | With the SmolLM2 tokenizer: place extraction 764 tokens, plan with 10 candidates 838 tokens. Both fit the 8k context; latency is mostly prefill. |
| Constrained plans | OK, needs a port | The JSON schema from `plan_schema()` converts to a GBNF grammar (llama.cpp's `json-schema-to-grammar.mjs`); wllama takes a grammar. IDs stay limited to the candidates. |
| Plan compiler | OK, needs a port | `src/gazet/plan.py` is about 250 lines of string building with no dependencies; a TypeScript port is easy to check against the Python output. |
| SQL engine | OK | DuckDB-WASM with spatial (GEOS) runs every function the compiler emits. `jaro_winkler_similarity` and `strip_accents` are core functions. |
| Overture S3 from the browser | OK | `overturemaps-us-west-2` sends `Access-Control-Allow-Origin: *` and exposes `Content-Range`/`Accept-Ranges`. Range reads work. |
| Fuzzy name search on raw Overture | **Too slow** | division_area is 8 files, 4.9 GB, 488 row groups. Scanning the names took 20 s from native DuckDB in the cloud; in WASM it would be slower still. `subtype`, `country` and `id` carry no row-group statistics, so they cannot prune. |
| Geometry by bbox on raw Overture | OK | `bbox.*` has row-group statistics; files are spatially sorted. A bbox-bounded fetch (counties around Loja) took 7.5 s natively and read only a few row groups. |
| Release pinning | Risk | S3 keeps only the latest releases (now 2026-09-23.0 to 2026-10-21.0); the 2026-02-18.0 release gazet was built from is gone. The model never memorised IDs (they come from candidates at run time), so a newer release works as long as the index and geometries come from the same release. |

## Architecture

```
browser
 ├─ LLM worker: plan-v1-smol135, q8_0 GGUF in wllama (~145 MB, cached in the browser after the first load)
 │    1. places  = extract(query)                  (same system prompt as gazet.lm)
 │    3. plan    = generate(candidates, query, grammar from plan_schema(candidate ids))
 ├─ DuckDB-WASM + spatial
 │    2. candidates = fuzzy search over the INDEX   (same SQL as gazet.search)
 │    4. sql = compilePlan(plan) -> run against GEOMETRY sources
 └─ MapLibre map + panels showing places, candidates, plan, SQL (mirrors gazet_demo.py)

static data
 ├─ index.parquet (~8 MB): built by us from one Overture release + Natural Earth. Loaded once.
 │    id, name, search_names, subtype, country, region, admin_level, is_land, is_territorial,
 │    population, bbox, area, source
 ├─ geometry, divisions: Overture S3 division_area, same release, read by bbox (row-group pruning)
 └─ geometry, natural: Natural Earth parquet (53 MB), hosted by us (HF dataset, CORS ok)
```

The index is the key piece. With it:

- **Fuzzy search** runs on 8 MB in memory, not on 4.9 GB over the network.
- **Attribute-only plans** (`{"find":"divisions","subtype":["county"],"country":["PE"]}`) choose their IDs from the
  index; only those geometries are fetched.
- **largest / smallest** rank on the stored `area` (gazet uses `ST_Area` on degrees, the same value), so no
  geometry is read just to rank.
- **Every geometry fetch has a bbox**: the union of the bboxes of the IDs it needs, or, for spatial filters, the
  bbox of the anchor (grown by the buffer distance). The compiler adds
  `bbox.xmin <= :xmax AND bbox.xmax >= :xmin AND ...` so DuckDB skips the row groups outside it.
- **Subtypes**: raw Overture also holds localities, neighbourhoods, macroregions and more. The compiler adds
  `subtype IN ('country','region','county')` to every divisions read, matching what the model was trained on.

## The cost to watch: big geometries

Overture geometries are full resolution: a row group carries about 10 MB of geometry, and a large country is tens of
MB on its own. "Regions touching Russia" or "union of Brazil's states" will pull hundreds of MB into a browser tab
(WASM memory is capped at 4 GB). Two options:

1. **Raw Overture only** (what was asked): fine for county- and region-scale questions, slow or failing for
   country-scale ones. Show download progress and a size warning.
2. **Fallback: our own simplified copy.** The 43k gazet features simplified at 0.001 degrees (about 100 m) are
   124 MB instead of 1.8 GB. Hosted on HF, sorted by Hilbert curve, small row groups. The answers change slightly
   at borders (the evals were run on full geometry, so `touches` may differ on simplified shapes; simplify with
   `ST_CoverageSimplify` on a whole coverage to keep shared borders shared).

Recommendation: build for option 1 with a source switch, and measure both on the eval questions before choosing the
default.

## Work plan

| # | Step | Size | Check |
| --- | --- | --- | --- |
| -1 | Convert `plan-v1-smol135/merged` to GGUF (q8_0 and q4_k_m) with `convert_hf_to_gguf.py` + `llama-quantize`; rerun gazet_compat with llama-server and the json_schema constraint, as was done for the LFM2.5 laptop run. For the SQL model, q4_k_m lost against q8_0 on this 135M model (exact 3.3 vs 8.3 on 60 questions), so default to q8_0 unless q4 holds up. Publish the chosen file to a public HF repo. | S | Same geometry near 78.9% |
| 0 | Spike (1 day): one HTML page with wllama + the GGUF + the grammar; one with DuckDB-WASM reading an Overture row group by bbox. Measure tokens/s and seconds per request on a laptop (Chrome, Firefox, Safari). | S | Go/no-go on latency |
| 1 | `build_browser_index.py`: index.parquet from the current Overture release (same filters as `normalize_geodata`: subtypes, `search_names`, population) + Natural Earth; write a manifest with the release and file URLs. | S | Row counts equal gazet's normalized data |
| 2 | TypeScript port of `plan.py` (`compilePlan`, `planSchema`, `validate`) with the bbox and subtype additions. | M | Golden test: the 256 reference plans compile to SQL that returns the same geometry as Python's on a local copy; the fuzz test ported too |
| 3 | Port the fuzzy search SQL and `_per_source_limit`, candidate CSV formatting and both prompts byte for byte from `gazet.lm`. | S | Same candidates and same prompt text as Python for 50 queries |
| 4 | Model loading in a Web Worker: wllama, GGUF from HF (`developmentseed/...`), cached; schema -> GBNF per request. Temperature 0, `enable_thinking` off. | M | Plans match llama-server's for the 42 bench questions |
| 5 | UI: query box, example queries, streamed stages (places, candidates, plan, SQL, geometry) on MapLibre; GeoJSON download. | M | Manual |
| 6 | Eval in the browser: run `gazet_compat`'s 350 questions headless (Playwright) and compare with the 83.1% same-geometry of the laptop run; record seconds per request. | M | Within a couple of points of 83% |
| 7 | Deploy as a static site (GitHub Pages or an HF static Space). wllama's multithreading needs cross-origin isolation (COOP/COEP headers); use `COEP: credentialless` or check that S3 and HF CORS responses load under `require-corp`. | S | Works from a clean profile |

## Risks and open questions

- **Latency.** Not measured for SmolLM2 plans. Each request is two calls of about 800 prompt tokens and 15-50 output
  tokens on a 135M model; on the laptop CPU the SmolLM2 SQL model took 1.3 s per query at q8_0. WASM CPU is typically
  2-3x slower, so expect a few seconds per request. Place extraction's system prompt is the same on every request:
  SmolLM2 is a plain transformer, so wllama can keep that prefix in the KV cache (the LFM2 conv-layer problem in
  REVIEW.md does not apply). If CPU is too slow: WebLLM on WebGPU, which reuses its prebuilt SmolLM2-135M model
  library with our converted weights and has JSON-schema decoding built in (XGrammar). One WebLLM issue reports a
  custom-compiled SmolLM2 library missing the grammar functions; the prebuilt one worked.
- **Accuracy gap.** SmolLM2 is 4 points below LFM2.5 on same geometry and 2 points below on places, mostly on
  disambiguation and buffers. Showing the candidates and plan in the UI makes wrong picks visible. LFM2.5-350M
  (229 MB q4_k_m GGUF, already built) stays a drop-in alternative behind the same code.
- **DuckDB-WASM version and httpfs.** The compiler works around a DuckDB 1.4 + GEOS crash on empty geometries; pin a
  DuckDB-WASM build and run the fuzz test against it. Confirm that reads go through range requests and not whole
  files (the DuckDB-WASM docs differ on this).
- **Release drift.** Re-running gazet's eval needs the 2026-02-18 IDs, which are no longer on S3. Run the browser
  eval against the local normalized copy (same IDs) and use S3 only for the live demo, or rebuild the eval answers
  on the new release.
- **Model hosting.** The GGUF must be public (HF model repo); pick the name and license.
