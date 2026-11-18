# Gazet in the browser

gazet's natural-language geocoder as a static web page: no server, no API. A 135M-parameter model and DuckDB run in
the browser tab, and division geometry is read straight from Overture's public S3 bucket.

```
query ──► model: place names ──► DuckDB: fuzzy search on the index ──► model: JSON plan ──► plan compiled to SQL
      ──► geometry fetched for just the features the plan can touch ──► DuckDB runs the SQL ──► map
```

- **Model:** `plan-v1-smol135`, SmolLM2-135M fine-tuned on gazet's place extraction and plans (the plan language in
  `src/gazet/plan.py`). q8_0 GGUF, 145 MB, run by [wllama](https://github.com/ngxson/wllama) (llama.cpp's
  server core in WebAssembly). Plans are decoded under the plan JSON schema with IDs limited to the candidates, as
  gazet does with llama-server. On gazet_compat (GPU): places 97.7%, same geometry 78.9%.
- **Compiler:** `src/plan.ts`, a port of `src/gazet/plan.py`. Tests check it writes the same SQL as Python.
- **Data:** `scripts/build_data.py` builds a 10 MB index (names, attributes, bbox, area, and the file and row of
  each of the 43k divisions and 32k Natural Earth features; no geometry) and a 40 MB Natural Earth geometry file.
  Division geometry stays on Overture's S3. Before a plan runs, the page works out from the index which features it
  can touch (`Db.resolve`), reads just their `id` and `geometry` column chunks with HTTP range requests
  ([hyparquet](https://github.com/hyparam/hyparquet)), and loads them into DuckDB as WKB. The compiled SQL then runs
  unchanged on those rows. DuckDB-WASM is not used for the remote reads: in every mode tried it downloaded whole
  files from S3 (200-800 MB each), and the 1.33 dev build failed on them outright.
- **DuckDB-WASM 1.32.0** (DuckDB 1.4.3, the version line gazet runs), pinned.

## Run it locally

```bash
cd web
npm install

# Model: convert the merged weights from the HF repo (or download the GGUF once it is published)
mkdir -p public/models
#   python llama.cpp/convert_hf_to_gguf.py <plan-v1-smol135/merged> --outtype q8_0 \
#     --outfile public/models/gazet-plan-v1-smol135-q8_0.gguf

# Data: index and Natural Earth file for an Overture release still on S3 (they keep the latest few).
# --local reads a downloaded copy of the release's division_area/ and division/ files (5 GB), much faster
# than reading S3 through DuckDB.
cd .. && uv run python web/scripts/build_data.py --release 2026-10-21.0 [--local DIR] && cd web

npm run dev        # http://localhost:5173
```

Where the page finds the model and data: `VITE_MODEL_URL` (default `./models/gazet-plan-v1-smol135-q8_0.gguf`) and
`VITE_DATA_URL` (default `./data/`), at build time. Both hosts must send CORS headers; Hugging Face does.

The page must be cross-origin isolated (COOP `same-origin`, COEP `require-corp`) for wllama to run multithreaded.
`vite.config.ts` sets the headers for `dev` and `preview`; a static host must set them too (GitHub Pages cannot; an
HF static Space, Netlify or Cloudflare Pages can).

## Tests

```bash
npm test                          # compiler, schema and prompts equal the Python ones (fixtures from scripts/)
node tests/e2e.mjs model          # in headless Chromium: replay 56 plan + 42 place prompts from gazet_compat
node tests/e2e.mjs search "counties that border Loja" "northern half of Ecuador"   # full searches, live S3
PREVIEW=1 node tests/e2e.mjs search "..."   # the same against the production build (not yet run)
```

Fixtures: `uv run python web/scripts/make_plan_fixtures.py` (plans), `tests/fixtures/prompts.json` (prompts and 40
evaluation prompts, from `gazet.lm` and `dataset/generative/runs/gazet_compat/prompts_plan.jsonl`),
`tests/fixtures/compat_sample.json` (prompts and the GPU answers of `plan-v1-smol135`).

## Results (2026-11-18, ChromeOS Linux container, 8-core ARM, 13 GB RAM, headless Chromium, CPU only)

- Model in the browser vs the same model on a GPU, same prompts: plans give the same SQL for 55 of 56 (the other is
  a valid, different plan), none invalid; places identical for 42 of 42.
- Speed: model load 2.5 s from a local file; places 1.2 s (the system prompt stays cached); plan 5-6 s, almost all
  of it reading the ~500 new prompt tokens of the candidates table at about 110 tokens/s. The default thread count
  is best here; 8 threads were 13x slower on this machine. wllama uses WebGPU where the browser has it, which should
  cut the prompt time substantially (not measured: headless Chromium here has no GPU).

- Same answers as gazet: for six searches run in the browser, the same plans run by `gazet.plan` on the full local
  dataset returned exactly the same feature IDs (counties bordering Loja, northern half of Ecuador, three largest
  counties in Albania, provinces within 50 km of Quito, 10 km around Lake Titicaca, counties in the Galapagos).
- End to end: 8-27 s per search here. Data: 8-37 MB from S3 for county- and region-scale questions (3-10 s);
  Peru's regions 171 MB.

## Limits

- Accuracy is the model's: 79% same geometry on gazet's test questions; it is weakest on disambiguation and
  buffers. The page shows the places, candidates, plan and SQL so a wrong pick is visible.
- A plan that needs geometry for more than 3,000 features, or more than 1.2 GB of it, stops with a message
  (`MAX_FEATURES`, `MAX_DOWNLOAD` in `src/db.ts`). Overture geometry is full resolution, and a read fetches whole
  row-group column chunks (Overture has no page index), so one feature costs 5-15 MB and country-scale questions
  download hundreds of MB.
- Some operations on full-resolution geometry run out of memory, such as a 50 km buffer around the Loja-Piura border
  (it also ran out of memory natively on the test machine). The page reports it and restarts DuckDB.
- The index pins one Overture release. S3 keeps releases for 60 days (2026-10-21.0 expires 2026-12-21), so the data
  must be rebuilt for a newer release before then. IDs are stable: 99.9% of the 2026-02-18 IDs gazet was trained with are in 2026-10-21.
- Same scope as gazet's natural-language mode: countries, regions and counties (no cities), plus Natural Earth.

## Not done yet

- Publishing: the GGUF was converted locally from the merged weights in the private `developmentseed/gazet-sql-qwen35`
  repo, and the data is local. A public deployment needs the GGUF and `public/data/` on a public host with CORS
  (e.g. an HF model and dataset repo) and a static host that can send COOP/COEP headers.
- The production build compiles but was not run through the e2e test.
- Speed in a real browser with WebGPU is not measured.
