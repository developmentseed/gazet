---
title: Gazet
emoji: "\U0001F5FA"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# [Gazet](https://gazet.ds.io)

<img src="assets/gazet-logo.svg" alt="Gazet logo" width="64" />

Lean natural-language geocoder with GIS operations over Overture and Natural Earth parquet datasets.

Gazet  is built to be easily packagable and minimal in setup, trying to push the boundaries on how small we can go in setup for LLM driven data applications. It is built for working with small language models and parquet files.

The name inspired by [Gazetteer](https://en.wikipedia.org/wiki/Gazetteer). A gazetteer is a geographical dictionary or directory used in conjunction with a map or atlas.

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/developmentseed/gazet-model) [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/developmentseed/gazet-dataset)

## Local setup

### Python setup

Install python dependencies using [uv](https://docs.astral.sh/uv/)

```bash
uv sync --extra dev --extra demo
```

### Data preparation

1. Download Overture divisions data
2. Download the 10m physical layer from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
3. Unzip the data
4. Convert natural earth data to parquet

Example for downloading overture

```bash
aws s3 sync s3://overturemaps-us-west-2/release/2026-02-18.0/theme=divisions/type=division_area/ data/overture/divisions_area
```

Example for running conversion script for natural earth

```bash
unzip ~/Downloads/10m_physical.zip -d data/natural_earth
python -m ingest.convert_natural_earth data/natural_earth
```

### Based on ollama

For now, gazet relies on [ollama](https://ollama.com/). For remote (cloud) models, ensure you are loged into Ollama.

## Usage

```bash
python -m gazet
# then GET http://localhost:8000/search?q=Border%20between%20Loja%20and%20Piura
# pure fuzzy match, no LLM, raw geometry only:
# GET http://localhost:8000/search/fuzzy?q=Loja&limit=5
# cheap ID-only search, then fetch geometry for just the one you want:
# GET http://localhost:8000/search/fuzzy?q=Loja&limit=5&ids_only=true
# GET http://localhost:8000/geometry/25f85439-e054-44e3-9ad4-9fbf755e3183
```

### API + Streamlit demo

```bash
uv run uvicorn gazet.api:app --reload   # API on :8000
uv run streamlit run gazet_demo.py   # demo UI
```



## Modules

| Module | Contents |
| --- | --- |
| `config.py` | data paths, model name, SQL schema description |
| `schemas.py` | `SUBTYPES`, `COUNTRIES`, `Place`, `PlacesResult` |
| `lm.py` | DSPy signatures + LM init (`extract`, `write_sql`) |
| `search.py` | fuzzy search against `divisions_area` / `natural_earth` |
| `geometry.py` | geometry simplification / GeoJSON normalization (no LLM dependency) |
| `sql.py` | code-act SQL generation loop |
| `export.py` | GeoJSON FeatureCollection writer |
| `api.py` | FastAPI app: `/search` (LLM), `/search/fuzzy` (fuzzy-only), `/geometry/{id}` (direct ID lookup) |

## Design notes

- `api.py` exposes GET `/search?q=<query>`; returns GeoJSON FeatureCollection and logs intermediate output. This path runs two LLM calls (place extraction, then SQL generation for candidate selection/disambiguation and GIS ops).
- `api.py` also exposes GET `/search/fuzzy?q=<place name>&limit=5&simplify=true&sources=divisions_area,natural_earth` — pure Jaro-Winkler fuzzy match with no LLM involved. `q` is a place-name string, not a natural-language query (there's no place-extraction step). Returns the combined top-`limit` matches across sources as a GeoJSON FeatureCollection with raw (or simplified, if `simplify=true`) geometry attached. Pass `ids_only=true` to skip full geometry and get back `{"ids": [{"source", "id", "name", "matched_name", "country", "subtype", "admin_level", "bbox", "similarity", "is_substring_match"}, ...]}` — `country`/`subtype`/`admin_level` disambiguate same-named places (e.g. multiple real "Loja"s across Ecuador and Spain, or Ecuador's "Loja" region vs. its nested "Loja" county), `bbox` is `[minx, miny, maxx, maxy]`, a much smaller payload than full geometry, for picking a candidate before fetching its full geometry, and `similarity` is the score the list was ranked on — Jaro-Winkler always returns a best row, so without it a caller cannot tell a hit from the closest thing to a miss.
- Fuzzy matching (`search.py`) ranks exact-substring matches (query literally contained in the name, e.g. "Loja" in "Loja Province") ahead of pure edit-distance near-misses (e.g. "Lodja"), even when the near-miss scores a slightly higher raw Jaro-Winkler similarity. Each record is matched under two names and keeps the better score: its own (`names.primary`) and its English one (`names.common.en` for `divisions_area`, `names.en` for `natural_earth`). Neither works alone — `names.common.en` is null for 77% of `divisions_area` rows, which matching on English alone would drop, while matching on `names.primary` alone leaves a place unreachable by the English name it is usually asked for ("Copenhagen" for `Københavns Kommune`). `matched_name` reports which of the two the query hit; `name` is the English one where the record has one, the same expression `/geometry/{id}` uses, so a candidate and a later lookup of it agree on what the place is called.
- `api.py` also exposes GET `/geometry/{id}?source=divisions_area&simplify=true` — fetch a single feature's geometry directly by ID, no fuzzy matching or LLM involved. `source` is inferred from the ID if omitted (Natural Earth IDs are prefixed `ne_`). Returns a single GeoJSON Feature, 404 if not found.
- The spatial extension is loaded once at API startup (FastAPI `lifespan`) onto a shared DuckDB connection stored on `app.state`; each request gets a cheap `cursor()` off it instead of paying DuckDB's ~90ms `LOAD spatial` cost per call.
- LM is initialised at import time in `lm.py`, suitable for a long-lived server process.
- Data lives in `data/overture/` and `data/natural_earth_geoparquet/` (not tracked in git).

## Attributions

Logo icon: search globe by popcornarts from <a href="https://thenounproject.com/browse/icons/term/search-globe/" target="_blank" title="search globe Icons">Noun Project</a> (CC BY 3.0)
