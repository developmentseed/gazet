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
| `sql.py` | code-act SQL generation loop |
| `export.py` | GeoJSON FeatureCollection writer |
| `api.py` | FastAPI app with `/search?q=...` returning GeoJSON FeatureCollection |

## Design notes

- `api.py` exposes GET `/search?q=<query>`; returns GeoJSON FeatureCollection and logs intermediate output.
- LM is initialised at import time in `lm.py`, suitable for a long-lived server process.
- Data lives in `data/overture/` and `data/natural_earth_geoparquet/` (not tracked in git).

## Attributions

Logo icon: search globe by popcornarts from <a href="https://thenounproject.com/browse/icons/term/search-globe/" target="_blank" title="search globe Icons">Noun Project</a> (CC BY 3.0)
