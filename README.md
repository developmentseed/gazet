# gazet

Lean natural-language geocoder with GIS operations over Overture and Natural Earth parquet datasets. In an industry trending toward ever-larger models and heavier infrastructure, gazet takes the opposite path: small language models, DuckDB, and local Parquet files — no PostGIS, no cloud geocoding APIs, no bloat.

Name inspired by [Gazetteer](https://en.wikipedia.org/wiki/Gazetteer). A gazetteer is a geographical dictionary or directory used in conjunction with a map or atlas.

## Modules

| Module | Contents |
| --- | --- |
| `config.py` | data paths, model name, SQL schema description |
| `types.py` | `SUBTYPES`, `COUNTRIES`, `Place`, `PlacesResult` |
| `lm.py` | DSPy signatures + LM init (`extract`, `write_sql`) |
| `search.py` | fuzzy search against `divisions_area` / `natural_earth` |
| `sql.py` | code-act SQL generation loop |
| `export.py` | GeoJSON FeatureCollection writer |
| `api.py` | FastAPI app with `/search?q=...` returning GeoJSON FeatureCollection |

## Local setup

Install python dependencies

```bash
uv sync --extra dev --extra demo
```

Ensure you are loged into Ollama to use remote models.

## Usage

```bash
python -m gazet
# then GET http://localhost:8000/search?q=Border%20between%20Loja%20and%20Piura
```

### API + Streamlit demo

```bash
uv run uvicorn gazet.api:app --reload   # API on :8000
uv run streamlit run demo_app.py   # demo UI
```

## Data preparation

1. Download Overture divisions data
2. Download the 10m physical layer from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
3. Unzip the data
4. Convert natural earth data to parquet

Example for downloading overture

```bash
aws s3 sync 
s3 sync s3://overturemaps-us-west-2/release/2026-02-18.0/theme=divisions/type=division_area/ data/overture/divisions_area
```

Example for running conversion script for natural earth

```bash
python -m ingest.convert_natural_earth ~/Downloads/10m_physical
```

## Design notes

- `api.py` exposes GET `/search?q=<query>`; returns GeoJSON FeatureCollection and logs intermediate output.
- LM is initialised at import time in `lm.py`, suitable for a long-lived server process.
- Data lives in `data/overture/` and `data/natural_earth_geoparquet/` (not tracked in git).
