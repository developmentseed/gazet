import os
import pathlib

# Data lives at project root (gazet/data/), not inside the package.
# Override with GAZET_DATA_DIR env var for remote execution (e.g. Modal volume at /data).
_DATA_DIR = pathlib.Path(os.environ.get("GAZET_DATA_DIR", str(
    pathlib.Path(__file__).resolve().parent.parent.parent / "data"
)))
DIVISIONS_AREA_PATH = str(_DATA_DIR / "overture/divisions_area/*.parquet")
NATURAL_EARTH_PATH = str(_DATA_DIR / "natural_earth_geoparquet/ne_geography.parquet")

# MODEL = "qwen3.5:cloud"
# MODEL = "granite4:350m"
# MODEL = "gemma3:12b-cloud"
# MODEL = "qwen3.5:397b-cloud"
# MODEL = "gpt-oss:20b-cloud"
# MODEL = "qwen3:4b"
# MODEL = "qwen3-coder-next:cloud"
# MODEL = "deepseek-coder:1.3b"
# MODEL = "qwen3.5:2b"
# MODEL = "qwen3.5:0.8b"
# MODEL = "qwen2.5-coder:1.5b"

PLACE_EXTRACTION_MODEL = "gpt-oss:20b-cloud"
SQL_GENERATION_MODEL = "gpt-oss:20b-cloud"

MAX_SQL_ITERATIONS = 5

# ── GGUF / llama-server config ────────────────────────────────────────────────
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "350"))
LLAMA_TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0"))

SCHEMA_INFO = f"""
Available DuckDB datasets (read via read_parquet):

1. divisions_area  — Overture polygon/multipolygon admin boundaries
   path: '{DIVISIONS_AREA_PATH}'
   columns:
     id VARCHAR              -- unique feature id (use this to filter precisely)
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | dependency | county | macrohood |
                               localadmin | locality | neighborhood | microhood
     class VARCHAR
     region VARCHAR          -- region code (e.g. 'EC-L' for Loja Ecuador)
     admin_level INTEGER
     division_id VARCHAR
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- boundary polygon/multipolygon (WKB, spatial ext loaded)

2. natural_earth  — Natural Earth geography polygons (oceans, seas, terrain regions, islands)
   path: '{NATURAL_EARTH_PATH}'
   columns:
     id VARCHAR              -- unique feature id prefixed 'ne_'
     names STRUCT("primary" VARCHAR, ...)
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'Terrain area', 'Island group'
     class VARCHAR
     country VARCHAR
     region VARCHAR
     admin_level INTEGER
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- polygon/multipolygon (WKB, spatial ext loaded)

Spatial extension is already loaded — use ST_AsGeoJSON(geometry) or ST_AsText(geometry).
To access names use: names."primary"

The candidates table has a 'source' column: 'divisions_area' or 'natural_earth'.
Use the matching path for each candidate's source when querying.

Example patterns:
  -- single region boundary from divisions_area
  SELECT id, names."primary" AS name, ST_AsGeoJSON(geometry) AS geojson
  FROM read_parquet('{DIVISIONS_AREA_PATH}')
  WHERE id = '<candidate_id>'

  -- feature from natural_earth
  SELECT id, names."primary" AS name, ST_AsGeoJSON(geometry) AS geojson
  FROM read_parquet('{NATURAL_EARTH_PATH}')
  WHERE id = '<candidate_id>'

  -- shared border between two adjacent regions
  WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '<id_a>'),
       b AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '<id_b>')
  SELECT ST_AsGeoJSON(ST_Intersection(a.geometry, b.geometry)) AS border
  FROM a, b
"""
