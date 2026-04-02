"""Compare demo: run a query through place extraction → SQL generation via both
DSPy (Ollama/cloud LM) and finetuned GGUF (llama-server), side by side.

Usage
-----
# Start llama-server in one terminal:
#   llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable

# Run this demo:
#   uv run streamlit run compare_demo.py
"""

import difflib
import json
import math
import os
import pathlib
import sys
import time

import duckdb
import httpx
import numpy as np
import pandas as pd
import pydeck as pdk
import sqlparse
import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from gazet.config import (
    DIVISIONS_AREA_PATH,
    LLAMA_SERVER_URL,
    NATURAL_EARTH_PATH,
    SCHEMA_INFO,
)
from gazet.lm import extract, generate_sql, is_llama_server_available, write_sql
from gazet.search import search_divisions_area, search_natural_earth

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = pathlib.Path(os.environ.get("GAZET_DATA_DIR", str(PROJECT_ROOT / "data")))

EXAMPLES = [
    "Angola and Mozambique",
    "Mediterranean Sea",
    "Which counties are within Guatemala?",
    "Top 5 smallest regions in Argentina",
    "What borders Comuna de San Felipe y Santa Bárbara?",
    "The northern half of India",
]


# ── DuckDB helpers ────────────────────────────────────────────────────────────


def get_duckdb_connection():
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    return con


def rewrite_data_paths(sql: str) -> str:
    """Replace hardcoded /data/ paths with the local data directory."""
    return sql.replace("/data/", f"{DATA_DIR}/")


def execute_sql(con, sql: str) -> pd.DataFrame:
    """Execute SQL, converting geometry columns to GeoJSON strings."""
    rel = con.sql(sql)
    cols = rel.columns
    types = [str(t) for t in rel.dtypes]

    select_parts = []
    for col, dtype in zip(cols, types):
        if "GEOMETRY" in dtype.upper():
            select_parts.append(
                f'ST_AsGeoJSON(ST_SimplifyPreserveTopology("{col}", 0.001)) AS "{col}"'
            )
        else:
            select_parts.append(f'"{col}"')

    wrapped = f"SELECT {', '.join(select_parts)} FROM ({sql})"
    return con.execute(wrapped).fetchdf()


def _strip_fences(sql: str) -> str:
    import re

    if not sql:
        return ""
    sql = re.sub(r"^```\w*\s*\n?", "", sql.strip())
    sql = re.sub(r"\n?```\s*$", "", sql)
    return sql.strip()


# ── SQL formatting / diff ─────────────────────────────────────────────────────


def format_sql(sql: str) -> str:
    return sqlparse.format(sql, reindent=True, keyword_case="upper")


def sql_diff_html(sql_a: str, sql_b: str, label_a: str = "DSPy", label_b: str = "GGUF") -> str:
    lines_a = format_sql(sql_a).splitlines()
    lines_b = format_sql(sql_b).splitlines()
    diff = difflib.HtmlDiff(tabsize=2, wrapcolumn=80)
    return diff.make_table(lines_a, lines_b, fromdesc=label_a, todesc=label_b, context=False)


_DIFF_CSS = """
<style>
.diff_add { background-color: rgba(40, 167, 69, 0.15); }
.diff_sub { background-color: rgba(220, 53, 69, 0.15); }
.diff_chg { background-color: rgba(255, 193, 7, 0.15); }
.diff_header { background-color: rgba(128, 128, 128, 0.1); font-weight: bold; }
table.diff { border-collapse: collapse; width: 100%; font-family: monospace; color: inherit; }
table.diff td, table.diff th { padding: 4px 8px; border: 1px solid rgba(128, 128, 128, 0.2); }
</style>
"""


# ── GeoJSON helpers ───────────────────────────────────────────────────────────


def _to_python(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _is_notna(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) > 0
    return pd.notna(val)


def to_feature_collection(result_df: pd.DataFrame) -> dict:
    geom_cols = []
    for c in result_df.columns:
        vals = [v for v in result_df[c].head(5) if isinstance(v, str)]
        if vals and all(v.lstrip().startswith('{"type":') for v in vals):
            geom_cols.append(c)

    prop_cols = [c for c in result_df.columns if c not in geom_cols]
    features = []
    for _, row in result_df.iterrows():
        geometry = None
        if geom_cols:
            raw = row[geom_cols[0]]
            if raw and isinstance(raw, str):
                geometry = json.loads(raw)
        properties = {}
        for c in prop_cols:
            val = row[c]
            if _is_notna(val):
                properties[c] = _to_python(val)
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    return {"type": "FeatureCollection", "features": features}


def _extract_coords(geom):
    t = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if t == "Point":
        yield coords
    elif t in ("LineString", "MultiPoint"):
        yield from coords
    elif t == "Polygon":
        for ring in coords:
            yield from ring
    elif t in ("MultiLineString", "MultiPolygon"):
        for part in coords:
            if t == "MultiLineString":
                yield from part
            else:
                for ring in part:
                    yield from ring
    elif t == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from _extract_coords(g)


def bbox_from_geojson(geojson):
    lngs, lats = [], []
    for f in geojson.get("features", []):
        geom = f.get("geometry")
        if geom:
            for coord in _extract_coords(geom):
                lngs.append(coord[0])
                lats.append(coord[1])
    if not lngs:
        return None
    return min(lngs), min(lats), max(lngs), max(lats)


def render_map(geojson, color, key):
    n = len(geojson.get("features", []))
    if not n:
        st.info("No features returned.")
        return

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            get_fill_color=color,
            get_line_color=[100, 100, 100, 200],
            get_line_width=2,
            pickable=True,
        ),
    ]

    bbox = bbox_from_geojson(geojson)
    if bbox:
        min_lng, min_lat, max_lng, max_lat = bbox
        span = max(max_lng - min_lng, max_lat - min_lat, 1e-6)
        zoom = max(0, min(18, math.log2(360 / span) - 0.8))
        view = pdk.ViewState(
            latitude=(min_lat + max_lat) / 2,
            longitude=(min_lng + max_lng) / 2,
            zoom=zoom,
        )
    else:
        view = pdk.ViewState(latitude=0, longitude=0, zoom=1)

    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=view, map_style=None),
        use_container_width=True,
        height=400,
        key=key,
    )


# ── SQL generation helpers ────────────────────────────────────────────────────


def generate_sql_dspy(query: str, candidates_df: pd.DataFrame) -> tuple[str, float]:
    """Generate SQL via DSPy (Ollama / cloud LM). Returns (sql, elapsed_secs)."""
    candidates_str = candidates_df.to_string(index=False)
    t0 = time.perf_counter()
    pred = write_sql(
        user_query=query,
        schema=SCHEMA_INFO,
        candidates=candidates_str,
        previous_sql="",
        execution_error="",
    )
    elapsed = time.perf_counter() - t0
    return _strip_fences(pred.sql), elapsed


def generate_sql_gguf(query: str, candidates_df: pd.DataFrame) -> tuple[str, float]:
    """Generate SQL via finetuned GGUF (llama-server). Returns (sql, elapsed_secs)."""
    t0 = time.perf_counter()
    sql = generate_sql(query, candidates_df)
    elapsed = time.perf_counter() - t0
    return sql, elapsed


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Gazet · Compare", page_icon="⚖️", layout="wide")
st.title("⚖️ Gazet · Compare")
st.caption("Side-by-side SQL generation: DSPy (cloud LM) vs finetuned GGUF (llama-server)")

# ── Status indicators ────────────────────────────────────────────────────────
gguf_ok = is_llama_server_available()
col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("**DSPy** · Ollama / cloud LM → :green[ready]")
with col_s2:
    if gguf_ok:
        st.markdown(f"**GGUF** · llama-server (`{LLAMA_SERVER_URL}`) → :green[ready]")
    else:
        st.markdown(f"**GGUF** · llama-server (`{LLAMA_SERVER_URL}`) → :red[not running]")
        st.code(
            "llama-server -m finetune/models/gemma-270m-q8.gguf -ngl 99 --port 8080 --log-disable",
            language="bash",
        )

st.divider()

# ── Query input ──────────────────────────────────────────────────────────────
if "run_q" not in st.session_state:
    st.session_state.run_q = None

inp_col, btn_col = st.columns([5, 1])
with inp_col:
    q = st.text_input("Query", placeholder="e.g. Top 5 smallest regions in Argentina", label_visibility="collapsed")
with btn_col:
    search_clicked = st.button("Compare!", type="primary")

if search_clicked and q:
    st.session_state.run_q = q

cols_ex = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    with cols_ex[i % 3]:
        if st.button(ex, key=f"ex_{i}"):
            st.session_state.run_q = ex

# ── Run comparison ───────────────────────────────────────────────────────────
to_run = st.session_state.run_q
if to_run:
    st.session_state.run_q = None
    st.subheader(f"🔍 {to_run}")

    # 1. Extract places
    with st.status("Extracting places...", expanded=False) as status:
        pred = extract(query=to_run)
        places = pred.result
        status.update(label=f"Extracted {len(places.places)} place(s)", state="complete")

    with st.expander("Extracted places", expanded=False):
        st.dataframe(
            pd.DataFrame([p.model_dump() for p in places.places]),
            use_container_width=True,
            hide_index=True,
        )

    # 2. Fuzzy search candidates
    con = get_duckdb_connection()
    with st.status("Searching candidates...", expanded=False) as status:
        all_candidates = []
        for place in places.places:
            for search_fn in (search_divisions_area, search_natural_earth):
                df = search_fn(con, place)
                if not df.empty:
                    all_candidates.append(df)

        if not all_candidates:
            st.error("No candidates found.")
            st.stop()

        candidates_df = (
            pd.concat(all_candidates, ignore_index=True)
            .drop_duplicates(subset=["source", "id"])
            .sort_values(["similarity", "admin_level"], ascending=[False, True])
            .reset_index(drop=True)
        )
        status.update(label=f"Found {len(candidates_df)} candidate(s)", state="complete")

    with st.expander("Candidates", expanded=False):
        st.dataframe(candidates_df, use_container_width=True, hide_index=True)

    st.divider()

    # 3. Generate SQL side-by-side
    col_dspy, col_gguf = st.columns(2)

    dspy_sql = None
    gguf_sql = None

    # DSPy
    with col_dspy:
        st.markdown("### 🧠 DSPy · cloud LM")
        with st.spinner("Generating SQL..."):
            try:
                dspy_sql, dspy_time = generate_sql_dspy(to_run, candidates_df)
                st.caption(f"⏱ {dspy_time:.2f}s")
                st.code(format_sql(dspy_sql), language="sql")

                dspy_sql_local = rewrite_data_paths(dspy_sql)
                df = execute_sql(con, dspy_sql_local)
                geojson = to_feature_collection(df)
                render_map(geojson, [40, 180, 160, 140], key="map_dspy")
                with st.expander("Result table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # GGUF
    with col_gguf:
        st.markdown("### ⚡ GGUF · llama-server")
        if not gguf_ok:
            st.warning("llama-server not running — skipping GGUF.")
        else:
            with st.spinner("Generating SQL..."):
                try:
                    gguf_sql, gguf_time = generate_sql_gguf(to_run, candidates_df)
                    st.caption(f"⏱ {gguf_time:.2f}s")
                    st.code(format_sql(gguf_sql), language="sql")

                    gguf_sql_local = rewrite_data_paths(gguf_sql)
                    df = execute_sql(con, gguf_sql_local)
                    geojson = to_feature_collection(df)
                    render_map(geojson, [180, 80, 60, 140], key="map_gguf")
                    with st.expander("Result table"):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # SQL diff
    if dspy_sql and gguf_sql:
        is_match = format_sql(dspy_sql).strip() == format_sql(gguf_sql).strip()
        if is_match:
            st.success("✅ Both models generated identical SQL.")
        else:
            with st.expander("SQL Diff", expanded=True):
                diff_html = sql_diff_html(dspy_sql, gguf_sql)
                st.html(
                    f"{_DIFF_CSS}<div style='overflow-x:auto; font-size:13px;'>{diff_html}</div>"
                )

    con.close()
