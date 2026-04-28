"""Streamlit eval viewer: compare expected vs predicted SQL and view results on a map.

Usage: streamlit run finetune/eval_demo.py
"""

import difflib
import json
import math
import os
import pathlib

import duckdb
import numpy as np
import pandas as pd
import pydeck as pdk
import sqlparse
import streamlit as st

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(
    os.environ.get("GAZET_DATA_DIR", str(PROJECT_ROOT / "data"))
)
EVAL_DIR = pathlib.Path(
    os.environ.get("GAZET_EVAL_DIR", str(PROJECT_ROOT / "results"))
)


def load_eval_results(path):
    with open(path) as f:
        return json.load(f)


def rewrite_data_paths(sql):
    """Replace symbolic and legacy paths with the configured runtime data paths."""
    # Legacy fixed Docker paths must be replaced first to avoid double-expansion.
    sql = sql.replace("/data/overture/division_area/*.parquet", DIVISIONS_AREA_PATH)
    sql = sql.replace("/data/overture/divisions_area/*.parquet", DIVISIONS_AREA_PATH)
    sql = sql.replace(
        "/data/natural_earth_geoparquet/ne_geography.parquet",
        NATURAL_EARTH_PATH,
    )
    sql = sql.replace("/data/", f"{DATA_DIR}/")
    sql = sql.replace("read_parquet('divisions_area')", f"read_parquet('{DIVISIONS_AREA_PATH}')")
    sql = sql.replace("read_parquet('natural_earth')", f"read_parquet('{NATURAL_EARTH_PATH}')")
    return sql


def format_sql(sql):
    """Pretty-print SQL with sqlparse."""
    return sqlparse.format(sql, reindent=True, keyword_case="upper")


def sql_diff_html(expected, predicted):
    """Return an HTML diff of two SQL strings."""
    expected_lines = format_sql(expected).splitlines()
    predicted_lines = format_sql(predicted).splitlines()
    diff = difflib.HtmlDiff(tabsize=2, wrapcolumn=80)
    return diff.make_table(
        expected_lines, predicted_lines,
        fromdesc="Expected", todesc="Predicted",
        context=False,
    )


def get_duckdb_connection():
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    return con


def execute_sql(con, sql):
    """Execute SQL, converting geometry columns to simplified GeoJSON strings."""
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


def _is_notna(val):
    """Check if a value is not NA, handling arrays/lists/numpy arrays safely."""
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) > 0
    return pd.notna(val)


def _to_python(val):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def to_feature_collection(result_df):
    """Build GeoJSON FeatureCollection from a DataFrame with GeoJSON string columns."""
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


def _centroids_from_geojson(geojson):
    """Extract centroid [lng, lat] for each feature to use as scatter markers."""
    centroids = []
    for f in geojson.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        lngs, lats = [], []
        for coord in _extract_coords(geom):
            lngs.append(coord[0])
            lats.append(coord[1])
        if lngs:
            centroids.append({"lng": sum(lngs) / len(lngs), "lat": sum(lats) / len(lats)})
    return centroids


def render_map(geojson, color, key):
    n = len(geojson.get("features", []))
    if not n:
        st.info("Query returned no features.")
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

        # Add scatter markers when polygons would be too small to see
        if zoom < 4:
            centroids = _centroids_from_geojson(geojson)
            if centroids:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=centroids,
                        get_position=["lng", "lat"],
                        get_fill_color=color[:3] + [220],
                        get_radius=50000,
                        radius_min_pixels=6,
                        pickable=True,
                    )
                )

        view = pdk.ViewState(
            latitude=(min_lat + max_lat) / 2,
            longitude=(min_lng + max_lng) / 2,
            zoom=zoom,
        )
    else:
        view = pdk.ViewState(latitude=0, longitude=0, zoom=1)

    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=view, map_style=None),
        width="stretch",
        height=400,
        key=key,
    )


# --- App ---

st.set_page_config(page_title="Eval Viewer", layout="wide")
st.title("Eval Viewer")

eval_files = sorted(EVAL_DIR.glob("eval-*.json"))
if not eval_files:
    st.error(f"No eval result files found in {EVAL_DIR}")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Eval file",
    eval_files,
    format_func=lambda p: p.stem,
)

data = load_eval_results(selected_file)
summary = data["summary"]
results = data["results"]

st.sidebar.markdown(f"""
**Model**: `{summary.get('label', '')}`
**Exact match**: {summary['exact_matches']}/{summary['num_samples']} ({summary['exact_match_rate']:.1%})
""")

filter_option = st.sidebar.radio("Filter", ["All", "Matches only", "Mismatches only"])
if filter_option == "Matches only":
    results = [r for r in results if r["exact_match"]]
elif filter_option == "Mismatches only":
    results = [r for r in results if not r["exact_match"]]

if not results:
    st.warning("No results match the current filter.")
    st.stop()

questions = [
    f"[{r['index']}] {r.get('question', 'Sample ' + str(r['index']))}"
    for r in results
]
selected_idx = st.selectbox("Select a query", range(len(questions)), format_func=lambda i: questions[i])
row = results[selected_idx]

match_label = "MATCH" if row["exact_match"] else "MISMATCH"
match_color = "green" if row["exact_match"] else "red"
st.markdown(f"### :{match_color}[{match_label}]")

is_sql = summary.get("task", "sql") == "sql"
expected = row["expected"]
predicted = row["predicted"]

# Formatted output side-by-side
col_expected, col_predicted = st.columns(2)
with col_expected:
    st.markdown("**Expected**")
    if is_sql:
        st.code(format_sql(expected), language="sql")
    else:
        st.code(expected, language="json")
with col_predicted:
    st.markdown("**Predicted**")
    if is_sql:
        st.code(format_sql(predicted), language="sql")
    else:
        st.code(predicted, language="json")

# Diff view
if not row["exact_match"]:
    with st.expander("Diff", expanded=True):
        diff_html = sql_diff_html(expected, predicted)
        diff_css = """
        <style>
        .diff_add { background-color: rgba(40, 167, 69, 0.15); }
        .diff_sub { background-color: rgba(220, 53, 69, 0.15); }
        .diff_chg { background-color: rgba(255, 193, 7, 0.15); }
        .diff_header { background-color: rgba(128, 128, 128, 0.1); font-weight: bold; }
        table.diff { border-collapse: collapse; width: 100%; font-family: monospace; color: inherit; }
        table.diff td, table.diff th { padding: 4px 8px; border: 1px solid rgba(128, 128, 128, 0.2); }
        </style>
        """
        st.html(f"{diff_css}<div style='overflow-x:auto; font-size:13px;'>{diff_html}</div>")

# Auto-execute SQL and show maps (only for sql task)
if is_sql:
    con = get_duckdb_connection()

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        st.markdown("**Expected result**")
        sql = rewrite_data_paths(expected)
        try:
            df = execute_sql(con, sql)
            geojson = to_feature_collection(df)
            render_map(geojson, [40, 180, 160, 140], key="map_expected")
            with st.expander("Result table"):
                st.dataframe(df, width="stretch")
        except Exception as e:
            st.error(f"Execution error: {e}")

    with map_col2:
        st.markdown("**Predicted result**")
        sql = rewrite_data_paths(predicted)
        try:
            df = execute_sql(con, sql)
            geojson = to_feature_collection(df)
            render_map(geojson, [180, 80, 60, 140], key="map_predicted")
            with st.expander("Result table"):
                st.dataframe(df, width="stretch")
        except Exception as e:
            st.error(f"Execution error: {e}")

    con.close()
