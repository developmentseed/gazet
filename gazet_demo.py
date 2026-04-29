"""Demo Streamlit app for gazet API. Run API first: uv run uvicorn gazet.api:app --reload"""

import json
import math
import os
import re

import pandas as pd
import requests
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None


def _coords_from_geom(geom):
    """Yield (lng, lat) from a GeoJSON geometry."""
    if geom is None:
        return
    t = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return
    if t == "Point":
        yield coords
    elif t in ("LineString", "MultiPoint"):
        for c in coords:
            yield c
    elif t == "Polygon":
        for ring in coords:
            for c in ring:
                yield c
    elif t in ("MultiLineString", "MultiPolygon"):
        for part in coords:
            for c in part if t == "MultiLineString" else part[0]:
                yield c
    elif t == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from _coords_from_geom(g)


def bbox_from_geojson(geojson):
    """Return (min_lng, min_lat, max_lng, max_lat) or None if no coordinates."""
    lngs, lats = [], []
    for f in geojson.get("features", []):
        geom = (
            f.get("geometry") if isinstance(f, dict) else getattr(f, "geometry", None)
        )
        for lng, lat in _coords_from_geom(geom):
            lngs.append(lng)
            lats.append(lat)
    if not lngs:
        return None
    return min(lngs), min(lats), max(lngs), max(lats)


def view_state_for_bbox(bbox, padding_zoom=0.8):
    """Return pydeck ViewState (lat, lon, zoom) to fit bbox (min_lng, min_lat, max_lng, max_lat)."""
    min_lng, min_lat, max_lng, max_lat = bbox
    lat = (min_lat + max_lat) / 2
    lng = (min_lng + max_lng) / 2
    lon_span = max(max_lng - min_lng, 1e-6)
    lat_span = max(max_lat - min_lat, 1e-6)
    span_deg = max(lon_span, lat_span)
    zoom = math.log2(360 / span_deg) - padding_zoom
    zoom = max(0, min(18, zoom))
    return pdk.ViewState(latitude=lat, longitude=lng, zoom=zoom)


def _has_line_geometries(features):
    """Return True if features are predominantly line/point (non-polygon) geometries."""
    line_types = {"LineString", "MultiLineString", "Point", "MultiPoint"}
    count = sum(
        1 for f in features
        if f.get("geometry", {}).get("type") in line_types
    )
    return count > len(features) / 2


def _render_map(geojson, placeholder):
    features = geojson.get("features", [])
    n = len(features)
    if pdk and n:
        is_linear = _has_line_geometries(features)
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            stroked=True,
            filled=not is_linear,
            get_fill_color=[40, 180, 160, 120],
            get_line_color=[0, 140, 255, 255] if is_linear else [10, 50, 46, 255],
            get_line_width=500 if is_linear else 80,
            line_width_min_pixels=2 if is_linear else 1,
            pickable=True,
        )
        with placeholder.container():
            selected_idx = None
            if n > 1:
                names = [
                    f.get("properties", {}).get("name", f"Feature {i}")
                    for i, f in enumerate(features)
                ]
                choice = st.selectbox(
                    "Zoom to feature",
                    ["All features"] + names,
                    key="feature_zoom",
                )
                if choice != "All features":
                    selected_idx = names.index(choice)

            if selected_idx is not None:
                single = {"type": "FeatureCollection", "features": [features[selected_idx]]}
                bbox = bbox_from_geojson(single)
            else:
                bbox = bbox_from_geojson(geojson)

            view = (
                view_state_for_bbox(bbox)
                if bbox
                else pdk.ViewState(latitude=0, longitude=0, zoom=1)
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    map_style=None,
                    tooltip={"text": "{name}"},
                ),
                width="stretch",
                height=500,
            )
    elif n:
        with placeholder.container():
            st.json(geojson)


API = os.environ.get("GAZET_API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "Odisha, India",
    "Neighbouring states of Odisha",
    "Odisha excluding Cuttack",
    "Coastal districts of Odisha",
    "1 km buffer along the border of Odisha and West Bengal",
    "Western half of Odisha",
    "Rivers flowing through Odisha",
    "Districts along the Indravati river",
]

st.set_page_config(page_title="Gazet", page_icon="🌍", layout="wide")
st.markdown("""<style>
[data-testid="stBaseButton-tertiary"] {
    border: 1px dashed rgba(128,128,128,0.3) !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.7rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stBaseButton-tertiary"]:hover {
    background: rgba(40,180,160,0.1) !important;
    border: 1px solid rgb(40,180,160) !important;
}
</style>""", unsafe_allow_html=True)

st.title("Gazet")
st.caption(
    "/ask plain english to geometry"
)

backend = "gguf"

if "run_q" not in st.session_state:
    st.session_state.run_q = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

col1, col2 = st.columns([1, 2])
with col1:
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        q = st.text_input(
            "Query",
            placeholder="e.g. Southern half of Florida",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("Go!", type="primary")
    if search_clicked and q:
        st.session_state.run_q = q
    st.caption("Click an example to get started ->")
    for ex in EXAMPLES:
        if st.button(ex, key=ex, type="tertiary"):
            st.session_state.run_q = ex

with col2:
    to_run = st.session_state.run_q
    if to_run:
        st.session_state.run_q = None
        st.session_state.last_result = None

        status_ph = st.empty()
        download_ph = st.empty()
        map_ph = st.empty()
        places_ph = st.empty()
        candidates_ph = st.empty()
        sql_ph = st.empty()

        status_ph.info("Extracting places…")

        result = {"query": to_run, "places": None, "candidates": None, "sql": None, "geojson": None}

        try:
            with requests.get(
                f"{API}/search/stream", params={"q": to_run, "backend": backend}, stream=True, timeout=120
            ) as r:
                r.raise_for_status()

                for raw in r.iter_lines():
                    if not raw:
                        continue
                    event = json.loads(raw)
                    t = event["type"]

                    if t == "warming_up":
                        status_ph.warning(event["data"])
                        continue

                    if t == "places":
                        places = event["data"].get("places", [])
                        result["places"] = places
                        status_ph.info("Fuzzy-matching candidates…")
                        if places:
                            with places_ph.container():
                                with st.expander(
                                    "Extracted place names", expanded=True
                                ):
                                    st.dataframe(
                                        pd.DataFrame(places).rename(
                                            columns={
                                                "place": "Place",
                                                "country": "Country",
                                                "subtype": "Subtype",
                                            }
                                        ),
                                        width="stretch",
                                        hide_index=True,
                                    )

                    elif t == "candidates":
                        result["candidates"] = event["data"]
                        status_ph.info("Generating SQL…")
                        with candidates_ph.container():
                            with st.expander("Candidate datasets", expanded=True):
                                st.dataframe(
                                    pd.DataFrame(event["data"]),
                                    width="stretch",
                                    hide_index=True,
                                )

                    elif t == "sql_attempt":
                        iteration = event.get("iteration", "")
                        result["sql"] = event["data"]
                        status_ph.info(f"Running SQL (attempt {iteration})…")
                        with sql_ph.container():
                            with st.expander("SQL", expanded=True):
                                st.code(event["data"], language="sql")

                    elif t == "sql_error":
                        status_ph.warning(
                            f"SQL error on attempt {event.get('iteration', '')}, retrying… "
                            f"`{event['data'][:120]}`"
                        )

                    elif t == "geojson":
                        geojson = event["data"]
                        result["geojson"] = geojson
                        n = len(geojson.get("features", []))
                        status_ph.success(f"**{to_run}** → {n} feature(s)")
                        _slug = re.sub(r"[^\w]+", "_", to_run.lower()).strip("_") or "result"
                        download_ph.download_button(
                            "Download GeoJSON",
                            data=json.dumps(geojson),
                            file_name=f"{_slug}.geojson",
                            mime="application/geo+json",
                            key=f"dl_{_slug}",
                        )
                        _render_map(geojson, map_ph)

                    elif t == "error":
                        status_ph.error(event["data"])

        except requests.RequestException as e:
            status_ph.error(
                f"API error: {e}. Is the API running? `uv run uvicorn gazet.api:app --reload`"
            )

        st.session_state.last_result = result

    elif st.session_state.last_result:
        result = st.session_state.last_result
        query = result["query"]
        n_feat = len((result["geojson"] or {}).get("features", []))
        st.success(f"**{query}** -> {n_feat} feature(s)")
        if result["geojson"]:
            _slug = re.sub(r"[^\w]+", "_", query.lower()).strip("_") or "result"
            st.download_button(
                "Download GeoJSON",
                data=json.dumps(result["geojson"]),
                file_name=f"{_slug}.geojson",
                mime="application/geo+json",
                key=f"dl_cached_{_slug}",
            )
        _render_map(result["geojson"], st.empty())
        if result["places"]:
            with st.expander("Extracted place names"):
                st.dataframe(
                    pd.DataFrame(result["places"]).rename(
                        columns={"place": "Place", "country": "Country", "subtype": "Subtype"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
        if result["candidates"]:
            with st.expander("Candidate datasets"):
                st.dataframe(
                    pd.DataFrame(result["candidates"]),
                    width="stretch",
                    hide_index=True,
                )
        if result["sql"]:
            with st.expander("SQL"):
                st.code(result["sql"], language="sql")
