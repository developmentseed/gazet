import json
import logging
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Generator
from fastapi.responses import Response

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .export import to_feature_collection
from .geometry import normalize_geometry_to_geojson
from .lm import extract, generate_places
from .schemas import Place
from .search import get_by_id, search_candidates
from .sql import run_geo_sql_dspy, run_geo_sql_gguf

_FUZZY_SOURCES = ("divisions_area", "natural_earth")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the spatial extension once at startup; per-request handlers get
    a cheap cursor() off this connection instead of paying the ~90ms
    LOAD spatial cost on every call."""
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    app.state.duckdb_con = con
    yield
    con.close()


app = FastAPI(lifespan=lifespan)

logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach X-Request-Id header and log each request."""
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    response = await call_next(request)
    logger.info("%s %s %d %s %s", request.method, request.url.path, response.status_code, request_id, request.query_params)
    return response


def _per_source_limit(num_places: int) -> int:
    """Candidates to fetch per source per place, scaled by number of places.

    Keeps the total candidate count in the prompt manageable:
      1 place  → 5 per source → 10 total
      2 places → 4 per source → 16 total
      3 places → 3 per source → 18 total
      4+ places → 3 per source → 12+ total (floors at reasonable minimum)
    """
    if num_places <= 1:
        return 5
    if num_places <= 2:
        return 4
    return 3


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to list of dicts for JSON; handle non-JSON-serializable types."""
    # pandas-stubs types to_dict(orient="records") as list[dict[Hashable, Any]],
    # but DataFrame column names are always strings, so the result is list[dict[str, Any]].
    return df.replace({float("nan"): None}).to_dict(orient="records")  # type: ignore[return-value]


def _run_stream(
    base_con: duckdb.DuckDBPyConnection, query: str, backend: str = "gguf"
) -> Generator[str, None, None]:
    """Yield NDJSON lines as each stage of the search completes.

    Event ``type`` values (in order of emission):
    - ``places``      – extracted place names
    - ``candidates``  – merged fuzzy-match table
    - ``sql_attempt`` – SQL generated in the current loop iteration
    - ``sql_error``   – execution/generation error in the current iteration
    - ``geojson``     – final FeatureCollection
    - ``error``       – fatal error (no result)
    """
    if backend == "gguf":
        from .lm import is_llama_server_available
        if not is_llama_server_available():
            yield json.dumps({
                "type": "warming_up",
                "data": "Starting model server, this takes 30-60s on cold start",
            }) + "\n"
        places_result = generate_places(query)
    else:
        pred = extract(query=query)
        places_result = pred.result

    yield json.dumps({"type": "places", "data": places_result.model_dump()}) + "\n"

    con = base_con.cursor()

    try:
        limit = _per_source_limit(len(places_result.places))
        all_candidates: list[pd.DataFrame] = []
        for place in places_result.places:
            all_candidates.extend(search_candidates(con, place, limit=limit))

        if not all_candidates:
            yield json.dumps({"type": "error", "data": "No candidates found"}) + "\n"
            return

        candidates_df = (
            pd.concat(all_candidates, ignore_index=True)
            .drop_duplicates(subset=["source", "id"])
            .sort_values(
                ["is_substring_match", "similarity", "admin_level"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

        yield (
            json.dumps({"type": "candidates", "data": _df_to_records(candidates_df)})
            + "\n"
        )

        sql_fn = run_geo_sql_gguf if backend == "gguf" else run_geo_sql_dspy
        result_df: pd.DataFrame | None = None
        for event in sql_fn(con, query, candidates_df):
            if event["type"] == "sql_attempt":
                yield (
                    json.dumps(
                        {
                            "type": "sql_attempt",
                            "data": event["sql"],
                            "iteration": event["iteration"],
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "sql_error":
                yield (
                    json.dumps(
                        {
                            "type": "sql_error",
                            "data": event["error"],
                            "iteration": event["iteration"],
                        }
                    )
                    + "\n"
                )
            elif event["type"] == "result":
                result_df = event["df"]

        if result_df is None or result_df.empty:
            yield json.dumps({"type": "error", "data": "No result from SQL"}) + "\n"
            return

        yield (
            json.dumps({"type": "geojson", "data": to_feature_collection(result_df)})
            + "\n"
        )

    finally:
        con.close()


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Health check — DuckDB connection alive + llama-server status."""
    con = request.app.state.duckdb_con
    duckdb_ok = False
    try:
        con.execute("SELECT 1")
        duckdb_ok = True
    except Exception:
        pass

    llama_ok = False
    try:
        from .lm import is_llama_server_available
        llama_ok = is_llama_server_available()
    except Exception:
        pass

    status = "ok" if duckdb_ok and llama_ok else ("degraded" if duckdb_ok else "unhealthy")
    return {
        "status": status,
        "duckdb": "ok" if duckdb_ok else "error",
        "llama_server": "ok" if llama_ok else "unavailable",
    }


@app.get("/sources")
def sources(request: Request) -> dict[str, Any]:
    """List available data sources with row counts and name ranges."""
    con = request.app.state.duckdb_con
    from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH

    info = {}
    for name, path in [("divisions_area", DIVISIONS_AREA_PATH), ("natural_earth", NATURAL_EARTH_PATH)]:
        try:
            row = con.execute(
                f"SELECT COUNT(*) as count, MIN(names.primary) as min_name, MAX(names.primary) as max_name FROM read_parquet('{path}')"
            ).fetchone()
            info[name] = {
                "path": path,
                "row_count": row[0],
                "name_range": [row[1], row[2]],
            }
        except Exception as e:
            info[name] = {"path": path, "error": str(e)}

    return info


@app.get("/search/stream")
def search_stream(request: Request, q: str, backend: str = "gguf") -> StreamingResponse:
    """Stream search progress as NDJSON (one JSON object per line)."""
    con = request.app.state.duckdb_con
    return StreamingResponse(
        _run_stream(con, q, backend), media_type="application/x-ndjson"
    )


@app.get("/search", response_model=None)
def search(request: Request, q: str, backend: str = "gguf") -> dict[str, Any]:
    """Run geo search for natural-language query (non-streaming).

    Returns GeoJSON FeatureCollection, the executed SQL, and the identified
    dataframes (candidates) as JSON-serializable records.
    """
    places: dict = {}
    candidates: list = []
    sql = ""
    geojson: dict | None = None

    con = request.app.state.duckdb_con
    for line in _run_stream(con, q, backend):
        if not line.strip():
            continue
        event = json.loads(line)
        t = event["type"]
        if t == "places":
            places = event["data"]
        elif t == "candidates":
            candidates = event["data"]
        elif t == "sql_attempt":
            sql = event["data"]
        elif t == "geojson":
            geojson = event["data"]

    if geojson is None:
        raise HTTPException(status_code=404, detail="No result")

    return {
        "geojson": geojson,
        "sql": sql,
        "places": places,
        "dataframes": {"candidates": candidates},
    }


@app.get("/search/fuzzy", response_model=None)
def search_fuzzy(
    request: Request,
    q: str,
    limit: int = 5,
    simplify: bool = True,
    sources: str | None = None,
    ids_only: bool = False,
) -> dict[str, Any]:
    """Pure fuzzy-name search with geometry, no LLM involved.

    ``q`` is a place-name string (not a natural-language query) — this
    endpoint has no place-extraction step. Matches are ranked by
    Jaro-Winkler similarity across the requested ``sources`` (comma-separated
    subset of divisions_area/natural_earth; defaults to both), combined and
    truncated to the top ``limit``. Returns a GeoJSON FeatureCollection.

    Pass ``ids_only=true`` to skip fetching full geometry and get back
    ``{"ids": [{"source", "id", "name", "country", "subtype", "admin_level", "bbox"}, ...]}``
    instead — ``country``/``subtype``/``admin_level`` disambiguate same-named
    places (e.g. multiple real-world "Loja"s across Ecuador and Spain, or
    Ecuador's "Loja" region vs. its nested "Loja" county — same ``subtype``
    can occur at different ``admin_level``s, and locality-type subtypes have
    no ``admin_level`` at all). ``bbox`` is ``[minx, miny, maxx, maxy]``
    computed via ST_XMin/YMin/XMax/YMax, a much smaller payload than full
    geometry, giving minimal spatial context before fetching the full
    geometry for one candidate via ``GET /geometry/{id}``.
    """
    requested_sources = (
        tuple(s.strip() for s in sources.split(",")) if sources else _FUZZY_SOURCES
    )
    invalid = set(requested_sources) - set(_FUZZY_SOURCES)
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Unknown source(s): {sorted(invalid)}"
        )

    con = request.app.state.duckdb_con.cursor()

    try:
        # Fetch a pool larger than `limit` per source so the combined,
        # similarity-ranked top-`limit` isn't skewed by per-source cutoffs.
        per_source_limit = max(limit * 3, 15)
        candidate_dfs = search_candidates(
            con,
            Place(place=q),
            limit=per_source_limit,
            include_geometry=not ids_only,
            include_bbox=ids_only,
            sources=requested_sources,
        )
        if not candidate_dfs:
            return to_feature_collection(pd.DataFrame())

        candidates_df = (
            pd.concat(candidate_dfs, ignore_index=True)
            .drop_duplicates(subset=["source", "id"])
            .sort_values(
                ["is_substring_match", "similarity"], ascending=[False, False]
            )
            .head(limit)
            .reset_index(drop=True)
        )

        if ids_only:
            scalar_cols = ["source", "id", "name", "country", "subtype", "admin_level"]
            ids_df = candidates_df[scalar_cols].copy()
            ids_df = ids_df.astype(object).where(ids_df.notna(), None)
            ids_df["bbox"] = candidates_df["bbox"].apply(
                lambda arr: [float(x) for x in arr] if arr is not None else None
            )
            return {
                "geojson": {"type": "FeatureCollection", "features": []},
                "ids": ids_df.to_dict(orient="records"),
            }

        if simplify:
            candidates_df = normalize_geometry_to_geojson(con, candidates_df)

        return to_feature_collection(candidates_df)
    finally:
        con.close()


@app.get("/geometry/{id}", response_model=None)
def get_geometry(
    request: Request,
    id: str,
    source: str | None = None,
    simplify: bool = True,
) -> dict[str, Any]:
    """Fetch a single feature's geometry directly by ID — no fuzzy matching, no LLM.

    ``source`` restricts the lookup to ``divisions_area`` or ``natural_earth``;
    if omitted, it's inferred from the ID (Natural Earth IDs are prefixed
    ``ne_``). Returns a single GeoJSON Feature, or 404 if the ID doesn't exist
    in the resolved source.
    """
    if source is not None and source not in _FUZZY_SOURCES:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    con = request.app.state.duckdb_con.cursor()

    try:
        df = get_by_id(con, id, source=source, include_geometry=True)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No feature found for id={id!r}")

        if simplify:
            df = normalize_geometry_to_geojson(con, df)

        return to_feature_collection(df)["features"][0]
    finally:
        con.close()
