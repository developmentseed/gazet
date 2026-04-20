import json
import math
from typing import Any, Generator

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .export import to_feature_collection
from .lm import extract, generate_places
from .search import search_candidates
from .sql import run_geo_sql_dspy, run_geo_sql_gguf

app = FastAPI()


def _per_source_limit(num_places: int) -> int:
    """Candidates to fetch per source per place, scaled by number of places.

    Keeps the total candidate count in the prompt manageable:
      1 place  → 5 per source → 10 total
      2 places → 4 per source → 16 total
      3 places → 3 per source → 18 total
      4 places → 2 per source → 16 total
      5 places → 2 per source → 20 total
    """
    table = {1: 5, 2: 4, 3: 3, 4: 2, 5: 2}
    return table.get(num_places, max(1, math.ceil(5 / num_places)))


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to list of dicts for JSON; handle non-JSON-serializable types."""
    return df.replace({float("nan"): None}).to_dict(orient="records")


def _run_stream(query: str, backend: str = "gguf") -> Generator[str, None, None]:
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
        places_result = generate_places(query)
    else:
        pred = extract(query=query)
        places_result = pred.result
    print("places:", places_result)

    yield json.dumps({"type": "places", "data": places_result.model_dump()}) + "\n"

    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")

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
            .sort_values(["similarity", "admin_level"], ascending=[False, True])
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


@app.get("/search/stream")
def search_stream(q: str, backend: str = "gguf") -> StreamingResponse:
    """Stream search progress as NDJSON (one JSON object per line)."""
    return StreamingResponse(_run_stream(q, backend), media_type="application/x-ndjson")


@app.get("/search", response_model=None)
def search(q: str, backend: str = "gguf") -> dict[str, Any]:
    """Run geo search for natural-language query (non-streaming).

    Returns GeoJSON FeatureCollection, the executed SQL, and the identified
    dataframes (candidates) as JSON-serializable records.
    """
    places: dict = {}
    candidates: list = []
    sql = ""
    geojson: dict | None = None

    for line in _run_stream(q, backend):
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
