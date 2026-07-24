import json
from typing import Any

import duckdb
import pandas as pd

SIMPLIFY_TOLERANCE = 0.001  # ~100m; adequate for web map display
COORD_PRECISION = 5  # ~1.1m; sufficient for web maps, shrinks payload 30-50%


def _round_coords(obj: Any, precision: int) -> Any:
    """Recursively round numeric coordinates in a GeoJSON geometry dict."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, list):
        return [_round_coords(x, precision) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_coords(v, precision) for k, v in obj.items()}
    return obj


def normalize_geometry_to_geojson(
    con: duckdb.DuckDBPyConnection,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Simplify geometries and convert to compact GeoJSON text.

    Accepts either binary WKB blobs or GeoJSON strings in the `geometry` column.
    Runs ST_SimplifyPreserveTopology then rounds coordinates to reduce payload.

    ``con`` must already have the spatial extension loaded
    (``INSTALL spatial; LOAD spatial;``).
    """
    if "geometry" not in result_df.columns or result_df.empty:
        return result_df

    sample = result_df["geometry"].dropna().head(5)
    if sample.empty:
        return result_df

    def _simplify(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, (bytes, bytearray, memoryview)):
            geom_expr = "ST_GeomFromWKB(?::BLOB)"
            arg: Any = bytes(val)
        elif isinstance(val, str) and val.lstrip().startswith('{"'):
            geom_expr = "ST_GeomFromGeoJSON(?)"
            arg = val
        else:
            return val
        row = con.execute(
            f"SELECT ST_AsGeoJSON(ST_SimplifyPreserveTopology({geom_expr}, ?))",
            [arg, SIMPLIFY_TOLERANCE],
        ).fetchone()
        if not row or not row[0]:
            return None
        return json.dumps(_round_coords(json.loads(row[0]), COORD_PRECISION))

    normalized_df = result_df.copy()
    normalized_df["geometry"] = normalized_df["geometry"].apply(_simplify)
    return normalized_df
