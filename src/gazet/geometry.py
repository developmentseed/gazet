from typing import Any

import duckdb
import pandas as pd

SIMPLIFY_TOLERANCE = 0.001  # ~100m; adequate for web map display
COORD_PRECISION = 5  # ~1.1m; sufficient for web maps, shrinks payload 30-50%
_GRID = 10.0**-COORD_PRECISION

# Precision reduction has to happen inside GEOS, not by rounding the GeoJSON
# afterwards: snapping vertices to the grid collapses near-coincident points and
# folds rings, which silently undoes ST_SimplifyPreserveTopology's guarantee and
# shipped most country-scale AOIs self-intersecting. ST_ReducePrecision snaps and
# repairs in one step. It raises on input GEOS already considers invalid, so retry
# through ST_MakeValid, and fall back to the un-reduced geometry if that fails or
# if the feature is small enough to vanish into a single grid cell.
_NORMALIZE_SQL = """
WITH simplified AS (
    SELECT ST_SimplifyPreserveTopology({geom_expr}, ?) AS g
), reduced AS (
    SELECT g, COALESCE(
        TRY(ST_ReducePrecision(g, ?)),
        TRY(ST_ReducePrecision(ST_MakeValid(g), ?))
    ) AS r FROM simplified
)
SELECT ST_AsGeoJSON(CASE WHEN r IS NULL OR ST_IsEmpty(r) THEN g ELSE r END) FROM reduced
"""


def normalize_geometry_to_geojson(
    con: duckdb.DuckDBPyConnection,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Simplify geometries and convert to compact GeoJSON text.

    Accepts either binary WKB blobs or GeoJSON strings in the `geometry` column.
    Runs ST_SimplifyPreserveTopology then reduces coordinate precision, both in
    GEOS, so the result is valid and small.

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
            _NORMALIZE_SQL.format(geom_expr=geom_expr),
            [arg, SIMPLIFY_TOLERANCE, _GRID, _GRID],
        ).fetchone()
        if not row or not row[0]:
            return None
        return row[0]

    normalized_df = result_df.copy()
    normalized_df["geometry"] = normalized_df["geometry"].apply(_simplify)
    return normalized_df
