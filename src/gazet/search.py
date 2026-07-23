import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH
from .schemas import Place


def simple_fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    place: Place,
    name_expr: str = 'names.common.en',
    extra_select: str = "",
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Jaro-Winkler similarity search using only the place name.

    ``include_geometry``/``include_bbox`` require the spatial extension to
    already be loaded on ``con`` (``INSTALL spatial; LOAD spatial;``).
    ``include_bbox`` computes ``[minx, miny, maxx, maxy]`` via
    ST_XMin/YMin/XMax/YMax — a much smaller payload than full geometry
    (no coordinate arrays or GeoJSON serialization), for lightweight
    context (e.g. ``ids_only`` responses).
    """
    params = [place.place, path, limit]

    extra_clause = f", {extra_select}" if extra_select else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    bbox_clause = (
        ", [ST_XMin(geometry), ST_YMin(geometry), ST_XMax(geometry), ST_YMax(geometry)] AS bbox"
        if include_bbox
        else ""
    )
    rel = con.execute(
        f"""
        SELECT
            id,
            {name_expr} AS name,
            country,
            subtype,
            class,
            region,
            admin_level,
            is_land,
            is_territorial{extra_clause}{geometry_clause}{bbox_clause},
            jaro_winkler_similarity(lower({name_expr}), lower(?)) AS similarity
        FROM read_parquet(?)
        WHERE {name_expr} IS NOT NULL AND trim({name_expr}) != ''
        ORDER BY similarity DESC, admin_level ASC
        LIMIT ?
        """,
        params,
    )
    df = rel.fetchdf()
    df.insert(0, "source", source)
    if df.empty:
        print(f"\n{source} - \"{place.place}\": no matches")
    else:
        print(f"\n{source} - \"{place.place}\" (top {len(df)} by Jaro-Winkler):")
        preview_cols = [c for c in df.columns if c != "geometry"]
        print(df[preview_cols].to_string(index=False))
    return df


def search_divisions_area(
    con: duckdb.DuckDBPyConnection,
    place: Place,
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match a place against divisions_area (Overture admin boundaries)."""
    return simple_fuzzy_search(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        place,
        extra_select="division_id",
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
    )


def search_natural_earth(
    con: duckdb.DuckDBPyConnection,
    place: Place,
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match a place against Natural Earth geography polygons."""
    return simple_fuzzy_search(
        con,
        NATURAL_EARTH_PATH,
        "natural_earth",
        place,
        name_expr='names.primary',
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
    )


def fetch_by_id(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    id: str,
    name_expr: str = 'names.common.en',
    extra_select: str = "",
    include_geometry: bool = True,
) -> pd.DataFrame:
    """Look up a single record by exact ID — no fuzzy matching.

    ``include_geometry`` requires the spatial extension to already be
    loaded on ``con`` (``INSTALL spatial; LOAD spatial;``).
    """
    extra_clause = f", {extra_select}" if extra_select else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    rel = con.execute(
        f"""
        SELECT
            id,
            {name_expr} AS name,
            country,
            subtype,
            class,
            region,
            admin_level,
            is_land,
            is_territorial{extra_clause}{geometry_clause}
        FROM read_parquet(?)
        WHERE id = ?
        LIMIT 1
        """,
        [path, id],
    )
    df = rel.fetchdf()
    df.insert(0, "source", source)
    return df


def get_division_by_id(
    con: duckdb.DuckDBPyConnection, id: str, include_geometry: bool = True
) -> pd.DataFrame:
    """Look up a single divisions_area record by exact ID."""
    return fetch_by_id(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        id,
        extra_select="division_id",
        include_geometry=include_geometry,
    )


def get_natural_earth_by_id(
    con: duckdb.DuckDBPyConnection, id: str, include_geometry: bool = True
) -> pd.DataFrame:
    """Look up a single Natural Earth record by exact ID."""
    return fetch_by_id(
        con,
        NATURAL_EARTH_PATH,
        "natural_earth",
        id,
        name_expr='names.primary',
        include_geometry=include_geometry,
    )


_SOURCE_SEARCH_FNS = {
    "divisions_area": search_divisions_area,
    "natural_earth": search_natural_earth,
}

_SOURCE_FETCH_FNS = {
    "divisions_area": get_division_by_id,
    "natural_earth": get_natural_earth_by_id,
}


def get_by_id(
    con: duckdb.DuckDBPyConnection,
    id: str,
    source: str | None = None,
    include_geometry: bool = True,
) -> pd.DataFrame:
    """Look up a single record by ID, inferring the source if not given.

    Natural Earth IDs are always prefixed ``ne_`` (see ``config.SCHEMA_INFO``);
    anything else is assumed to be a divisions_area ID.
    """
    resolved_source = source or ("natural_earth" if id.startswith("ne_") else "divisions_area")
    return _SOURCE_FETCH_FNS[resolved_source](con, id, include_geometry=include_geometry)


def search_candidates(
    con: duckdb.DuckDBPyConnection,
    place: Place,
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
    sources: tuple[str, ...] = ("divisions_area", "natural_earth"),
) -> list[pd.DataFrame]:
    """Return candidate DataFrames for a place from the requested sources.

    Defaults to always searching divisions_area and natural_earth to avoid
    missing natural features when the model assigns an incorrect admin
    subtype. Pass ``sources`` to restrict to a subset.
    """
    results = []
    for source in sources:
        df = _SOURCE_SEARCH_FNS[source](
            con,
            place,
            limit=limit,
            include_geometry=include_geometry,
            include_bbox=include_bbox,
        )
        if not df.empty:
            results.append(df)
    return results
