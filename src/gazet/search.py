import logging

import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH
from .schemas import Place

logger = logging.getLogger(__name__)

# Overture divisions store the endonym (native name) in names.primary and
# localized names in the names.common map. Match against both the native name
# and the English exonym so a place is findable either way (e.g. "Beijing"
# finds the row whose primary is "北京市"); display the English name when it
# exists, otherwise fall back to the native primary.
_DIV_PRIMARY = 'names."primary"'
_DIV_EN = "names.common['en']"
_DIV_DISPLAY = f"coalesce({_DIV_EN}, {_DIV_PRIMARY})"
_DIV_MATCH_EXPRS = (_DIV_PRIMARY, _DIV_EN)


def simple_fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    place: Place,
    name_expr: str = "names.common.en",
    match_exprs: tuple[str, ...] | None = None,
    extra_select: str = "",
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Jaro-Winkler similarity search using only the place name.

    Ranks exact-substring matches (query literally contained in the name,
    e.g. "Loja" in "Loja Province") ahead of pure edit-distance near-misses
    (e.g. "Lodja"), even when the near-miss scores a slightly higher raw
    Jaro-Winkler similarity — substring containment is a stronger signal of
    a true match than character-level similarity alone.

    Matching is performed against every expression in ``match_exprs``
    (defaulting to ``(name_expr,)``): the ranked similarity is the best
    (``greatest``) Jaro-Winkler across them, and a substring match in ANY of
    them counts. This lets a row be found by more than one name — e.g. its
    native ``names.primary`` OR its English ``names.common['en']`` — so
    querying "Beijing" matches the division whose primary name is "北京市",
    without dropping the many rows that have no English common name (their
    NULL name expressions are simply ignored by ``greatest``). ``name_expr``
    is only the *displayed* name and need not be one of the match expressions.

    ``include_geometry``/``include_bbox`` require the spatial extension to
    already be loaded on ``con`` (``INSTALL spatial; LOAD spatial;``).
    ``include_bbox`` computes ``[minx, miny, maxx, maxy]`` via
    ST_XMin/YMin/XMax/YMax — a much smaller payload than full geometry
    (no coordinate arrays or GeoJSON serialization), for lightweight
    context (e.g. ``ids_only`` responses).
    """
    match_exprs = tuple(match_exprs) if match_exprs else (name_expr,)

    # Best Jaro-Winkler across all match expressions; greatest() ignores NULLs,
    # so a missing name variant (e.g. no English common name) never excludes a row.
    sim_terms = [f"jaro_winkler_similarity(lower({m}), lower(?))" for m in match_exprs]
    similarity_sql = (
        f"greatest({', '.join(sim_terms)})" if len(sim_terms) > 1 else sim_terms[0]
    )
    # coalesce to false so a NULL name variant yields a boolean (not SQL NULL),
    # preserving the True-before-False ordering of is_substring_match.
    substring_sql = (
        "coalesce("
        + " OR ".join(f"contains(lower({m}), lower(?))" for m in match_exprs)
        + ", false)"
    )
    present_sql = " OR ".join(f"({m} IS NOT NULL AND trim({m}) != '')" for m in match_exprs)

    extra_clause = f", {extra_select}" if extra_select else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    bbox_clause = (
        ", [ST_XMin(geometry), ST_YMin(geometry), ST_XMax(geometry), ST_YMax(geometry)] AS bbox"
        if include_bbox
        else ""
    )
    # Placeholders appear in SQL text order: similarity terms, then substring
    # terms (one query value each), then path, then limit.
    params = (
        [place.place] * len(match_exprs)
        + [place.place] * len(match_exprs)
        + [path, limit]
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
            {similarity_sql} AS similarity,
            ({substring_sql}) AS is_substring_match
        FROM read_parquet(?)
        WHERE {present_sql}
        ORDER BY is_substring_match DESC, similarity DESC, admin_level ASC
        LIMIT ?
        """,
        params,
    )
    df = rel.fetchdf()
    df.insert(0, "source", source)
    if df.empty:
        logger.debug("%s - %r: no matches", source, place.place)
    else:
        logger.debug("%s - %r (top %d by Jaro-Winkler)", source, place.place, len(df))
    return df


def search_divisions_area(
    con: duckdb.DuckDBPyConnection,
    place: Place,
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match a place against divisions_area (Overture admin boundaries).

    Matches against both ``names.primary`` (the native/endonym name) and
    ``names.common['en']`` (the English exonym). Overture stores the primary
    name in the local script — e.g. Beijing's primary is "北京市" — so matching
    primary alone silently fails for every non-Latin-script place; adding the
    English common name recovers them. Primary is retained as a match field so
    the many rows that have no English common name are still found.
    """
    return simple_fuzzy_search(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        place,
        name_expr=_DIV_DISPLAY,
        match_exprs=_DIV_MATCH_EXPRS,
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
        name_expr="names.primary",
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
    )


def fetch_by_id(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    id: str,
    name_expr: str = "names.common.en",
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
        name_expr=_DIV_DISPLAY,  # English name when present, else native primary
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
        name_expr="names.primary",
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
    resolved_source = source or (
        "natural_earth" if id.startswith("ne_") else "divisions_area"
    )
    return _SOURCE_FETCH_FNS[resolved_source](
        con, id, include_geometry=include_geometry
    )


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
