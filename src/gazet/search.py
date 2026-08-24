import logging

import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH
from .schemas import Place

logger = logging.getLogger(__name__)

#: Where each source keeps a record's own name and its English one. Overture
#: holds translations in a ``names.common`` map; the Natural Earth conversion
#: gives every language a struct field of its own (``ingest/convert_natural_earth.py``).
PRIMARY_NAME = "names.primary"
DIVISIONS_AREA_ENGLISH_NAME = "names.common.en"
NATURAL_EARTH_ENGLISH_NAME = "names.en"


def _readable(expr: str) -> str:
    """SQL that reads a name, treating a blank one as absent."""
    return f"NULLIF(trim({expr}), '')"


def simple_fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    place: Place,
    name_expr: str = PRIMARY_NAME,
    english_name_expr: str | None = None,
    extra_select: str = "",
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
) -> pd.DataFrame:
    """Jaro-Winkler similarity search using only the place name.

    Each record is matched under two names: ``name_expr``, its own, and
    ``english_name_expr``, its English one where the source carries it. A
    record scores the better of the two, so a place is reachable by its
    English exonym ("Copenhagen") as well as by its local name ("Københavns
    Kommune"), and a record with no English name matches on its own name
    alone. ``matched_name`` is whichever of the two produced the score.

    ``name`` is the English name where there is one and the record's own name
    otherwise — the expression :func:`fetch_by_id` uses too, so a candidate
    and a later lookup of that same candidate agree on what it is called.

    Ranks exact-substring matches (query literally contained in the name,
    e.g. "Loja" in "Loja Province") ahead of pure edit-distance near-misses
    (e.g. "Lodja"), even when the near-miss scores a slightly higher raw
    Jaro-Winkler similarity — substring containment is a stronger signal of
    a true match than character-level similarity alone.

    ``include_geometry``/``include_bbox`` require the spatial extension to
    already be loaded on ``con`` (``INSTALL spatial; LOAD spatial;``).
    ``include_bbox`` computes ``[minx, miny, maxx, maxy]`` via
    ST_XMin/YMin/XMax/YMax — a much smaller payload than full geometry
    (no coordinate arrays or GeoJSON serialization), for lightweight
    context (e.g. ``ids_only`` responses).
    """
    params = [path, place.place, place.place, place.place, place.place, limit]

    english_expr = english_name_expr or "CAST(NULL AS VARCHAR)"
    extra_clause = f", {extra_select}" if extra_select else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    bbox_clause = (
        ", [ST_XMin(geometry), ST_YMin(geometry), ST_XMax(geometry), ST_YMax(geometry)] AS bbox"
        if include_bbox
        else ""
    )
    # Geometry and bbox are computed in the first stage, so the final select
    # only has to name them.
    carried_clause = (", geometry" if include_geometry else "") + (
        ", bbox" if include_bbox else ""
    )
    rel = con.execute(
        f"""
        WITH named AS (
            SELECT
                id,
                country,
                subtype,
                class,
                region,
                admin_level,
                is_land,
                is_territorial{extra_clause}{geometry_clause}{bbox_clause},
                {_readable(name_expr)} AS primary_name,
                {_readable(english_expr)} AS english_name
            FROM read_parquet(?)
        ),
        scored AS (
            SELECT
                *,
                COALESCE(
                    jaro_winkler_similarity(lower(primary_name), lower(?)), 0.0
                ) AS primary_similarity,
                COALESCE(
                    jaro_winkler_similarity(lower(english_name), lower(?)), 0.0
                ) AS english_similarity,
                COALESCE(contains(lower(primary_name), lower(?)), false)
                    AS primary_substring,
                COALESCE(contains(lower(english_name), lower(?)), false)
                    AS english_substring
            FROM named
            WHERE primary_name IS NOT NULL OR english_name IS NOT NULL
        )
        SELECT
            id,
            COALESCE(english_name, primary_name) AS name,
            country,
            subtype,
            class,
            region,
            admin_level,
            is_land,
            is_territorial{extra_clause}{carried_clause},
            -- Compared as a pair so the winning name is the one the ranking
            -- below would pick: a substring hit beats a higher raw score.
            CASE
                WHEN (english_substring, english_similarity)
                     > (primary_substring, primary_similarity)
                THEN english_name
                ELSE COALESCE(primary_name, english_name)
            END AS matched_name,
            GREATEST(primary_similarity, english_similarity) AS similarity,
            primary_substring OR english_substring AS is_substring_match
        FROM scored
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

    Matched on ``names.primary`` and on ``names.common.en`` together: 77% of
    divisions_area rows have no English common name and would drop out of
    every search if English were the only match field, while ``names.primary``
    on its own leaves a place unreachable by the English name it is usually
    asked for.
    """
    return simple_fuzzy_search(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        place,
        english_name_expr=DIVISIONS_AREA_ENGLISH_NAME,
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
        english_name_expr=NATURAL_EARTH_ENGLISH_NAME,
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
    )


def fetch_by_id(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    id: str,
    name_expr: str = PRIMARY_NAME,
    english_name_expr: str | None = None,
    extra_select: str = "",
    include_geometry: bool = True,
) -> pd.DataFrame:
    """Look up a single record by exact ID — no fuzzy matching.

    ``name`` is built as :func:`simple_fuzzy_search` builds it — the English
    name where the record has one, its own name otherwise — so a record
    answers to a single name whichever way it is reached.

    ``include_geometry`` requires the spatial extension to already be
    loaded on ``con`` (``INSTALL spatial; LOAD spatial;``).
    """
    english_expr = english_name_expr or "CAST(NULL AS VARCHAR)"
    extra_clause = f", {extra_select}" if extra_select else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    rel = con.execute(
        f"""
        SELECT
            id,
            COALESCE({_readable(english_expr)}, {_readable(name_expr)}) AS name,
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
        english_name_expr=DIVISIONS_AREA_ENGLISH_NAME,
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
        english_name_expr=NATURAL_EARTH_ENGLISH_NAME,
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
