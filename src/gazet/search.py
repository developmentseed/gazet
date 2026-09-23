import logging
from functools import cache, partial

import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, LOCALITIES_PATH, NATURAL_EARTH_PATH
from .schemas import Place

logger = logging.getLogger(__name__)

#: Where each source keeps a record's own name and its English one. Overture
#: holds translations in a ``names.common`` map; the Natural Earth conversion
#: gives every language a struct field of its own (``ingest/convert_natural_earth.py``).
PRIMARY_NAME = "names.primary"
DIVISIONS_AREA_ENGLISH_NAME = "names.common.en"
NATURAL_EARTH_ENGLISH_NAME = "names.en"
#: ``[minx, miny, maxx, maxy]`` computed from the geometry, for a source
#: without a bbox of its own.
GEOMETRY_BBOX = (
    "[ST_XMin(geometry), ST_YMin(geometry), ST_XMax(geometry), ST_YMax(geometry)]"
)
#: Overture stores each record's bbox beside its geometry. Reading it leaves
#: the geometry column unopened, which is most of the cost of a bbox search.
DIVISIONS_AREA_BBOX = "[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]"


def _readable(expr: str) -> str:
    """SQL that reads a name, treating a blank one as absent."""
    return f"NULLIF(trim({expr}), '')"


def _folded(expr: str) -> str:
    """SQL that compares a name without case or accents: "Bogota" is "Bogotá"."""
    return f"lower(strip_accents({expr}))"


def search_names_sql(
    name_expr: str, english_name_expr: str, alternate_names_expr: str | None = None
) -> str:
    """SQL for every name a record answers to, each beside its folded form.

    The record's own name comes first and its English one second, so they
    win a tie; the alternates follow with exact repeats dropped. Blank names
    are left out.
    """

    def entry(name: str) -> str:
        return f"{{'name': trim({name}), 'folded': {_folded(f'trim({name})')}}}"

    alternates = (
        f"list_distinct(list_transform({alternate_names_expr}, n -> {entry('n')}))"
        if alternate_names_expr
        else "[]"
    )
    return (
        f"list_filter(list_concat([{entry(name_expr)}, {entry(english_name_expr)}], "
        f"{alternates}), s -> NULLIF(s.name, '') IS NOT NULL)"
    )


#: The column normalize_geodata stores search_names_sql's result in, so a
#: search reads it instead of rebuilding and folding every name per query.
SEARCH_NAMES_COLUMN = "search_names"
#: Overture's translations (``names.common``) and its alternate, short and
#: official names (``names.rules``).
DIVISIONS_AREA_SEARCH_NAMES = search_names_sql(
    PRIMARY_NAME,
    DIVISIONS_AREA_ENGLISH_NAME,
    "list_concat(COALESCE(map_values(names.common), []), "
    "COALESCE(list_transform(names.rules, r -> r.value), []))",
)
NATURAL_EARTH_SEARCH_NAMES = search_names_sql(PRIMARY_NAME, NATURAL_EARTH_ENGLISH_NAME)


@cache
def _stores_search_names(path: str) -> bool:
    """Whether a parquet file (or glob) carries the stored search names."""
    columns = duckdb.connect().execute("DESCRIBE SELECT * FROM read_parquet(?)", [path])
    return SEARCH_NAMES_COLUMN in {row[0] for row in columns.fetchall()}


def _search_names(path: str | list[str], computed: str) -> str:
    """The stored search names where every file has them, else ``computed``.

    Raw Overture downloads and the test fixtures have no stored column.
    """
    paths = [path] if isinstance(path, str) else path
    if all(_stores_search_names(p) for p in paths):
        return SEARCH_NAMES_COLUMN
    return computed


def _divisions_area_paths(include_localities: bool) -> str | list[str]:
    """The divisions_area file, and the localities beside it where asked for.

    Localities are kept apart from the file the natural-language pipeline
    reads, because its model was not trained on them.
    """
    if include_localities and LOCALITIES_PATH:
        return [DIVISIONS_AREA_PATH, LOCALITIES_PATH]
    return DIVISIONS_AREA_PATH


def simple_fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str | list[str],
    source: str,
    place: Place,
    search_names_expr: str,
    name_expr: str = PRIMARY_NAME,
    english_name_expr: str | None = None,
    extra_select: str = "",
    limit: int = 5,
    include_geometry: bool = False,
    include_bbox: bool = False,
    bbox_expr: str = GEOMETRY_BBOX,
) -> pd.DataFrame:
    """Jaro-Winkler similarity search using only the place name.

    Each record is matched under every name in ``search_names_expr`` (see
    :func:`search_names_sql`): its own, its English one, and any others the
    source holds. A record scores its best name, so a place is reachable by
    its English exonym ("Copenhagen") as well as by its local name
    ("Københavns Kommune") or an alternate one ("Chittagong" for "Chattogram
    District"). Names are compared without case or accents, so "Bogota"
    finds "Bogotá". ``matched_name`` is whichever name produced the score;
    on a tie, the record's own name.

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
    ``include_bbox`` returns ``[minx, miny, maxx, maxy]`` from ``bbox_expr``
    — a much smaller payload than full geometry (no coordinate arrays or
    GeoJSON serialization), for lightweight context (e.g. ``ids_only``
    responses).
    """
    english_expr = english_name_expr or "CAST(NULL AS VARCHAR)"
    extra_clause = f", {extra_select}" if extra_select else ""
    bbox_clause = f", {bbox_expr} AS bbox" if include_bbox else ""
    # Geometry is carried raw and only serialised for the rows returned.
    carried_geometry = ", geometry" if include_geometry else ""
    geometry_clause = ", ST_AsGeoJSON(geometry) AS geometry" if include_geometry else ""
    query = _folded("$1")
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
                is_territorial{extra_clause}{bbox_clause}{carried_geometry},
                {_readable(name_expr)} AS primary_name,
                {_readable(english_expr)} AS english_name,
                {search_names_expr} AS search_names
            FROM read_parquet($2)
        ),
        scored AS (
            SELECT
                *,
                -- Struct order is the ranking: a substring hit beats a
                -- higher raw score, and an earlier name breaks a tie.
                list_max(
                    list_transform(
                        search_names,
                        (s, i) -> {{
                            'substring': contains(s.folded, {query}),
                            'similarity': jaro_winkler_similarity(s.folded, {query}),
                            'rank': -i,
                            'name': s.name
                        }}
                    )
                ) AS best
            FROM named
            WHERE len(search_names) > 0
        ),
        ranked AS (
            SELECT * FROM scored
            -- id last, so rows that tie on everything else (such as
            -- same-named towns, which have no admin_level) come back
            -- in the same order on every run.
            ORDER BY best.substring DESC, best.similarity DESC, admin_level ASC, id
            LIMIT $3
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
            is_territorial{extra_clause}{", bbox" if include_bbox else ""}{geometry_clause},
            best.name AS matched_name,
            best.similarity AS similarity,
            best.substring AS is_substring_match
        FROM ranked
        ORDER BY is_substring_match DESC, similarity DESC, admin_level ASC, id
        """,
        [place.place, path, limit],
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
    include_localities: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match a place against divisions_area (Overture admin boundaries).

    Matched on ``names.primary``, ``names.common`` and ``names.rules``
    together: 77% of divisions_area rows have no English common name and
    would drop out of every search if English were the only match field,
    while ``names.primary`` on its own leaves a place unreachable by the
    English name it is usually asked for.

    ``include_localities`` searches cities and towns as well. Only the fuzzy
    endpoint passes it; see :func:`_divisions_area_paths`.
    """
    path = _divisions_area_paths(include_localities)
    return simple_fuzzy_search(
        con,
        path,
        "divisions_area",
        place,
        _search_names(path, DIVISIONS_AREA_SEARCH_NAMES),
        english_name_expr=DIVISIONS_AREA_ENGLISH_NAME,
        extra_select="division_id",
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
        bbox_expr=DIVISIONS_AREA_BBOX,
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
        NATURAL_EARTH_SEARCH_NAMES,
        english_name_expr=NATURAL_EARTH_ENGLISH_NAME,
        limit=limit,
        include_geometry=include_geometry,
        include_bbox=include_bbox,
    )


def fetch_by_id(
    con: duckdb.DuckDBPyConnection,
    path: str | list[str],
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
    """Look up a single divisions_area record by exact ID, localities included.

    A locality's ID only ever comes from a fuzzy search, so looking it up
    here does not open localities to the natural-language pipeline.
    """
    return fetch_by_id(
        con,
        _divisions_area_paths(include_localities=True),
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
    include_localities: bool = False,
) -> list[pd.DataFrame]:
    """Return candidate DataFrames for a place from the requested sources.

    Defaults to always searching divisions_area and natural_earth to avoid
    missing natural features when the model assigns an incorrect admin
    subtype. Pass ``sources`` to restrict to a subset, and
    ``include_localities`` to search cities and towns too.
    """
    results = []
    for source in sources:
        search = _SOURCE_SEARCH_FNS[source]
        if source == "divisions_area" and include_localities:
            search = partial(search_divisions_area, include_localities=True)
        df = search(
            con,
            place,
            limit=limit,
            include_geometry=include_geometry,
            include_bbox=include_bbox,
        )
        if not df.empty:
            results.append(df)
    return results
