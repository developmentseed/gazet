import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH
from .schemas import Place


def simple_fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    place: Place,
    extra_select: str = "",
    limit: int = 5,
) -> pd.DataFrame:
    """Jaro-Winkler similarity search using only the place name."""
    params = [place.place, path, limit]

    extra_clause = f", {extra_select}" if extra_select else ""
    rel = con.execute(
        f"""
        SELECT
            id,
            names."primary" AS name,
            country,
            subtype,
            class,
            region,
            admin_level,
            is_land,
            is_territorial{extra_clause},
            jaro_winkler_similarity(lower(names."primary"), lower(?)) AS similarity
        FROM read_parquet(?)
        WHERE names."primary" IS NOT NULL AND trim(names."primary") != ''
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
        print(df.to_string(index=False))
    return df


def search_divisions_area(
    con: duckdb.DuckDBPyConnection, place: Place, limit: int = 5
) -> pd.DataFrame:
    """Fuzzy-match a place against divisions_area (Overture admin boundaries)."""
    return simple_fuzzy_search(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        place,
        extra_select="division_id",
        limit=limit,
    )


def search_natural_earth(
    con: duckdb.DuckDBPyConnection, place: Place, limit: int = 5
) -> pd.DataFrame:
    """Fuzzy-match a place against Natural Earth geography polygons."""
    return simple_fuzzy_search(
        con,
        NATURAL_EARTH_PATH,
        "natural_earth",
        place,
        limit=limit,
    )


def search_candidates(
    con: duckdb.DuckDBPyConnection, place: Place, limit: int = 5
) -> list[pd.DataFrame]:
    """Return candidate DataFrames for a place from both sources.

    Always searches divisions_area and natural_earth to avoid missing
    natural features when the model assigns an incorrect admin subtype.
    """
    results = []
    for fn in (search_divisions_area, search_natural_earth):
        df = fn(con, place, limit=limit)
        if not df.empty:
            results.append(df)
    return results
