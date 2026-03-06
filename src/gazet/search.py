import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH
from .schemas import Place


def _fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str,
    source: str,
    place: Place,
    extra_select: str = "",
    limit: int = 5,
    is_overture: bool = False,
) -> pd.DataFrame:
    """Generic Levenshtein fuzzy search against any parquet with a names.primary column."""
    country_filter = ""
    country_params: list = []
    if is_overture and place.country:
        country_filter = "AND country = ?"
        country_params = [place.country]

    subtype_filter = ""
    subtype_params: list = []
    if is_overture and place.subtype:
        subtype_filter = "AND subtype = ?"
        subtype_params = [place.subtype]

    params = (
        [place.place, place.place, path] + country_params + subtype_params + [limit]
    )

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
            1.0 - (levenshtein(lower(names."primary"), lower(?))::float
                   / greatest(length(names."primary"), length(?), 1)) AS similarity
        FROM read_parquet(?)
        WHERE names."primary" IS NOT NULL AND trim(names."primary") != ''
        {country_filter}
        {subtype_filter}
        ORDER BY similarity DESC, admin_level ASC
        LIMIT ?
        """,
        params,
    )
    df = rel.fetchdf()
    df.insert(0, "source", source)
    label = f'"{place.place}"' + (f" [{place.country}]" if place.country else "")
    if df.empty:
        print(f"\n{source} – {label}: no matches")
    else:
        print(f"\n{source} – {label} (top {len(df)} by name similarity):")
        print(df.to_string(index=False))
    return df


def search_divisions_area(
    con: duckdb.DuckDBPyConnection, place: Place, limit: int = 5
) -> pd.DataFrame:
    """Fuzzy-match a place against divisions_area (Overture admin boundaries)."""
    return _fuzzy_search(
        con,
        DIVISIONS_AREA_PATH,
        "divisions_area",
        place,
        extra_select="division_id",
        limit=limit,
        is_overture=True,
    )


def search_natural_earth(
    con: duckdb.DuckDBPyConnection, place: Place, limit: int = 5
) -> pd.DataFrame:
    """Fuzzy-match a place against Natural Earth geography polygons."""
    return _fuzzy_search(
        con,
        NATURAL_EARTH_PATH,
        "natural_earth",
        place,
        limit=limit,
    )
