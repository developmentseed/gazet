"""
Precompute spatial relation tables for efficient anchor sampling.

This script computes:
- Adjacency pairs (touching features)
- Containment pairs (features within other features)
- Intersection pairs (overlapping features)
- Cross-source relations (divisions_area ↔ natural_earth)

Output:
- intermediate/adjacency_pairs.parquet
- intermediate/containment_pairs.parquet
- intermediate/intersection_pairs.parquet
- intermediate/cross_source_relations.parquet
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH


# (container_subtype, contained_subtype) combos used by the chained,
# containment, and disambiguation templates. The normalized Overture input is
# now restricted to country / region / county, so keep relation building in
# sync and avoid wasting work on removed subtypes.
_CONTAINMENT_SUBTYPE_PAIRS = (
    ("country", "region"),
    ("country", "county"),
    ("region", "county"),
)


# Natural Earth subtype vocabulary normalized to lowercase.
# We lowercase the source subtype values while building relation tables so
# mixed casing in upstream data (e.g. Lake vs lake, Range/mtn vs range/mtn)
# does not fragment anchor pools or break template matching.
_NE_CROSS_SOURCE_SUBTYPES = (
    "sea",
    "ocean",
    "lake",
    "river",
    "basin",
    "gulf",
    "bay",
    "strait",
    "island group",
    "peninsula",
    "range/mtn",
    "plateau",
    "plain",
    "lowland",
    "valley",
    "depression",
    "gorge",
)

_DIVISION_SUBTYPES = ("country", "region", "county")


def _country_filter(countries: list) -> tuple[str, list]:
    """Return (SQL WHERE clause, params) handling 'all' sentinel."""
    if countries == ["all"]:
        return "", []
    return "WHERE country IN (SELECT unnest(?))", [countries]


def _country_filter_for_join(countries: list) -> tuple[str, list]:
    """Return a country filter for self-joins over normalized admin data."""
    if countries == ["all"]:
        return "", []
    return "WHERE country IN (SELECT unnest(?))", [countries]


def _country_chunks(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    chunk_size: int = 40,
) -> list[list[str]]:
    """Return explicit country batches for safer global containment joins."""
    if countries != ["all"]:
        return [countries]

    rows = con.execute(
        """
        SELECT DISTINCT country
        FROM read_parquet(?)
        WHERE country IS NOT NULL
          AND trim(country) != ''
        ORDER BY country
        """,
        [DIVISIONS_AREA_PATH],
    ).fetchall()
    codes = [row[0] for row in rows]
    return [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]


def compute_adjacency_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find all pairs of features that touch (share a boundary)."""
    print("Computing adjacency pairs (optimized with spatial index)...")
    
    cfilter, cparams = _country_filter_for_join(countries)
    
    # Use bounding box pre-filter to avoid full cartesian product
    query = f"""
    WITH features AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            admin_level,
            geometry,
            ST_Envelope(geometry) AS bbox
        FROM read_parquet(?)
        {cfilter}
    )
    SELECT 
        a.id AS anchor_id,
        a.name AS anchor_name,
        a.subtype AS anchor_subtype,
        a.country AS anchor_country,
        b.id AS target_id,
        b.name AS target_name,
        b.subtype AS target_subtype,
        b.country AS target_country,
        'adjacency' AS relation_type
    FROM features AS a
    JOIN features AS b ON (
        a.id < b.id
        AND ST_Intersects(a.bbox, b.bbox)
        AND ST_Touches(a.geometry, b.geometry)
    )
    LIMIT ?
    """
    
    df = con.execute(query, [DIVISIONS_AREA_PATH] + cparams + [limit]).fetchdf()
    print(f"Found {len(df)} adjacency pairs")
    
    return df


def _stratified_containment(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int,
    relation_type: str,
    extra_where: str = "",
    extra_params: list = None,
) -> pd.DataFrame:
    """Compute containment pairs stratified by (container_subtype, contained_subtype).

    A single global self-join with LIMIT fills up with coarse country->region
    pairs before emitting country->county and region->county pairs. We run one
    focused query per subtype combo instead so every combo receives a fair
    share of the overall limit.

    ``extra_where`` / ``extra_params`` let the coastal and landlocked variants
    inject their country-set filter without duplicating the whole body.
    """
    extra_params = extra_params or []
    # Use a lower target per subtype combo for global runs; they are the most
    # memory-intensive part of the pipeline and don't need huge anchor tables.
    if countries == ["all"]:
        per_combo = min(max(limit // len(_CONTAINMENT_SUBTYPE_PAIRS), 100), 1500)
    else:
        per_combo = max(limit // len(_CONTAINMENT_SUBTYPE_PAIRS), 100)

    country_batches = _country_chunks(con, countries)
    frames: list[pd.DataFrame] = []
    for container_st, contained_st in _CONTAINMENT_SUBTYPE_PAIRS:
        remaining = per_combo
        combo_parts: list[pd.DataFrame] = []

        for batch in country_batches:
            if remaining <= 0:
                break

            cfilter, cparams = _country_filter(batch)
            query = f"""
            WITH a AS (
                SELECT src.id, src.names."primary" AS name, src.subtype, src.country, src.admin_level,
                       src.geometry, ST_Envelope(src.geometry) AS bbox
                FROM read_parquet(?) AS src
                WHERE src.subtype = '{container_st}'
                  {cfilter.replace("WHERE", "AND") if cfilter else ""}
                  {extra_where}
            ),
            b AS (
                SELECT dst.id, dst.names."primary" AS name, dst.subtype, dst.country, dst.admin_level,
                       dst.geometry, ST_Envelope(dst.geometry) AS bbox
                FROM read_parquet(?) AS dst
                WHERE dst.subtype = '{contained_st}'
                  {cfilter.replace("WHERE", "AND") if cfilter else ""}
            )
            SELECT
                a.id AS container_id,
                a.name AS container_name,
                a.subtype AS container_subtype,
                b.id AS contained_id,
                b.name AS contained_name,
                b.subtype AS contained_subtype,
                a.country AS container_country,
                '{relation_type}' AS relation_type
            FROM a JOIN b ON (
                a.id != b.id
                AND ST_Intersects(a.bbox, b.bbox)
                AND ST_Within(b.geometry, a.geometry)
            )
            LIMIT {remaining}
            """
            params = [DIVISIONS_AREA_PATH] + extra_params + cparams + [DIVISIONS_AREA_PATH] + cparams
            df_part = con.execute(query, params).fetchdf()
            if not df_part.empty:
                combo_parts.append(df_part)
                remaining -= len(df_part)

        df_combo = (
            pd.concat(combo_parts, ignore_index=True)
            if combo_parts else pd.DataFrame()
        )
        print(f"  {relation_type} {container_st:>10s} -> {contained_st:<10s}: {len(df_combo)} pairs")
        frames.append(df_combo)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_containment_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find containment pairs stratified across admin-level combinations."""
    print("\nComputing containment pairs (stratified by subtype combo)...")
    df = _stratified_containment(con, countries, limit, relation_type="containment")
    print(f"Found {len(df)} containment pairs")
    return df


def compute_intersection_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find pairs that intersect but don't touch or contain."""
    print("\nComputing intersection pairs (optimized)...")
    
    cfilter, cparams = _country_filter_for_join(countries)
    
    query = f"""
    WITH features AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            admin_level,
            geometry,
            ST_Envelope(geometry) AS bbox
        FROM read_parquet(?)
        {cfilter}
    )
    SELECT 
        a.id AS anchor_id,
        a.name AS anchor_name,
        a.subtype AS anchor_subtype,
        b.id AS target_id,
        b.name AS target_name,
        b.subtype AS target_subtype,
        'intersection' AS relation_type
    FROM features AS a
    JOIN features AS b ON (
        a.id < b.id
        AND ST_Intersects(a.bbox, b.bbox)
        AND ST_Intersects(a.geometry, b.geometry)
        AND NOT ST_Touches(a.geometry, b.geometry)
        AND NOT ST_Within(a.geometry, b.geometry)
        AND NOT ST_Within(b.geometry, a.geometry)
    )
    LIMIT ?
    """
    
    df = con.execute(query, [DIVISIONS_AREA_PATH] + cparams + [limit]).fetchdf()
    print(f"Found {len(df)} same-source intersection pairs")
    
    return df


def compute_cross_source_relations(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find relations between divisions_area and natural_earth.

    The join is skewed both by abundant Natural Earth subtypes and by coarse
    admin features. We therefore stratify by (natural_subtype,
    division_subtype) so country / region / county anchors all make it into
    the pool used by mixed-source, NE-intersection, and NE-adjacency templates.
    """
    print("\nComputing cross-source relations (stratified by NE subtype and admin subtype)...")

    cfilter, cparams = _country_filter(countries)
    num_combos = len(_NE_CROSS_SOURCE_SUBTYPES) * len(_DIVISION_SUBTYPES)
    per_combo = max(limit // num_combos, 10)

    frames: list[pd.DataFrame] = []
    for natural_subtype in _NE_CROSS_SOURCE_SUBTYPES:
        for division_subtype in _DIVISION_SUBTYPES:
            query = f"""
            WITH divisions AS (
                SELECT
                    id,
                    names."primary" AS name,
                    subtype,
                    country,
                    geometry
                FROM read_parquet(?)
                WHERE geometry IS NOT NULL
                  AND names."primary" IS NOT NULL
                  AND trim(names."primary") != ''
                  AND subtype = '{division_subtype}'
                  {cfilter.replace("WHERE", "AND") if cfilter else ''}
            ),
            natural_features AS (
                SELECT
                    id,
                    names."primary" AS name,
                    lower(subtype) AS natural_subtype,
                    geometry
                FROM read_parquet(?)
                WHERE geometry IS NOT NULL
                  AND names."primary" IS NOT NULL
                  AND trim(names."primary") != ''
                  AND lower(subtype) = '{natural_subtype}'
            )
            SELECT
                d.id AS division_id,
                d.name AS division_name,
                d.subtype AS division_subtype,
                d.country AS division_country,
                n.id AS natural_id,
                n.name AS natural_name,
                n.natural_subtype AS natural_subtype,
                CASE
                    WHEN ST_Touches(d.geometry, n.geometry) THEN 'touches'
                    WHEN ST_Within(d.geometry, n.geometry) THEN 'within'
                    WHEN ST_Contains(d.geometry, n.geometry) THEN 'contains'
                    WHEN ST_Intersects(d.geometry, n.geometry) THEN 'intersects'
                END AS relation_type
            FROM divisions AS d
            JOIN natural_features AS n
              ON ST_Intersects(d.geometry, n.geometry)
            LIMIT {per_combo}
            """
            df_part = con.execute(
                query,
                [DIVISIONS_AREA_PATH] + cparams + [NATURAL_EARTH_PATH],
            ).fetchdf()
            print(
                f"  cross_source {natural_subtype:>12s} x {division_subtype:<7s}: "
                f"{len(df_part)} rows"
            )
            frames.append(df_part)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"Found {len(df)} cross-source relations")
    return df


def compute_coastal_containment_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int,
) -> pd.DataFrame:
    """Stratified containment pairs limited to coastal-country containers.

    Used by chained templates so sampled anchors actually have sea-adjacent
    sub-features. Stratification guarantees coverage of every supported
    admin-level combination (country->region, country->county, region->county).
    """
    print("\nComputing coastal containment pairs (stratified)...")
    extra_where = f"""
              AND EXISTS (
                  SELECT 1
                  FROM read_parquet('{NATURAL_EARTH_PATH}') AS n
                  WHERE n.geometry IS NOT NULL
                    AND n.names."primary" IS NOT NULL
                    AND trim(n.names."primary") != ''
                    AND n.subtype IN ('sea', 'ocean')
                    AND ST_Intersects(src.geometry, n.geometry)
              )
    """
    df = _stratified_containment(
        con, countries, limit,
        relation_type="coastal_containment",
        extra_where=extra_where,
    )
    print(f"Found {len(df)} coastal containment pairs")
    return df


def compute_landlocked_containment_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int,
) -> pd.DataFrame:
    """Stratified containment pairs limited to landlocked-country containers.

    Used by chained templates that need inland anchors. Stratification by
    subtype combo ensures county-level pairs are actually present in the
    output instead of being starved by coarse country->region pairs.
    """
    print("\nComputing landlocked containment pairs (stratified)...")
    extra_where = f"""
              AND NOT EXISTS (
                  SELECT 1
                  FROM read_parquet('{NATURAL_EARTH_PATH}') AS n
                  WHERE n.geometry IS NOT NULL
                    AND n.names."primary" IS NOT NULL
                    AND trim(n.names."primary") != ''
                    AND n.subtype IN ('sea', 'ocean')
                    AND ST_Intersects(src.geometry, n.geometry)
              )
    """
    df = _stratified_containment(
        con, countries, limit,
        relation_type="landlocked_containment",
        extra_where=extra_where,
    )
    print(f"Found {len(df)} landlocked containment pairs")
    return df


def compute_common_neighbor_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int,
) -> pd.DataFrame:
    """Pairs of anchors that share at least one common touching neighbour.

    Used by multi_adj_01 (borders both X and Y) so that the generated SQL
    is guaranteed to return at least one result rather than failing constantly
    on random pairs that have no common neighbour.

    Derived by self-joining adjacency_pairs on the shared target_id.
    """
    print("\nComputing common-neighbor pairs...")

    adj_path = Path(__file__).parent.parent / "intermediate" / "adjacency_pairs.parquet"
    if not adj_path.exists():
        print("  adjacency_pairs.parquet not found — skipping (run adjacency first)")
        return pd.DataFrame(columns=[
            "anchor_id_1", "anchor_name_1", "anchor_subtype_1",
            "anchor_id_2", "anchor_name_2", "anchor_subtype_2",
            "shared_neighbor_id", "shared_neighbor_name", "shared_neighbor_subtype",
        ])

    query = """
    WITH undirected AS (
        SELECT
            anchor_id,
            anchor_name,
            anchor_subtype,
            target_id,
            target_name,
            target_subtype
        FROM read_parquet(?)
        UNION ALL
        SELECT
            target_id      AS anchor_id,
            target_name    AS anchor_name,
            target_subtype AS anchor_subtype,
            anchor_id      AS target_id,
            anchor_name    AS target_name,
            anchor_subtype AS target_subtype
        FROM read_parquet(?)
    )
    SELECT DISTINCT
        a1.anchor_id      AS anchor_id_1,
        a1.anchor_name    AS anchor_name_1,
        a1.anchor_subtype AS anchor_subtype_1,
        a2.anchor_id      AS anchor_id_2,
        a2.anchor_name    AS anchor_name_2,
        a2.anchor_subtype AS anchor_subtype_2,
        a1.target_id      AS shared_neighbor_id,
        a1.target_name    AS shared_neighbor_name,
        a1.target_subtype AS shared_neighbor_subtype
    FROM undirected AS a1
    JOIN undirected AS a2
      ON a1.target_id = a2.target_id
     AND a1.anchor_id < a2.anchor_id
    LIMIT ?
    """

    df = con.execute(query, [str(adj_path), str(adj_path), limit]).fetchdf()
    print(f"Found {len(df)} common-neighbor pairs")
    return df


def _make_connection():
    """Create a new DuckDB connection with spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    memory_limit = os.environ.get("GAZET_DUCKDB_MEMORY_LIMIT", "12GB")
    threads = int(os.environ.get("GAZET_DUCKDB_THREADS", "1"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    con.execute(f"SET threads={threads}")
    return con


def _compute_and_save(compute_fn, countries, limit, output_path):
    """Compute a relation table and save it to parquet. Uses its own DuckDB connection."""
    con = _make_connection()
    try:
        df = compute_fn(con, countries, limit)
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")
        return df
    finally:
        con.close()


RELATION_FUNCTIONS = {
    "adjacency":              compute_adjacency_pairs,
    "containment":            compute_containment_pairs,
    "intersection":           compute_intersection_pairs,
    "cross_source":           compute_cross_source_relations,
    "coastal_containment":    compute_coastal_containment_pairs,
    "landlocked_containment": compute_landlocked_containment_pairs,
    "common_neighbor":        compute_common_neighbor_pairs,
}

# Single source of truth for the on-disk filename for each relation.
# Both local and Modal paths must use this so the sample generator loads
# the same file regardless of where the pipeline ran.
RELATION_FILENAMES = {
    "adjacency":              "adjacency_pairs.parquet",
    "containment":            "containment_pairs.parquet",
    "intersection":           "intersection_pairs.parquet",
    "cross_source":           "cross_source_relations.parquet",
    "coastal_containment":    "coastal_containment_pairs.parquet",
    "landlocked_containment": "landlocked_containment_pairs.parquet",
    "common_neighbor":        "common_neighbor_pairs.parquet",
}


def compute_single_relation(
    relation_type: str,
    countries: list,
    limit: int,
    output_dir: Path,
) -> int:
    """Compute one relation type and save to output_dir.

    Returns the number of rows computed. Usable from Modal or locally.
    """
    compute_fn = RELATION_FUNCTIONS.get(relation_type)
    if compute_fn is None:
        raise ValueError(
            f"Unknown relation type: {relation_type}. "
            f"Expected one of {list(RELATION_FUNCTIONS)}"
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / RELATION_FILENAMES[relation_type]
    df = _compute_and_save(compute_fn, countries, limit, output_path)
    return len(df)


def main(countries: list = None, relation_limits: dict = None):
    """Compute and save all relation tables in parallel.
    
    Args:
        countries: List of country codes to process
        relation_limits: Dict with keys: adjacency, containment, intersection, cross_source
    """
    # Defaults
    if countries is None:
        countries = ['EC', 'BE', 'KE', 'AE', 'SG', 'CH']
    if relation_limits is None:
        relation_limits = {
            'adjacency':              50000,
            'containment':            3000,
            'intersection':           3000,
            'cross_source':           1800,
            'coastal_containment':    3000,
            'landlocked_containment': 1500,
            'common_neighbor':        5000,
        }

    output_dir = Path(__file__).parent.parent / "intermediate"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define all relation tasks. Filenames come from RELATION_FILENAMES so
    # local and Modal pipelines produce identically-named parquet files.
    # common_neighbor depends on adjacency_pairs so it runs after adjacency.
    tasks = [
        (rel_type, RELATION_FUNCTIONS[rel_type], relation_limits[rel_type], output_dir / RELATION_FILENAMES[rel_type])
        for rel_type in (
            "adjacency", "containment", "intersection", "cross_source",
            "coastal_containment", "landlocked_containment", "common_neighbor",
        )
    ]
    
    # common_neighbor reads adjacency_pairs.parquet so it must run after
    # adjacency finishes.  Split into two waves.
    independent_tasks = [t for t in tasks if t[0] != "common_neighbor"]
    dependent_tasks   = [t for t in tasks if t[0] == "common_neighbor"]

    print(f"Computing {len(independent_tasks)} relation types in parallel...")
    with ThreadPoolExecutor(max_workers=len(independent_tasks)) as executor:
        futures = {
            executor.submit(_compute_and_save, compute_fn, countries, limit, path): name
            for name, compute_fn, limit, path in independent_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"ERROR computing {name}: {e}")
                raise

    for name, compute_fn, limit, path in dependent_tasks:
        print(f"\nComputing {name} (depends on adjacency)...")
        try:
            _compute_and_save(compute_fn, countries, limit, path)
        except Exception as e:
            print(f"ERROR computing {name}: {e}")
            raise

    print("\nRelation tables build complete")


if __name__ == "__main__":
    main()
