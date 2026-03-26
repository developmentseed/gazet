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

import duckdb
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH


# Subtypes too granular for spatial self-joins at global scale
_EXCLUDED_SUBTYPES_FOR_GLOBAL = ("locality", "neighborhood", "microhood", "macrohood")


def _country_filter(countries: list) -> tuple[str, list]:
    """Return (SQL WHERE clause, params) handling 'all' sentinel."""
    if countries == ["all"]:
        return "", []
    return "WHERE country IN (SELECT unnest(?))", [countries]


def _country_filter_for_join(countries: list) -> tuple[str, list]:
    """Like _country_filter but also excludes fine-grained subtypes for global runs.

    When joining all 1M+ entities, localities/neighborhoods/microhoods cause
    OOM. Excluding them keeps ~110K higher-level admin entities.
    """
    excluded = "', '".join(_EXCLUDED_SUBTYPES_FOR_GLOBAL)
    subtype_clause = f"AND subtype NOT IN ('{excluded}')"
    if countries == ["all"]:
        return f"WHERE 1=1 {subtype_clause}", []
    return f"WHERE country IN (SELECT unnest(?)) {subtype_clause}", [countries]


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


def compute_containment_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find all pairs where one feature contains another."""
    print("\nComputing containment pairs (optimized)...")
    
    cfilter, cparams = _country_filter(countries)
    
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
        a.id AS container_id,
        a.name AS container_name,
        a.subtype AS container_subtype,
        b.id AS contained_id,
        b.name AS contained_name,
        b.subtype AS contained_subtype,
        'containment' AS relation_type
    FROM features AS a
    JOIN features AS b ON (
        a.id != b.id
        AND a.admin_level < b.admin_level
        AND ST_Intersects(a.bbox, b.bbox)
        AND ST_Within(b.geometry, a.geometry)
    )
    LIMIT ?
    """
    
    df = con.execute(query, [DIVISIONS_AREA_PATH] + cparams + [limit]).fetchdf()
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
    """Find relations between divisions_area and natural_earth."""
    print("\nComputing cross-source relations...")
    
    cfilter, cparams = _country_filter(countries)
    
    query = f"""
    WITH divisions AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            geometry
        FROM read_parquet(?)
        {cfilter}
    ),
    natural_features AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            ST_SetCRS(geometry, 'OGC:CRS84') AS geometry
        FROM read_parquet(?)
        WHERE subtype IN ('sea', 'ocean', 'Lake', 'River', 'Basin', 'gulf', 'bay')
        LIMIT 200
    )
    SELECT 
        d.id AS division_id,
        d.name AS division_name,
        d.subtype AS division_subtype,
        d.country AS division_country,
        n.id AS natural_id,
        n.name AS natural_name,
        n.subtype AS natural_subtype,
        CASE 
            WHEN ST_Touches(d.geometry, n.geometry) THEN 'touches'
            WHEN ST_Within(d.geometry, n.geometry) THEN 'within'
            WHEN ST_Contains(d.geometry, n.geometry) THEN 'contains'
            WHEN ST_Intersects(d.geometry, n.geometry) THEN 'intersects'
        END AS relation_type
    FROM divisions AS d
    JOIN natural_features AS n ON ST_Intersects(d.geometry, n.geometry)
    LIMIT ?
    """
    
    df = con.execute(query, [DIVISIONS_AREA_PATH] + cparams + [NATURAL_EARTH_PATH, limit]).fetchdf()
    print(f"Found {len(df)} cross-source relations")
    
    return df


def _make_connection():
    """Create a new DuckDB connection with spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("SET memory_limit='24GB'")
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    con.execute("SET threads=4")
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
    "adjacency": compute_adjacency_pairs,
    "containment": compute_containment_pairs,
    "intersection": compute_intersection_pairs,
    "cross_source": compute_cross_source_relations,
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
    output_path = output_dir / f"{relation_type}_pairs.parquet"
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
            'adjacency': 50000,
            'containment': 1000,
            'intersection': 500,
            'cross_source': 500
        }
    
    output_dir = Path(__file__).parent.parent / "intermediate"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define all relation tasks
    tasks = [
        ("adjacency", compute_adjacency_pairs, relation_limits['adjacency'], output_dir / "adjacency_pairs.parquet"),
        ("containment", compute_containment_pairs, relation_limits['containment'], output_dir / "containment_pairs.parquet"),
        ("intersection", compute_intersection_pairs, relation_limits['intersection'], output_dir / "intersection_pairs.parquet"),
        ("cross_source", compute_cross_source_relations, relation_limits['cross_source'], output_dir / "cross_source_relations.parquet"),
    ]
    
    print(f"Computing {len(tasks)} relation types in parallel...")
    
    # Run all relation computations concurrently
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(_compute_and_save, compute_fn, countries, limit, path): name
            for name, compute_fn, limit, path in tasks
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"ERROR computing {name}: {e}")
                raise
    
    print("\nRelation tables build complete")


if __name__ == "__main__":
    main()
