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


def compute_adjacency_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find all pairs of features that touch (share a boundary)."""
    print("Computing adjacency pairs (optimized with spatial index)...")
    
    # Use bounding box pre-filter to avoid full cartesian product
    query = """
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
        WHERE country IN (SELECT unnest(?))
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
    
    df = con.execute(query, [DIVISIONS_AREA_PATH, countries, limit]).fetchdf()
    print(f"Found {len(df)} adjacency pairs")
    
    return df


def compute_containment_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find all pairs where one feature contains another."""
    print("\nComputing containment pairs (optimized)...")
    
    query = """
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
        WHERE country IN (SELECT unnest(?))
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
    
    df = con.execute(query, [DIVISIONS_AREA_PATH, countries, limit]).fetchdf()
    print(f"Found {len(df)} containment pairs")
    
    return df


def compute_intersection_pairs(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find pairs that intersect but don't touch or contain."""
    print("\nComputing intersection pairs (optimized)...")
    
    query = """
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
        WHERE country IN (SELECT unnest(?))
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
    
    df = con.execute(query, [DIVISIONS_AREA_PATH, countries, limit]).fetchdf()
    print(f"Found {len(df)} same-source intersection pairs")
    
    return df


def compute_cross_source_relations(
    con: duckdb.DuckDBPyConnection,
    countries: list,
    limit: int
) -> pd.DataFrame:
    """Find relations between divisions_area and natural_earth."""
    print("\nComputing cross-source relations...")
    
    query = """
    WITH divisions AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            geometry
        FROM read_parquet(?)
        WHERE country IN (SELECT unnest(?))
    ),
    natural_features AS (
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            geometry
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
    
    df = con.execute(query, [DIVISIONS_AREA_PATH, countries, NATURAL_EARTH_PATH, limit]).fetchdf()
    print(f"Found {len(df)} cross-source relations")
    
    return df


def _make_connection():
    """Create a new DuckDB connection with spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
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
    
    print("\n✓ Relation tables build complete")


if __name__ == "__main__":
    main()
