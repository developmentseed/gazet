"""
Build entity inventory from divisions_area and natural_earth parquet files.

This script creates compact inventory tables containing only the fields needed
for candidate sampling and distractor generation.

Output:
- intermediate/divisions_area_inventory.parquet
- intermediate/natural_earth_inventory.parquet
"""

import duckdb
import pandas as pd
from pathlib import Path

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH


def build_divisions_area_inventory(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract compact inventory from divisions_area."""
    query = """
    SELECT 
        'divisions_area' AS source,
        id,
        names."primary" AS name,
        subtype,
        country,
        region,
        admin_level,
        class,
        is_land,
        is_territorial,
        division_id,
        ST_Area(geometry) AS area_sq_deg,
        ST_XMin(geometry) AS xmin,
        ST_YMin(geometry) AS ymin,
        ST_XMax(geometry) AS xmax,
        ST_YMax(geometry) AS ymax
    FROM read_parquet(?)
    WHERE names."primary" IS NOT NULL
      AND trim(names."primary") != ''
      AND geometry IS NOT NULL
    """

    df = con.execute(query, [DIVISIONS_AREA_PATH]).fetchdf()
    print(f"Divisions area inventory: {len(df)} entities")
    print(f"Subtypes: {df['subtype'].value_counts().to_dict()}")
    print(f"Countries: {df['country'].nunique()} unique")
    
    return df


def build_natural_earth_inventory(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract compact inventory from natural_earth."""
    query = """
    SELECT 
        'natural_earth' AS source,
        id,
        names."primary" AS name,
        subtype,
        country,
        region,
        admin_level,
        class,
        is_land,
        is_territorial,
        ST_Area(geometry) AS area_sq_deg,
        ST_XMin(geometry) AS xmin,
        ST_YMin(geometry) AS ymin,
        ST_XMax(geometry) AS xmax,
        ST_YMax(geometry) AS ymax
    FROM read_parquet(?)
    WHERE names."primary" IS NOT NULL
      AND trim(names."primary") != ''
      AND geometry IS NOT NULL
    """

    df = con.execute(query, [NATURAL_EARTH_PATH]).fetchdf()
    print(f"\nNatural earth inventory: {len(df)} entities")
    print(f"Subtypes: {df['subtype'].value_counts().to_dict()}")
    
    return df


def build_inventory_to_dir(output_dir: Path) -> dict:
    """Build and save all inventory tables to output_dir.

    Reusable entry point for both local CLI and Modal.

    Returns:
        Dict with counts: {"divisions_area": int, "natural_earth": int}
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")

    print("Building divisions_area inventory...")
    divisions_df = build_divisions_area_inventory(con)
    divisions_path = output_dir / "divisions_area_inventory.parquet"
    divisions_df.to_parquet(divisions_path, index=False)
    print(f"Saved to {divisions_path}")

    print("\nBuilding natural_earth inventory...")
    natural_earth_df = build_natural_earth_inventory(con)
    natural_earth_path = output_dir / "natural_earth_inventory.parquet"
    natural_earth_df.to_parquet(natural_earth_path, index=False)
    print(f"Saved to {natural_earth_path}")

    con.close()

    total = len(divisions_df) + len(natural_earth_df)
    print(f"\nInventory build complete")
    print(f"  Total entities: {total}")
    return {"divisions_area": len(divisions_df), "natural_earth": len(natural_earth_df)}


def main():
    """Build and save inventory tables."""
    output_dir = Path(__file__).parent.parent / "intermediate"
    build_inventory_to_dir(output_dir)


if __name__ == "__main__":
    main()
