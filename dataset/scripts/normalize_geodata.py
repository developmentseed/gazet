"""Normalize source GeoParquet files to a shared CRS-neutral geometry encoding.

The training pipeline mixes Overture divisions_area and Natural Earth geometry.
Across environments these sources can advertise different CRS metadata labels
(`EPSG:4326` vs `OGC:CRS84`), which causes DuckDB spatial joins to fail even
when coordinates are already compatible lon/lat values.

This script rewrites both datasets into normalized copies whose geometry column
is rebuilt from WKB. That preserves coordinates while dropping conflicting CRS
metadata, so downstream joins behave consistently locally and on Modal.

Output layout under data/ by default:
    overture_normalized/divisions_area/part-000.parquet
    overture_normalized/localities/part-000.parquet
    natural_earth_normalized/ne_geography.parquet

Cities and towns (Overture ``locality`` and ``localadmin``) go to their own
file. Fuzzy search reads it; the training pipeline and the natural-language
search read only ``divisions_area``, because the model was not trained on
those subtypes.
"""

from pathlib import Path

import duckdb

from gazet.config import _DATA_DIR

#: The subtypes the natural-language model was trained on.
TRAINED_SUBTYPES = ("country", "region", "county")
#: Cities and towns, for fuzzy search only.
LOCALITY_SUBTYPES = ("localadmin", "locality")


def normalize_geodata(output_root: Path | None = None) -> dict[str, str]:
    """Write normalized copies of both source datasets.

    Args:
        output_root: Base directory to write normalized datasets into.
            Defaults to the project data dir.

    Returns:
        Mapping of dataset name to written path/glob.
    """
    root = output_root or _DATA_DIR
    overture_dir = root / "overture_normalized" / "divisions_area"
    localities_dir = root / "overture_normalized" / "localities"
    natural_earth_dir = root / "natural_earth_normalized"
    for directory in (overture_dir, localities_dir, natural_earth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    natural_earth_path = natural_earth_dir / "ne_geography.parquet"

    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")

    # Rebuild geometry from WKB so conflicting CRS metadata is dropped.
    for subtypes, directory in (
        (TRAINED_SUBTYPES, overture_dir),
        (LOCALITY_SUBTYPES, localities_dir),
    ):
        con.execute(
            f"""
            COPY (
                SELECT * REPLACE (
                    ST_GeomFromWKB(ST_AsWKB(geometry)) AS geometry
                )
                FROM read_parquet('{root / "overture/divisions_area/*.parquet"}')
                WHERE geometry IS NOT NULL
                  AND subtype IN {subtypes}
                  AND is_land = true
            ) TO '{directory / "part-000.parquet"}' (FORMAT PARQUET)
            """
        )

    con.execute(
        f"""
        COPY (
            SELECT * REPLACE (
                ST_GeomFromWKB(ST_AsWKB(geometry)) AS geometry
            )
            FROM read_parquet('{root / "natural_earth_geoparquet/ne_geography.parquet"}')
            WHERE geometry IS NOT NULL
        ) TO '{natural_earth_path}' (FORMAT PARQUET)
        """
    )
    con.close()

    return {
        "divisions_area": str(overture_dir / "*.parquet"),
        "localities": str(localities_dir / "*.parquet"),
        "natural_earth": str(natural_earth_path),
    }


def main() -> None:
    result = normalize_geodata()
    print("Normalized datasets written:")
    for name, path in result.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
