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
    urban_centres_normalized/part-000.parquet

Cities and towns (Overture ``locality`` and ``localadmin``) go to their own
file. Fuzzy search reads it; the training pipeline and the natural-language
search read only ``divisions_area``, because the model was not trained on
those subtypes.

Both Overture files also store ``search_names``: every name a record
answers to, folded for comparison, so fuzzy search does not rebuild and
fold them on every query. They also store ``population``, joined from
Overture's ``division`` records, so fuzzy search can rank a city above a
same-named village.

Urban centres come from the GHSL Urban Centre Database (GHS-UCDB R2024A):
named city polygons for places Overture has only as a point, such as
London. They are a source of their own, never merged into an Overture
record, so every result traces back to one dataset.
"""

from pathlib import Path

import duckdb

from gazet.config import _DATA_DIR
from gazet.search import (
    DIVISIONS_AREA_SEARCH_NAMES,
    SEARCH_NAMES_COLUMN,
    search_names_sql,
)

#: The subtypes the natural-language model was trained on.
TRAINED_SUBTYPES = ("country", "region", "county")
#: Cities and towns, for fuzzy search only.
LOCALITY_SUBTYPES = ("localadmin", "locality")
#: The GHS-UCDB layer that carries each urban centre's names, country and
#: population. The other layers repeat the same polygons with other attributes.
GHSL_UCDB_LAYER = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"


def _ghsl(column: str) -> str:
    """A GHS-UCDB column, whose name and text values start with a byte-order mark."""
    return f'ltrim("\ufeff{column}", chr(65279))'


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
    urban_centres_dir = root / "urban_centres_normalized"
    for directory in (
        overture_dir,
        localities_dir,
        natural_earth_dir,
        urban_centres_dir,
    ):
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
                SELECT
                    area.* REPLACE (ST_GeomFromWKB(ST_AsWKB(area.geometry)) AS geometry),
                    division.population,
                    {DIVISIONS_AREA_SEARCH_NAMES} AS {SEARCH_NAMES_COLUMN}
                FROM read_parquet('{root / "overture/divisions_area/*.parquet"}') AS area
                LEFT JOIN (
                    SELECT id, population
                    FROM read_parquet('{root / "overture/division/*.parquet"}')
                ) AS division ON division.id = area.division_id
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

    # Shaped like the Overture files, so search and lookup read every source
    # the same way. Names come as one "; "-separated list of alternates.
    alternates = f"string_split({_ghsl('GC_UCN_LIS_2025')}, '; ')"
    con.execute(
        f"""
        COPY (
            SELECT
                * EXCLUDE (alternate_names),
                {search_names_sql("names.primary", "CAST(NULL AS VARCHAR)", "alternate_names")}
                    AS {SEARCH_NAMES_COLUMN}
            FROM (
                SELECT
                    'ghsl_' || CAST("\ufeffID_UC_G0" AS VARCHAR) AS id,
                    ST_GeomFromWKB(ST_AsWKB(geometry)) AS geometry,
                    {{
                        'xmin': ST_XMin(geometry), 'xmax': ST_XMax(geometry),
                        'ymin': ST_YMin(geometry), 'ymax': ST_YMax(geometry)
                    }} AS bbox,
                    {_ghsl("GC_CNT_GAD_2025")} AS country,
                    'urban_centre' AS subtype,
                    CAST(NULL AS VARCHAR) AS class,
                    {{'primary': {_ghsl("GC_UCN_MAI_2025")}}} AS names,
                    CAST(NULL AS VARCHAR) AS region,
                    CAST(NULL AS INTEGER) AS admin_level,
                    true AS is_land,
                    CAST(NULL AS BOOLEAN) AS is_territorial,
                    CAST(round("\ufeffGC_POP_TOT_2025") AS BIGINT) AS population,
                    {alternates} AS alternate_names
                FROM (
                    SELECT
                        *,
                        -- Mollweide to lon/lat.
                        ST_Transform(geom, 'ESRI:54009', 'EPSG:4326', always_xy := true)
                            AS geometry
                    FROM st_read(
                        '{root / "ghsl/GHS_UCDB_GLOBE_R2024A.gpkg"}',
                        layer = '{GHSL_UCDB_LAYER}'
                    )
                )
            )
        ) TO '{urban_centres_dir / "part-000.parquet"}' (FORMAT PARQUET)
        """
    )
    con.close()

    return {
        "divisions_area": str(overture_dir / "*.parquet"),
        "localities": str(localities_dir / "*.parquet"),
        "natural_earth": str(natural_earth_path),
        "urban_centres": str(urban_centres_dir / "*.parquet"),
    }


def main() -> None:
    result = normalize_geodata()
    print("Normalized datasets written:")
    for name, path in result.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
