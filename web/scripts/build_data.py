"""Build the static data the browser demo reads.

Writes, under web/public/data/ by default:

    index.parquet          every place the model can name: divisions (Overture division_area, the subtypes gazet was
                           trained on) and Natural Earth features. Names for fuzzy search, attributes for plan filters,
                           bbox, area, and where the geometry is (file and row). No geometry. Loaded whole by the
                           browser (about 10 MB).
    natural_earth.parquet  Natural Earth geometry, Hilbert-sorted, with a page index so the browser reads only the
                           pages it needs.
    manifest.json          the Overture release and the division_area file URLs the browser reads geometry from.

Division geometry is not copied: the browser reads it from Overture's public S3 bucket, just the pages that hold
the rows it needs.
The index must come from the same release as those files, so this script reads the release straight from S3.

    uv run python web/scripts/build_data.py --release 2026-10-21.0 [--local DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb

from gazet.search import DIVISIONS_AREA_SEARCH_NAMES, NATURAL_EARTH_SEARCH_NAMES

BUCKET = "overturemaps-us-west-2"
HTTPS = f"https://{BUCKET}.s3.us-west-2.amazonaws.com"
#: The subtypes the plan model was trained on (dataset/scripts/normalize_geodata.py).
TRAINED_SUBTYPES = ("country", "region", "county")
REPO = Path(__file__).resolve().parents[2]


def list_files(release: str, type_: str) -> list[str]:
    """HTTPS URLs of a release's parquet files for one Overture type."""
    prefix = f"release/{release}/theme=divisions/type={type_}/"
    with urllib.request.urlopen(f"{HTTPS}/?list-type=2&prefix={prefix}") as r:
        keys = re.findall(r"<Key>([^<]+\.parquet)</Key>", r.read().decode())
    if not keys:
        raise SystemExit(f"no files under s3://{BUCKET}/{prefix} (is the release still published?)")
    return [f"{HTTPS}/{k}" for k in sorted(keys)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="Overture release, e.g. 2026-10-21.0")
    ap.add_argument("--natural-earth", default=str(REPO / "data/natural_earth_normalized/ne_geography.parquet"))
    ap.add_argument("--out", default=str(REPO / "web/public/data"))
    ap.add_argument("--memory-limit", default="3GB")
    ap.add_argument(
        "--local",
        help="directory holding a downloaded copy of the release's division_area/ and division/ files (same names); "
        "much faster than reading S3 from DuckDB. The manifest still points at S3.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    area_files = list_files(args.release, "division_area")
    division_files = list_files(args.release, "division")

    def readable(urls: list[str], type_: str) -> list[str]:
        """Where to read each file from: the local copy if given, else S3."""
        if not args.local:
            return urls
        paths = [Path(args.local) / type_ / u.rsplit("/", 1)[1] for u in urls]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing local files: {missing}")
        return [str(p) for p in paths]

    con = duckdb.connect()
    for sql in ("INSTALL spatial", "LOAD spatial", "INSTALL httpfs", "LOAD httpfs"):
        con.execute(sql)
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")

    t = time.time()
    area_read = readable(area_files, "division_area")
    files = ", ".join(f"'{f}'" for f in area_read)
    # Which manifest file each row is in: the browser reads a feature's geometry from that file only.
    file_index = " ".join(f"WHEN '{f}' THEN {i}" for i, f in enumerate(area_read))
    # Same rows as gazet's normalized divisions_area: trained subtypes, land only, with geometry.
    con.execute(
        f"""
        CREATE TABLE divisions AS
        SELECT
            'divisions_area' AS source,
            id,
            names."primary" AS primary_name,
            names.common['en'] AS english_name,
            {DIVISIONS_AREA_SEARCH_NAMES} AS search_names,
            subtype, class, country, region, admin_level, is_land, is_territorial,
            division_id,
            {{'xmin': bbox.xmin, 'ymin': bbox.ymin, 'xmax': bbox.xmax, 'ymax': bbox.ymax}} AS bbox,
            ST_Area(geometry) AS area,
            CAST(CASE filename {file_index} END AS TINYINT) AS file,
            file_row_number AS "row"
        FROM read_parquet([{files}], filename = true, file_row_number = true)
        WHERE subtype IN {TRAINED_SUBTYPES} AND is_land = true AND geometry IS NOT NULL
        """
    )
    print(f"divisions: {con.execute('SELECT count(*) FROM divisions').fetchone()[0]} rows, {time.time() - t:.0f}s")

    dfiles = ", ".join(f"'{f}'" for f in readable(division_files, "division"))
    con.execute(
        f"""
        CREATE TABLE population AS
        SELECT id, population FROM read_parquet([{dfiles}])
        WHERE id IN (SELECT division_id FROM divisions) AND population IS NOT NULL
        """
    )

    ne = args.natural_earth
    con.execute(
        f"""
        CREATE TABLE ne_rows AS
        SELECT
            row_number() OVER () AS ne_key,  -- Natural Earth IDs repeat (gazet returns every row of an ID)
            * EXCLUDE (names, geometry),
            {{'primary': names."primary"}} AS names,
            names.en AS english_name,
            {NATURAL_EARTH_SEARCH_NAMES} AS search_names,
            geometry,
            {{'xmin': ST_XMin(geometry), 'ymin': ST_YMin(geometry),
              'xmax': ST_XMax(geometry), 'ymax': ST_YMax(geometry)}} AS bbox,
            ST_Area(geometry) AS area
        FROM read_parquet('{ne}')
        WHERE geometry IS NOT NULL
        """
    )

    # Natural Earth geometry: Hilbert order and small row groups, so a bbox filter skips most of the file.
    con.execute(
        f"""
        COPY (
            SELECT ne_key, id, names, subtype, class, country, region, admin_level, is_land, is_territorial, bbox, geometry
            FROM ne_rows
            ORDER BY ST_Hilbert(geometry, ST_Extent(ST_MakeEnvelope(-180, -90, 180, 90)))
        ) TO '{out / "natural_earth.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)
        """
    )

    # Where each Natural Earth feature landed in the sorted file.
    con.execute(
        f"""
        CREATE TABLE ne_rows_at AS
        SELECT ne_key, file_row_number AS "row" FROM read_parquet('{out / "natural_earth.parquet"}', file_row_number = true)
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                d.source, d.id, COALESCE(NULLIF(trim(d.english_name), ''), NULLIF(trim(d.primary_name), '')) AS name,
                d.primary_name,
                d.search_names, d.subtype, d.class, d.country, d.region, d.admin_level, d.is_land, d.is_territorial,
                CAST(p.population AS BIGINT) AS population, d.bbox, d.area, d.file, d."row"
            FROM divisions AS d LEFT JOIN population AS p ON p.id = d.division_id
            UNION ALL
            SELECT
                'natural_earth', id, COALESCE(NULLIF(trim(english_name), ''), NULLIF(trim(names."primary"), '')),
                names."primary",
                search_names, subtype, class, country, region, admin_level, is_land, is_territorial,
                CAST(NULL AS BIGINT), bbox, area, CAST(NULL AS TINYINT), r."row"
            FROM ne_rows JOIN ne_rows_at AS r USING (ne_key)
            ORDER BY source, id
        ) TO '{out / "index.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    counts = dict(con.execute("SELECT source, count(*) FROM read_parquet(?) GROUP BY 1", [str(out / "index.parquet")]).fetchall())
    manifest = {
        "release": args.release,
        "division_area": area_files,
        "natural_earth": "natural_earth.parquet",
        "index": "index.parquet",
        "counts": counts,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("index.parquet", "natural_earth.parquet"):
        print(f"{name}: {(out / name).stat().st_size / 1e6:.1f} MB")
    print(json.dumps(counts), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
