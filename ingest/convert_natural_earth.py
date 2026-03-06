"""Convert Natural Earth shapefiles to a single GeoParquet with Overture-compatible schema.

Input: directory of *.shp files (passed as CLI argument)

Output: path to write the .parquet file (passed as CLI argument, default: data/natural_earth/ne_geography.parquet)
"""

import argparse
import pathlib

import geopandas as gpd
import pandas as pd

DEFAULT_OUTPUT = pathlib.Path("data/natural_earth_geoparquet/ne_geography.parquet")

# Stems (or substrings) to skip — pure cartographic / utility layers with no
# geographic search value, or point layers that need a separate schema.
SKIP_PATTERNS = (
    "graticules",  # cartographic grid lines
    "_label_points",  # point label layers
    "_scale_rank",  # scale-rank rendering duplicates (base layers kept)
)
SKIP_EXACT = {
    "ne_10m_land_ocean_seams",
    "ne_10m_wgs84_bounding_box",
    "ne_10m_geography_regions_points",
    "ne_10m_geography_regions_elevation_points",
}

LANG_COLS = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "el",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "sv",
    "tr",
    "vi",
    "zh",
    "fa",
    "he",
    "uk",
    "ur",
    "zht",
]


def _names_struct(gdf: gpd.GeoDataFrame, name_col: str) -> pd.Series:
    """Build a names struct column matching Overture's names STRUCT(primary, ...)."""

    def _row(row: pd.Series) -> dict:
        entry: dict[str, str | None] = {"primary": row.get(name_col) or None}
        for lang in LANG_COLS:
            val = row.get(f"name_{lang}")
            entry[lang] = (
                str(val) if val and str(val) not in ("", "nan", "None") else None
            )
        return entry

    return gdf.apply(_row, axis=1)


def _pick_name_col(gdf: gpd.GeoDataFrame) -> str | None:
    """Pick best name column: 'name' or first name_* or 'NAME' etc."""
    cols = [c.lower() for c in gdf.columns]
    if "name" in cols:
        return "name"
    for lang in ["en", "name"] + LANG_COLS:
        cand = f"name_{lang}"
        if cand in cols:
            return cand
    for c in gdf.columns:
        if c.lower().startswith("name"):
            return c
    return None


def _load_shapefile(src: pathlib.Path, source_key: str) -> gpd.GeoDataFrame:
    """Load any Natural Earth shapefile and normalize to Overture-like schema."""
    gdf = gpd.read_file(src)
    gdf.columns = [c.lower() for c in gdf.columns]
    n = len(gdf)

    # id: ne_id if present else source_index
    if "ne_id" in gdf.columns:
        ids = "ne_" + gdf["ne_id"].astype(str)
    else:
        ids = pd.Series([f"ne_10m_{source_key}_{i}" for i in range(n)])

    name_col = _pick_name_col(gdf)
    if name_col is None:
        names = pd.Series([{"primary": None, **{lang: None for lang in LANG_COLS}}] * n)
    else:
        names = _names_struct(gdf, name_col)

    # subtype: featurecla or source key
    if "featurecla" in gdf.columns:
        subtype = gdf["featurecla"]
    else:
        subtype = pd.Series([source_key] * n)

    return gpd.GeoDataFrame(
        {
            "id": ids,
            "source_layer": pd.array([source_key] * n, dtype=pd.StringDtype()),
            "names": names,
            "subtype": subtype,
            "class": pd.array([None] * n, dtype=pd.StringDtype()),
            "country": gdf.get("sov_a3", pd.array([None] * n, dtype=pd.StringDtype()))
            if "sov_a3" in gdf.columns
            else pd.array([None] * n, dtype=pd.StringDtype()),
            "region": gdf.get("region", pd.array([None] * n, dtype=pd.StringDtype())),
            "admin_level": pd.array([None] * n, dtype=pd.Int32Dtype()),
            "is_land": _infer_is_land(source_key, gdf),
            "is_territorial": pd.array([None] * n, dtype=pd.BooleanDtype()),
            "geometry": gdf.geometry,
        },
        crs=gdf.crs,
    )


def _infer_is_land(source_key: str, gdf: gpd.GeoDataFrame) -> pd.Series:
    """Infer is_land from source name when possible."""
    n = len(gdf)
    ocean_marine = ("ocean", "marine", "bathymetry", "coastline", "seams", "reefs")
    if any(x in source_key for x in ocean_marine):
        return pd.Series([False] * n)
    if "land" in source_key or "lakes" in source_key or "regions" in source_key:
        return pd.Series([True] * n)
    return pd.array([None] * n, dtype=pd.BooleanDtype())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shp_dir", type=pathlib.Path, help="Directory containing *.shp files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .parquet path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    all_shp = sorted(args.shp_dir.glob("*.shp"))
    if not all_shp:
        raise SystemExit(f"No .shp files in {args.shp_dir}")

    def _should_skip(stem: str) -> bool:
        if stem in SKIP_EXACT:
            return True
        return any(p in stem for p in SKIP_PATTERNS)

    shp_files = [p for p in all_shp if not _should_skip(p.stem)]
    skipped = [p.stem for p in all_shp if _should_skip(p.stem)]
    if skipped:
        print(f"Skipping {len(skipped)} utility layers: {', '.join(skipped)}\n")

    frames = []
    for path in shp_files:
        source_key = path.stem  # e.g. ne_10m_geography_marine_polys
        gdf = _load_shapefile(path, source_key)
        frames.append(gdf)
        print(f"  {path.name}: {len(gdf)} features")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=frames[0].crs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output)
    print(f"\nSaved {len(combined)} features → {args.output}")
