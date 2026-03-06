import json
import pathlib
import re

import pandas as pd


def _is_geojson_col(series: pd.Series) -> bool:
    """Heuristic: a string column whose non-null values start with '{"type":'."""
    sample = series.dropna().head(5)
    return (
        sample.apply(
            lambda v: isinstance(v, str) and v.lstrip().startswith('{"type":')
        ).all()
        and len(sample) > 0
    )


def save_geojson(
    result_df: pd.DataFrame, query: str, output_dir: pathlib.Path | str = "."
) -> pathlib.Path:
    """Wrap result_df into a GeoJSON FeatureCollection and save to disk.

    Any column whose values are GeoJSON geometry strings (output of ST_AsGeoJSON)
    is used as the feature geometry; remaining columns become properties.
    If multiple geometry columns exist the first one wins.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^\w]+", "_", query.lower()).strip("_")
    out_path = output_dir / f"{slug}.geojson"

    fc = _to_feature_collection(result_df)
    out_path.write_text(json.dumps(fc, indent=2))
    print(f"\nSaved {len(fc['features'])} feature(s) → {out_path.resolve()}")
    return out_path


def to_feature_collection(result_df: pd.DataFrame) -> dict:
    """Build a GeoJSON FeatureCollection dict from a result DataFrame."""
    return _to_feature_collection(result_df)


def _to_feature_collection(result_df: pd.DataFrame) -> dict:
    geom_cols = [c for c in result_df.columns if _is_geojson_col(result_df[c])]
    prop_cols = [c for c in result_df.columns if c not in geom_cols]
    features = []
    for _, row in result_df.iterrows():
        geometry = None
        if geom_cols:
            raw = row[geom_cols[0]]
            if raw and isinstance(raw, str):
                try:
                    geometry = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        properties = {c: row[c] for c in prop_cols if pd.notna(row[c])}
        for c in geom_cols[1:]:
            if pd.notna(row[c]):
                properties[c] = row[c]
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    return {"type": "FeatureCollection", "features": features}
