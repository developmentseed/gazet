import json
import pathlib
import re

import numpy as np
import pandas as pd


def _to_serializable(val):
    """Convert a value to a JSON-serializable Python type."""
    if isinstance(val, (bytearray, bytes)):
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _is_geojson_col(series: pd.Series) -> bool:
    """Heuristic: a string column whose non-null values start with '{"type":'."""
    sample = series.dropna().head(5)
    return (
        sample.apply(
            lambda v: isinstance(v, str) and v.lstrip().startswith('{"type":')
        ).all()
        and len(sample) > 0
    )


def _is_wkb_col(series: pd.Series) -> bool:
    """Heuristic: a column whose non-null values are bytearray or bytes (WKB geometry)."""
    sample = series.dropna().head(5)
    return (
        sample.apply(lambda v: isinstance(v, (bytearray, bytes))).all()
        and len(sample) > 0
    )


def _wkb_to_geojson(wkb: bytearray | bytes) -> dict | None:
    """Convert WKB geometry to GeoJSON dict via DuckDB."""
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
        result = con.execute(
            "SELECT ST_AsGeoJSON(ST_GeomFromWKB(?::BLOB)) AS geojson",
            [bytes(wkb)],
        ).fetchone()
        if result and result[0]:
            return json.loads(result[0])
    except Exception:
        pass
    finally:
        con.close()
    return None


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
    geojson_cols = [c for c in result_df.columns if _is_geojson_col(result_df[c])]
    wkb_cols = [c for c in result_df.columns if _is_wkb_col(result_df[c])]
    geom_cols = geojson_cols + wkb_cols
    prop_cols = [c for c in result_df.columns if c not in geom_cols]
    features = []
    for _, row in result_df.iterrows():
        geometry = None
        if geojson_cols:
            raw = row[geojson_cols[0]]
            if raw and isinstance(raw, str):
                try:
                    geometry = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        elif wkb_cols:
            raw = row[wkb_cols[0]]
            if raw and isinstance(raw, (bytearray, bytes)):
                geometry = _wkb_to_geojson(raw)
        properties = {}
        for c in prop_cols:
            v = row[c]
            try:
                if not pd.notna(v):
                    continue
            except ValueError:
                pass  # pd.notna fails on arrays — treat as present
            properties[c] = _to_serializable(v)
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    return {"type": "FeatureCollection", "features": features}
