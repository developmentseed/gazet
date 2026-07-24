import json
import logging
import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_serializable(val: Any) -> Any:
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
    return bool(
        sample.apply(
            lambda v: isinstance(v, str) and v.lstrip().startswith('{"type":')
        ).all()
        and len(sample) > 0
    )


def save_geojson(
    result_df: pd.DataFrame, query: str, output_dir: pathlib.Path | str = "."
) -> pathlib.Path:
    """Wrap result_df into a GeoJSON FeatureCollection and save to disk.

    Expects the ``geometry`` column to already be normalized to GeoJSON
    strings (run ``geometry.normalize_geometry_to_geojson`` first).
    Remaining columns become properties.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^\w]+", "_", query.lower()).strip("_")
    out_path = output_dir / f"{slug}.geojson"

    fc = to_feature_collection(result_df)
    out_path.write_text(json.dumps(fc, indent=2))
    logger.info("Saved %d feature(s) to %s", len(fc["features"]), out_path.resolve())
    return out_path


def to_feature_collection(result_df: pd.DataFrame) -> dict:
    """Build a GeoJSON FeatureCollection dict from a result DataFrame.

    Expects the ``geometry`` column to already be normalized to GeoJSON
    strings. Remaining columns become properties.
    """
    if "geometry" not in result_df.columns or result_df.empty:
        return {"type": "FeatureCollection", "features": []}

    prop_cols = [c for c in result_df.columns if c != "geometry"]
    records = result_df.to_dict(orient="records")

    features = []
    for record in records:
        raw = record.pop("geometry", None)
        geometry = None
        if raw and isinstance(raw, str):
            try:
                geometry = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        properties = {}
        for c in prop_cols:
            v = record.get(c)
            try:
                if v is None or pd.isna(v):
                    continue
            except (ValueError, TypeError):
                pass  # pd.isna fails on arrays — treat as present
            properties[c] = _to_serializable(v)

        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )

    return {"type": "FeatureCollection", "features": features}
