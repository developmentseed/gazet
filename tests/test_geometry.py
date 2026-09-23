"""Tests for gazet.geometry — geometry normalization.

The validity tests cover the regression that shipped country-scale AOIs
self-intersecting: rounding the GeoJSON after ST_SimplifyPreserveTopology
undid the topology guarantee it had just bought.
"""

import json

import pandas as pd
import pytest
from shapely.geometry import shape

from gazet.geometry import COORD_PRECISION, normalize_geometry_to_geojson


def _normalized(con, geom):
    df = pd.DataFrame({"geometry": [geom]})
    return normalize_geometry_to_geojson(con, df)["geometry"].iloc[0]


def _comb(teeth, gap=1e-6):
    """A polygon whose teeth are separated by less than one grid cell, so
    snapping to the grid folds them into each other."""
    coords = [(0.0, 0.0)]
    for i in range(teeth):
        x = i * gap
        coords += [(x, 1.0), (x + gap / 2, 1.0), (x + gap / 2, 0.0)]
    coords.append((0.0, 0.0))
    return json.dumps({"type": "Polygon", "coordinates": [[list(c) for c in coords]]})


class TestNormalizeGeometryToGeojson:
    def test_no_geometry_column(self, con):
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        result = normalize_geometry_to_geojson(con, df)
        assert "geometry" not in result.columns

    def test_empty_dataframe(self, con):
        df = pd.DataFrame(columns=["geometry"])
        result = normalize_geometry_to_geojson(con, df)
        assert result.empty

    def test_rounds_coordinates(self, con):
        point = json.dumps({"type": "Point", "coordinates": [12.1234567, 34.9876543]})
        parsed = json.loads(_normalized(con, point))
        assert parsed["type"] == "Point"
        assert parsed["coordinates"] == [
            round(12.1234567, COORD_PRECISION),
            round(34.9876543, COORD_PRECISION),
        ]

    def test_precision_reduction_leaves_a_valid_polygon(self, con):
        # Rounding this after the fact self-intersects; ST_ReducePrecision does not.
        cleaned = shape(json.loads(_normalized(con, _comb(40))))
        assert cleaned.is_valid

    def test_repairs_an_invalid_input(self, con):
        bowtie = json.dumps(
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
        )
        assert shape(json.loads(_normalized(con, bowtie))).is_valid

    def test_keeps_a_feature_smaller_than_the_grid(self, con):
        # Precision reduction empties this one, so the un-reduced geometry stands.
        speck = json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1e-6, 0], [1e-6, 1e-6], [0, 1e-6], [0, 0]]],
            }
        )
        assert not shape(json.loads(_normalized(con, speck))).is_empty

    def test_null_geometry_preserved(self, con):
        df = pd.DataFrame({"geometry": [None, None]})
        result = normalize_geometry_to_geojson(con, df)
        assert result["geometry"].isna().all()

    def test_mixed_geojson_and_null(self, con):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"geometry": [point, None]})
        result = normalize_geometry_to_geojson(con, df)
        assert result["geometry"].iloc[0] is not None
        assert result["geometry"].iloc[1] is None

    @pytest.mark.parametrize("value", [42, "not geojson"])
    def test_passes_through_non_geometry_values(self, con, value):
        assert _normalized(con, value) == value
