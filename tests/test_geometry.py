"""Tests for gazet.geometry — coordinate rounding and geometry normalization."""

import json

from gazet.geometry import _round_coords, normalize_geometry_to_geojson


class TestRoundCoords:
    def test_single_float(self):
        assert _round_coords(1.23456789, 1) == 1.2

    def test_zero_precision(self):
        assert _round_coords(1.9, 0) == 2.0

    def test_nested_list(self):
        coords = [1.11111, 2.22222, 3.33333]
        result = _round_coords(coords, 2)
        assert result == [1.11, 2.22, 3.33]

    def test_dict_recursion(self):
        obj = {"lat": 1.99999, "lng": 2.11111}
        result = _round_coords(obj, 1)
        assert result == {"lat": 2.0, "lng": 2.1}

    def test_nested_structure(self):
        geojson = {
            "type": "Point",
            "coordinates": [12.123456, 34.987654],
        }
        result = _round_coords(geojson, 3)
        assert result == {"type": "Point", "coordinates": [12.123, 34.988]}

    def test_multipolygon(self):
        coords = [[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]]
        result = _round_coords(coords, 1)
        assert result == [[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]]

    def test_string_passthrough(self):
        assert _round_coords("hello", 2) == "hello"

    def test_int_passthrough(self):
        assert _round_coords(42, 2) == 42

    def test_none_passthrough(self):
        assert _round_coords(None, 2) is None

    def test_mixed_types_in_list(self):
        data = [1.111, "type", 2.222]
        result = _round_coords(data, 1)
        assert result == [1.1, "type", 2.2]


class TestNormalizeGeometryToGeojson:
    def test_no_geometry_column(self, con):
        import pandas as pd

        df = pd.DataFrame({"id": [1], "name": ["test"]})
        result = normalize_geometry_to_geojson(con, df)
        assert "geometry" not in result.columns

    def test_empty_dataframe(self, con):
        import pandas as pd

        df = pd.DataFrame(columns=["geometry"])
        result = normalize_geometry_to_geojson(con, df)
        assert result.empty

    def test_geojson_string_simplified(self, con):
        import pandas as pd

        # A simple point — simplify should pass it through
        point_geojson = json.dumps(
            {
                "type": "Point",
                "coordinates": [12.1234567, 34.9876543],
            }
        )
        df = pd.DataFrame({"geometry": [point_geojson]})
        result = normalize_geometry_to_geojson(con, df)
        assert result["geometry"].iloc[0] is not None
        parsed = json.loads(result["geometry"].iloc[0])
        assert parsed["type"] == "Point"
        # Coordinates should be rounded
        coords = parsed["coordinates"]
        assert coords[0] == round(12.1234567, 5)
        assert coords[1] == round(34.9876543, 5)

    def test_null_geometry_preserved(self, con):
        import pandas as pd

        df = pd.DataFrame({"geometry": [None, None]})
        result = normalize_geometry_to_geojson(con, df)
        assert pd.isna(result["geometry"]).all() or result["geometry"].isna().all()

    def test_mixed_geojson_and_null(self, con):
        import pandas as pd

        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"geometry": [point, None]})
        result = normalize_geometry_to_geojson(con, df)
        assert result["geometry"].iloc[0] is not None
        assert result["geometry"].iloc[1] is None
