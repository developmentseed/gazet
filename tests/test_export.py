"""Tests for gazet.export — GeoJSON FeatureCollection serialization."""

import json

import numpy as np
import pandas as pd
import pytest

from gazet.export import (
    _is_geojson_col,
    _to_serializable,
    save_geojson,
    to_feature_collection,
)


class TestToSerializable:
    def test_bytearray(self):
        assert _to_serializable(bytearray(b"hello")) is None

    def test_bytes(self):
        assert _to_serializable(b"hello") is None

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        assert _to_serializable(arr) == [1, 2, 3]

    def test_numpy_int(self):
        val = np.int64(42)
        assert _to_serializable(val) == 42
        assert isinstance(_to_serializable(val), int)

    def test_numpy_float(self):
        val = np.float64(3.14)
        assert _to_serializable(val) == 3.14
        assert isinstance(_to_serializable(val), float)

    def test_numpy_bool(self):
        val = np.bool_(True)
        assert _to_serializable(val) is True

    def test_plain_python_passthrough(self):
        assert _to_serializable("hello") == "hello"
        assert _to_serializable(42) == 42
        assert _to_serializable(None) is None

    def test_dict_passthrough(self):
        d = {"a": 1}
        assert _to_serializable(d) == d


class TestIsGeojsonCol:
    def test_all_geojson_strings(self):
        s = pd.Series(['{"type": "Point"}', '{"type": "Line"}'])
        assert _is_geojson_col(s) is True

    def test_mixed_content(self):
        s = pd.Series(['{"type": "Point"}', "not geojson"])
        assert not _is_geojson_col(s)

    def test_empty_series(self):
        s = pd.Series([], dtype=str)
        assert _is_geojson_col(s) is False

    def test_all_null(self):
        s = pd.Series([None, None])
        assert _is_geojson_col(s) is False

    def test_geojson_with_whitespace(self):
        s = pd.Series(['  {"type": "Point"}'])
        assert _is_geojson_col(s) is True


class TestToFeatureCollection:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        fc = to_feature_collection(df)
        assert fc["type"] == "FeatureCollection"
        assert fc["features"] == []

    def test_no_geometry_column(self):
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        fc = to_feature_collection(df)
        assert fc["type"] == "FeatureCollection"
        assert fc["features"] == []

    def test_single_feature(self):
        point = json.dumps({"type": "Point", "coordinates": [12.0, 34.0]})
        df = pd.DataFrame({"id": ["x1"], "name": ["Paris"], "geometry": [point]})
        fc = to_feature_collection(df)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1

        feat = fc["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        assert feat["geometry"]["coordinates"] == [12.0, 34.0]
        assert feat["properties"]["id"] == "x1"
        assert feat["properties"]["name"] == "Paris"

    def test_null_geometry(self):
        df = pd.DataFrame({"id": ["x1"], "geometry": [None]})
        fc = to_feature_collection(df)
        assert fc["features"][0]["geometry"] is None

    def test_invalid_geojson_string(self):
        df = pd.DataFrame({"id": ["x1"], "geometry": ["not valid json"]})
        fc = to_feature_collection(df)
        assert fc["features"][0]["geometry"] is None

    def test_null_properties_excluded(self):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"id": ["x1"], "name": [None], "geometry": [point]})
        fc = to_feature_collection(df)
        assert "name" not in fc["features"][0]["properties"]

    def test_numpy_values_serialized(self):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({
            "id": [np.int64(1)],
            "score": [np.float64(3.14)],
            "geometry": [point],
        })
        fc = to_feature_collection(df)
        props = fc["features"][0]["properties"]
        assert props["id"] == 1
        assert isinstance(props["id"], int)
        assert props["score"] == pytest.approx(3.14)

    def test_bytearray_property_becomes_null(self):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({
            "id": ["x1"],
            "data": [bytearray(b"raw")],
            "geometry": [point],
        })
        fc = to_feature_collection(df)
        # bytearray should be converted to None via _to_serializable
        assert fc["features"][0]["properties"]["data"] is None

    def test_multiple_features(self):
        geojsons = [
            json.dumps({"type": "Point", "coordinates": [1.0, 2.0]}),
            json.dumps({"type": "Point", "coordinates": [3.0, 4.0]}),
        ]
        df = pd.DataFrame({"id": ["a", "b"], "geometry": geojsons})
        fc = to_feature_collection(df)
        assert len(fc["features"]) == 2
        assert fc["features"][0]["properties"]["id"] == "a"
        assert fc["features"][1]["properties"]["id"] == "b"


class TestSaveGeojson:
    def test_writes_file(self, tmp_path):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"id": ["x1"], "geometry": [point]})
        out = save_geojson(df, "test_query", output_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".geojson"
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

    def test_slug_generation(self, tmp_path):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"id": ["x1"], "geometry": [point]})
        out = save_geojson(df, "What's the boundary of Paris?", output_dir=tmp_path)
        # Slug should be lowercase with non-word chars replaced
        assert "what_s_the_boundary_of_paris" in out.name

    def test_creates_parent_dirs(self, tmp_path):
        point = json.dumps({"type": "Point", "coordinates": [0.0, 0.0]})
        df = pd.DataFrame({"id": ["x1"], "geometry": [point]})
        nested = tmp_path / "a" / "b" / "c"
        out = save_geojson(df, "test", output_dir=nested)
        assert out.parent == nested
        assert out.exists()

    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        out = save_geojson(df, "empty", output_dir=tmp_path)
        data = json.loads(out.read_text())
        assert data["features"] == []
