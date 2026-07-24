"""Tests for gazet.sql — SQL rewriting, normalization, and helpers."""

from unittest.mock import MagicMock, patch

import pandas as pd

from gazet.sql import (
    _normalize_ne_subtypes,
    _rewrite_data_paths,
    _strip_fences,
    run_geo_sql_dspy,
    run_geo_sql_gguf,
)


class TestStripFences:
    def test_plain_sql(self):
        assert _strip_fences("SELECT * FROM foo") == "SELECT * FROM foo"

    def test_sql_backtick_fences(self):
        raw = "```sql\nSELECT id FROM bar\n```"
        assert _strip_fences(raw) == "SELECT id FROM bar"

    def test_backtick_fences_no_lang(self):
        raw = "```\nSELECT 1\n```"
        assert _strip_fences(raw) == "SELECT 1"

    def test_none_input(self):
        assert _strip_fences(None) == ""

    def test_empty_string(self):
        assert _strip_fences("") == ""

    def test_partial_fence_leading(self):
        raw = "```sql\nSELECT id"
        assert _strip_fences(raw) == "SELECT id"

    def test_partial_fence_trailing(self):
        raw = "SELECT id\n```"
        assert _strip_fences(raw) == "SELECT id"

    def test_preserves_inner_backticks(self):
        raw = "```sql\nSELECT `column` FROM table\n```"
        assert _strip_fences(raw) == "SELECT `column` FROM table"


class TestRewriteDataPaths:
    def test_symbolic_divisions_area(self):
        sql = "SELECT * FROM read_parquet('divisions_area')"
        result = _rewrite_data_paths(sql)
        assert "divisions_area" in result
        assert "read_parquet('divisions_area')" not in result

    def test_symbolic_natural_earth(self):
        sql = "SELECT * FROM read_parquet('natural_earth')"
        result = _rewrite_data_paths(sql)
        assert "natural_earth" in result
        assert "read_parquet('natural_earth')" not in result

    def test_hallucinated_division_path(self):
        sql = "SELECT * FROM read_parquet('/data/overture/division_area/foo.parquet')"
        result = _rewrite_data_paths(sql)
        # The original hallucinated path should be gone
        assert "/data/overture/division_area/foo.parquet" not in result

    def test_hallucinated_natural_earth_path(self):
        sql = "SELECT * FROM read_parquet('/some/natural_earth_geoparquet/data.parquet')"
        result = _rewrite_data_paths(sql)
        assert "/some/natural_earth_geoparquet/data.parquet" not in result

    def test_double_quotes(self):
        sql = 'SELECT * FROM read_parquet("divisions_area")'
        result = _rewrite_data_paths(sql)
        assert 'read_parquet("divisions_area")' not in result

    def test_no_false_positive_unrelated_table(self):
        sql = "SELECT * FROM read_parquet('some_other_table')"
        result = _rewrite_data_paths(sql)
        # Should be unchanged
        assert "some_other_table" in result


class TestNormalizeNeSubtypes:
    def test_lowercase_river(self):
        sql = "WHERE n.subtype = 'River'"
        result = _normalize_ne_subtypes(sql)
        assert "'river'" in result

    def test_lowercase_lake(self):
        sql = "WHERE n.subtype = 'Lake'"
        result = _normalize_ne_subtypes(sql)
        assert "'lake'" in result

    def test_lowercase_ocean(self):
        sql = "WHERE subtype = 'Ocean'"
        result = _normalize_ne_subtypes(sql)
        assert "'ocean'" in result

    def test_lowercase_sea(self):
        sql = "WHERE subtype = 'Sea'"
        result = _normalize_ne_subtypes(sql)
        assert "'sea'" in result

    def test_lowercase_range_mtn(self):
        sql = "WHERE subtype = 'Range/mtn'"
        result = _normalize_ne_subtypes(sql)
        assert "'range/mtn'" in result

    def test_terrain_area_replacement(self):
        sql = "WHERE n.subtype = 'Terrain area'"
        result = _normalize_ne_subtypes(sql)
        assert "range/mtn" in result
        assert "peninsula" in result
        assert "depression" in result

    def test_terrain_area_in_clause(self):
        sql = "WHERE n.subtype IN ('Terrain area')"
        result = _normalize_ne_subtypes(sql)
        assert "range/mtn" in result

    def test_already_lowercase_unchanged(self):
        sql = "WHERE n.subtype = 'river'"
        result = _normalize_ne_subtypes(sql)
        assert result == sql

    def test_island_group(self):
        sql = "WHERE subtype = 'Island group'"
        result = _normalize_ne_subtypes(sql)
        assert "'island group'" in result


class TestRunGeoSqlGguf:
    def test_empty_candidates_returns_none_result(self, con):
        empty_df = pd.DataFrame()
        events = list(run_geo_sql_gguf(con, "get Paris", empty_df))
        assert len(events) >= 1
        assert events[-1]["type"] == "result"
        assert events[-1]["df"] is None

    @patch("gazet.sql.generate_sql")
    def test_execution_flow(self, mock_generate, con):
        mock_generate.return_value = "SELECT 1"
        # A query that succeeds but returns no useful geometry rows
        events = list(run_geo_sql_gguf(con, "test", pd.DataFrame({"id": ["1"]})))
        # Should emit at least sql_attempt and result
        types = [e["type"] for e in events]
        assert "sql_attempt" in types


class TestRunGeoSqlDspy:
    def test_empty_candidates_returns_none_result(self, con):
        empty_df = pd.DataFrame()
        events = list(run_geo_sql_dspy(con, "get Paris", empty_df))
        assert events[-1]["type"] == "result"
        assert events[-1]["df"] is None

    @patch("gazet.sql.write_sql")
    def test_successful_sql(self, mock_write, con):
        # Return a result object with .sql attribute
        mock_pred = MagicMock()
        mock_pred.sql = "SELECT 1 as id"
        mock_write.return_value = mock_pred
        df = pd.DataFrame({"id": ["x1"], "name": ["test"], "source": ["divisions_area"]})
        events = list(run_geo_sql_dspy(con, "test", df, max_iterations=1))
        types = [e["type"] for e in events]
        assert "sql_attempt" in types

    @patch("gazet.sql.write_sql")
    def test_exhausts_iterations(self, mock_write, con):
        mock_pred = MagicMock()
        mock_pred.sql = "INVALID SQL"  # will cause execution error
        mock_write.return_value = mock_pred
        df = pd.DataFrame({"id": ["x1"], "name": ["test"], "source": ["divisions_area"]})
        events = list(run_geo_sql_dspy(con, "test", df, max_iterations=2))
        # Should exhaust iterations and yield final result
        assert events[-1]["type"] == "result"
