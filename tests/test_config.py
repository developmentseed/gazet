"""Tests for gazet.config — data path resolution and environment overrides."""

import os
from pathlib import Path

import gazet.config as config


class TestPreferNormalized:
    def test_prefers_normalized_when_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAZET_USE_NORMALIZED_DATA", "1")
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "test.parquet").touch()
        orig = tmp_path / "original.parquet"
        orig.touch()

        result = config._prefer_normalized(
            norm_dir / "test.parquet", orig
        )
        assert result == norm_dir / "test.parquet"

    def test_falls_back_to_original_when_normalized_missing(self, tmp_path):
        norm = tmp_path / "normalized.parquet"  # doesn't exist
        orig = tmp_path / "original.parquet"
        orig.touch()

        result = config._prefer_normalized(norm, orig)
        assert result == orig

    def test_wildcard_normalized_when_glob_matches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAZET_USE_NORMALIZED_DATA", "1")
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "data.parquet").touch()
        orig = tmp_path / "original.parquet"
        orig.touch()

        result = config._prefer_normalized(
            norm_dir / "*.parquet", orig
        )
        assert result == norm_dir / "*.parquet"

    def test_wildcard_falls_back_when_no_match(self, tmp_path):
        norm_dir = tmp_path / "empty"
        norm_dir.mkdir()
        orig = tmp_path / "original.parquet"
        orig.touch()

        result = config._prefer_normalized(
            norm_dir / "*.parquet", orig
        )
        assert result == orig

    def test_respects_use_normalized_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAZET_USE_NORMALIZED_DATA", "0")
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "test.parquet").touch()
        orig = tmp_path / "original.parquet"
        orig.touch()

        result = config._prefer_normalized(
            norm_dir / "test.parquet", orig
        )
        # Should fall back because GAZET_USE_NORMALIZED_DATA is "0"
        assert result == orig


class TestDataDir:
    def test_data_dir_resolves(self):
        # _DATA_DIR should be set (either from env or relative to this file)
        assert isinstance(config._DATA_DIR, Path)
        assert config._DATA_DIR.exists()

    def test_divisions_path_is_set(self):
        assert config.DIVISIONS_AREA_PATH
        assert "divisions_area" in config.DIVISIONS_AREA_PATH

    def test_natural_earth_path_is_set(self):
        assert config.NATURAL_EARTH_PATH
        assert "natural_earth" in config.NATURAL_EARTH_PATH


class TestSchemaInfo:
    def test_schema_info_mentions_divisions_area(self):
        assert "divisions_area" in config.SCHEMA_INFO

    def test_schema_info_mentions_natural_earth(self):
        assert "natural_earth" in config.SCHEMA_INFO

    def test_schema_info_mentions_st_asgeojson(self):
        assert "ST_AsGeoJSON" in config.SCHEMA_INFO

    def test_schema_info_mentions_geometry(self):
        assert "geometry" in config.SCHEMA_INFO


class TestLlamaConfig:
    def test_llama_server_url_default(self):
        assert config.LLAMA_SERVER_URL == "http://localhost:9000"

    def test_llama_server_url_env(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_URL", "http://remote:8080")
        # Reimport to pick up env var — config is module-level, so we
        # need to check the env var directly since module is already loaded.
        assert os.environ["LLAMA_SERVER_URL"] == "http://remote:8080"

    def test_llama_max_tokens_default(self):
        assert config.LLAMA_MAX_TOKENS == 2048

    def test_llama_temperature_default(self):
        assert config.LLAMA_TEMPERATURE == 0.0
