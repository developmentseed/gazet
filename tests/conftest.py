"""Shared fixtures for the gazet test suite."""

import os
from pathlib import Path

import duckdb
import pytest

# Force data dir to project root for tests
os.environ["GAZET_DATA_DIR"] = str(Path(__file__).resolve().parent.parent / "data")
# Prefer original (non-normalized) paths — test suite ships without normalized copies
os.environ["GAZET_USE_NORMALIZED_DATA"] = "0"

# ---------------------------------------------------------------------------
# DuckDB connection fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def con():
    """Provide a DuckDB connection with the spatial extension loaded."""
    c = duckdb.connect()
    c.execute("INSTALL spatial")
    c.execute("LOAD spatial")
    yield c
    c.close()


@pytest.fixture()
def con_no_spatial():
    """Provide a bare DuckDB connection (no spatial extension)."""
    c = duckdb.connect()
    yield c
    c.close()
