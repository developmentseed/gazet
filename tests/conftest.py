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


@pytest.fixture(scope="module")
def con():
    """Provide a DuckDB connection with the spatial extension loaded.

    Module-scoped so INSTALL/LOAD spatial happens once per test file
    instead of on every individual test."""
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


# ---------------------------------------------------------------------------
# English-exonym fixture
# ---------------------------------------------------------------------------

#: Places whose English name differs from the name the record is filed under,
#: shaped like Overture divisions_area rows: ``names.primary`` is the local
#: name and ``names.common`` is a translation map that may or may not carry
#: ``en``. ``Coper`` is the Colombian county a search for "Copenhagen" used to
#: land on, and it has no English name — the 77% of the real table that a
#: match on English alone would drop. ``Genève`` carries a blank one, which is
#: absent rather than a name the record can be shown under.
_EXONYM_ROWS = [
    (
        "cph",
        "Københavns Kommune",
        "Copenhagen Municipality",
        "DK",
        "county",
        2,
        12.5,
        55.6,
    ),
    ("dnk", "Danmark", "Denmark", "DK", "country", 0, 9.5, 55.5),
    ("muc", "München", "Munich", "DE", "county", 2, 11.5, 48.1),
    ("vie", "Wien", "Vienna", "AT", "region", 1, 16.3, 48.2),
    ("coper", "Coper", None, "CO", "county", 2, -74.0, 5.4),
    ("ratnapura", None, "Ratnapura District", "LK", "county", 2, 80.4, 6.7),
    ("gva", "Genève", "", "CH", "county", 2, 6.1, 46.2),
]


def _literal(value):
    """A string as a SQL literal, or NULL where there is no value."""
    return "NULL" if value is None else f"'{value}'"


def _name_map(english):
    """``names.common``: the English name where there is one, otherwise empty."""
    return "MAP([], [])" if english is None else f"MAP(['en'], ['{english}'])"


@pytest.fixture(scope="session")
def exonym_parquet(tmp_path_factory):
    """Write the exonym rows to a parquet with the divisions_area schema."""
    path = tmp_path_factory.mktemp("exonym") / "divisions_area.parquet"
    c = duckdb.connect()
    c.execute("INSTALL spatial")
    c.execute("LOAD spatial")
    rows = ",\n".join(
        f"({_literal(id)}, {_literal(primary)}, {_name_map(english)}, "
        f"{_literal(country)}, {_literal(subtype)}, {admin_level}, {lon}, {lat})"
        for id, primary, english, country, subtype, admin_level, lon, lat in _EXONYM_ROWS
    )
    c.execute(
        f"""
        COPY (
            SELECT
                id,
                ST_AsWKB(
                    ST_GeomFromText(
                        'POLYGON((' || lon || ' ' || lat || ', '
                                    || (lon + 0.1) || ' ' || lat || ', '
                                    || (lon + 0.1) || ' ' || (lat + 0.1) || ', '
                                    || lon || ' ' || (lat + 0.1) || ', '
                                    || lon || ' ' || lat || '))'
                    )
                ) AS geometry,
                country,
                subtype,
                'land' AS class,
                {{'primary': primary_name, 'common': common}} AS names,
                NULL AS region,
                admin_level,
                true AS is_land,
                false AS is_territorial,
                id AS division_id
            FROM (VALUES
                {rows}
            ) AS t(id, primary_name, common, country, subtype, admin_level, lon, lat)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    c.close()
    return str(path)


@pytest.fixture()
def exonym_source(monkeypatch, exonym_parquet):
    """Point divisions_area at the exonym parquet for the duration of a test."""
    monkeypatch.setattr("gazet.search.DIVISIONS_AREA_PATH", exonym_parquet)
    return exonym_parquet
