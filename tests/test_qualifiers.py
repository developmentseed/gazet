"""Qualified fuzzy queries ("Paris, Texas") and near-miss rejection.

The fixture reproduces the reported failures: each trap row is the wrong place
the unqualified search used to return (Panama's Parita District for "Paris,
Texas", Newfoundland's Raleigh for "Raleigh, NC", Mount Eba for "Mount Etna").
"""

import duckdb
import pytest
from fastapi.testclient import TestClient

from gazet.api import app
from gazet.qualifiers import (
    jaro_winkler,
    parse_qualified_query,
    strip_directional,
    token_score,
)

# id, primary, english, country, region, subtype, admin_level, lon, lat, size
_ROWS = [
    ("us", "United States", None, "US", None, "country", 0, -125.0, 25.0, 50.0),
    ("tx", "Texas", None, "US", "US-TX", "region", 1, -106.0, 26.0, 10.0),
    ("nc", "North Carolina", None, "US", "US-NC", "region", 1, -84.0, 34.0, 5.0),
    ("it", "Italia", "Italy", "IT", None, "country", 0, 7.0, 37.0, 10.0),
    ("br", "Brasil", "Brazil", "BR", None, "country", 0, -73.0, -33.0, 30.0),
    ("paris-fr", "Paris", None, "FR", "FR-IDF", "county", 2, 2.2, 48.8, 0.1),
    ("paris-tx", "Paris", None, "US", "US-TX", "locality", None, -95.6, 33.6, 0.1),
    ("parita", "Parita District", None, "PA", "PA-6", "county", 2, -80.5, 8.0, 0.1),
    ("raleigh-nl", "Raleigh", None, "CA", "CA-NL", "county", 2, -55.7, 51.5, 0.1),
    ("raleigh-nc", "Raleigh", None, "US", "US-NC", "locality", None, -78.7, 35.8, 0.1),
    ("eba", "Mount Eba", None, "AU", "AU-SA", "locality", None, 135.2, -30.3, 0.1),
]


def _literal(value):
    return "NULL" if value is None else f"'{value}'"


@pytest.fixture(scope="module")
def qualifier_parquet(tmp_path_factory):
    path = tmp_path_factory.mktemp("qualifiers") / "divisions_area.parquet"
    c = duckdb.connect()
    c.execute("INSTALL spatial")
    c.execute("LOAD spatial")
    rows = ",\n".join(
        f"({_literal(id)}, {_literal(primary)}, "
        + ("MAP([], [])" if english is None else f"MAP(['en'], ['{english}'])")
        + f", {_literal(country)}, {_literal(region)}, {_literal(subtype)}, "
        f"{'NULL' if level is None else level}::INTEGER, {lon}, {lat}, {size})"
        for id, primary, english, country, region, subtype, level, lon, lat, size in _ROWS
    )
    c.execute(
        f"""
        COPY (
            SELECT
                id,
                ST_AsWKB(ST_MakeEnvelope(lon, lat, lon + size, lat + size)) AS geometry,
                country,
                subtype,
                'land' AS class,
                {{'primary': primary_name, 'common': common}} AS names,
                region,
                admin_level,
                true AS is_land,
                false AS is_territorial,
                id AS division_id
            FROM (VALUES
                {rows}
            ) AS t(id, primary_name, common, country, region, subtype, admin_level,
                   lon, lat, size)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    c.close()
    return str(path)


@pytest.fixture()
def qualifier_source(monkeypatch, qualifier_parquet):
    monkeypatch.setattr("gazet.search.DIVISIONS_AREA_PATH", qualifier_parquet)
    return qualifier_parquet


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _search(client, q):
    resp = client.get(
        "/search",
        params={
            "q": q,
            "mode": "fuzzy",
            "ids_only": "true",
            "sources": "divisions_area",
            "limit": 3,
        },
    )
    assert resp.status_code == 200
    return resp.json().get("ids", [])


class TestJaroWinkler:
    @pytest.mark.parametrize(
        "a,b",
        [("etna", "eba"), ("brazil", "alps"), ("sprabery", "spree"), ("paris", "parita"),
         ("martha", "marhta"), ("a", "b"), ("same", "same")],
    )
    def test_matches_duckdb(self, con, a, b):
        expected = con.execute("SELECT jaro_winkler_similarity(?, ?)", [a, b]).fetchone()[0]
        assert jaro_winkler(a, b) == pytest.approx(expected)


class TestTokenScore:
    def test_rejects_a_near_miss_on_one_word(self):
        # Whole-string Jaro-Winkler scores this 0.94; "etna" vs "eba" does not.
        assert token_score("Mount Etna", "Mount Eba") < 0.8

    def test_folds_accents(self):
        assert token_score("Sao Paulo", "São Paulo") == 1.0

    def test_ignores_a_generic_word_a_natural_feature_left_out(self):
        assert token_score("Lake Titicaca", "Titicaca", drop_generic=True) == 1.0

    def test_keeps_generic_words_for_admin_names(self):
        assert token_score("Gulf of Mexico", "State of Mexico") < 0.8

    def test_scores_the_worst_matched_word(self):
        assert token_score("Southern Brazil", "Southern Alps") < 0.8


class TestStripDirectional:
    def test_strips_a_leading_direction(self):
        assert strip_directional("Southern Brazil") == "Brazil"

    def test_leaves_a_single_word(self):
        assert strip_directional("South") is None

    def test_leaves_other_names(self):
        assert strip_directional("Rio Grande do Sul") is None


class TestParseQualifiedQuery:
    def test_state_name(self, con, qualifier_source):
        name, area = parse_qualified_query(con, "Paris, Texas")
        assert name == "Paris"
        assert (area.country, area.region) == ("US", "US-TX")

    def test_postal_code(self, con, qualifier_source):
        _, area = parse_qualified_query(con, "Raleigh, NC")
        assert area.region == "US-NC"

    def test_english_country_name(self, con, qualifier_source):
        _, area = parse_qualified_query(con, "Mount Etna, Italy")
        assert (area.country, area.region) == ("IT", None)

    def test_most_specific_qualifier_wins(self, con, qualifier_source):
        _, area = parse_qualified_query(con, "Paris, Texas, United States")
        assert area.region == "US-TX"

    def test_country_alias(self, con, qualifier_source):
        _, area = parse_qualified_query(con, "Paris, USA")
        assert (area.country, area.region) == ("US", None)

    def test_unresolved_qualifier_keeps_the_whole_query(self, con, qualifier_source):
        assert parse_qualified_query(con, "Paris, Nowhere") == ("Paris, Nowhere", None)

    def test_no_comma(self, con, qualifier_source):
        assert parse_qualified_query(con, "Paris") == ("Paris", None)


class TestQualifiedSearch:
    def test_state_picks_the_place_in_that_state(self, client, qualifier_source):
        ids = _search(client, "Paris, Texas")
        assert [i["id"] for i in ids] == ["paris-tx"]
        assert ids[0]["match_type"] == "name"

    def test_postal_code_picks_the_place_in_that_state(self, client, qualifier_source):
        assert [i["id"] for i in _search(client, "Raleigh, NC")] == ["raleigh-nc"]

    def test_unknown_name_falls_back_to_the_qualifier(self, client, qualifier_source):
        ids = _search(client, "Sprabery, Texas, United States")
        assert [i["id"] for i in ids] == ["tx"]
        assert ids[0]["match_type"] == "qualifier_fallback"

    def test_near_miss_returns_nothing(self, client, qualifier_source):
        assert _search(client, "Mount Etna") == []

    def test_directional_prefix_falls_back_to_the_rest(self, client, qualifier_source):
        ids = _search(client, "Southern Brazil")
        assert ids[0]["id"] == "br"
        assert ids[0]["match_type"] == "directional_fallback"

    def test_unqualified_search_is_unchanged(self, client, qualifier_source):
        ids = _search(client, "Raleigh")
        assert [i["id"] for i in ids] == ["raleigh-nl", "raleigh-nc"]
