"""Tests for gazet.search — fuzzy search and ID lookup against real parquet data."""

import pandas as pd

from gazet.schemas import Place
from gazet.search import (
    get_by_id,
    get_division_by_id,
    get_natural_earth_by_id,
    search_candidates,
    search_divisions_area,
    search_natural_earth,
)


class TestSearchDivisionsArea:
    def test_returns_dataframe(self, con):
        df = search_divisions_area(con, Place(place="India"))
        assert isinstance(df, pd.DataFrame)
        assert "source" in df.columns
        assert "id" in df.columns

    def test_matches_known_country(self, con):
        df = search_divisions_area(con, Place(place="India"))
        assert not df.empty
        # At least one result should be India
        names = df["name"].str.lower().tolist()
        assert any("india" in name for name in names)

    def test_limits_results(self, con):
        df = search_divisions_area(con, Place(place="a"), limit=3)
        assert len(df) <= 3

    def test_empty_result(self, con):
        # Extremely unlikely string — Jaro-Winkler always returns top N,
        # but similarity should be very low
        df = search_divisions_area(con, Place(place="Xyzz98765"))
        # Results may still be returned (always top-5) but with low similarity
        assert len(df) <= 5

    def test_source_column(self, con):
        df = search_divisions_area(con, Place(place="India"))
        if not df.empty:
            assert (df["source"] == "divisions_area").all()

    def test_case_insensitive(self, con):
        df_upper = search_divisions_area(con, Place(place="INDIA"))
        df_lower = search_divisions_area(con, Place(place="india"))
        # Both should return results (even if different similarity scores)
        assert not df_upper.empty
        assert not df_lower.empty

    def test_with_country_filter(self, con):
        df = search_divisions_area(con, Place(place="Loja", country="EC"))
        # Should find Loja in Ecuador
        if not df.empty:
            assert "source" in df.columns

    def test_similar_names_different_country(self, con):
        df = search_divisions_area(con, Place(place="Manchester"))
        # Should return results (multiple Manchesters exist)
        assert isinstance(df, pd.DataFrame)

    def test_include_geometry_flag(self, con):
        df = search_divisions_area(
            con, Place(place="India"), include_geometry=True, limit=1
        )
        if not df.empty:
            assert "geometry" in df.columns

    def test_include_bbox_flag(self, con):
        df = search_divisions_area(
            con, Place(place="India"), include_bbox=True, limit=1
        )
        if not df.empty:
            assert "bbox" in df.columns

    def test_column_presence(self, con):
        df = search_divisions_area(con, Place(place="India"), limit=1)
        if not df.empty:
            expected_cols = ["source", "id", "name", "country", "subtype", "similarity"]
            for col in expected_cols:
                assert col in df.columns


class TestSearchNaturalEarth:
    def test_returns_dataframe(self, con):
        df = search_natural_earth(con, Place(place="Nile"))
        assert isinstance(df, pd.DataFrame)

    def test_searches_ocean(self, con):
        df = search_natural_earth(con, Place(place="Pacific"))
        assert isinstance(df, pd.DataFrame)

    def test_source_column(self, con):
        df = search_natural_earth(con, Place(place="Nile"))
        if not df.empty:
            assert (df["source"] == "natural_earth").all()

    def test_empty_result(self, con):
        df = search_natural_earth(con, Place(place="Xyzz98765"))
        # Jaro-Winkler always returns top 5, similarity is very low
        assert len(df) <= 5

    def test_limits_results(self, con):
        df = search_natural_earth(con, Place(place="a"), limit=2)
        assert len(df) <= 2

    def test_include_geometry(self, con):
        df = search_natural_earth(con, Place(place="Nile"), include_geometry=True)
        if not df.empty:
            assert "geometry" in df.columns


class TestSearchCandidates:
    def test_searches_both_sources(self, con):
        results = search_candidates(con, Place(place="India"))
        # Should get at least one source
        assert len(results) >= 1
        sources = {df["source"].iloc[0] for df in results if not df.empty}
        assert "divisions_area" in sources

    def test_restricts_to_single_source(self, con):
        results = search_candidates(
            con,
            Place(place="Nile"),
            sources=("natural_earth",),
        )
        for df in results:
            if not df.empty:
                assert df["source"].iloc[0] == "natural_earth"

    def test_empty_place(self, con):
        # Even random strings get fuzzy matches — check results exist
        # but have low similarity
        results = search_candidates(con, Place(place="Xyzz98765"))
        # Results may be non-empty due to fuzzy matching, but similarity low
        for df in results:
            assert len(df) <= 5

    def test_combine_multiple_places(self, con):
        # Search for a place that exists in both sources
        results = search_candidates(
            con,
            Place(place="Brazil"),
            sources=("divisions_area", "natural_earth"),
        )
        for df in results:
            assert isinstance(df, pd.DataFrame)


class TestGetById:
    def test_get_division_by_id(self, con):
        # First get a valid ID from search
        df = search_divisions_area(con, Place(place="India"), limit=1)
        if not df.empty:
            rid = df["id"].iloc[0]
            result = get_division_by_id(con, rid)
            assert not result.empty
            assert result["id"].iloc[0] == rid

    def test_get_natural_earth_by_id(self, con):
        df = search_natural_earth(con, Place(place="Nile"), limit=1)
        if not df.empty:
            rid = df["id"].iloc[0]
            result = get_natural_earth_by_id(con, rid)
            assert not result.empty
            assert result["id"].iloc[0] == rid

    def test_get_by_id_auto_infer_divisions(self, con):
        df = search_divisions_area(con, Place(place="India"), limit=1)
        if not df.empty:
            rid = df["id"].iloc[0]
            result = get_by_id(con, rid)
            assert not result.empty

    def test_get_by_id_auto_infer_natural_earth(self, con):
        df = search_natural_earth(con, Place(place="Nile"), limit=1)
        if not df.empty:
            rid = df["id"].iloc[0]
            # NE IDs start with "ne_"
            assert rid.startswith("ne_")
            result = get_by_id(con, rid)
            assert not result.empty

    def test_get_by_id_invalid(self, con):
        result = get_by_id(con, "nonexistent_id_999")
        assert result.empty

    def test_get_by_id_with_source(self, con):
        df = search_divisions_area(con, Place(place="India"), limit=1)
        if not df.empty:
            rid = df["id"].iloc[0]
            result = get_by_id(con, rid, source="divisions_area")
            assert not result.empty
            assert result["source"].iloc[0] == "divisions_area"


class TestEnglishNames:
    """A place is reachable by its English name as well as its own.

    Fuzzy search matched only ``names.primary``, so "Copenhagen" scored
    against "Københavns Kommune" and lost to unrelated places that happen to
    start with "Cop-"; the English name was on the record the whole time.
    """

    def test_english_exonym_finds_the_place(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Copenhagen"), limit=3)
        assert df.iloc[0]["id"] == "cph"

    def test_local_name_still_finds_the_place(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="København"), limit=3)
        assert df.iloc[0]["id"] == "cph"

    def test_the_two_names_agree_on_one_record(self, con, exonym_source):
        english = search_divisions_area(con, Place(place="Copenhagen"), limit=1)
        local = search_divisions_area(con, Place(place="København"), limit=1)
        assert english.iloc[0]["id"] == local.iloc[0]["id"]

    def test_the_wrong_place_no_longer_wins(self, con, exonym_source):
        # "Coper" (Colombia) is what the top hit for "Copenhagen" used to be.
        df = search_divisions_area(con, Place(place="Copenhagen"), limit=5)
        assert df.iloc[0]["id"] != "coper"

    def test_geometry_is_the_right_one(self, con, exonym_source):
        df = search_divisions_area(
            con, Place(place="Copenhagen"), limit=1, include_bbox=True
        )
        # Denmark is east of Greenwich; the county this used to resolve to is
        # in the Americas, and a bbox is the only part of the answer a caller
        # passing it on to a data request ever reads.
        assert df.iloc[0]["bbox"][0] > 0

    def test_matched_name_reports_which_name_hit(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="København"), limit=1)
        assert df.iloc[0]["matched_name"] == "Københavns Kommune"
        assert df.iloc[0]["name"] == "Copenhagen Municipality"

    def test_record_without_an_english_name_still_matches(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Coper"), limit=1)
        assert df.iloc[0]["id"] == "coper"
        assert df.iloc[0]["name"] == "Coper"
        assert df.iloc[0]["similarity"] == 1.0

    def test_record_with_only_an_english_name_is_reachable(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Ratnapura"), limit=1)
        assert df.iloc[0]["id"] == "ratnapura"

    def test_blank_english_name_is_not_a_name(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Genève"), limit=1)
        assert df.iloc[0]["name"] == "Genève"

    def test_search_and_fetch_agree_on_the_name(self, con, exonym_source):
        found = search_divisions_area(con, Place(place="Copenhagen"), limit=1)
        fetched = get_division_by_id(con, found.iloc[0]["id"], include_geometry=False)
        assert fetched.iloc[0]["name"] == found.iloc[0]["name"]

    def test_exact_english_match_scores_one(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Munich"), limit=1)
        assert df.iloc[0]["similarity"] == 1.0
        assert bool(df.iloc[0]["is_substring_match"]) is True

    def test_a_miss_scores_low(self, con, exonym_source):
        df = search_divisions_area(con, Place(place="Xyzz98765"), limit=1)
        assert df.iloc[0]["similarity"] < 0.5
        assert bool(df.iloc[0]["is_substring_match"]) is False
