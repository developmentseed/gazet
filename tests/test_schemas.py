"""Tests for gazet.schemas — Pydantic models."""

from gazet.schemas import Place, PlacesResult


class TestPlace:
    def test_minimal(self):
        p = Place(place="Paris")
        assert p.place == "Paris"
        assert p.country is None
        assert p.subtype is None

    def test_all_fields(self):
        p = Place(place="Odisha", country="IN", subtype="region")
        assert p.place == "Odisha"
        assert p.country == "IN"
        assert p.subtype == "region"

    def test_default_none_explicit(self):
        p = Place(place="Loja", country=None, subtype=None)
        assert p.country is None
        assert p.subtype is None

    def test_model_dump(self):
        p = Place(place="Brazil", country="BR")
        d = p.model_dump()
        assert d["place"] == "Brazil"
        assert d["country"] == "BR"
        assert d["subtype"] is None

    def test_rejects_missing_place(self):
        import pytest
        with pytest.raises(Exception):  # pydantic validation error
            Place()  # pragma: no cover


class TestPlacesResult:
    def test_single_place(self):
        r = PlacesResult(places=[Place(place="India")])
        assert len(r.places) == 1
        assert r.places[0].place == "India"

    def test_multiple_places(self):
        r = PlacesResult(
            places=[
                Place(place="Ecuador"),
                Place(place="Amazon basin"),
            ]
        )
        assert len(r.places) == 2
        assert r.places[0].place == "Ecuador"
        assert r.places[1].place == "Amazon basin"

    def test_empty(self):
        r = PlacesResult(places=[])
        assert len(r.places) == 0

    def test_model_validate_from_dict(self):
        data = {
            "places": [
                {"place": "Chad"},
                {"place": "Lake Chad", "country": "TD", "subtype": "lake"},
            ]
        }
        r = PlacesResult.model_validate(data)
        assert len(r.places) == 2
        assert r.places[1].country == "TD"
        assert r.places[1].subtype == "lake"

    def test_places_are_place_instances(self):
        r = PlacesResult(places=[Place(place="Nairobi")])
        assert isinstance(r.places[0], Place)
