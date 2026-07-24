from typing import Any, Literal

from pydantic import BaseModel, Field


class Place(BaseModel):
    place: str
    country: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code (if known)",
    )
    subtype: str | None = Field(
        default=None,
        description="Administrative type: country, region, county, physical feature",
    )


class PlacesResult(BaseModel):
    places: list[Place]


class Feature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict[str, Any] | None = Field(
        default=None, description="GeoJSON geometry object"
    )
    properties: dict[str, Any] = Field(default_factory=dict)


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature] = Field(default_factory=list)


class FuzzyIdItem(BaseModel):
    """One candidate from a ``mode=fuzzy&ids_only=true`` search.

    ``country``/``subtype``/``admin_level`` disambiguate same-named places
    (e.g. multiple real-world "Loja"s across Ecuador and Spain, or Ecuador's
    "Loja" region vs. its nested "Loja" county). ``bbox`` is
    ``[minx, miny, maxx, maxy]``, a much smaller payload than full geometry —
    fetch the full geometry for one candidate via ``GET /geometry/{id}``.
    """

    source: str
    id: str
    name: str | None = None
    country: str | None = None
    subtype: str | None = None
    admin_level: int | None = None
    bbox: list[float] | None = None


class FuzzyIdsResult(BaseModel):
    geojson: FeatureCollection
    ids: list[FuzzyIdItem]


class NLSearchResult(BaseModel):
    """Result of a ``mode=nl`` (natural-language) search."""

    geojson: FeatureCollection
    sql: str = Field(description="The SQL query executed to produce the result")
    places: dict[str, Any] = Field(
        description="Place names extracted from the query by the LLM"
    )
    dataframes: dict[str, Any] = Field(
        description="Intermediate dataframes (e.g. fuzzy-matched candidates)"
    )


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    duckdb: Literal["ok", "error"]
    llama_server: Literal["ok", "unavailable"]


class SourceInfo(BaseModel):
    path: str
    row_count: int | None = None
    name_range: list[str | None] | None = None
    error: str | None = None
