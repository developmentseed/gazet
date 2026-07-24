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
