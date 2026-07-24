from pydantic import BaseModel


class Place(BaseModel):
    place: str


class PlacesResult(BaseModel):
    places: list[Place]
