import dspy

from .config import MODEL
from .schemas import PlacesResult


class ExtractPlaces(dspy.Signature):
    """Extract place names from a query.

    Data is available from overture and natural earth datasets.

    Overture has divisions and natural earth has physical features.
    - divisions are administrative units like countries, states, counties, cities, towns, villages, etc.
    - physical features are natural features like oceans, seas, lakes, rivers, mountains, etc.

    When extracting a place name, you can use the overture divisions or natural earth physical features.
    - If the user mentions an overture division, use the overture divisions.
    - If the user mentions a natural earth physical feature, use the natural earth physical features.
    - If the user mentions a place name that is not in the overture divisions or natural earth physical features, return the place name as is.

    Where possible and relevant, also extract the ISO country code for each place.

    Do not repeat the same place name in the result.

    If the user does not explicitly mention a country, dont add the country code to the result.

    If the user does not mention an admin level, dont add the subtype to the result.

    If the query asks for some kind of subdivision (e.g. 'municipalities in Bern', 'States in Brazil'),
    return the subdivision type in the places result.

    When identifying a place name from the user's query, also infer the most appropriate
    Overture division subtype from the list below. Only include a subtype if the query
    makes it reasonably clear what geographic level is intended. If ambiguous, omit it.

    SUBTYPES:
    - country      : Sovereign nation. E.g. "France", "Brazil"
    - dependency   : Territory dependent on a country but not a full sub-region. E.g. "Puerto Rico", "Guam"
    - region       : Largest admin unit within a country; state, province, canton, etc. E.g. "California", "Alberta", "Bavaria"
    - county       : Second-level admin subdivision within a region. E.g. "Kings County", "Kent"
    - localadmin   : A governing layer (common in Europe) that contains localities which have no authority of their own. E.g. a French commune or Belgian municipality. Use when the place is clearly an admin unit but not a city itself.
    - locality     : A populated place — city, town, village. The most common subtype for named settlements. E.g. "Lisbon", "Taipei", "Salt Lake City"
    - macrohood    : A large super-neighborhood grouping smaller neighborhoods. E.g. "BoCoCa" in Brooklyn
    - neighborhood : A named community area within a city or town. E.g. "Cobble Hill", "Alfama"
    - microhood    : A mini-neighborhood within a neighborhood. Very fine-grained, rarely referenced explicitly.

    HIERARCHY (coarse to fine):
    country → dependency / region → county → localadmin → locality → macrohood → neighborhood → microhood

    GUIDANCE:
    - "Paris" with no qualifier → locality
    - "Île-de-France" or "Catalonia" → region
    - "the 11th arrondissement" → neighborhood (or localadmin)
    - "Greater London" style phrasing → county or region depending on context
    - If the user says "neighborhood in X" or "district of X" → neighborhood
    - Default to locality for any named city/town if unsure
    - Omit subtype entirely if the query gives no signal (e.g. bare coordinates or a POI name)
    """

    query: str = dspy.InputField(
        desc="Natural language query mentioning one or more place names"
    )
    result: PlacesResult = dspy.OutputField(
        desc="Extracted places with optional country codes and optional subtype"
    )


class WriteGeoSQL(dspy.Signature):
    """Write a DuckDB SQL SELECT query that extracts the geometry answering a geo query.

    You are given:
    - The user's original natural language query
    - A schema description of the available Overture divisions parquet datasets
    - A table of fuzzy-matched candidate divisions with their IDs and metadata

    Write a single, read-only DuckDB SQL SELECT statement that returns the geometry
    (and key attributes like name, subtype, country) that best answers the query.
    Use candidate IDs from the candidates table to filter precisely — avoid full scans
    when you have exact IDs. The spatial extension is already loaded; use
    ST_AsGeoJSON(geometry) for geometry output.

    The user might ask for GIS operations such as intersections, buffering, or
    sections of geometries. You can use the spatial extension to perform these operations.

    Return ONLY the SQL — no markdown fences, no explanation.
    """

    user_query: str = dspy.InputField(desc="Original natural language geo query")
    schema: str = dspy.InputField(
        desc="Available datasets, column types, and example patterns"
    )
    candidates: str = dspy.InputField(
        desc="Fuzzy-matched candidate divisions (id, name, country, subtype, similarity, ...)"
    )
    previous_sql: str = dspy.InputField(
        desc="SQL from the previous attempt — empty string if this is the first try"
    )
    execution_error: str = dspy.InputField(
        desc="Error raised by the previous SQL — empty string if no error yet"
    )
    sql: str = dspy.OutputField(
        desc="Valid read-only DuckDB SQL SELECT statement, no markdown fences"
    )


lm = dspy.LM(
    f"ollama_chat/{MODEL}", api_base="http://localhost:11434", api_key="", temperature=0.1, cache=False,
)
dspy.configure(lm=lm)

extract = dspy.Predict(ExtractPlaces)
write_sql = dspy.Predict(WriteGeoSQL)
