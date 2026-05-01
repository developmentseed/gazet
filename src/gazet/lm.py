import json
import logging
import time

import dspy
import httpx
import pandas as pd

from .config import (
    LLAMA_MAX_TOKENS,
    LLAMA_SERVER_URL,
    LLAMA_TEMPERATURE,
    PLACE_EXTRACTION_MODEL,
    SQL_GENERATION_MODEL,
)
from .schemas import Place, PlacesResult

logger = logging.getLogger(__name__)


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

    Only extract place names that are explicitly mentioned in the query.
    Do NOT generate or infer place names from your own knowledge.
    For example:
    - "north half of India" -> extract "India", NOT individual state names
    - "coastal cities of France" -> extract "France", NOT city names
    - "neighbouring states of Odisha" -> extract "Odisha", NOT neighbouring state names

    Do not repeat the same place name in the result.
    Return only the place names, in the order they appear in the query.
    """

    query: str = dspy.InputField(
        desc="Natural language query mentioning one or more place names"
    )
    result: PlacesResult = dspy.OutputField(
        desc="Extracted place names in query order"
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


place_extraction_lm = dspy.LM(
    f"ollama_chat/{PLACE_EXTRACTION_MODEL}", 
    api_base="http://localhost:11434", 
    api_key="", 
    temperature=0.1, 
    cache=False,
)

sql_generation_lm = dspy.LM(
    f"ollama_chat/{SQL_GENERATION_MODEL}", 
    api_base="http://localhost:11434", 
    api_key="", 
    temperature=0.1, 
    cache=False,
    think=False
)


class PlaceExtractor(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.Predict(ExtractPlaces)
    
    def forward(self, query: str):
        with dspy.context(lm=self.lm):
            return self.predictor(query=query)


class SQLWriter(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.predictor = dspy.Predict(WriteGeoSQL)
    
    def forward(self, user_query: str, schema: str, candidates: str, 
                previous_sql: str = "", execution_error: str = ""):
        with dspy.context(lm=self.lm):
            return self.predictor(
                user_query=user_query,
                schema=schema,
                candidates=candidates,
                previous_sql=previous_sql,
                execution_error=execution_error
            )


extract = PlaceExtractor(lm=place_extraction_lm)
write_sql = SQLWriter(lm=sql_generation_lm)


# ── GGUF SQL generation via llama-server ──────────────────────────────────────

_SYSTEM_PROMPT = """You are a text to SQL query translator that helps in natural language geocoding.

You have access to two DuckDB parquet tables. Given a set of candidate entities and a user query, generate the SQL to retrieve the desired geometry.

<SCHEMA>
1. divisions_area  -- Overture polygon/multipolygon admin boundaries
   query: read_parquet('divisions_area')
   columns:
     id VARCHAR              -- unique feature id
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | county
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     division_id VARCHAR
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)

2. natural_earth  -- Natural Earth geography polygons (oceans, seas, rivers, terrain)
   query: read_parquet('natural_earth')
   columns:
     id VARCHAR              -- unique feature id prefixed 'ne_'
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'range/mtn', 'island group'
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)
</SCHEMA>

The candidates table has a 'source' column: 'divisions_area' or 'natural_earth'.
Use read_parquet('divisions_area') or read_parquet('natural_earth') accordingly.
Use ST_AsGeoJSON(geometry) for all geometry outputs."""

_USER_PROMPT_TEMPLATE = """<CANDIDATES>
{candidates_csv}
</CANDIDATES>

<USER_QUERY>
{question}
</USER_QUERY>
"""


def _postprocess_sql(text: str) -> str:
    """Strip markdown fences and whitespace from generated SQL."""
    cleaned = text.strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()


def is_llama_server_available() -> bool:
    """Check if the llama-server is running and healthy."""
    try:
        resp = httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _llama_chat_complete(messages: list[dict]) -> str:
    """Call llama-server /v1/chat/completions, waiting through cold starts."""
    payload = {
        "messages": messages,
        "n_predict": LLAMA_MAX_TOKENS,
        "temperature": LLAMA_TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    deadline = time.monotonic() + 120.0
    backoff = 2.0
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=90,
            )
            if resp.status_code == 503:
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 8.0)
                continue
            if resp.status_code != 200:
                logger.error("llama-server %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 8.0)
    raise RuntimeError(f"llama-server did not become ready within 120s: {last_err}")


_PLACES_SYSTEM_PROMPT = """You are a geographic entity extractor. Extract the place names the user is asking about and return valid JSON only.

OUTPUT FORMAT:
{"places": [{"place": "<name>"}]}

RULES:
- Extract the place or places that are the actual anchors of the query.
- Physical features are valid places: oceans, seas, gulfs, bays, straits, rivers, lakes, basins, mountain ranges, peninsulas, island groups, deserts, and terrain regions.
- When a place is followed by its containing region, state, or country as disambiguation context ("Puri, Odisha", "Lisboa, Portugal", "Goa, India", "Manchester in US"), extract ONLY the specific place. Do not return the container as a separate place.
- When a query names two or more distinct anchors joined by words like "and", "both", "between", or mixes an admin area with a physical feature as separate anchors, extract every anchor in the order they appear.
- Do not infer or expand category nouns like "regions", "districts", "counties", "rivers", or "mountains" when they refer to a type rather than a specific named place ("regions of India" -> extract "India" only).
- Only extract places explicitly mentioned.
- No duplicate place names.

EXAMPLES:
Query: "Puri, Odisha"
-> {"places": [{"place": "Puri"}]}

Query: "Lisboa, Portugal"
-> {"places": [{"place": "Lisboa"}]}

Query: "Goa, India"
-> {"places": [{"place": "Goa"}]}

Query: "Manchester in US"
-> {"places": [{"place": "Manchester"}]}

Query: "Springfield, Illinois"
-> {"places": [{"place": "Springfield"}]}

Query: "coastal districts of Brazil"
-> {"places": [{"place": "Brazil"}]}

Query: "northern half of India"
-> {"places": [{"place": "India"}]}

Query: "what's within 50 km of Paris?"
-> {"places": [{"place": "Paris"}]}

Query: "countries the Nile crosses"
-> {"places": [{"place": "Nile"}]}

Query: "which countries touch the Gulf of Maine"
-> {"places": [{"place": "Gulf of Maine"}]}

Query: "10 km buffer around Odisha"
-> {"places": [{"place": "Odisha"}]}

Query: "part of Ecuador in the Amazon basin"
-> {"places": [{"place": "Ecuador"}, {"place": "Amazon basin"}]}

Query: "Amazon basin inside Ecuador"
-> {"places": [{"place": "Amazon basin"}, {"place": "Ecuador"}]}

Query: "the part of Chad in Lake Chad"
-> {"places": [{"place": "Chad"}, {"place": "Lake Chad"}]}

Query: "which regions border both France and Germany?"
-> {"places": [{"place": "France"}, {"place": "Germany"}]}

Query: "merge Nairobi and Mombasa"
-> {"places": [{"place": "Nairobi"}, {"place": "Mombasa"}]}"""


def generate_places(user_query: str) -> PlacesResult:
    """Extract place names from a query using the finetuned GGUF model.

    Uses the same prompt format the model was trained on.
    Returns a PlacesResult; falls back to an empty result on parse failure.
    """
    messages = [
        {"role": "system", "content": _PLACES_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    raw_output = _llama_chat_complete(messages).strip()

    # Strip markdown fences if the model wrapped the JSON
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    try:
        data = json.loads(raw_output)
        return PlacesResult.model_validate(data)
    except Exception as exc:
        logger.warning("generate_places: failed to parse output %r: %s", raw_output, exc)
        # Best-effort: treat entire query as a single unnamed place
        return PlacesResult(places=[Place(place=user_query)])


def generate_sql(user_query: str, candidates_df: pd.DataFrame) -> str:
    """Generate SQL from a natural language query using the finetuned GGUF model.

    Uses the same prompt format the model was trained on:
    SYSTEM_PROMPT (includes schema) + USER_PROMPT_TEMPLATE with candidates CSV and question.
    Single-shot — no retry loop (the finetuned model can't improve from error feedback).
    """
    # Keep only columns the model was trained on
    keep_cols = ["source", "id", "name", "subtype", "country", "region", "admin_level"]
    cols = [c for c in keep_cols if c in candidates_df.columns]
    candidates_csv = candidates_df[cols].to_csv(index=False)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        candidates_csv=candidates_csv.strip(),
        question=user_query.strip(),
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw_output = _llama_chat_complete(messages)
    return _postprocess_sql(raw_output)
