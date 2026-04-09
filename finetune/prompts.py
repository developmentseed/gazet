"""Prompt templates and message formatting for natural language geocoding."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

SYSTEM_PROMPT = (
    "You are a text to SQL query translator that helps in natural language geocoding."
)

USER_PROMPT_TEMPLATE = """GIVEN the <SCHEMA_DETAILS>, <CANDIDATES> and <USER_QUERY>, generate the corresponding SQL command to retrieve the desired geometry.

<SCHEMA_DETAILS>
{schema_details}
</SCHEMA_DETAILS>

<CANDIDATES>
{candidates_csv}
</CANDIDATES>

<USER_QUERY>
{question}
</USER_QUERY>
"""

DEFAULT_SCHEMA_DETAILS = """1. divisions_area  -- Overture polygon/multipolygon admin boundaries
   query: read_parquet('divisions_area')
   columns:
     id VARCHAR              -- unique feature id (use to filter precisely)
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | dependency | county | localadmin |
                               locality | macrohood | neighborhood | microhood
     class VARCHAR
     region VARCHAR          -- region code e.g. 'IN-OR'
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
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'Terrain area', 'Island group'
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)

The candidates table has a 'source' column: 'divisions_area' or 'natural_earth'.
Use read_parquet('divisions_area') or read_parquet('natural_earth') accordingly."""


def candidates_to_csv(candidates: Sequence[Dict[str, Any]]) -> str:
    df = pd.DataFrame(list(candidates))
    if "candidate_id" in df.columns:
        df = df.drop(columns=["candidate_id"])
    return df.to_csv(index=False)


def build_user_prompt(
    question: str,
    candidates: Sequence[Dict[str, Any]],
    schema_details: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        schema_details=schema_details.strip(),
        candidates_csv=candidates_to_csv(candidates).strip(),
        question=question.strip(),
    )


def make_prompt_completion(
    sample: Dict[str, Any],
    schema_details: str,
) -> Dict[str, str]:
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(
        question=sample["question"],
        candidates=sample["candidates"],
        schema_details=schema_details,
    )
    completion = sample.get("target", {}).get("sql", "")
    return {"prompt": prompt, "completion": completion}
