"""Prompt templates and message formatting for natural language geocoding."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

SYSTEM_PROMPT = """You are a text to SQL query translator that helps in natural language geocoding.

You have access to two DuckDB parquet tables. Given a set of candidate entities and a user query, generate the SQL to retrieve the desired geometry.

<SCHEMA>
1. divisions_area  -- Overture polygon/multipolygon admin boundaries
   query: read_parquet('divisions_area')
   columns:
     id VARCHAR              -- unique feature id
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | dependency | county | localadmin |
                               locality | macrohood | neighborhood | microhood
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
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'Terrain area', 'Island group'
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

USER_PROMPT_TEMPLATE = """<CANDIDATES>
{candidates_csv}
</CANDIDATES>

<USER_QUERY>
{question}
</USER_QUERY>
"""


def candidates_to_csv(candidates: Sequence[Dict[str, Any]]) -> str:
    df = pd.DataFrame(list(candidates))
    if "candidate_id" in df.columns:
        df = df.drop(columns=["candidate_id"])
    return df.to_csv(index=False)


def build_user_prompt(
    question: str,
    candidates: Sequence[Dict[str, Any]],
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        candidates_csv=candidates_to_csv(candidates).strip(),
        question=question.strip(),
    )


def make_prompt_completion(
    sample: Dict[str, Any],
) -> Dict[str, str]:
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(
        question=sample["question"],
        candidates=sample["candidates"],
    )
    completion = sample.get("target", {}).get("sql", "")
    return {"prompt": prompt, "completion": completion}
