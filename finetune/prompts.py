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
   path: '/data/overture/division_area/*.parquet'
   columns:
     id VARCHAR
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR
     subtype VARCHAR
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     division_id VARCHAR
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY

2. natural_earth  -- Natural Earth geography polygons
   path: '/data/natural_earth_geoparquet/ne_geography.parquet'
   columns:
     id VARCHAR
     name VARCHAR
     featurecla VARCHAR
     scalerank INTEGER
     min_zoom DOUBLE
     geometry GEOMETRY"""


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
