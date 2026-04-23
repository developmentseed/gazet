import json
import re
from typing import Any, Generator, Optional


_CANDIDATE_PROMPT_COLS = [
    "source",
    "id",
    "name",
    "subtype",
    "country",
    "region",
    "admin_level",
]

import duckdb
import pandas as pd
from shapely import wkb
from shapely.geometry import mapping

from .config import DIVISIONS_AREA_PATH, MAX_SQL_ITERATIONS, NATURAL_EARTH_PATH, SCHEMA_INFO
from .lm import generate_sql, write_sql


def _needs_geometry_normalization(result_df: pd.DataFrame) -> bool:
    """Return True when `geometry` is a binary blob likely from DuckDB GEOMETRY."""
    if "geometry" not in result_df.columns or result_df.empty:
        return False
    sample = result_df["geometry"].dropna().head(5)
    if sample.empty:
        return False
    return sample.apply(lambda v: isinstance(v, (bytes, bytearray, memoryview))).all()


def _normalize_geometry_to_geojson(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """If result has binary geometry, convert it to GeoJSON text in Python."""
    if not _needs_geometry_normalization(result_df):
        return result_df

    def _blob_to_geojson_text(val: Any) -> Optional[str]:
        if not isinstance(val, (bytes, bytearray, memoryview)):
            return val
        try:
            geom = wkb.loads(bytes(val))
            return json.dumps(mapping(geom))
        except Exception:
            return None

    normalized_df = result_df.copy()
    normalized_df["geometry"] = normalized_df["geometry"].apply(_blob_to_geojson_text)
    return normalized_df


def _rewrite_data_paths(sql: str) -> str:
    """Replace any read_parquet table reference with the correct runtime path.

    Handles three generations of model output:
      - Symbolic:  read_parquet('divisions_area')
      - Old paths: read_parquet('/data/overture/division_area/...')
      - Hallucinated variants: any quoted path containing 'division' or 'natural_earth'

    Legacy replacements run FIRST so the absolute path is never re-matched.
    """
    # Any quoted path that looks like a divisions_area reference
    sql = re.sub(
        r"read_parquet\(['\"][^'\"]*(?:division_area|divisions_area)[^'\"]*['\"]\)",
        f"read_parquet('{DIVISIONS_AREA_PATH}')",
        sql,
    )
    # Any quoted path that looks like a natural_earth reference
    sql = re.sub(
        r"read_parquet\(['\"][^'\"]*natural_earth[^'\"]*['\"]\)",
        f"read_parquet('{NATURAL_EARTH_PATH}')",
        sql,
    )
    # Symbolic names (current training format)
    sql = sql.replace(
        "read_parquet('divisions_area')", f"read_parquet('{DIVISIONS_AREA_PATH}')"
    )
    sql = sql.replace(
        "read_parquet('natural_earth')", f"read_parquet('{NATURAL_EARTH_PATH}')"
    )
    return sql


# Title-cased NE subtype literals the trained model may emit.
# Data is now fully lowercased, so we normalise at query time.
_NE_SUBTYPE_FIXES = {
    "'River'": "'river'",
    "'Lake'": "'lake'",
    "'Basin'": "'basin'",
    "'Range/mtn'": "'range/mtn'",
    "'Peninsula'": "'peninsula'",
    "'Depression'": "'depression'",
    "'Island group'": "'island group'",
    "'Ocean'": "'ocean'",
    "'Sea'": "'sea'",
}

_TERRAIN_AREA_PATTERN = re.compile(
    r"n\.subtype\s*(=|IN\s*\()\s*'Terrain area'\s*\)?",
    flags=re.IGNORECASE,
)


def _normalize_ne_subtypes(sql: str) -> str:
    """Lowercase known NE subtype literals and fix common terrain hallucinations."""
    for old, new in _NE_SUBTYPE_FIXES.items():
        sql = sql.replace(old, new)

    sql = _TERRAIN_AREA_PATTERN.sub(
        "n.subtype IN ('range/mtn', 'peninsula', 'depression')",
        sql,
    )
    return sql


def _strip_fences(sql: Optional[str]) -> str:
    """Remove markdown code fences that the LM may wrap the SQL in."""
    if not sql:
        return ""
    sql = re.sub(r"^```\w*\s*\n?", "", sql.strip())
    sql = re.sub(r"\n?```\s*$", "", sql)
    return sql.strip()


def _execute_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    label: str,
    iteration: int,
) -> Generator[dict[str, Any], None, None]:
    """Execute SQL and yield result/error events. Shared by both paths."""
    try:
        result_df = con.execute(sql).fetchdf()
        result_df = _normalize_geometry_to_geojson(result_df)
        if result_df.empty:
            print(f"[{label}] Query returned no rows.")
            yield {"type": "sql_error", "error": "Query returned no rows", "iteration": iteration}
            yield {"type": "result", "df": None, "sql": sql}
        else:
            print(f"[{label}] Result ({len(result_df)} row(s))")
            yield {"type": "result", "df": result_df, "sql": sql}
    except Exception as exc:
        error = str(exc)
        print(f"[{label}] Execution error: {error}")
        yield {"type": "sql_error", "error": error, "iteration": iteration}
        yield {"type": "result", "df": None, "sql": sql}


# ── GGUF path: finetuned model via llama-server (single-shot) ─────────────────


def run_geo_sql_gguf(
    con: duckdb.DuckDBPyConnection,
    user_query: str,
    candidates_df: pd.DataFrame,
) -> Generator[dict[str, Any], None, None]:
    """Single-shot text-to-SQL via the finetuned GGUF model (llama-server).

    Event types:
    - ``sql_attempt``  – ``{"type": "sql_attempt", "sql": str, "iteration": int}``
    - ``sql_error``    – ``{"type": "sql_error", "error": str, "iteration": int}``
    - ``result``       – ``{"type": "result", "df": DataFrame | None, "sql": str}``
    """
    if candidates_df.empty:
        print("\n[SQL·GGUF] No candidates to work with — skipping.")
        yield {"type": "result", "df": None, "sql": ""}
        return

    try:
        sql = generate_sql(user_query, candidates_df)
    except Exception as exc:
        error = f"GGUF generation failed: {exc}"
        print(f"[SQL·GGUF] {error}")
        yield {"type": "sql_error", "error": error, "iteration": 1}
        yield {"type": "result", "df": None, "sql": ""}
        return

    if not sql:
        print("[SQL·GGUF] Model returned empty SQL.")
        yield {"type": "sql_error", "error": "Empty SQL response", "iteration": 1}
        yield {"type": "result", "df": None, "sql": ""}
        return

    sql = _rewrite_data_paths(sql)
    sql = _normalize_ne_subtypes(sql)
    print(f"\n[SQL·GGUF] Generated:\n{sql}\n")
    yield {"type": "sql_attempt", "sql": sql, "iteration": 1}
    yield from _execute_sql(con, sql, "SQL·GGUF", iteration=1)


# ── DSPy path: cloud/local LM with retry loop ────────────────────────────────


def run_geo_sql_dspy(
    con: duckdb.DuckDBPyConnection,
    user_query: str,
    candidates_df: pd.DataFrame,
    max_iterations: int = MAX_SQL_ITERATIONS,
) -> Generator[dict[str, Any], None, None]:
    """Code-act retry loop using the DSPy SQL writer (Ollama / cloud LM).

    Same event types as ``run_geo_sql_gguf``.
    """
    if candidates_df.empty:
        print("\n[SQL·DSPy] No candidates to work with — skipping.")
        yield {"type": "result", "df": None, "sql": ""}
        return

    cols = [c for c in _CANDIDATE_PROMPT_COLS if c in candidates_df.columns]
    candidates_str = candidates_df[cols].to_string(index=False)
    previous_sql = ""
    error = ""

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"[SQL·DSPy] Iteration {iteration}/{max_iterations}")

        try:
            pred = write_sql(
                user_query=user_query,
                schema=SCHEMA_INFO,
                candidates=candidates_str,
                previous_sql=previous_sql,
                execution_error=error,
            )
            sql = _strip_fences(pred.sql)
            sql = _rewrite_data_paths(sql)
            sql = _normalize_ne_subtypes(sql)
        except Exception as exc:
            error = f"LM generation failed: {exc}"
            print(f"Generation error: {error}")
            yield {"type": "sql_error", "error": error, "iteration": iteration}
            continue

        if not sql:
            error = "LM returned an empty SQL response."
            print(f"Generation error: {error}")
            yield {"type": "sql_error", "error": error, "iteration": iteration}
            continue

        print(f"\nGenerated SQL:\n{sql}\n")
        yield {"type": "sql_attempt", "sql": sql, "iteration": iteration}

        try:
            result_df = con.execute(sql).fetchdf()
            result_df = _normalize_geometry_to_geojson(result_df)
            if result_df.empty:
                error = "The query executed successfully but returned no rows. Revise the query to return at least one result."
                previous_sql = sql
                print(f"Empty result: {error}")
                yield {"type": "sql_error", "error": error, "iteration": iteration}
                continue
            print(f"Result ({len(result_df)} row(s)):")
            print(result_df.to_string(index=False, max_colwidth=120))
            yield {"type": "result", "df": result_df, "sql": sql}
            return
        except Exception as exc:
            error = str(exc)
            previous_sql = sql
            print(f"Execution error: {error}")
            yield {"type": "sql_error", "error": error, "iteration": iteration}

    print(
        f"\n[SQL·DSPy] Exhausted {max_iterations} iterations without a successful query."
    )
    yield {"type": "result", "df": None, "sql": ""}
