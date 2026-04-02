import re
from typing import Any, Generator, Optional

import duckdb
import pandas as pd

from .config import DIVISIONS_AREA_PATH, MAX_SQL_ITERATIONS, NATURAL_EARTH_PATH, SCHEMA_INFO
from .lm import generate_sql, write_sql


def _rewrite_data_paths(sql: str) -> str:
    """Replace training-time /data/ paths with local data paths."""
    sql = sql.replace(
        "/data/overture/division_area/*.parquet", DIVISIONS_AREA_PATH
    )
    sql = sql.replace(
        "/data/overture/divisions_area/*.parquet", DIVISIONS_AREA_PATH
    )
    sql = sql.replace(
        "/data/natural_earth_geoparquet/ne_geography.parquet", NATURAL_EARTH_PATH
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

    candidates_str = candidates_df.to_string(index=False)
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
