import re
from typing import Any, Generator, Optional

import duckdb
import pandas as pd

from .config import MAX_SQL_ITERATIONS, SCHEMA_INFO
from .lm import write_sql


def _strip_fences(sql: Optional[str]) -> str:
    """Remove markdown code fences that the LM may wrap the SQL in."""
    if not sql:
        return ""
    sql = re.sub(r"^```\w*\s*\n?", "", sql.strip())
    sql = re.sub(r"\n?```\s*$", "", sql)
    return sql.strip()


def run_geo_sql_loop(
    con: duckdb.DuckDBPyConnection,
    user_query: str,
    candidates_df: pd.DataFrame,
    max_iterations: int = MAX_SQL_ITERATIONS,
) -> Generator[dict[str, Any], None, None]:
    """Code-act loop yielding progress events.

    Event types:
    - ``sql_attempt``  – ``{"type": "sql_attempt", "sql": str, "iteration": int}``
    - ``sql_error``    – ``{"type": "sql_error", "error": str, "iteration": int}``
    - ``result``       – ``{"type": "result", "df": DataFrame | None, "sql": str}``
    """
    if candidates_df.empty:
        print("\n[SQL-Act] No candidates to work with — skipping.")
        yield {"type": "result", "df": None, "sql": ""}
        return

    candidates_str = candidates_df.to_string(index=False)
    previous_sql = ""
    error = ""

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"[SQL-Act] Iteration {iteration}/{max_iterations}")

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
        f"\n[SQL-Act] Exhausted {max_iterations} iterations without a successful query."
    )
    yield {"type": "result", "df": None, "sql": ""}
