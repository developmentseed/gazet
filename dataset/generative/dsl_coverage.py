"""How much of the SQL corpus could a structured plan language express?

Each query is parsed with sqlglot and every construct it uses is mapped to a category of a draft plan language:

- tier 1, core plan: select/filter/join/nested steps (CTEs, subqueries in FROM, IN/EXISTS semi-joins), grouping
  and aggregates, order/limit/distinct, arithmetic and casts, struct field access, and the common spatial
  predicates and operations;
- tier 2, extensions: window functions (rank within group), conditional labels and aggregates (CASE/IF,
  FILTER), text and date functions, set operations, scalar subqueries, expanding a list column into rows,
  list membership, less common aggregates;
- tier 3, free SQL: list and map processing (lambdas, unnest, list functions), JSON, regular expressions, inline
  VALUES tables, struct construction, series generation and anything unmapped.

A query's tier is the highest tier of its constructs. For queries outside tier 1, `blocking` counts the categories
they need, and `only` the queries that a single category would bring into tier 1 or 2 on its own.

Usage: python dsl_coverage.py NAME=FILE.jsonl [NAME=FILE.jsonl ...]   (chat files; the SQL is the last message)
"""
import json
import sys
from collections import Counter

import sqlglot
from sqlglot import exp

SPATIAL_CORE = """ST_POINT ST_MAKEENVELOPE ST_INTERSECTS ST_WITHIN ST_CONTAINS ST_COVERS ST_COVEREDBY ST_TOUCHES ST_CROSSES
ST_OVERLAPS ST_DISJOINT ST_EQUALS ST_DWITHIN ST_DWITHIN_SPHEROID ST_DISTANCE ST_DISTANCE_SPHEROID ST_DISTANCE_SPHERE
ST_AREA ST_AREA_SPHEROID ST_LENGTH ST_LENGTH_SPHEROID ST_PERIMETER ST_PERIMETER_SPHEROID ST_CENTROID
ST_POINTONSURFACE ST_BUFFER ST_INTERSECTION ST_DIFFERENCE ST_UNION ST_UNION_AGG ST_EXTENT ST_EXTENT_AGG
ST_ENVELOPE ST_ENVELOPE_AGG ST_CONVEXHULL ST_TRANSFORM ST_X ST_Y ST_XMIN ST_XMAX ST_YMIN ST_YMAX ST_ASGEOJSON
ST_ASTEXT ST_ASWKB ST_GEOMFROMTEXT ST_GEOMFROMWKB ST_GEOMFROMGEOJSON ST_BOUNDARY ST_COLLECT ST_MAKELINE
ST_SIMPLIFY ST_EXPAND ST_ISEMPTY ST_GEOMETRYTYPE ST_NUMGEOMETRIES ST_NPOINTS ST_STARTPOINT ST_ENDPOINT
ST_CLOSESTPOINT ST_EXTERIORRING ST_POINTN ST_COLLECTIONEXTRACT READ_PARQUET""".split()
CORE_FUNCS = set(SPATIAL_CORE) | set("""AND OR NOT CAST TRY_CAST COUNT SUM AVG MIN MAX ROUND ABS FLOOR CEIL
COALESCE NULLIF LEAST GREATEST ARRAY_AGG EXISTS PI SQRT POWER LN LOG STRUCT_EXTRACT""".split())
TIER2_FUNCS = {
    "window": "ROW_NUMBER RANK DENSE_RANK NTILE LAG LEAD FIRST_VALUE LAST_VALUE PERCENT_RANK CUME_DIST".split(),
    "conditional": "CASE IF IIF".split(),
    "text": """LOWER UPPER TRIM LTRIM RTRIM LENGTH SUBSTRING SUBSTR LEFT RIGHT CONCAT PRINTF FORMAT STARTS_WITH
               ENDS_WITH CONTAINS STR_POSITION SPLIT_PART REPLACE STRIP_ACCENTS""".split(),
    "date": """EXTRACT YEAR MONTH DAY HOUR DATE_PART DATE_TRUNC TIMESTAMP_TRUNC DATEDIFF DATE_DIFF STR_TO_TIME
               TIME_TO_STR STRFTIME STRPTIME DATE_FROM_PARTS MONTHNAME DAYNAME TO_MINUTES EPOCH""".split(),
    "expand_list": "UNNEST EXPLODE".split(),
    "list_membership": """ARRAY_CONTAINS LIST_CONTAINS LIST_HAS LIST_HAS_ALL LIST_HAS_ANY ARRAY_OVERLAPS
               ARRAY_HAS_ANY""".split(),
    "aggregate_more": """MEDIAN PERCENTILE_CONT PERCENTILE_DISC QUANTILE_CONT STDDEV VARIANCE GROUP_CONCAT STRING_AGG
               LISTAGG ANY_VALUE ARG_MIN ARG_MAX LOGICAL_OR LOGICAL_AND MODE""".split(),
}
TIER3_FUNCS = {
    "lists": """SPLIT TRANSFORM LIST_FILTER LIST_TRANSFORM LIST_SORT LIST_VALUE LIST_DISTINCT ARRAY ARRAY_SORT ARRAY_DISTINCT
                LIST_SLICE LIST_EXTRACT LIST_COUNT LIST_AGGREGATE LIST_POSITION ARRAY_TO_STRING ARRAY_SIZE
                CARDINALITY ARRAY_MAX ARRAY_CONCAT MAP_KEYS MAP_ENTRIES MAP_EXTRACT_VALUE STRUCT""".split(),
    "json": """TO_JSON J_S_O_N_OBJECT J_S_O_N_ARRAY JSON_EXTRACT JSON_EXTRACT_SCALAR JSON_KEYS JSON_GROUP_OBJECT
               JSON_GROUP_ARRAY JSON_ARRAY_LENGTH""".split(),
    "regex": "REGEXP_EXTRACT REGEXP_LIKE REGEXP_REPLACE REGEXP_MATCHES".split(),
    "series": "GENERATE_SERIES RANGE".split(),
}
TIER = {c: 2 for c in TIER2_FUNCS} | {c: 3 for c in TIER3_FUNCS} | {
    "set_operation": 2, "scalar_subquery": 2, "text_match": 2, "inline_values": 3, "lateral": 3,
    "lambda": 3, "recursive_cte": 3, "unmapped_function": 3}
FUNC_CAT = {f: c for c, fs in TIER2_FUNCS.items() for f in fs} | {f: c for c, fs in TIER3_FUNCS.items() for f in fs}


def categories(sql, unmapped):
    """Set of non-core categories a query needs (empty = tier 1)."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    cats = set()
    for node in tree.walk():
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            cats.add("set_operation")
        elif isinstance(node, (exp.Like, exp.ILike)):
            cats.add("text_match")
        elif isinstance(node, exp.Window):
            cats.add("window")
        elif isinstance(node, (exp.Case, exp.If)):
            cats.add("conditional")
        elif isinstance(node, exp.Values):
            cats.add("inline_values")
        elif isinstance(node, exp.Lateral):
            cats.add("expand_list" if node.find(exp.Unnest, exp.Explode) else "lateral")
        elif isinstance(node, exp.Filter):  # count(...) FILTER (WHERE ...): conditional aggregate
            cats.add("conditional")
        elif isinstance(node, exp.Lambda):
            cats.add("lambda")
        elif isinstance(node, exp.With) and node.args.get("recursive"):
            cats.add("recursive_cte")
        elif isinstance(node, exp.Subquery) and not isinstance(node.parent, (exp.From, exp.Join, exp.In, exp.Exists, exp.CTE)):
            cats.add("scalar_subquery")
        elif isinstance(node, exp.DPipe):
            cats.add("text")
        if isinstance(node, exp.Func) and not isinstance(node, (exp.Case, exp.If, exp.Window)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).upper()
            if name in CORE_FUNCS:
                continue
            if name in FUNC_CAT:
                cats.add(FUNC_CAT[name])
            else:
                unmapped[name] += 1
                cats.add("unmapped_function")
    return cats


def report(name, path):
    sqls = [json.loads(line)["messages"][-1]["content"] for line in open(path) if line.strip()]
    tiers, blocking, only, unmapped = Counter(), Counter(), Counter(), Counter()
    for s in sqls:
        try:
            cats = categories(s, unmapped)
        except Exception:
            cats = {"unparseable"}
            TIER["unparseable"] = 3
        tier = max([1] + [TIER[c] for c in cats])
        tiers[tier] += 1
        blocking.update(cats)
        if len(cats) == 1:
            only.update(cats)
    n = len(sqls)
    print(f"\n== {name}: {n:,} queries")
    print(f"tier 1 (core plan) {100 * tiers[1] / n:5.1f}%   tier 1+2 (with extensions) {100 * (tiers[1] + tiers[2]) / n:5.1f}%"
          f"   tier 3 (free SQL) {100 * tiers[3] / n:5.1f}%")
    print(f"{'category':20s} {'tier':>4s} {'queries using it':>17s} {'only blocker':>13s}")
    for c, k in blocking.most_common():
        print(f"{c:20s} {TIER.get(c, 3):4d} {100 * k / n:16.1f}% {100 * only[c] / n:12.1f}%")
    if unmapped:
        print("unmapped functions: " + ", ".join(f"{f} {k}" for f, k in unmapped.most_common(25)))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        report(name, path)
