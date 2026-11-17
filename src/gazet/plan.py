"""Structured query plans for gazet: a small JSON language the model writes instead of SQL, compiled here to DuckDB SQL.

A plan is one expression. Every expression stands for a set of features (rows with id, name, subtype, country,
region and geometry), so expressions nest freely:

    {"ids": ["<id>", ...]}                      features by candidate ID (source read from the ID: 'ne_' = natural_earth)
    {"find": "divisions" | "natural",           features of one table, filtered:
     "subtype": [...], "country": [...],          attribute filters (lists mean "any of")
     "is_land": bool, "is_territorial": bool,
     "touches" | "intersects" | "within" |        spatial filters against another expression: true when the feature
     "contains" | "not_intersects": E | [E, ...],   relates to ANY feature of E; a list of expressions means ALL of them
     "largest" | "smallest": n, "per": "region" | "country"}   top n by area, optionally within each group
    {"buffer": E, "m": metres}                  each feature of E grown by a distance
    {"union": E}                                all features of E merged into one geometry
    {"intersection": [A, B]}                    the part of A inside B
    {"difference": [A, B]}                      A minus B
    {"half": E, "side": "north" | "south" | "east" | "west"}   half of each feature's bounding box, clipped to it

The compiler mirrors gazet's SQL templates (dataset/scripts/sql_templates.py), so a plan returns the same geometry as
the template SQL it was parsed from. Features named by ID in a "touches" or "intersects" filter are left out of the
result ("regions that touch X" never returns X).
"""

from __future__ import annotations

import json
import re
from typing import Any

SOURCES = {"divisions": "divisions_area", "natural": "natural_earth"}
PREDICATES = {
    "touches": "ST_Touches({f}, {o})",
    "intersects": "ST_Intersects({f}, {o})",
    "within": "ST_Within({f}, {o})",
    "contains": "ST_Contains({f}, {o})",
    "not_intersects": "ST_Intersects({f}, {o})",  # negated below
}
SIDES = {
    "north": "xmin, (ymin + ymax) / 2.0, xmax, ymax",
    "south": "xmin, ymin, xmax, (ymin + ymax) / 2.0",
    "east": "(xmin + xmax) / 2.0, ymin, xmax, ymax",
    "west": "xmin, ymin, (xmin + xmax) / 2.0, ymax",
}
GROUPS = ("region", "country")
# Filters whose anchors are left out of the result (a feature always intersects itself); "within" and "contains"
# keep them, as gazet's templates do ("smallest regions in X" may be X itself).
EXCLUDING = ("touches", "intersects")
FIND_KEYS = {
    "find",
    "subtype",
    "country",
    "is_land",
    "is_territorial",
    "largest",
    "smallest",
    "per",
    *PREDICATES,
}
COLS = "id, name, subtype, country, region, geometry"
DEG_PER_M = "111320.0"


class PlanError(ValueError):
    """The plan is not valid JSON or does not follow the plan language."""


# ── JSON schema (for constrained decoding with llama-server's json_schema) ────────


def _expr_ref() -> dict:
    return {"$ref": "#/$defs/expr"}


def _list(item: dict, min_items: int = 1) -> dict:
    return {"type": "array", "items": item, "minItems": min_items}


_STR = {"type": "string"}
_SPATIAL = {"anyOf": [_expr_ref(), _list(_expr_ref(), 2)]}
_PAIR = {**_list(_expr_ref(), 2), "maxItems": 2}


def plan_schema(candidate_ids: list[str] | None = None) -> dict[str, Any]:
    """JSON schema of a plan. With `candidate_ids`, "ids" may only name those IDs (constrained decoding cannot
    invent an ID)."""
    ids = sorted(set(candidate_ids or []))
    id_item = {"enum": ids} if ids else {"type": "string", "minLength": 1}
    # At most every candidate once: also bounds how long a constrained answer can run.
    id_list = {**_list(id_item), "maxItems": len(ids)} if ids else _list(id_item)
    return {
        "$defs": {
            "expr": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"ids": id_list},
                        "required": ["ids"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "find": {"enum": list(SOURCES)},
                            "subtype": _list(_STR),
                            "country": _list(_STR),
                            "is_land": {"type": "boolean"},
                            "is_territorial": {"type": "boolean"},
                            **{p: _SPATIAL for p in PREDICATES},
                            "largest": {"type": "integer", "minimum": 1},
                            "smallest": {"type": "integer", "minimum": 1},
                            "per": {"enum": list(GROUPS)},
                        },
                        "required": ["find"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"buffer": _expr_ref(), "m": {"type": "number"}},
                        "required": ["buffer", "m"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"union": _expr_ref()},
                        "required": ["union"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"intersection": _PAIR},
                        "required": ["intersection"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"difference": _PAIR},
                        "required": ["difference"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "half": _expr_ref(),
                            "side": {"enum": list(SIDES)},
                        },
                        "required": ["half", "side"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "$ref": "#/$defs/expr",
    }


PLAN_SCHEMA = plan_schema()


# ── parsing and validation ───────────────────────────────────────────────────


def parse_plan(text: str) -> dict:
    """Read a plan from model output (tolerates code fences and text around the JSON object)."""
    s = text.strip()
    if "```" in s:
        s = re.sub(
            r"^```\w*\s*|\s*```$",
            "",
            s.split("```", 2)[1] if s.startswith("```") else s,
        )
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        raise PlanError(f"no JSON object in output: {text[:200]!r}")
    try:
        plan = json.loads(s[start : end + 1])
    except json.JSONDecodeError as e:
        raise PlanError(f"invalid JSON: {e}") from e
    validate(plan)
    return plan


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def validate(e: Any, path: str = "plan") -> None:
    if not isinstance(e, dict) or not e:
        raise PlanError(f"{path}: expected an object")
    kind = op_of(e)
    if kind == "ids":
        if (
            set(e) != {"ids"}
            or not _as_list(e["ids"])
            or not all(isinstance(i, str) and i for i in _as_list(e["ids"]))
        ):
            raise PlanError(f"{path}: 'ids' needs a list of ID strings")
    elif kind == "find":
        if e["find"] not in SOURCES:
            raise PlanError(f"{path}: 'find' must be one of {list(SOURCES)}")
        extra = set(e) - FIND_KEYS
        if extra:
            raise PlanError(f"{path}: unknown keys {sorted(extra)}")
        for k in ("subtype", "country"):
            if k in e and not all(isinstance(v, str) for v in _as_list(e[k])):
                raise PlanError(f"{path}.{k}: expected strings")
        for k in ("is_land", "is_territorial"):
            if k in e and not isinstance(e[k], bool):
                raise PlanError(f"{path}.{k}: expected true or false")
        # Rules the JSON schema cannot express are resolved by the compiler instead of rejected, so every
        # schema-valid plan compiles: 'largest' wins over 'smallest', 'per' without either is ignored.
        for k in ("largest", "smallest"):
            if k in e and (not isinstance(e[k], int) or isinstance(e[k], bool)):
                raise PlanError(f"{path}.{k}: expected an integer")
        if "per" in e and e["per"] not in GROUPS:
            raise PlanError(f"{path}.per: expected one of {list(GROUPS)}")
        for p in PREDICATES:
            for i, sub in enumerate(_as_list(e.get(p, []))):
                validate(sub, f"{path}.{p}[{i}]")
    elif kind == "buffer":
        if (
            set(e) != {"buffer", "m"}
            or not isinstance(e["m"], (int, float))
            or isinstance(e["m"], bool)
        ):
            raise PlanError(f"{path}: 'buffer' needs an expression and 'm' (metres)")
        validate(e["buffer"], f"{path}.buffer")
    elif kind == "union":
        if set(e) != {"union"}:
            raise PlanError(f"{path}: 'union' takes one expression")
        validate(e["union"], f"{path}.union")
    elif kind in ("intersection", "difference"):
        if set(e) != {kind} or not isinstance(e[kind], list) or len(e[kind]) != 2:
            raise PlanError(f"{path}: '{kind}' takes a list of two expressions")
        for i, sub in enumerate(e[kind]):
            validate(sub, f"{path}.{kind}[{i}]")
    elif kind == "half":
        if set(e) != {"half", "side"} or e["side"] not in SIDES:
            raise PlanError(
                f"{path}: 'half' needs an expression and 'side' in {list(SIDES)}"
            )
        validate(e["half"], f"{path}.half")


def op_of(e: dict) -> str:
    for k in ("ids", "find", "buffer", "union", "intersection", "difference", "half"):
        if k in e:
            return k
    raise PlanError(f"unknown expression with keys {sorted(e)}")


# ── compiler ─────────────────────────────────────────────────────────────────


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _in(col: str, values: list) -> str:
    values = _as_list(values)
    if len(values) == 1:
        return f"{col} = {_q(values[0])}"
    return f"{col} IN ({', '.join(_q(v) for v in values)})"


def _source_of_id(i: str, sources: dict[str, str]) -> str:
    if i in sources:
        return sources[i]
    return "natural_earth" if i.startswith("ne_") else "divisions_area"


def _table(source: str, where: str) -> str:
    return (
        f'SELECT id, names."primary" AS name, subtype, country, region, geometry '
        f"FROM read_parquet('{source}') WHERE {where}"
    )


def _anchor_ids(e: dict) -> list[str]:
    """IDs a spatial filter refers to directly (possibly grown by a buffer): excluded from the filter's result."""
    if "ids" in e:
        return _as_list(e["ids"])
    if "buffer" in e:
        return _anchor_ids(e["buffer"])
    return []


GEOMETRY_OPS = ("buffer", "union", "intersection", "difference", "half")


def _compile(e: dict, sources: dict[str, str]) -> str:
    """SQL for one expression. Geometry operations drop empty and NULL results, so they never feed the next
    operation (DuckDB 1.4 + GEOS can crash on nested operations over empty geometries)."""
    sql = _compile_op(e, sources)
    if op_of(e) in GEOMETRY_OPS:
        sql = f"SELECT * FROM ({sql}) WHERE geometry IS NOT NULL AND NOT ST_IsEmpty(geometry)"
    return sql


def _compile_op(e: dict, sources: dict[str, str]) -> str:
    kind = op_of(e)
    if kind == "ids":
        by_src: dict[str, list[str]] = {}
        for i in _as_list(e["ids"]):
            by_src.setdefault(_source_of_id(i, sources), []).append(i)
        return " UNION ALL ".join(
            _table(src, _in("id", ids)) for src, ids in by_src.items()
        )

    if kind == "find":
        source = SOURCES[e["find"]]
        conds = [_in("subtype", e["subtype"])] if e.get("subtype") else []
        if e.get("country"):
            conds.append(_in("country", e["country"]))
        for k in ("is_land", "is_territorial"):
            if k in e:
                conds.append(f"{k} = {'TRUE' if e[k] else 'FALSE'}")
        exclude = sorted(
            {
                i
                for p in EXCLUDING
                for sub in _as_list(e.get(p, []))
                for i in _anchor_ids(sub)
                if _source_of_id(i, sources) == source
            }
        )
        if exclude:
            conds.insert(0, f"NOT ({_in('id', exclude)})")
        body = _table(source, " AND ".join(conds) if conds else "TRUE")
        # Spatial filters join the attribute-filtered features with the other set and keep each feature once.
        # (A correlated EXISTS would make DuckDB hash every candidate geometry and run out of memory.)
        for p, tmpl in PREDICATES.items():
            for sub in _as_list(e.get(p, [])):
                test = tmpl.format(f="f.geometry", o="o.geometry")
                join = "ANTI JOIN" if p == "not_intersects" else "SEMI JOIN"
                body = f"SELECT f.* FROM ({body}) AS f {join} ({_compile(sub, sources)}) AS o ON {test}"
        top = "largest" if "largest" in e else "smallest" if "smallest" in e else None
        if top is None:
            return body
        n, order = max(1, e[top]), "DESC" if top == "largest" else "ASC"
        if e.get("per"):
            return (
                f"SELECT {COLS} FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY {e['per']} "
                f"ORDER BY ST_Area(geometry) {order}) AS rn FROM ({body})) WHERE rn <= {int(n)}"
            )
        return f"SELECT {COLS} FROM ({body}) ORDER BY ST_Area(geometry) {order} LIMIT {int(n)}"

    if kind == "buffer":
        return (
            f"SELECT id, name, subtype, country, region, ST_Buffer(geometry, {float(e['m'])} / {DEG_PER_M}) AS geometry "
            f"FROM ({_compile(e['buffer'], sources)})"
        )

    if kind == "union":
        return (
            f"SELECT NULL AS id, NULL AS name, NULL AS subtype, NULL AS country, NULL AS region, "
            f"ST_Union_Agg(geometry) AS geometry FROM ({_compile(e['union'], sources)})"
        )

    if kind in ("intersection", "difference"):
        a, b = (_compile(x, sources) for x in e[kind])
        fn = "ST_Intersection" if kind == "intersection" else "ST_Difference"
        return (
            f"SELECT a.id, a.name, a.subtype, a.country, a.region, {fn}(a.geometry, b.geometry) AS geometry "
            f"FROM ({a}) AS a, ({b}) AS b WHERE ST_Intersects(a.geometry, b.geometry)"
        )

    if kind == "half":
        inner = _compile(e["half"], sources)
        return (
            f"SELECT id, name, subtype, country, region, ST_Intersection(geometry, ST_MakeEnvelope({SIDES[e['side']]})) "
            f"AS geometry FROM (SELECT *, ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, "
            f"ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM ({inner}))"
        )
    raise PlanError(f"unknown expression {kind}")


def compile_plan(plan: dict, sources: dict[str, str] | None = None) -> str:
    """Plan -> one DuckDB SELECT returning id, name, subtype, country and the geometry as GeoJSON.

    `sources` maps candidate IDs to their table ('divisions_area' or 'natural_earth'); IDs not in it are placed by
    their prefix. Table names stay symbolic (read_parquet('divisions_area')); gazet.sql rewrites them to paths.
    """
    validate(plan)
    body = _compile(plan, sources or {})
    return (
        f"SELECT id, name, subtype, country, ST_AsGeoJSON(geometry) AS geometry FROM ({body}) "
        f"WHERE geometry IS NOT NULL AND NOT ST_IsEmpty(geometry)"
    )


def dumps(plan: dict) -> str:
    """Canonical compact plan text (training targets use this form)."""
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
