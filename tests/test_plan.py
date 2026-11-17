"""Tests for gazet.plan — the JSON plan language and its compiler to DuckDB SQL.

Plans run against two tiny in-memory tables shaped like divisions_area and natural_earth: a 3x1 row of unit squares
(A, B, C, west to east) inside a country X, and a sea S that overlaps C.
"""

import json
import random

import duckdb
import pytest

from gazet.plan import PlanError, compile_plan, parse_plan, plan_schema, validate

SQUARE = "POLYGON(({x0} 0, {x1} 0, {x1} 1, {x0} 1, {x0} 0))"


def make_con():
    c = duckdb.connect()
    c.execute("INSTALL spatial; LOAD spatial")
    cols = (
        'id VARCHAR, names STRUCT("primary" VARCHAR), subtype VARCHAR, country VARCHAR, region VARCHAR, '
        "is_land BOOLEAN, is_territorial BOOLEAN, geometry GEOMETRY"
    )
    c.execute(f"CREATE TABLE divisions_area ({cols})")
    c.execute(f"CREATE TABLE natural_earth ({cols})")
    rows = [
        ("X", "country", None, "POLYGON((0 0, 3 0, 3 1, 0 1, 0 0))"),
        ("A", "region", "R1", SQUARE.format(x0=0, x1=1)),
        ("B", "region", "R1", SQUARE.format(x0=1, x1=2)),
        (
            "C",
            "region",
            "R2",
            "POLYGON((2 0, 3 0, 3 1.5, 2 1.5, 2 0))",
        ),  # largest region
    ]
    for i, sub, reg, wkt in rows:
        c.execute(
            "INSERT INTO divisions_area VALUES (?, {'primary': ?}, ?, 'XX', ?, TRUE, TRUE, ST_GeomFromText(?))",
            [i, i, sub, reg, wkt],
        )
    c.execute(
        "INSERT INTO natural_earth VALUES ('ne_1', {'primary': 'S'}, 'sea', NULL, NULL, FALSE, FALSE, "
        "ST_GeomFromText('POLYGON((2.5 0, 4 0, 4 1, 2.5 1, 2.5 0))'))"
    )
    return c


@pytest.fixture(scope="module")
def con():
    return make_con()


def run(con, plan):
    sql = compile_plan(plan).replace("read_parquet('divisions_area')", "divisions_area")
    sql = sql.replace("read_parquet('natural_earth')", "natural_earth")
    return con.execute(sql).fetchall()


def ids(con, plan):
    return sorted(r[0] for r in run(con, plan))


def area(con, plan):
    sql = compile_plan(plan).replace("read_parquet('divisions_area')", "divisions_area")
    sql = sql.replace("read_parquet('natural_earth')", "natural_earth")
    return con.execute(
        f"SELECT sum(ST_Area(ST_GeomFromGeoJSON(geometry))) FROM ({sql})"
    ).fetchone()[0]


class TestCompile:
    def test_ids(self, con):
        assert ids(con, {"ids": ["A", "ne_1"]}) == ["A", "ne_1"]

    def test_touches_excludes_anchor(self, con):
        assert ids(
            con, {"find": "divisions", "subtype": ["region"], "touches": {"ids": ["B"]}}
        ) == ["A", "C"]

    def test_intersects_excludes_anchor(self, con):
        assert ids(con, {"find": "divisions", "intersects": {"ids": ["A"]}}) == [
            "B",
            "X",
        ]

    def test_within(self, con):
        plan = {"find": "divisions", "subtype": ["region"], "within": {"ids": ["X"]}}
        assert ids(con, plan) == ["A", "B"]  # C sticks out of X

    def test_all_of_list(self, con):
        plan = {
            "find": "divisions",
            "subtype": ["region"],
            "touches": [{"ids": ["A"]}, {"ids": ["C"]}],
        }
        assert ids(con, plan) == ["B"]

    def test_not_intersects_natural(self, con):
        plan = {
            "find": "divisions",
            "subtype": ["region"],
            "not_intersects": {"find": "natural", "subtype": ["sea"]},
        }
        assert ids(con, plan) == ["A", "B"]

    def test_buffer_filter(self, con):
        plan = {
            "find": "divisions",
            "subtype": ["region"],
            "intersects": {"buffer": {"ids": ["A"]}, "m": 1000},
        }
        assert ids(con, plan) == ["B"]

    def test_largest_and_per(self, con):
        assert ids(con, {"find": "divisions", "subtype": ["region"], "largest": 1}) == [
            "C"
        ]
        plan = {
            "find": "divisions",
            "subtype": ["region"],
            "smallest": 1,
            "per": "region",
        }
        assert len(ids(con, plan)) == 2  # one per region R1, R2

    def test_union_difference_half(self, con):
        assert area(con, {"union": {"ids": ["A", "B"]}}) == pytest.approx(2.0)
        assert area(
            con, {"difference": [{"ids": ["C"]}, {"ids": ["ne_1"]}]}
        ) == pytest.approx(1.0)
        assert area(con, {"half": {"ids": ["X"]}, "side": "west"}) == pytest.approx(1.5)
        assert area(
            con, {"intersection": [{"ids": ["C"]}, {"ids": ["ne_1"]}]}
        ) == pytest.approx(0.5)

    def test_corridor(self, con):
        plan = {"buffer": {"intersection": [{"ids": ["A"]}, {"ids": ["B"]}]}, "m": 1000}
        assert area(con, plan) > 0

    def test_quotes_are_escaped(self, con):
        assert ids(con, {"ids": ["it's"]}) == []

    def test_sources_override_prefix(self):
        sql = compile_plan({"ids": ["n1"]}, {"n1": "natural_earth"})
        assert "read_parquet('natural_earth')" in sql


class TestParseAndValidate:
    def test_parse_fenced(self):
        assert parse_plan('```json\n{"ids": ["A"]}\n```') == {"ids": ["A"]}

    @pytest.mark.parametrize(
        "plan",
        [
            {"find": "places"},
            {"find": "divisions", "colour": "red"},
            {"ids": []},
            {"buffer": {"ids": ["A"]}},
            {"difference": [{"ids": ["A"]}]},
            {"find": "divisions", "per": "town"},
            {"half": {"ids": ["A"]}, "side": "up"},
        ],
    )
    def test_invalid(self, plan):
        with pytest.raises(PlanError):
            validate(plan)

    def test_not_json(self):
        with pytest.raises(PlanError):
            parse_plan("SELECT 1")

    def test_schema_limits_ids(self):
        schema = plan_schema(["A", "B"])
        ids_item = schema["$defs"]["expr"]["anyOf"][0]["properties"]["ids"]["items"]
        assert ids_item == {"enum": ["A", "B"]}
        json.dumps(schema)


def random_instance(schema, node, rnd, depth=0):
    """A random JSON value that follows `node` (a part of the plan schema); deep nesting falls back to "ids"."""
    if "$ref" in node:
        return random_instance(
            schema, schema["$defs"][node["$ref"].split("/")[-1]], rnd, depth
        )
    if "anyOf" in node:
        options = node["anyOf"]
        return random_instance(
            schema, options[0] if depth > 3 else rnd.choice(options), rnd, depth
        )
    if "enum" in node:
        return rnd.choice(node["enum"])
    t = node.get("type")
    if t == "object":
        props = node["properties"]
        keys = set(node.get("required", [])) | {k for k in props if rnd.random() < 0.3}
        return {k: random_instance(schema, props[k], rnd, depth + 1) for k in keys}
    if t == "array":
        lo = node.get("minItems", 0)
        n = rnd.randint(lo, min(lo + 2, node.get("maxItems", lo + 2)))
        return [
            random_instance(schema, node["items"], rnd, depth + 1) for _ in range(n)
        ]
    if t == "integer":
        return rnd.randint(node.get("minimum", -3), 5)
    if t == "number":
        return rnd.choice([-500, 0, 1000, 2500.5])
    if t == "boolean":
        return rnd.random() < 0.5
    return rnd.choice(["sea", "region", "XX", "it's"])


def test_every_schema_valid_plan_compiles_and_runs(con):
    """The guarantee behind constrained decoding: whatever the schema allows, the compiler accepts and DuckDB runs."""
    schema = plan_schema(["A", "B", "C", "X", "ne_1"])
    rnd = random.Random(0)
    for _ in range(2000):
        plan = random_instance(schema, schema, rnd)
        validate(plan)
        run(con, plan)
