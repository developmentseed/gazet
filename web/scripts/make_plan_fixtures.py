"""Write tests/fixtures/plans.json: plans with the SQL, schema and errors Python's gazet.plan gives for them.

The TypeScript port (src/plan.ts) must reproduce each exactly. Plans: the gazet-dataset test plans plus random
schema-valid plans from the generator in tests/test_plan.py.

    uv run python web/scripts/make_plan_fixtures.py
"""

import json
import random
import sys
from pathlib import Path

from gazet.plan import PlanError, compile_plan, parse_plan, plan_schema

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tests.test_plan import random_instance  # noqa: E402

plans = [json.loads(line)["plan"] for line in (REPO / "dataset/generative/runs/gazet_plan/plans_test.jsonl").open()][:400]
schema = plan_schema()
rnd = random.Random(7)
plans += [random_instance(schema, schema, rnd) for _ in range(300)]
sources = {"ne_1159116143": "divisions_area", "abc": "natural_earth"}

cases = []
for p in plans:
    case = {"plan": p}
    try:
        case["sql"] = compile_plan(p)
        case["sql_sources"] = compile_plan(p, sources)
    except PlanError as e:
        case["error"] = str(e)
    cases.append(case)

texts = ['```json\n{"ids": ["a"]}\n```', 'Plan: {"union": {"ids": ["a", "b"]}} done', '{"ids": []}', "nope", '{"find": "x"}',
         '{"buffer": {"ids": ["a"]}, "m": true}', '{"find": "divisions", "largest": 2.5}', '{"half": {"ids": ["a"]}, "side": "up"}']
parse = []
for t in texts:
    try:
        parse.append({"text": t, "plan": parse_plan(t)})
    except PlanError:
        parse.append({"text": t, "error": True})

out = {"cases": cases, "parse": parse, "sources": sources,
       "schema": plan_schema(["b", "a", "ne_1", "a"]), "schema_open": schema}
path = REPO / "web/tests/fixtures/plans.json"
path.write_text(json.dumps(out, ensure_ascii=False))
print(f"{len(cases)} plans ({sum('error' in c for c in cases)} errors), {path.stat().st_size / 1e6:.1f} MB -> {path}")
