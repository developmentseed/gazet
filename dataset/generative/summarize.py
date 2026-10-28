"""Per-model bake-off table: parse rate, execution rate, PostGIS-dialect errors, measured cost."""
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN_DIR = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
POSTGIS = re.compile(r"ST_MakeEnvelope\([^)]*INTEGER_LITERAL\)|ST_GeomFromText\(STRING_LITERAL, INTEGER|"
                     r"ST_Transform\(GEOMETRY[^)]*\), INTEGER|st_setsrid|GEOGRAPHY|"
                     r"ST_Distance\(GEOMETRY[^)]*\), GEOMETRY[^)]*\), STRING|ST_Area\(GEOMETRY[^)]*\), STRING|"
                     r"ST_Collect\(GEOMETRY\(|only accepts POINT", re.IGNORECASE)

print(f"{'model':42s} {'prompt':6s} {'pairs':>5s} {'exec':>9s} {'postgis':>7s} {'$/call':>8s} {'$/1k exec pairs':>15s} {'reason tok':>10s}")
for f in sorted(glob.glob(str(RUN_DIR / "out_*.validated.jsonl"))):
    name = Path(f).name.removeprefix("out_").removesuffix(".validated.jsonl")
    model, _, tag = name.partition("__v")
    tag = f"v{tag}" if tag else "v1"
    rs = [json.loads(line) for line in open(f)]
    ok = sum(r["ok"] and (r.get("ms") or 0) <= 30000 for r in rs)
    pg = sum(bool(POSTGIS.search(r.get("error") or "")) for r in rs)
    usage = json.loads((RUN_DIR / f"out_{name}" / "_usage.json").read_text())
    cost = sum(u["cost"] or 0 for u in usage)
    calls = len(usage)
    reason = sum(u["reasoning_tokens"] or 0 for u in usage) / max(1, calls)
    print(f"{model:42s} {tag:6s} {len(rs):>5d} {ok:>3d} ({ok/72:4.0%}) {pg:>7d} {cost/max(1,calls):>8.4f} "
          f"{1000*cost/max(1,ok):>15.2f} {reason:>10.0f}")
