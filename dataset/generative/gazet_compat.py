"""Can a model replace the fine-tuned model in gazet's two-step geocoding pipeline without changing the API?

Runs gazet's own production calls, `gazet.lm.generate_places` and `gazet.lm.generate_sql` (the llama-server/GGUF
path behind the API), on a stratified sample of the old test set (`developmentseed/gazet-dataset`, `places/` and
`sql/test.jsonl`), then scores:

- places: the extracted place names equal the reference set (case-insensitive, order-insensitive);
- sql: the predicted SQL, after gazet's own path rewriting and subtype fixes, returns the same geometry as the
  reference SQL. All output geometries of a query are unioned and compared: polygons by intersection over union
  >= 0.99, other geometries by Hausdorff distance <= 1e-4 degrees. Attribute columns are ignored, because the API
  returns geometry. Every query runs in its own memory-capped subprocess (harness.run_capped).

Run with gazet's environment (`~/repos/gazet/.venv/bin/python`), llama-server serving the model on PORT:
    python gazet_compat.py prep [--per-family 25]   # fix the SQL sample (references that run on current data)
    python gazet_compat.py gen LABEL PORT [--per-family 25]
    python gazet_compat.py score LABEL [LABEL...]
Outputs: runs/gazet_compat/<label>__{places,sql}.jsonl, scored files *.scored.jsonl.
"""
import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = HERE / "runs" / "gazet_compat"
TEST = Path(os.environ.get("GAZET_TEST_DIR", Path.home() / ".cache" / "gazet-exp" / "gazet-dataset"))
SEED = 5


def prep(per_family, pool):
    """Fix the SQL sample: questions whose reference SQL returns a geometry on the current data, per_family each.

    The test set was built on an older Overture release; references whose candidate IDs no longer exist return no
    rows and cannot score either model, so they are skipped (production candidates always come from current data).
    """
    cache_p = OUT / "ref_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    chosen, stats = [], Counter()
    for fam, items in sorted(_by_family("sql", pool).items()):
        ok = 0
        for i, r in items:
            res = _run(r["messages"][2]["content"], cache)
            stats["ran" if "wkt" in res else (res.get("error") or "?")[:40]] += 1
            if "wkt" in res:
                chosen.append(i)
                ok += 1
                if ok == per_family:
                    break
        cache_p.write_text(json.dumps(cache))
        print(f"{fam}: {ok} valid", flush=True)
    (OUT / "sql_sample.json").write_text(json.dumps(sorted(chosen)))
    print(f"{len(chosen)} SQL questions; reference outcomes: {dict(stats)}")


def _by_family(task, pool):
    rows = [json.loads(line) for line in (TEST / task / "test.jsonl").read_text().splitlines() if line.strip()]
    by_fam = defaultdict(list)
    for i, r in enumerate(rows):
        by_fam[r["metadata"].get("task_family", "other")].append((i, r))
    rnd = random.Random(SEED)
    return {f: rnd.sample(v, min(pool, len(v))) for f, v in by_fam.items()}


def sample(task, per_family):
    rows = [json.loads(line) for line in (TEST / task / "test.jsonl").read_text().splitlines() if line.strip()]
    if task == "sql" and (OUT / "sql_sample.json").exists():
        return [(i, rows[i]) for i in json.loads((OUT / "sql_sample.json").read_text())]
    by_fam = defaultdict(list)
    for i, r in enumerate(rows):
        by_fam[r["metadata"].get("task_family", "other")].append((i, r))
    rnd, out = random.Random(SEED), []
    for fam in sorted(by_fam):
        out += rnd.sample(by_fam[fam], min(per_family, len(by_fam[fam])))
    return out


def parse_sql_prompt(user):
    import pandas as pd
    cands = user.split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
    question = user.split("<USER_QUERY>")[1].split("</USER_QUERY>")[0].strip()
    return question, pd.read_csv(io.StringIO(cands), dtype=str, keep_default_na=False)


def gen(label, port, per_family):
    os.environ["LLAMA_SERVER_URL"] = f"http://127.0.0.1:{port}"
    sys.path.insert(0, str(REPO / "src"))
    from gazet import lm  # production prompts and post-processing
    OUT.mkdir(parents=True, exist_ok=True)
    for task in ("places", "sql"):
        path = OUT / f"{label}__{task}.jsonl"
        done = {json.loads(line)["idx"] for line in path.read_text().splitlines()} if path.exists() else set()
        todo = [(i, r) for i, r in sample(task, per_family) if i not in done]
        t_all = time.time()
        with path.open("a") as f:
            for i, r in todo:
                t0 = time.time()
                user = r["messages"][1]["content"]
                if task == "places":
                    pred = [p.place for p in lm.generate_places(user).places]
                    ref = [p["place"] for p in json.loads(r["messages"][2]["content"])["places"]]
                    rec = dict(idx=i, family=r["metadata"].get("task_family"), query=user, pred=pred, ref=ref)
                else:
                    question, cands = parse_sql_prompt(user)
                    pred = lm.generate_sql(question, cands)
                    rec = dict(idx=i, family=r["metadata"]["task_family"], query=question, pred=pred,
                               ref=r["messages"][2]["content"])
                rec["secs"] = round(time.time() - t0, 2)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        print(f"{label} {task}: {len(todo)} new answers in {(time.time() - t_all) / 60:.1f} min")


# ── scoring ──────────────────────────────────────────────────────────────────

def exec_geom(sql):
    """Child process: run SQL the way gazet does, return the union of its output geometries as WKT."""
    sys.path.insert(0, str(REPO / "src"))
    import duckdb
    import shapely
    from shapely import wkb as swkb
    from gazet.geometry import normalize_geometry_to_geojson
    from gazet.sql import _normalize_ne_subtypes, _rewrite_data_paths
    con = duckdb.connect()
    con.execute("LOAD spatial")
    con.execute("SET threads=2")
    con.execute(f"SET memory_limit='{os.environ.get('GAZET_COMPAT_DUCKDB_MB', '2500')}MB'")
    sql = _normalize_ne_subtypes(_rewrite_data_paths(sql))
    try:
        df = normalize_geometry_to_geojson(con, con.execute(sql).fetchdf())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"}
    if df is None or df.empty:
        return {"error": "no rows"}
    geoms = []
    for col in df.columns:
        for v in df[col]:
            if isinstance(v, str) and v.lstrip().startswith("{") and '"type"' in v:
                try:
                    geoms.append(shapely.from_geojson(v))
                except Exception:
                    pass
            elif isinstance(v, (bytes, bytearray)):
                try:
                    geoms.append(swkb.loads(bytes(v)))
                except Exception:
                    pass
    if not geoms:
        return {"error": "no geometry column", "rows": len(df)}
    u = shapely.union_all([shapely.make_valid(g) for g in geoms])
    return {"wkt": shapely.to_wkt(u, rounding_precision=7), "rows": len(df)}


def same_geom(a_wkt, b_wkt):
    import shapely
    a, b = shapely.from_wkt(a_wkt), shapely.from_wkt(b_wkt)
    if a.is_empty or b.is_empty:
        return a.is_empty and b.is_empty
    if a.area > 0 and b.area > 0:
        inter = shapely.intersection(a, b).area
        union = shapely.union(a, b).area
        return union > 0 and inter / union >= 0.99
    return shapely.hausdorff_distance(a, b) <= 1e-4


def _run(sql, cache):
    if sql in cache:
        return cache[sql]
    sys.path.insert(0, str(HERE))
    import harness
    # One query at a time (callers use one worker): full divisions scans need ~2.5 GB, and the laptop VM has ~5 GB.
    rc, out, err, killed = harness.run_capped([sys.executable, "-I", __file__, "exec"], sql, 180, 3500)
    if killed:
        res = {"error": killed}
    elif rc != 0 or not out.strip():
        res = {"error": f"crash (exit {rc}): {err.strip().splitlines()[-1][:120] if err.strip() else ''}"}
    else:
        res = json.loads(out)
    cache[sql] = res
    return res


def score(labels):
    cache_p = OUT / "ref_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for label in labels:
        # places
        rows = [json.loads(line) for line in (OUT / f"{label}__places.jsonl").read_text().splitlines()]
        norm = lambda xs: sorted({x.strip().lower() for x in xs})
        for r in rows:
            r["ok"] = norm(r["pred"]) == norm(r["ref"])
        (OUT / f"{label}__places.scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        p_ok, p_n = sum(r["ok"] for r in rows), len(rows)
        # sql
        rows = [json.loads(line) for line in (OUT / f"{label}__sql.jsonl").read_text().splitlines()]
        refs = sorted({r["ref"] for r in rows} - set(cache))
        for sql in refs:
            _run(sql, cache)
        cache_p.write_text(json.dumps(cache))
        preds = {}
        for r in rows:
            _run(r["pred"], preds)
        for r in rows:
            ref, pred = cache[r["ref"]], preds[r["pred"]]
            r["ref_error"], r["pred_error"] = ref.get("error"), pred.get("error")
            r["executes"] = "wkt" in pred
            r["ok"] = "wkt" in ref and "wkt" in pred and same_geom(ref["wkt"], pred["wkt"])
        (OUT / f"{label}__sql.scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        valid = [r for r in rows if not r["ref_error"]]
        print(f"\n== {label}\nplaces: same place set {p_ok}/{p_n} = {100 * p_ok / p_n:.1f}%")
        print(f"sql: reference ran for {len(valid)}/{len(rows)}; executes {100 * sum(r['executes'] for r in valid) / len(valid):.1f}%, "
              f"same geometry {100 * sum(r['ok'] for r in valid) / len(valid):.1f}%; "
              f"median {sorted(r['secs'] for r in rows)[len(rows) // 2]:.1f} s/query")
        fam = defaultdict(list)
        for r in valid:
            fam[r["family"]].append(r["ok"])
        print("  " + "  ".join(f"{k} {100 * sum(v) / len(v):.0f}%" for k, v in sorted(fam.items())))
        errs = Counter((r["pred_error"] or "")[:60] for r in valid if r["pred_error"])
        print("  top errors: " + "; ".join(f"{n}x {e}" for e, n in errs.most_common(5)))


if __name__ == "__main__":
    if sys.argv[1] == "exec":
        print(json.dumps(exec_geom(sys.stdin.read())))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--per-family", type=int, default=25)
    p.add_argument("--pool", type=int, default=80)
    g = sub.add_parser("gen")
    g.add_argument("label")
    g.add_argument("port", type=int)
    g.add_argument("--per-family", type=int, default=25)
    s = sub.add_parser("score")
    s.add_argument("labels", nargs="+")
    a = ap.parse_args()
    if a.cmd == "prep":
        OUT.mkdir(parents=True, exist_ok=True)
        prep(a.per_family, a.pool)
    elif a.cmd == "gen":
        gen(a.label, a.port, a.per_family)
    else:
        score(a.labels)
