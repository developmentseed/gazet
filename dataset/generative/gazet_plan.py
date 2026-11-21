"""Gazet plan prototype: labels, verification and training export for the structured-plan model.

The plan language and its compiler live in gazet itself (src/gazet/plan.py); gazet.lm.generate_sql uses them when
GAZET_SQL_MODE=plan. This script turns gazet-dataset's template SQL into plans and checks the compiler against it:

    python gazet_plan.py labels                 # parse every SQL in train/val/test into a plan; coverage per template
    python gazet_plan.py verify [--per-template 6] [--split test]
                                                # run reference SQL and compiled plan; same geometry? (gazet_compat scoring)
    python gazet_plan.py export [--out DIR]     # ChatML: plan task (gazet.lm's plan prompt) + places task, train/val
    python gazet_plan.py export-joint           # phase 2a: gazet plans + places + generative SQL, one training set
    python gazet_plan.py bench LABEL PORT sql|plan [--per-family 3]
                                                # CPU latency of gazet.lm's calls against llama-server on PORT
Run with gazet's environment (~/repos/gazet/.venv/bin/python). Outputs go to runs/gazet_plan/.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from gazet.plan import PlanError, compile_plan, dumps, validate

DATA = Path.home() / ".cache" / "gazet-exp" / "gazet-dataset"
OUT = HERE / "runs" / "gazet_plan"

_UUID_OR_NE = r"'([0-9a-f]{8}-[0-9a-f-]{27}|ne_[\w.]+)'"


def _lits(s):
    return re.findall(r"'([^']*)'", s)


def _anchor_ids(sql):
    """Anchor IDs in order of first appearance (positive and negative selections repeat the same IDs)."""
    out = []
    for m in re.finditer(_UUID_OR_NE, sql):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def _subtypes(sql, alias):
    """Subtype literals filtered on `alias.subtype` (alias '' = unqualified)."""
    pre = rf"(?<![\w.]){re.escape(alias)}\." if alias else r"(?<![\w.])"
    m = re.search(pre + r"subtype\s*(=\s*'([^']*)'|IN\s*\(([^)]*)\))", sql)
    if not m:
        return None
    return [m.group(2)] if m.group(2) is not None else _lits(m.group(3))


def _cte_bodies(sql):
    """name -> body of each top-level CTE (bodies are balanced parentheses)."""
    out = {}
    for m in re.finditer(r"(?:\bWITH|,)\s*(\w+) AS \(", sql):
        depth, i = 1, m.end()
        while depth and i < len(sql):
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i += 1
        out[m.group(1)] = sql[m.end():i - 1]
    return out


def sql_to_plan(sql):
    """Parse one gazet template SQL into a plan (raises PlanError when the shape is unknown)."""
    s = re.sub(r"\s+", " ", sql).strip()
    ids = _anchor_ids(s)
    ctes = _cte_bodies(s)
    buf = re.search(r"ST_Buffer\(\w*\.?\w+, (\d+(?:\.\d+)?) (\* 1000\.0 )?/ 111320\.0\)", s)
    metres = None
    if buf:
        metres = float(buf.group(1)) * (1000 if buf.group(2) else 1)
        metres = int(metres) if metres == int(metres) else metres

    if "ST_Difference" in s:
        return {"difference": [{"ids": [ids[0]]}, {"ids": [ids[1]]}]}
    if "border" in ctes:
        return {"buffer": {"intersection": [{"ids": [ids[0]]}, {"ids": [ids[1]]}]}, "m": metres}
    if "ST_MakeEnvelope" in s:
        env = re.search(r"ST_MakeEnvelope\(([^;]*?)\) AS half_geom", s).group(1).replace(" ", "")
        side = {"xmin,(ymin+ymax)/2.0,xmax,ymax": "north", "xmin,ymin,xmax,(ymin+ymax)/2.0": "south",
                "(xmin+xmax)/2.0,ymin,xmax,ymax": "east", "xmin,ymin,(xmin+xmax)/2.0,ymax": "west"}[env]
        return {"half": {"ids": [ids[0]]}, "side": side}
    if re.search(r"ST_Intersection\(a\.g1, b\.g2\)", s):
        return {"intersection": [{"ids": [ids[0]]}, {"ids": [ids[1]]}]}
    if s.startswith("SELECT ST_AsGeoJSON(ST_Buffer("):
        return {"buffer": {"ids": ids}, "m": metres}

    # A selection from one table: attribute filters, a spatial filter against anchors, optional ranking or union.
    final = s[s.rfind(") SELECT") + 1:] if ctes else s
    body = ctes.get("ranked", s)  # ranking templates select inside a 'ranked' CTE
    main = re.search(r"FROM read_parquet\('(\w+)'\)(?: AS (\w+))?(?:, \w+)* (WHERE|$)", ctes.get("ranked", final))
    if not main:
        raise PlanError("unknown SQL shape")
    table, alias = main.group(1), main.group(2) or ""
    find = {"find": "divisions" if table == "divisions_area" else "natural"}
    st = _subtypes(body, alias)
    if st:
        find["subtype"] = st
    cm = re.search(r"(?<![\w.])country\s*(=\s*'([^']*)'|IN\s*\(([^)]*)\))", body)
    if cm:
        find["country"] = [cm.group(2)] if cm.group(2) is not None else _lits(cm.group(3))
    for k in ("is_land", "is_territorial"):
        km = re.search(rf"(?<![\w.]){k} = (TRUE|FALSE)", s)
        if km:
            find[k] = km.group(1) == "TRUE"

    anchors = [n for n in ctes if n != "ranked"]
    if anchors:
        a = alias or "b"
        exprs = []
        for n in anchors:
            ids_n = _anchor_ids(ctes[n])
            e = {"ids": ids_n}
            if "ST_Buffer" in ctes[n]:
                e = {"buffer": e, "m": metres}
            exprs.append((n, e))
        preds = {}
        for fn, x, y in re.findall(r"ST_(Touches|Within|Contains|Intersects)\((\w+)\.\w+, (\w+)\.\w+\)", s):
            if a not in (x, y) or not ({x, y} & set(ctes)):
                continue
            other = y if x == a else x
            p = fn.lower()
            if p in ("within", "contains") and x != a:  # ST_Within(anchor, b) = b contains anchor
                p = {"within": "contains", "contains": "within"}[p]
            preds.setdefault(p, []).append(dict(exprs)[other])
        if not preds:
            raise PlanError("no spatial filter found")
        for p, es in preds.items():
            find[p] = es[0] if len(es) == 1 else es
    ex = re.search(r"(NOT )?EXISTS \(SELECT 1 FROM read_parquet\('natural_earth'\) AS n WHERE (.*?) AND ST_Intersects", s)
    if ex:
        sub = {"find": "natural", "subtype": _subtypes(ex.group(2), "n")}
        find["not_intersects" if ex.group(1) else "intersects"] = sub
    lim = re.search(r"ORDER BY area (ASC|DESC) LIMIT (\d+)", s)
    if lim:
        find["largest" if lim.group(1) == "DESC" else "smallest"] = int(lim.group(2))
    rn = re.search(r"PARTITION BY (\w+) ORDER BY ST_Area\(geometry\) (ASC|DESC)\) AS rn .* WHERE rn = (\d+)", s)
    if rn:
        find["largest" if rn.group(2) == "DESC" else "smallest"] = int(rn.group(3))
        find["per"] = rn.group(1)
    lookup = {"ids": ids}  # no anchor CTE but IDs: the selection is the anchors themselves
    if "ST_Union_Agg" in final:
        return {"union": canonical(find) if anchors else lookup}
    if not anchors and ids:
        return lookup
    return canonical(find)


ORDER = ["ids", "find", "subtype", "country", "is_land", "is_territorial", "touches", "intersects", "within",
         "contains", "not_intersects", "largest", "smallest", "per", "buffer", "m", "union", "intersection",
         "difference", "half", "side"]


def canonical(e):
    """Fixed key order, recursively (training targets are deterministic)."""
    if isinstance(e, list):
        return [canonical(x) for x in e]
    if not isinstance(e, dict):
        return e
    return {k: canonical(e[k]) for k in sorted(e, key=ORDER.index)}


def load(split):
    return [json.loads(line) for line in (DATA / "sql" / f"{split}.jsonl").read_text().splitlines() if line.strip()]


def labels():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows, ok, bad, shapes = load(split), 0, Counter(), defaultdict(Counter)
        with (OUT / f"plans_{split}.jsonl").open("w") as f:
            for i, r in enumerate(rows):
                tid = r["metadata"]["template_id"]
                try:
                    plan = canonical(sql_to_plan(r["messages"][2]["content"]))
                    validate(plan)
                    compile_plan(plan)
                except Exception as e:  # record every failure
                    bad[f"{tid}: {type(e).__name__} {str(e)[:60]}"] += 1
                    continue
                ok += 1
                shapes[tid][_shape(plan)] += 1
                f.write(json.dumps({"idx": i, "template_id": tid, "family": r["metadata"]["task_family"],
                                    "plan": plan}, ensure_ascii=False) + "\n")
        print(f"{split}: {ok}/{len(rows)} parsed ({100 * ok / len(rows):.2f}%)")
        for k, n in bad.most_common(10):
            print(f"  FAIL {n}x {k}")
        multi = {t: c for t, c in shapes.items() if len(c) > 1}
        if multi:
            print(f"  templates with more than one plan shape: {multi}")
    if (OUT / "plans_train.jsonl").exists():
        ex = {}
        for line in (OUT / "plans_train.jsonl").read_text().splitlines():
            r = json.loads(line)
            ex.setdefault(r["template_id"], r["plan"])
        (OUT / "examples.txt").write_text("".join(f"{t}: {dumps(p)}\n" for t, p in sorted(ex.items())))
        print(f"one plan per template -> {OUT / 'examples.txt'}")


def _shape(e):
    if isinstance(e, list):
        return "[" + ",".join(_shape(x) for x in e) + "]"
    if isinstance(e, dict):
        return "{" + ",".join(k + ":" + _shape(v) for k, v in e.items() if k not in ("ids",)) + "}"
    return "_"


def verify(per_template, split):
    """Reference SQL vs compiled plan, both through gazet_compat's capped executor and geometry comparison."""
    import gazet_compat as gc
    rows = load(split)
    plans = {json.loads(line)["idx"]: json.loads(line) for line in (OUT / f"plans_{split}.jsonl").read_text().splitlines()}
    by_t = defaultdict(list)
    for i, p in plans.items():
        by_t[p["template_id"]].append(i)
    rnd = random.Random(7)
    cache_p = gc.OUT / "ref_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    res, out = defaultdict(Counter), []
    for t in sorted(by_t):
        for i in rnd.sample(by_t[t], min(per_template, len(by_t[t]))):
            ref_sql = rows[i]["messages"][2]["content"]
            ref = gc._run(ref_sql, cache)
            pred = gc._run(compile_plan(plans[i]["plan"]), {})
            if "wkt" not in ref:
                res[t]["ref_no_geometry"] += 1
                status = "ref_no_geometry"
            elif "wkt" not in pred:
                res[t]["plan_error"] += 1
                status = "plan_error: " + str(pred.get("error"))[:100]
            else:
                status = "same" if gc.same_geom(ref["wkt"], pred["wkt"]) else "different"
                res[t][status] += 1
            out.append({"idx": i, "template_id": t, "status": status, "plan": plans[i]["plan"]})
            print(f"{t} {i}: {status}", flush=True)
        cache_p.write_text(json.dumps(cache))
    (OUT / f"verify_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    tot = Counter()
    for c in res.values():
        tot.update(c)
    print("\n" + "\n".join(f"{t:18s} {dict(c)}" for t, c in sorted(res.items())))
    checked = tot["same"] + tot["different"] + tot["plan_error"]
    print(f"\nreference ran: {checked}; same geometry {tot['same']} ({100 * tot['same'] / max(checked, 1):.1f}%), "
          f"different {tot['different']}, plan error {tot['plan_error']}; reference without geometry {tot['ref_no_geometry']}")


def export(out_dir):
    """ChatML for train_hf_jobs_qwen35.py: sql/{train,val}.jsonl hold both tasks (plan answers and place lists)."""
    from gazet import lm
    out = Path(out_dir)
    (out / "sql").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    import io

    import pandas as pd
    for split, dst in (("train", "train"), ("val", "val")):
        rows = load(split)
        plans = {json.loads(line)["idx"]: json.loads(line)["plan"]
                 for line in (OUT / f"plans_{split}.jsonl").read_text().splitlines()}
        recs = []
        for i, r in enumerate(rows):
            if i not in plans:
                continue
            user = r["messages"][1]["content"]
            cands = user.split("<CANDIDATES>")[1].split("</CANDIDATES>")[0].strip()
            question = user.split("<USER_QUERY>")[1].split("</USER_QUERY>")[0].strip()
            msgs = lm.plan_messages(question, pd.read_csv(io.StringIO(cands), dtype=str, keep_default_na=False))
            recs.append({"messages": msgs + [{"role": "assistant", "content": dumps(plans[i])}],
                         "task": "plan", "family": r["metadata"]["task_family"]})
        places = [json.loads(line) for line in (DATA / "places" / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        for r in places:
            recs.append({"messages": r["messages"], "task": "places", "family": r.get("metadata", {}).get("task_family")})
        random.Random(11).shuffle(recs)
        if dst == "val":
            recs = recs[:1500]
        (out / "sql" / f"{dst}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
        print(f"{dst}: {len(recs)} records ({Counter(x['task'] for x in recs)})")
    # The job also answers these prompts after training (the compat test prompts, captured from gazet.lm in plan mode).
    prompts = HERE / "runs" / "gazet_compat" / "prompts_plan.jsonl"
    if prompts.exists():
        (out / "eval" / "gazet_compat.jsonl").write_text(prompts.read_text())
        print(f"eval prompts: {sum(1 for _ in prompts.open())}")


def export_joint(gazet_dir, generative_dir, out_dir, generative_repeat):
    """One model for both tasks: gazet plans + places (once) and the generative SQL pairs (`generative_repeat` times,
    i.e. its epochs), shuffled. The job answers gazet's compat prompts and every generative eval set."""
    gz, gen, out = Path(gazet_dir), Path(generative_dir), Path(out_dir)
    (out / "sql").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)

    def read(p):
        return [line for line in p.read_text().splitlines() if line.strip()]
    train = read(gz / "sql" / "train.jsonl") + read(gen / "sql" / "train.jsonl") * generative_repeat
    random.Random(13).shuffle(train)
    val = read(gz / "sql" / "val.jsonl")[:750] + read(gen / "sql" / "val.jsonl")[:750]
    (out / "sql" / "train.jsonl").write_text("\n".join(train) + "\n")
    (out / "sql" / "val.jsonl").write_text("\n".join(val) + "\n")
    for f in list((gz / "eval").glob("*.jsonl")) + list((gen / "eval").glob("*.jsonl")):
        (out / "eval" / f.name).write_text(f.read_text())
    print(f"train {len(train):,}  val {len(val):,}  eval files {len(list((out / 'eval').glob('*.jsonl')))}")


def bench(label, port, mode, per_family):
    """Seconds per gazet.lm call (places, then SQL or plan) on the same questions, first per_family of each family."""
    import os
    import time
    os.environ["LLAMA_SERVER_URL"] = f"http://127.0.0.1:{port}"
    os.environ["GAZET_SQL_MODE"] = mode
    import gazet_compat as gc

    from gazet import lm
    out = {}
    for task in ("places", "sql"):
        picked, seen = [], Counter()
        for i, r in gc.sample(task, 25):
            fam = r["metadata"].get("task_family")
            if seen[fam] < per_family:
                seen[fam] += 1
                picked.append(r)
        secs = []
        for r in picked:
            t0 = time.time()
            if task == "places":
                lm.generate_places(r["messages"][1]["content"])
            else:
                gc._sql_or_failure(lm, *gc.parse_sql_prompt(r["messages"][1]["content"]))
            secs.append(time.time() - t0)
        secs.sort()
        out[task] = {"n": len(secs), "median": round(secs[len(secs) // 2], 2), "mean": round(sum(secs) / len(secs), 2),
                     "p90": round(secs[int(0.9 * len(secs))], 2)}
    print(json.dumps({"label": label, "mode": mode, **out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    v = sub.add_parser("verify")
    v.add_argument("--per-template", type=int, default=6)
    v.add_argument("--split", default="test")
    e = sub.add_parser("export")
    e.add_argument("--out", default=str(Path.home() / ".cache" / "gazet-exp" / "chatml_plan_v1"))
    j = sub.add_parser("export-joint")
    j.add_argument("--gazet", default=str(Path.home() / ".cache" / "gazet-exp" / "chatml_plan_v1"))
    j.add_argument("--generative", default=str(Path.home() / ".cache" / "gazet-exp" / "chatml_v2"))
    j.add_argument("--out", default=str(Path.home() / ".cache" / "gazet-exp" / "chatml_joint_v1"))
    j.add_argument("--generative-repeat", type=int, default=2)
    b = sub.add_parser("bench")
    b.add_argument("label")
    b.add_argument("port", type=int)
    b.add_argument("mode", choices=["sql", "plan"])
    b.add_argument("--per-family", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "export-joint":
        export_joint(a.gazet, a.generative, a.out, a.generative_repeat)
    elif a.cmd == "bench":
        bench(a.label, a.port, a.mode, a.per_family)
    elif a.cmd == "labels":
        labels()
    elif a.cmd == "verify":
        verify(a.per_template, a.split)
    else:
        export(a.out)
