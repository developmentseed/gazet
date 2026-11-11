"""Execution-match evaluation of a model on question + schema (+ value hints) -> DuckDB spatial SQL.

The model's SQL and the reference SQL are both run; the result sets are compared ignoring row and column
order (exact: 3 significant figures; lenient: every reference column appears in the prediction, 2 figures).

Usage:
    python evaluate.py run MODEL [--data gold|dev|FILE] [--hints] [--system rules|minimal] [--port N] [--limit N]
    python evaluate.py run NAME --preds PREDS.jsonl [--data ...] [--hints] --system minimal   # score saved answers
    python evaluate.py report [FILE...]        # table over runs/eval/*.jsonl

MODEL is an OpenRouter id, or with --port the name of a model served by a local llama-server on that port.
With --preds, answers come from a file of {"id", "text"[, "secs"]} lines (e.g. written by a training job).
Results go to runs/eval/<model>__<data>__<hints|plain>.jsonl; a rerun only asks for the pairs still missing.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import consistency
import harness
import hints as hints_mod
import student
from or_generate import URL, api_key

HERE = Path(__file__).parent
DATA = {"gold": HERE / "gold" / "gold_v1.jsonl", "dev": HERE / "dev" / "dev_v1.jsonl",
        "devnew": HERE / "dev" / "dev_new_v1.jsonl"}
OUT = HERE / "runs" / "eval"


def extract_sql(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def ask(model, system, user, port, key):
    """-> (sql, seconds, usage). Local servers get a tight token limit; hosted reasoning models need room to think."""
    body = dict(model=model, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if port:
        url, headers = f"http://127.0.0.1:{port}/v1/chat/completions", {}
        # llama-server turns Qwen3.5 thinking on by default; the student was trained without it
        body.update(max_tokens=512, chat_template_kwargs={"enable_thinking": False})
    else:
        url, headers = URL, {"Authorization": f"Bearer {key}"}
        body.update(max_tokens=24000, usage={"include": True})
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=600)
            if r.status_code == 200 and "choices" in r.json():
                j = r.json()
                return extract_sql(j["choices"][0]["message"].get("content")), time.time() - t0, j.get("usage") or {}
            err = r.text[:300]
        except requests.RequestException as e:
            err = str(e)[:300]
        time.sleep(2 ** (attempt + 1))
    return None, 0.0, {"error": err}


def run(args):
    path = DATA.get(args.data, Path(args.data))
    pairs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:args.limit]
    name = f"{args.model.replace('/', '__')}__{path.stem}__{'hints' if args.hints else 'plain'}"
    name += "" if args.system == "rules" else f"__{args.system}"
    outp = OUT / f"{name}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if outp.exists():
        done = {r["id"]: r for r in map(json.loads, outp.read_text().splitlines()) if r.get("pred_sql")}
    c = harness.con()
    cards = {t: harness.schema_card(c, t) for p in pairs for t in p["tables"]}
    todo = []
    for p in pairs:
        if p["id"] in done:
            continue
        h = hints_mod.value_hints(c, p["tables"], p["question"]) if args.hints else None
        todo.append(dict(p, hints=h, prompt=student.user_prompt(cards, p["tables"], p["question"], h)))
    key = None if args.port or args.preds else api_key()
    preds = {r["id"]: r for r in map(json.loads, Path(args.preds).read_text().splitlines())} if args.preds else None

    def one(p):
        if preds is not None:
            got = preds.get(p["id"])
            p.pop("prompt")
            return dict(p, solver_sql=extract_sql(got["text"]) if got else None, secs=(got or {}).get("secs", 0), cost=0,
                        out_tokens=(got or {}).get("out_tokens"), in_tokens=None, api_error=None if got else "no prediction")
        sql, secs, usage = ask(args.model, student.SYSTEMS[args.system], p.pop("prompt"), args.port, key)
        return dict(p, solver_sql=sql, secs=round(secs, 2), cost=usage.get("cost") or 0,
                    out_tokens=usage.get("completion_tokens"), in_tokens=usage.get("prompt_tokens"), api_error=usage.get("error"))

    with ThreadPoolExecutor(1 if args.port else 8) as ex:
        answered = list(ex.map(one, todo))
    with ThreadPoolExecutor(4) as ex:
        compared = list(ex.map(consistency._compare_isolated, answered))
    rows = list(done.values()) + [dict(r, pred_sql=r.pop("solver_sql")) for r in compared]
    order = {p["id"]: i for i, p in enumerate(pairs)}
    rows.sort(key=lambda r: order.get(r["id"], 1e9))
    outp.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n")
    print(f"{len(answered)} new answers, cost ${sum(r['cost'] for r in answered):.3f} -> {outp}")
    report([outp], detail=True)


def _ci(flags, n_boot=2000):
    """Mean and 95% bootstrap interval, in percent."""
    if not flags:
        return "-"
    rnd, n = random.Random(0), len(flags)
    means = sorted(sum(rnd.choices(flags, k=n)) / n for _ in range(n_boot))
    return f"{100 * sum(flags) / n:5.1f} [{100 * means[int(.025 * n_boot)]:.0f}-{100 * means[int(.975 * n_boot)]:.0f}]"


def _held_out(r):
    return any(t in harness.HELD_OUT for t in r["tables"])


def report(paths, detail=False):
    paths = [Path(p) for p in paths] or sorted(OUT.glob("*.jsonl"))
    print(f"{'run':<58} {'n':>4} {'executes':>14} {'exact':>14} {'lenient':>14} {'s/query':>8}")
    for p in paths:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        ok = lambda r: bool(r.get("pred_sql")) and not r.get("pred_error")
        secs = sorted(r.get("secs") or 0 for r in rows)
        print(f"{p.stem:<58} {len(rows):>4} {_ci([ok(r) for r in rows]):>14} {_ci([r['exact'] for r in rows]):>14} "
              f"{_ci([r['lenient'] for r in rows]):>14} {secs[len(secs) // 2]:>8.1f}")
        if not detail:
            continue
        groups = defaultdict(list)
        for r in rows:
            groups["difficulty: " + r["difficulty"]].append(r)
            groups["skill: " + r["skill"].split(" (")[0][:45]].append(r)
            groups["tables: " + ("held-out" if _held_out(r) else "seen in training")].append(r)
            groups["hints: " + ("none requested" if r.get("hints") is None else "found" if r["hints"] else "nothing found")].append(r)
        for g in sorted(groups):
            rs = groups[g]
            print(f"  {g:<56} {len(rs):>4} {_ci([ok(r) for r in rs]):>14} {_ci([r['exact'] for r in rs]):>14} {_ci([r['lenient'] for r in rs]):>14}")
        errs = defaultdict(int)
        for r in rows:
            if r.get("pred_error"):
                errs[re.sub(r"[\"'`].*", "", r["pred_error"])[:70]] += 1
        for e, n in sorted(errs.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  error x{n}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--data", default="gold")
    r.add_argument("--hints", action="store_true")
    r.add_argument("--system", default="rules", choices=list(student.SYSTEMS))
    r.add_argument("--port", type=int)
    r.add_argument("--preds")
    r.add_argument("--limit", type=int)
    rep = sub.add_parser("report")
    rep.add_argument("files", nargs="*")
    rep.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a.files, a.detail)
