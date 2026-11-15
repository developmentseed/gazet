"""pass@k from sampled answers: how often a run gets a question right in at least one of k tries.

The gap between greedy accuracy (one answer, temperature 0) and pass@k is the headroom reinforcement learning can
reach: RL makes the model pick, more reliably, answers it can already produce. Questions it never gets right in
n tries are beyond RL; questions it gets right sometimes are the useful RL training signal.

Samples come from finetune/sample_hf_jobs.py (<run>/samples/<data>_plain.jsonl in the model repo). Identical SQL
strings for one question are executed once.

Usage: python passk.py RUN [RUN...]   -> runs/passk/<run>__<data>.jsonl and a table
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import consistency
import evaluate
import harness

HERE = Path(__file__).parent
W = Path.home() / ".cache" / "gazet-exp" / "models"
REPO = "developmentseed/gazet-sql-qwen35"
FILES = {"gold": "gold_plain", "gold2": "gold2_plain", "devnew": "devnew_plain"}
OUT = HERE / "runs" / "passk"


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def fetch(run):
    files = [f"{run}/samples/{f}.jsonl" for f in FILES.values()]
    subprocess.run(["hf", "download", REPO, *files, "--local-dir", str(W)], check=True, capture_output=True)


def score(run, data):
    out = OUT / f"{run}__{data}.jsonl"
    if out.exists():
        return [json.loads(line) for line in out.read_text().splitlines()]
    gold = {r["id"]: r for r in map(json.loads, evaluate.DATA[data].read_text().splitlines()) if r}
    samples = [json.loads(line) for line in (W / run / "samples" / f"{FILES[data]}.jsonl").read_text().splitlines()]
    jobs = {}
    for s in samples:
        for sql in {evaluate.extract_sql(t) for t in s["texts"]}:
            jobs[(s["id"], sql)] = {"sql": gold[s["id"]]["sql"], "solver_sql": sql}
    keys = list(jobs)
    with ThreadPoolExecutor(harness.COMPARE_WORKERS) as ex:
        results = dict(zip(keys, ex.map(consistency._compare_isolated, [jobs[k] for k in keys])))
    rows = []
    for s in samples:
        ok = [bool(results[(s["id"], evaluate.extract_sql(t))]["lenient"]) for t in s["texts"]]
        rows.append({"id": s["id"], "n": len(ok), "c": sum(ok), "distinct": len({evaluate.extract_sql(t) for t in s["texts"]})})
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def greedy(run, data):
    p = evaluate.OUT / f"{run}__{evaluate.DATA[data].stem}__plain__minimal.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return 100 * sum(bool(r.get("lenient")) for r in rows) / len(rows)


def main():
    print(f"{'run':24s} {'data':7s} {'greedy':>7s} {'pass@1':>7s} {'pass@2':>7s} {'pass@4':>7s} {'pass@8':>7s}"
          f" {'never':>6s} {'some':>6s} {'always':>6s} {'distinct':>8s}")
    for run in sys.argv[1:]:
        fetch(run)
        for data in FILES:
            rows = score(run, data)
            n = len(rows)
            pk = {k: 100 * sum(pass_at_k(r["n"], r["c"], k) for r in rows) / n for k in (1, 2, 4, 8)}
            never = 100 * sum(r["c"] == 0 for r in rows) / n
            always = 100 * sum(r["c"] == r["n"] for r in rows) / n
            dist = sum(r["distinct"] for r in rows) / n
            print(f"{run:24s} {data:7s} {greedy(run, data):7.1f} {pk[1]:7.1f} {pk[2]:7.1f} {pk[4]:7.1f} {pk[8]:7.1f}"
                  f" {never:6.1f} {100 - never - always:6.1f} {always:6.1f} {dist:8.1f}")


if __name__ == "__main__":
    main()
