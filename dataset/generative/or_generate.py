"""Run the scenario prompts against an OpenRouter model and save raw outputs + usage.

Usage: python or_generate.py MODEL --prompts 'runs/prompts_v3/*.txt' [--quant fp8,bf16] [--reasoning low] [--tag v3]
Key: $OPENROUTER_API_KEY or ~/.config/openrouter/key
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).parent
URL = "https://openrouter.ai/api/v1/chat/completions"


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        key = (Path.home() / ".config/openrouter/key").read_text().strip()
    return key


def call(model, prompt, quant, reasoning, key, temperature=0.7):
    body = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
                max_tokens=32000, usage={"include": True})
    if quant:
        # Only route to full/near-full precision deployments; fail rather than fall back to fp4/int4.
        body["provider"] = {"quantizations": quant, "allow_fallbacks": True}
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    for attempt in range(4):
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
        if r.status_code == 200 and "choices" in r.json():
            return r.json()
        time.sleep(2 ** (attempt + 1))
    return {"error": r.text[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--quant", default="fp8,bf16,fp16")
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. v2")
    ap.add_argument("--prompts", required=True, help="glob of prompt .txt files")
    ap.add_argument("--root", default="runs", help="directory (under this folder) for outputs")
    a = ap.parse_args()
    key = api_key()
    quant = [q for q in a.quant.split(",") if q]
    slug = a.model.replace("/", "__").replace(":", "_") + (f"__r{a.reasoning}" if a.reasoning else "") + (f"__{a.tag}" if a.tag else "")
    out = HERE / a.root / f"out_{slug}"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(a.prompts))

    def done_ok(sid):
        f = out / f"{sid}.json"
        try:
            return f.exists() and isinstance(json.loads(f.read_text().strip().strip("`").removeprefix("json")), list)
        except json.JSONDecodeError:
            return False

    files = [f for f in files if not done_ok(Path(f).stem)]
    print(f"{len(files)} scenarios to (re)generate")

    def run(f):
        sid = Path(f).stem
        t = time.time()
        resp = call(a.model, Path(f).read_text(), quant, a.reasoning, key)
        wall = time.time() - t
        if "error" in resp:
            print(f"{sid}: ERROR {resp['error'][:200]}")
            return None
        msg = resp["choices"][0]["message"]
        (out / f"{sid}.json").write_text(msg.get("content") or "")
        u = resp.get("usage", {})
        return dict(scenario=sid, provider=resp.get("provider"), wall_s=round(wall, 1),
                    prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                    reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    cost=u.get("cost"))

    with ThreadPoolExecutor(6) as ex:
        usage = [u for u in ex.map(run, files) if u]
    prev = json.loads((out / "_usage.json").read_text()) if (out / "_usage.json").exists() else []
    redone = {u["scenario"] for u in usage}
    (out / "_usage.json").write_text(json.dumps([u for u in prev if u["scenario"] not in redone] + usage, indent=1))
    cost = sum(u["cost"] or 0 for u in usage)
    print(f"{a.model}: {len(usage)}/{len(files)} ok, cost ${cost:.4f}, "
          f"providers {sorted({u['provider'] for u in usage})}, "
          f"avg completion {sum(u['completion_tokens'] or 0 for u in usage)/max(1,len(usage)):.0f} tok "
          f"(reasoning {sum(u['reasoning_tokens'] or 0 for u in usage)/max(1,len(usage)):.0f})")


if __name__ == "__main__":
    main()
