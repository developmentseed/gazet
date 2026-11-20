# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "torch==2.14.1",
#     "transformers==5.2.0",
#     "flash-linear-attention",
#     "accelerate",
#     "huggingface_hub",
# ]
# ///
"""Sample N answers per test prompt from a trained run's merged model, for pass@k (RL headroom).

Reads <run>/merged/ from the model repo and the test prompts from the dataset's <data>/eval/, writes
<run>/samples/<file>.jsonl ({"id", "texts": [...]}) back to the model repo. Prompts are built exactly as in
train_hf_jobs_qwen35.py (training-format text cut before the answer).

hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN finetune/sample_hf_jobs.py --run-name sql-v2-coder05
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

DATASET = "developmentseed/gazet-sql-data"
MODEL_REPO = "developmentseed/gazet-models"
_ANS = "\x00ANSWER\x00"


def prompt_text(tokenizer, messages):
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": _ANS}], tokenize=False)
    return full[:full.index(_ANS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default="chatml/v2")
    ap.add_argument("--files", default="gold_plain,gold2_plain,devnew_plain")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(a.seed)
    root = pathlib.Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{a.run_name}/merged/*"])) / a.run_name / "merged"
    data = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=[f"{a.data}/eval/*"])) / a.data / "eval"
    tok = AutoTokenizer.from_pretrained(root)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, device_map="cuda").eval()

    out = pathlib.Path("/tmp/samples") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    for name in a.files.split(","):
        rows = [json.loads(line) for line in (data / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        texts = [prompt_text(tok, r["messages"]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
        res, t0 = [None] * len(rows), time.time()
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=a.top_p,
                                     num_return_sequences=a.n, max_new_tokens=a.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                res[i] = {"id": rows[i]["id"],
                          "texts": [tok.decode(new[j * a.n + k], skip_special_tokens=True) for k in range(a.n)]}
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
        print(f"{name}: {len(rows)} prompts x {a.n} samples in {time.time() - t0:.0f} s; first: {res[0]['texts'][0][:150]!r}")

    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    HfApi().upload_folder(repo_id=MODEL_REPO, folder_path=str(out), path_in_repo=f"{a.run_name}/samples",
                          commit_message=f"Add pass@k samples for {a.run_name}")
    print(f"uploaded to {MODEL_REPO}/{a.run_name}/samples")


if __name__ == "__main__":
    main()
