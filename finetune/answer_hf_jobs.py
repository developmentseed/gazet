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
"""Answer captured chat prompts with one or more models on a GPU, as gazet's llama-server would.

Prompts ({"id", "messages"}) come from dataset/generative/gazet_compat.py `prompts`, which records the exact
messages gazet.lm sends. Each model renders them with its own chat template and thinking off (gazet sends
chat_template_kwargs enable_thinking=false) and answers greedily with up to 2,048 new tokens (gazet's
LLAMA_TEMPERATURE / LLAMA_MAX_TOKENS defaults). Writes gazet_compat/answers/<label>.jsonl ({"id", "text",
"out_tokens"}) to the dataset repo.

hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN finetune/answer_hf_jobs.py \
    --model gazet-v1=developmentseed/gazet-model:merged --model rl2-q3-06b=developmentseed/gazet-sql-qwen35:rl2-q3-06b/merged
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
DATASET = "developmentseed/gazet-generative-sql"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label=repo:subfolder")
    ap.add_argument("--prompts", default="gazet_compat/prompts.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in pathlib.Path(hf_hub_download(DATASET, a.prompts, repo_type="dataset")).read_text().splitlines()]
    out_dir = pathlib.Path("/tmp/answers")
    out_dir.mkdir(exist_ok=True)
    for spec in a.model:
        label, rest = spec.split("=", 1)
        repo, sub = rest.split(":", 1)
        root = pathlib.Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"])) / sub
        tok = AutoTokenizer.from_pretrained(root)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, device_map="cuda").eval()
        texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
        res, t0 = [None] * len(rows), time.time()
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
            for i, ids in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                n = int((ids != tok.pad_token_id).sum())
                res[i] = {"id": rows[i]["id"], "text": tok.decode(ids, skip_special_tokens=True), "out_tokens": n}
        (out_dir / f"{label}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
        print(f"{label}: {len(rows)} answers in {(time.time() - t0) / 60:.1f} min; first: {res[0]['text'][:120]!r}")
        del model
        torch.cuda.empty_cache()
    HfApi().upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=str(out_dir), path_in_repo="gazet_compat/answers",
                          commit_message="Add gazet compatibility answers")
    print("uploaded")


if __name__ == "__main__":
    main()
