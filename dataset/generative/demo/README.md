---
title: Gazet SQL demo
emoji: 🗺️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Gazet SQL demo

A 0.6B model (Qwen3-0.6B, fine-tuned, then two rounds of RL on an execution reward) that writes DuckDB spatial SQL from a question and table descriptions, running on CPU with
llama.cpp, the same way it would run on a laptop. Pick tables, ask a question, and the app shows the SQL, runs it
on the data, and maps the result.

Tables marked "new" were never seen in training (the gold evaluation set uses them).

Source: `dataset/generative/demo/` in the gazet repository (`deploy.sh` builds this Space). The model
(`developmentseed/gazet-models`) and data (`developmentseed/gazet-sql-data`) are public, so no token is needed.

## Run locally

Needs Python 3.12 and a llama.cpp build (release b11935 or later; `brew install llama.cpp` on a Mac).

```bash
cd dataset/generative
rm -rf /tmp/gazet-demo && mkdir /tmp/gazet-demo && cp demo/app.py demo/requirements.txt harness.py hints.py student.py /tmp/gazet-demo/
cd /tmp/gazet-demo && uv venv -p 3.12 && uv pip install -r requirements.txt
LLAMA_SERVER=$(which llama-server) .venv/bin/python app.py    # then open http://localhost:7860
```

The first start downloads the data (193 MB) and the model (640 MB).

## Hosting

Docker Spaces need a paid plan: PRO on a personal account, Team for the developmentseed organisation. With one,
`demo/deploy.sh [SPACE_ID]` creates the private Space. Pointing `MODEL_REPO` or `DATA_REPO` at a private repo needs a
read-only token as the Space secret `HF_TOKEN`.
