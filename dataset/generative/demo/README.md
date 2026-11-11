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

A 0.8B model fine-tuned to write DuckDB spatial SQL from a question and table descriptions, running on CPU with
llama.cpp, the same way it would run on a laptop. Pick tables, ask a question, and the app shows the SQL, runs it
on the data, and maps the result.

Tables marked "new" were never seen in training (the gold evaluation set uses them).

Source: `dataset/generative/demo/` in the gazet repository (`deploy.sh` builds this Space). Needs the Space secret
`HF_TOKEN` with read access to the private model and dataset repos.

## Run locally

Needs Python 3.12, a llama.cpp build with Qwen3.5 support (release b11935 or later; `brew install llama.cpp` on a Mac)
and an HF token with read access to the private repos (`hf auth login`).

```bash
cd dataset/generative
rm -rf /tmp/gazet-demo && mkdir /tmp/gazet-demo && cp demo/app.py demo/requirements.txt harness.py hints.py student.py /tmp/gazet-demo/
cd /tmp/gazet-demo && uv venv -p 3.12 && uv pip install -r requirements.txt
LLAMA_SERVER=$(which llama-server) .venv/bin/python app.py    # then open http://localhost:7860
```

The first start downloads the data (137 MB) and the model (834 MB).

## Hosting

Docker Spaces need a paid plan: PRO on a personal account, Team for the developmentseed organisation. With one,
`demo/deploy.sh [SPACE_ID]` creates the private Space; then add a read-only token as the Space secret `HF_TOKEN`.
