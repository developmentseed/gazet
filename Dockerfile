FROM ghcr.io/ggml-org/llama.cpp:server AS llama

FROM python:3.13-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl supervisor libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary and backend .so files (must stay together)
COPY --from=llama /app /usr/local/lib/llama
RUN ln -s /usr/local/lib/llama/llama-server /usr/local/bin/llama-server \
    && echo /usr/local/lib/llama > /etc/ld.so.conf.d/llama.conf \
    && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib/llama

# HF Spaces requires UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies (cache layer)
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra demo

# Copy application code
COPY --chown=user src/ src/
COPY --chown=user gazet_demo.py .
RUN uv sync --frozen --extra demo

# Download model from HF
RUN uv run python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('developmentseed/gazet-model', 'models/ckpt-001.gguf', local_dir='.')"

# Download geodata from HF (repo structure matches app's expected layout)
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('developmentseed/gazet-geodata', repo_type='dataset', local_dir='data')"

COPY --chown=user supervisord.conf .

ENV GAZET_DATA_DIR=$HOME/app/data \
    LLAMA_SERVER_URL=http://localhost:9000 \
    GAZET_API_URL=http://localhost:8000 \
    PATH="$HOME/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["supervisord", "-c", "supervisord.conf"]
