FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*

# llama.cpp CPU build, same release as the local evaluation (supports Qwen3.5)
ARG LLAMA_TAG=b11935
RUN curl -sL https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz \
      | tar xz -C /opt && ln -s /opt/llama-${LLAMA_TAG} /opt/llama

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && python -c "import duckdb; duckdb.connect().execute('INSTALL spatial')"
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
