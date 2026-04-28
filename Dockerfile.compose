FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra demo

# Copy application code
COPY src/ src/
COPY gazet_demo.py .

# Install the project itself
RUN uv sync --frozen --extra demo

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501
