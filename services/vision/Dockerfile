FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

FROM python:3.11-slim

COPY --from=uv /uv /uvx /bin/

ENV PATH="/opt/pickleball-vision/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pickleball-vision/src \
    PORT=8000

WORKDIR /opt/pickleball-vision

COPY requirements-api.txt pyproject.toml README.md ./
RUN uv venv .venv && uv pip install --python .venv/bin/python --requirement requirements-api.txt

COPY src ./src
RUN uv pip install --python .venv/bin/python --no-deps .

RUN addgroup --system api \
    && adduser --system --ingroup api --home /nonexistent --no-create-home api \
    && chown -R api:api /opt/pickleball-vision

USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=3).read()"]

CMD ["sh", "-c", "exec uvicorn pickleball_vision.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
