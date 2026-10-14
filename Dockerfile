FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY contracts ./contracts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi>=0.100.0 \
        uvicorn>=0.22.0 \
        pyyaml>=6.0 \
        pydantic>=2.0.0 \
        opentelemetry-api>=1.26.0 \
        opentelemetry-sdk>=1.26.0 \
        confluent-kafka>=2.3.0 \
        psycopg[binary]>=3.1.0 \
        httpx>=0.27.0 \
        prometheus-client>=0.17.0 && \
    pip install --no-cache-dir --no-deps .

FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["bash", "scripts/run.sh"]
