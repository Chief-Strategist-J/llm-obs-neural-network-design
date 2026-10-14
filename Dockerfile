FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY contracts ./contracts
COPY config ./config

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

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
