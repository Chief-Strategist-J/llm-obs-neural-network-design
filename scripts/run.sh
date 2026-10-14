#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "Starting Neural Network Design Service on ${HOST}:${PORT}..."
exec uvicorn src.api.rest.v1.app:app --host "${HOST}" --port "${PORT}"
