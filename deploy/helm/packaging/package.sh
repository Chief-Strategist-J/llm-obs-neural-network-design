#!/usr/bin/env bash
set -euo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${CHART_DIR}/dist}"

mkdir -p "${OUTPUT_DIR}"
helm package "${CHART_DIR}" --destination "${OUTPUT_DIR}"
echo "Successfully packaged Helm chart to ${OUTPUT_DIR}"
