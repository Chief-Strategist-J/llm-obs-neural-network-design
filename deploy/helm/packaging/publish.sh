#!/usr/bin/env bash
set -euo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${CHART_DIR}/dist"
REGISTRY="${1:-oci://ghcr.io/scaibu/charts}"

if [ ! -d "${DIST_DIR}" ]; then
  bash "${CHART_DIR}/packaging/package.sh" "${DIST_DIR}"
fi

PACKAGE_FILE=$(find "${DIST_DIR}" -name "*.tgz" | head -n 1)
if [ -z "${PACKAGE_FILE}" ]; then
  echo "Error: No chart archive found in ${DIST_DIR}"
  exit 1
fi

echo "Publishing ${PACKAGE_FILE} to ${REGISTRY}..."
helm push "${PACKAGE_FILE}" "${REGISTRY}"
echo "Successfully published chart."
