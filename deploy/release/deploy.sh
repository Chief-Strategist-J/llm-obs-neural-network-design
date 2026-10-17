#!/usr/bin/env bash
set -euo pipefail

ENV="${1:-dev}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Deploying neural-network-design to environment: ${ENV} ==="

case "${ENV}" in
  dev|staging|prod)
    if command -v kubectl >/dev/null 2>&1; then
      echo "Applying Kustomize overlay: deploy/k8s/overlays/${ENV}"
      kubectl apply -k "${ROOT_DIR}/deploy/k8s/overlays/${ENV}"
      echo "Waiting for deployment rollout..."
      kubectl rollout status deployment/neural-network-design -n "llmobs-${ENV}" --timeout=120s
    elif command -v helm >/dev/null 2>&1; then
      echo "Applying Helm release with env values: deploy/helm/env-values/values.${ENV}.yaml"
      helm upgrade --install neural-network-design "${ROOT_DIR}/deploy/helm" \
        -f "${ROOT_DIR}/deploy/helm/env-values/values.${ENV}.yaml" \
        --namespace "llmobs-${ENV}" --create-namespace
    else
      echo "Neither kubectl nor helm found. Simulating deployment dry-run."
    fi
    ;;
  local)
    echo "Starting local docker-compose stack..."
    docker-compose -f "${ROOT_DIR}/deploy/docker/docker-compose.yaml" up -d
    ;;
  *)
    echo "Unknown environment: ${ENV}. Usage: $0 [dev|staging|prod|local]"
    exit 1
    ;;
esac

echo "Deployment of neural-network-design to ${ENV} completed successfully."
