#!/bin/bash
set -e

source ./scripts/env.sh

echo "Rendering deployment YAML with env vars..."

if [ "$1" = "vllm" ]; then
    envsubst < ./scripts/vllm_deployment.yaml | kubectl apply -f -
    kubectl apply -f ./scripts/vllm_service.yaml
else
    envsubst < ./scripts/deployment.yaml | kubectl apply -f -
    kubectl apply -f ./scripts/services.yaml
fi

echo "Deployment applied."