#!/bin/bash
set -e

source ./scripts/env.sh

echo "Submitting build to Cloud Build..."

MODE=${1:-transformers}
CLOUD_CONFIG=./scripts/cloudbuild.yaml

if [ "$MODE" = "vllm" ]; then
    IMAGE_NAME="vllm-${IMAGE_NAME}"
    IMAGE_TAG="vllm-${IMAGE_TAG}"
    DOCKERFILE="Dockerfile.vllm"
else
    DOCKERFILE="Dockerfile"
fi

echo "Image name: $IMAGE_NAME"
echo "Image tag: $IMAGE_TAG"

echo "Dockerfile: $DOCKERFILE"

gcloud builds submit \
  --config="${CLOUD_CONFIG}" \
  --substitutions=_DOCKERFILE="${DOCKERFILE}",_REPO_NAME="${REPO_NAME}",_IMAGE_NAME="${IMAGE_NAME}",_IMAGE_TAG="${IMAGE_TAG}" \
  .

echo "Build completed."