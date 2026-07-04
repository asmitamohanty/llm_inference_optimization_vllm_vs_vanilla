# env.sh

export PROJECT_ID="llm-gke-498501"
export REGION="us-central1"
export ZONE="us-central1-a"

export REPO_NAME="llm-demo-repo"
export IMAGE_NAME="qwen-server"
export IMAGE_TAG="v15"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export ENABLED_CHUNKED_PREFILL="--no-enable-chunked-prefill" #"--enable-chunked-prefill"

export CLUSTER_NAME="llm-evals-demo-cluster"

export AR_HOST="${REGION}-docker.pkg.dev"
export IMAGE_URI="${AR_HOST}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
export IMAGE_VLLM_URI="${AR_HOST}/${PROJECT_ID}/${REPO_NAME}/vllm-${IMAGE_NAME}:vllm-${IMAGE_TAG}"