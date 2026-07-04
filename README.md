# LLM Inference Benchmark: Vanilla Transformers vs vLLM on GKE

## Overview

This project benchmarks vanilla Hugging Face Transformers against vLLM on Google Kubernetes Engine (GKE) using NVIDIA GPU L4.The goal is to evaluate how modern inference optimizations such as 1)Continuous batching & 2)Chunked prefill affect latency and throughput under realistic workloads.

The benchmark suite measures:

- `Context Sweep` – varying prompt length. Benchmark flag - `--context-sweep`
- `Request Sweep` – varying concurrent requests. Benchmark flag - `--request-sweep`
- `Mixed Context Sweep` – simultaneous heterogeneous prompt lengths. Benchmark flag - `--mixed-context-sweep`

### Stack
- Python
- FastAPI
- Hugging Face Transformers - Qwen2.5-7B-Instruct
- vLLM
- Kubernetes (GKE)
- NVIDIA L4 GPU
- Prometheus + DCGM Exporter
- Matplotlib
- Dataset - HotPotQA

### Metrics
Each benchmark records:
- End-to-end latency
- TTFT (Time To First Token)
- Average ITL (Inter-Token Latency)
- P99 ITL
- Decode throughput (tokens/sec)
- Aggregate throughput
- Prompt generation overhead
- GPU utilization (DCGM)
- Prompt length
- Output length

## Quickstart
- Setup environment & GCP project zones/regions/services
```
source scripts/env.sh
bash scripts/setup.sh
```
### Deployment - vLLM vs Vanilla Transformers
- Create 2 separate containers - `vllm-server` vs `llm-inference`
- For each container, run the benchmark suite separately
- For vLLM - set the `scripts/env.sh` to enable/disable `chunked-prefill` flag

### Run Benchmark
- Get your service endpoint once your pod is activated. Check `kubectl get pods -o wide`
- Get your `EXTERNAL-IP` aka `ENDPOINT`. Check `kubectl get svc`
- Run the following:
  - backend-type: `transformers` or `vllm`
  - benchmark-type: `context` or `request` or `mixed-context`
  - EXTERNAL-IP: Choose the address under the respective service - `llm-service` or `vllm-service`
```
ENDPOINT=<EXTERNAL-IP> python benchmark.py --backend <backend-type> --dataset-choice longbench --<benchmark-type>-sweep
```

## Results
