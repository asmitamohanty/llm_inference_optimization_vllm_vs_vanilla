# LLM Inference Benchmark: Vanilla Transformers vs vLLM on GKE

## Overview

This project benchmarks vanilla Hugging Face Transformers against vLLM on Google Kubernetes Engine (GKE) using NVIDIA GPU L4.The goal is to evaluate how modern inference optimizations such as 1)Continuous batching & 2)Chunked prefill affect latency and throughput under realistic workloads.

The benchmark suite measures:

- `Context Sweep` – varying prompt length under low concurrency. 
- `Request Sweep` – varying concurrent requests under high concurrency. 
- `Mixed Context Sweep` – simultaneous heterogeneous prompt lengths comparing vLLM with vs without chunked prefill.

### Stack
- Python
- FastAPI
- Hugging Face Transformers - Qwen2.5-7B-Instruct
- vLLM
- Kubernetes (GKE)
- NVIDIA L4 GPU (Single GPU)
- Prometheus + DCGM Exporter
- Matplotlib
- Dataset - [HotPotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/main/distractor)

## Quickstart

### Pre-Requisites (For GKE)
- Setup environment & GCP project zones/regions/services
```
source scripts/env.sh
bash scripts/setup.sh
```
### Deployment - vLLM vs Vanilla Transformers
- Create 2 separate containers - `vllm-server` vs `llm-inference`
- For each container, run the benchmark suite separately
- For vLLM - set the `scripts/env.sh` to enable/disable `chunked-prefill` flag
- For vanilla transformers: `BUILD -> DEPLOY`
  ```
  bash scripts/build_cloud.sh
  bash scripts/deploy.sh
  ```
- For vLLM: `DEPLOY` . We are using the [vLLM's docker image for NVIDIA GPU](https://docs.vllm.ai/en/stable/deployment/docker/)
  ```
  bash scripts/deploy.sh vllm
  ```

### Run Benchmark
- Get your service endpoint once your pod is activated. Check `kubectl get pods -o wide`
- Get your `EXTERNAL-IP` aka `ENDPOINT`. Check `kubectl get svc`
- Add the following while running the benchmark script:
  - backend-type: `transformers` or `vllm`
  - benchmark-type: `context` or `request` or `mixed-context`
  - EXTERNAL-IP: Choose the address under the respective service - `llm-service` or `vllm-service`
    
```
ENDPOINT=<EXTERNAL-IP> python benchmark.py --backend <backend-type> --dataset-choice longbench --<benchmark-type>-sweep
```

## Results
- Given model config & hardware constraints support the default `max_model_len=32768` in vLLM

<details>
  <summary><strong>Request Sweep</strong></summary>
  
 - All measurements are taken under batch size=1, context length=512 & max-new-tokens=128
  
| Metric| Results|
|:------:|:-------:|
|Avg ITL|  <img src="plots/request/request_avg_itl_sec.png" width="200">          |
|Tail/P99 Latency|  <img src="plots/request/request_p99_latency_sec.png" width="200">        |
|Aggregate Throughput|   <img src="plots/request/request_aggr_tps.png" width="200">          |
|Avg TTFT| <img src="plots/request/request_avg_ttft_sec.png" width="200">   |

</details>

<details>
  <summary><strong>Context Sweep</strong></summary>
  
  - All measurements are taken under batch size=1 & max-new-tokens=128 under low concurrency
  
| Metric| Results|
|:------:|:-------:|
|Latency| <img src="plots/context/context_latency_sec.png" width="200">|
|TTFT| <img src="plots/context/context_tokens_per_sec.png" width="200">            |
|Throughput| <img src="plots/context/context_ttft_sec.png" width="200">      |

</details>


<details>
  <summary><strong>Mixed Context Sweep</strong></summary>
  
  - All measurements are taken under max-new-tokens=128 & max-num-batched-tokens=4096 (chunk size) simulating batch size=36 mixed context requests

**(a) vLLM - With vs Without Chunked Prefill**
| Metric| Results|
|:------:|:-------:|
|Avg ITL|  <img src="plots/mixed/mixed_avg_itl_sec.png" width="200">          |
|Tail/P99 Latency|  <img src="plots/mixed/mixed_p99_latency_sec.png" width="200">        |
|Avg Throughput|   <img src="plots/mixed/mixed_avg_tps.png" width="200">          |
|Avg TTFT| <img src="plots/mixed/mixed_avg_ttft_sec.png" width="200">   |


**(b) vLLM With Chunked Prefill: Sweep Chunk Size aka max-num-batched-tokens**
- Best Chunk Size found at: max-num-batched-tokens=4096
  
| Metric| Results|
|:------:|:-------:|
|Avg ITL|  <img src="plots/mixed_batched_tokens/mixed_avg_itl_sec.png" width="200">          |
|Tail/P99 Latency|  <img src="plots/mixed_batched_tokens/mixed_p99_latency_sec.png" width="200">        |
|Avg Throughput|   <img src="plots/mixed_batched_tokens/mixed_avg_tps.png" width="200">          |
|Avg TTFT| <img src="plots/mixed_batched_tokens/mixed_avg_ttft_sec.png" width="200">   |
</details>

## Summary
- vLLM outperforms vanilla transformer under high concurrent loads by roughly **11x** higher in throughput, with **345x** gains at highest concurreny load of 50 & **26x** lower in ITL, sacrificing some TTFT to protect ITL for in-flight requests.
- For low concurrency, batch size=1 & varying context lengths, vLLM's **8%** decoding efficiency is eclipsed by **12-78%** scheduling overhead compared to vanilla transformer. This is expected from vLLM's architecture which is designed to benefit in maximum throughput at high concurrency.
- Under mixed context loads for a given batch size:

  (a) With vs Without Chunked Prefill: vLLM shows comparable performance in almost every metric except the tail latency showing **8%** drop. Root cause is due to the addition of scheduling steps when context length > chunk size.
  
  (b) Sweeping chunk size (max-num-batched-tokens) with chunked prefill: Higher chunk size shows improved performance in all the metrics due to reducing scheduling overhead/steps for larger contexts.

  - The above data shows that under mixed context loads for a single GPU, the real lever is the chunk size that drives the performance. 

## References
- vLLM - https://docs.vllm.ai/en/stable/

    
