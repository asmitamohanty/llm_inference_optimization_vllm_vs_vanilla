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
- Dataset - HotPotQA

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
- Given model config & hardware constraints support the default `max_model_len=32768` in vLLM

<details>
  <summary><strong>Request Sweep</strong></summary>
 - All measurements are taken under batch size=1, context length=512 & max-new-tokens=128
  
| Metric| Results|
|:------:|:-------:|
|Avg ITL|            |
|Tail/P99 Latency|          |
|Aggregate Throughput|             |
|TTFT|    |

</details>

<details>
  <summary><strong>Context Sweep</strong></summary>
  - All measurements are taken under batch size=1 & max-new-tokens=128 under low concurrency
  
| Metric| Results|
|:------:|:-------:|
|Latency|          |
|TTFT|             |
|Throughput|       |

</details>


<details>
  <summary><strong>Mixed Context Sweep</strong></summary>
  - All measurements are taken under max-new-tokens=128 & max-num-batched-tokens=4096 (chunk size) simulating batch size with mixed contexts

**(a) vLLM - With vs Without Chunked Prefill**
| Metric| Results|
|:------:|:-------:|
|Avg ITL|              |
|Tail/P99 Latency|          |
|Aggregate Throughput|             |
|TTFT|    |

**(b) vLLM With Chunked Prefill: Sweep Chunk Size aka max-num-batched-tokens**
| Metric| Results|
|:------:|:-------:|
|Avg ITL|              |
|Tail/P99 Latency|          |
|Aggregate Throughput|             |
|TTFT|    |
</details>

## Summary
- vLLM outperforms vanilla transformer under high concurrent loads by roughly **11x** higher in throughput, with **345x** gains at highest concurreny load of 50 & **26x** lower in ITL, sacrificing some TTFT to protect ITL for in-flight requests.
- For low concurrency, batch size=1 & varying context lengths, vLLM's 8% decoding efficiency is eclipsed by 12-78% scheduling overhead compared to vanilla transformer. This is expected from vLLM's architecture which is designed to benefit in maximum throughput at high concurrency.
- Under mixed context loads in a given batch size:
  (a) With vs Without Chunked Prefill: vLLM shows comparable performance in almost every metric except the tail latency showing 8% drop. Root cause is due to the addition of scheduling steps when context length > chunk size.
  (b) Sweeping chunk size (max-num-batched-tokens) with chunked prefill: Higher chunk size shows improved performance in all the metrics due to reducing scheduling overhead/steps for larger contexts.
  - The above data shows that under mixed context loads for a single GPU, the real lever is the chunk size that drives the performance. 

    
