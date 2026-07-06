#!/bin/bash
set -e

source ./scripts/env.sh

echo "Plotting results..."
CSV1="/home/asmita_itspersonal/llm-gke-demo/results/final_outputs/vllm-chunk-ON/vllm_chunkEN_2048_mixed_context_sweep.csv"
CSV2="/home/asmita_itspersonal/llm-gke-demo/results/final_outputs/vllm-chunk-ON/vllm_chunkEN_3072_mixed_context_sweep.csv"
CSV3="/home/asmita_itspersonal/llm-gke-demo/results/final_outputs/vllm-chunk-ON/vllm_chunkEN_4096_31reqs_mixed_context_sweep.csv"

python plot_benchmarks.py \
    --experiment mixed \
    --csv ${CSV1} ${CSV2} ${CSV3} \
    --labels vLLM-CHUNK-ON-2048 vLLM-CHUNK-ON-3072 vLLM-CHUNK-ON-4096\
    --output-dir plots/mixed_batched_tokens