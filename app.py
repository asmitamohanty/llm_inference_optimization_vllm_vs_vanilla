from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

import time
from threading import Thread
import torch
import os
import numpy as np

print("Numpy version:", np.__version__)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)

# =====================================================
# Prometheus Metrics
# =====================================================
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total inference requests"
)

TOKENS_GENERATED = Counter(
    "llm_generated_tokens_total",
    "Total generated tokens"
)

LATENCY = Histogram(
    "llm_latency_seconds",
    "End-to-end latency",
    ["context_length", "batch_size"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
)

TTFT = Histogram(
    "llm_ttft_seconds",
    "Time to first token",
    ["context_length", "batch_size"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
)

TPS = Histogram(
    "llm_tokens_per_second",
    "Generation throughput",
    ["context_length", "batch_size"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200)
)

ITL_AVG = Histogram(
    "llm_avg_itl",
    "Inter-token latency",
    ["context_length", "batch_size"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200)
)

PROMPT_TOKENS = Histogram(
    "llm_prompt_tokens",
    "Prompt/context length",
    ["context_length", "batch_size"],
    buckets=(64, 128, 256, 512, 1024, 2048, 4096)
)

OUTPUT_TOKENS = Histogram(
    "llm_output_tokens",
    "Generated output length",
    ["context_length", "batch_size"],
    buckets=(16, 32, 64, 128, 256, 512)
)

BATCH_SIZE_METRIC = Histogram(
    "llm_batch_size",
    "Batch size",
    ["context_length", "batch_size"],
    buckets=(1, 2, 4, 8, 16, 32)
)

INFLIGHT_REQUESTS = Gauge(
    "llm_inflight_requests",
    "Currently active requests"
)
# =====================================================
# FastAPI
# =====================================================

app = FastAPI()

# =====================================================
# Model Load Once
# =====================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INIT] Using device: {DEVICE}")

print("[INIT] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("[INIT] Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    ))
    
model.to(DEVICE)

model.eval()

model_loaded = True

print("[INIT] Model loaded")

# =====================================================
# Health
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(model.device)
    }

# =====================================================
# Metrics
# =====================================================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# =====================================================
# Request Schema
# =====================================================

class GenerateRequest(BaseModel):
    prompt: list[str]
    max_new_tokens: int = 50

# =====================================================
# Generate
# =====================================================

@app.post("/generate")
def generate(req: GenerateRequest):
    batch_size = len(req.prompt)
    #context_label = str(req.context_length)
    batch_label = str(batch_size)

    if batch_size == 1:
        use_streaming = True
    else:
        use_streaming = False

    INFLIGHT_REQUESTS.inc()

    try:

        REQUEST_COUNT.inc()

        start_time = time.time()

        chat_texts = []

        for prompt in req.prompt:

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            chat_texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            chat_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        print("Inputs size:", inputs["input_ids"].shape)

        prompt_tokens = inputs["input_ids"].shape[1]
        context_label = str(prompt_tokens)

        PROMPT_TOKENS.labels(
        context_length=context_label,
        batch_size=batch_label
        ).observe(prompt_tokens)

        # ====================================================
        # Batch Size = 1 (Streaming + TTFT)
        # =====================================================

        if use_streaming:

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            thread = Thread(
                target=model.generate,
                kwargs=generation_kwargs
            )

            thread.start()

            generated_text = ""
            first_token_time = None
            itls = []
            token_times = []

            for chunk in streamer:
                now = time.time()
                token_times.append(now)

                if first_token_time is None:
                    first_token_time = now
                else:
                    itls.append(
                        now - previous_token_time
                    )

                previous_token_time = now
                generated_text += chunk

            end_time = time.time()

            generated_token_count = len(
                tokenizer.encode(
                    generated_text,
                    add_special_tokens=False
                )
            )

            total_latency = end_time - start_time

            ttft = (
                first_token_time - start_time
                if first_token_time
                else total_latency
            )

            generation_time = (
                end_time - first_token_time
                if first_token_time
                else total_latency
            )

            tps = (
                generated_token_count / generation_time
                if generation_time > 0
                else 0
            )
            
            avg_itl = np.mean(itls) if len(itls) > 0 else np.nan

            response_payload = generated_text

        # =====================================================
        # Batch Size > 1 (No Streaming)
        # =====================================================

        else:

            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            end_time = time.time()

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs.input_ids,
                    outputs
                )
            ]

            responses = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            generated_token_count = sum(
                len(ids)
                for ids in generated_ids
            )

            total_latency = end_time - start_time

            ttft = None

            tps = (
                generated_token_count / total_latency
                if total_latency > 0
                else 0
            )

            response_payload = responses

        # =====================================================
        # Metrics
        # =====================================================

        OUTPUT_TOKENS.labels(
            context_length=context_label,
            batch_size=batch_label
        ).observe(generated_token_count)

        TOKENS_GENERATED.inc(
            generated_token_count
        )

        BATCH_SIZE_METRIC.labels(
            context_length=context_label,
            batch_size=batch_label
        ).observe(batch_size)

        LATENCY.labels(
            context_length=context_label,
            batch_size=batch_label
        ).observe(total_latency)

        TPS.labels(
            context_length=context_label,
            batch_size=batch_label
        ).observe(tps)

        ITL_AVG.labels(
            context_length=context_label,
            batch_size=batch_label
        ).observe(avg_itl)
        
        if ttft is not None:

            TTFT.labels(
                context_length=context_label,
                batch_size=batch_label
            ).observe(ttft)

        # =====================================================
        # Response
        # =====================================================

        return {
            "response": response_payload,
            "batch_size": batch_size,
            "latency_sec": round(total_latency, 4),
            "ttft_sec": (
                round(ttft, 4)
                if ttft is not None
                else None
            ),
            "tokens_generated": generated_token_count,
            "tokens_per_sec": round(tps, 2),
            "avg_itl_sec": round(avg_itl, 4) if avg_itl is not None else None,
            "prompt_tokens": prompt_tokens
        }

    finally:

        INFLIGHT_REQUESTS.dec()