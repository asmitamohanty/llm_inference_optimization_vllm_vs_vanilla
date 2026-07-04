import argparse
import requests
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import httpx
import asyncio
from prompt_loader import PromptBuilder
import json
import random

# --------------------------------------------------
# Endpoint
# --------------------------------------------------
BASE_URL = os.getenv(
    "ENDPOINT",
    "http://localhost:8000/generate"
)

ENDPOINT = BASE_URL
print(f"Using endpoint: {ENDPOINT}")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ENABLED_CHUNKED_PREFILL = os.getenv("ENABLED_CHUNKED_PREFILL", "")
DEFAULT_CTX_LEN = 512
CONTEXTS = [
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192
    ]
REQUEST_COUNTS = [
    1,
    5,
    10,
    20,
    50
]
# --------------------------------------------------
# Helpers
# --------------------------------------------------

def create_prompts(
    batch_size,
    context_length
):
    """
    Creates a batch of prompts.

    context_length is approximate token count.
    """

    PROMPT_BUILDER = PromptBuilder(
        args.dataset_choice
    )
    prompt = PROMPT_BUILDER.get_prompt(context_length=context_length)

    return [
        prompt
        for _ in range(batch_size)
    ]

def decode_chunk(line, stream=False) -> tuple[str, dict]:
    """
    Returns (delta_text, raw_chunk)
    delta_text: the text content of the chunk (empty string if none)
    raw_chunk:  the full parsed JSON object (None if unparseable or DONE)
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = line.strip()
    if not line:
        return "", None

    if stream:
        if not line.startswith("data: "):
            return "", None

    payload = line[len("data: "):] if stream else line

    if payload == "[DONE]":
        return "", None
    
    try:
        obj = json.loads(payload)
        #print("obj:", obj)
        if not obj.get("choices"):
            return "", obj
        if stream:
            delta = obj["choices"][0]["delta"].get("content", "")
        else:
            delta = obj["choices"][0]["message"].get("content", "")
        return delta, obj
    except Exception:
        return "", None

# --------------------------------------------------
# Requests
# --------------------------------------------------

def run_transformer_request(
    batch_size,
    context_length,
    max_new_tokens=128
    ):

    print("Running: run_transformer_request for context length=", context_length)
    prompt_start_time = time.perf_counter()
    prompts = create_prompts(batch_size=batch_size,context_length=context_length)
    prompt_end_time = time.perf_counter()
    prompt_duration = prompt_end_time - prompt_start_time
    print("Prompt duration:", prompt_duration)
    print("[DEBUG] prompt:", prompts)

    #print("Prompts:", prompts)
    response = requests.post(
        f"{ENDPOINT}/generate",
        json={
            "prompt": prompts,
            "max_new_tokens": max_new_tokens
        },
        timeout=900
    )

    print("STATUS:", response.status_code)
    print("BODY:", response.text)
    response.raise_for_status()
    result = response.json()
    result['prompt_duration'] = prompt_duration

    return result

def run_vllm_request(
    batch_size,
    context_length, 
    max_tokens=128):

    print("Running: run_vllm_request for context length=", context_length)
    prompt_start_time = time.perf_counter()
    prompts = create_prompts(batch_size=batch_size,context_length=context_length)
    prompt_end_time = time.perf_counter()
    prompt_duration = prompt_end_time - prompt_start_time
    print("Prompt duration:", prompt_duration)

    prompt = prompts[0]
    print("[DEBUG] prompt:", prompt)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    token_times=[]
    itls=[]
    output_tokens=0
    first_token_time = None
    last_token_time = None
    previous_token_time = None
    generated_text = ""
    last_chunk = None
    chunk_count = 0

    start = time.perf_counter()
    response = requests.post(
        f"{ENDPOINT}/v1/chat/completions",
        json=payload,
        stream=True
    )

    for line in response.iter_lines():
        #print("[DEBUG] line:", line)
        if not line:
            continue

        now = time.perf_counter()
        delta, chunk = decode_chunk(line, stream=True)

        if chunk is not None:
            last_chunk = chunk
            #print("[DEBUG] last chunk:", chunk)

        if not delta:
            continue
        
        token_times.append(now)
        chunk_count+=1
    
        if first_token_time is None:
            first_token_time = now
        else:
            itls.append(
                now - previous_token_time
            )
        previous_token_time = now
            
        #print("[RAW]", line.decode("utf-8"))
        generated_text += delta

    end = time.perf_counter()
    last_token_time = token_times[-1] if token_times else None
    #print("[DEBUG] last chunk:", last_chunk)

    usage = last_chunk.get("usage")
    if usage:
        output_tokens = usage.get("completion_tokens", output_tokens)
        #print("Token source is usage, output tokens=", output_tokens)
    else:
        output_tokens = chunk_count
        #print("Token source is chunk_count, output tokens=", output_tokens)

    print("[DEBUG] generated text:", generated_text)
    #print("[DEBUG] first_token_time:", first_token_time)
    #print("[DEBUG] response:", response)
    
    latency = end - start
    ttft = first_token_time - start if first_token_time else None
    #result = response.json() #can revert & switch to stream=True
    print("[DEBUG] response:", response)
    print("STATUS:", response.status_code)
    #print("BODY:", response.text)

    response.raise_for_status()
    
    decode_time = (
        last_token_time - first_token_time
        if first_token_time and last_token_time
        else None
    )
    # print("[DEBUG] first_token_time:", first_token_time)
    # print("[DEBUG] last_token_time:", last_token_time)
    # print("[DEBUG] decode_time:", decode_time)
    decode_tps = (output_tokens - 1) / decode_time \
                     if (decode_time and decode_time > 0 and output_tokens > 1) \
                     else None
    # print("[DEBUG] decode_tps:", decode_tps)
    avg_itl = (np.mean(itls) if len(itls) > 0 else np.nan)

    return {
        "latency_sec": latency,
        "ttft_sec": ttft,
        "tokens_per_sec": decode_tps,
        "avg_itl_sec": avg_itl,
        "tokens_generated": output_tokens,
        "prompt_duration": prompt_duration
    }

# --------------------------------------------------
# Context Length Sweep
# --------------------------------------------------

def context_sweep():

    rows = []

    print("\nRunning Context Length Sweep\n")

    for ctx in CONTEXTS:

        print(f"Context={ctx}")

        result = BACKEND_FN(
            batch_size=1,
            context_length=ctx
        )

        rows.append({
            "context_length": ctx,
            "batch_size": 1,
            "latency_sec": result["latency_sec"],
            "ttft_sec": result["ttft_sec"],
            "tokens_per_sec": result["tokens_per_sec"],
            "tokens_generated": result["tokens_generated"],
            "cpu_time_sec(prompt-builder)": result["prompt_duration"]

        })

    df = pd.DataFrame(rows)

    print(df)

    USE_VLLM = 'vllm_'
    if not args.backend=='vllm':
        USE_VLLM = ''
    
    if ENABLED_CHUNKED_PREFILL=="--enable-chunked-prefill":
        USE_VLLM += 'chunkEN_'

    df.to_csv(
        f"{args.output_csv}/{USE_VLLM}context_sweep.csv",
        index=False
    )

    print(
        f"\nSaved: {args.output_csv}/{USE_VLLM}context_sweep.csv"
    )


# --------------------------------------------------
# Batch Size Sweep
# --------------------------------------------------

def batch_sweep():

    BATCHES = [
        1,
        2,
        4,
        8
    ]

    rows = []

    print("\nRunning Batch Sweep\n")

    for batch in BATCHES:

        print(f"Batch={batch}")

        result = BACKEND_FN(
            batch_size=batch,
            context_length=512
        )

        rows.append({
            "batch_size": batch,
            "context_length": 512,
            "latency_sec": result["latency_sec"],
            "ttft_sec": result["ttft_sec"],
            "tokens_per_sec": result["tokens_per_sec"],
            "tokens_generated": result["tokens_generated"]
        })

    df = pd.DataFrame(rows)

    print(df)

    df.to_csv(
        "./results/batch_sweep.csv",
        index=False
    )

    print(
        "\nSaved: batch_sweep.csv"
    )


# --------------------------------------------------
# Number of Requests Sweep
# --------------------------------------------------
async def run_request_async(
    prompt,
    prompt_duration,
    max_tokens=128,
    client=None
    ):
    print("Running: run_request_async")
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens
    }

    start = time.time()

    #async with httpx.AsyncClient(timeout=900) as client:

    response = await client.post(
        f"{ENDPOINT}/generate",
        json=payload
    )

    latency = time.time() - start

    result = response.json()

    result["latency_sec"] = latency
    result["prompt_duration"] = prompt_duration

    return result

async def run_vllm_request_async(
    prompt,
    prompt_duration,
    max_tokens=128,
    client=None
    ):
    print("Running: run_vllm_request_async")
    #print("[DEBUG] prompt:", prompt[0])
    payload = {

        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt[0]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    first_token_time = None
    last_token_time = None
    previous_token_time = None
    generated_text = ""
    itls = []
    output_tokens=0
    token_times=[]
    last_chunk = None
    chunk_count = 0

    #async with httpx.AsyncClient(timeout=900) as client:
    #print("Is client closed?", client.is_closed)
    start = time.perf_counter()
    async with client.stream("POST",f"{ENDPOINT}/v1/chat/completions",json=payload) as response:
        async for line in response.aiter_lines():
            if not line:
                continue

            now = time.perf_counter()
            delta, chunk = decode_chunk(line, stream=True)

            if chunk is not None:
                last_chunk = chunk
                #print("[DEBUG] last chunk:", chunk)

            if not delta:
                continue
            
            token_times.append(now)
            chunk_count+=1
            
            if first_token_time is None:
                first_token_time = now
            else:
                itls.append(
                    now - previous_token_time
                )
            previous_token_time = now
            generated_text += delta
    
    end = time.perf_counter()
    last_token_time = token_times[-1] if token_times else None
    # print("[DEBUG] last chunk:", last_chunk)
    print("[DEBUG] generated_text:", generated_text)
    
    usage = last_chunk.get("usage")
    if usage:
        output_tokens = usage.get("completion_tokens", output_tokens)
        # print("Token source is usage, output tokens=", output_tokens)
    else:
        output_tokens = chunk_count
        # print("Token source is chunk_count, output tokens=", output_tokens)
    
    total_latency = end - start #E2E latency
    print("[DEBUG] response:",response)
    print("STATUS:", response.status_code)
    #print("BODY:", response.text)
    #print("[DEBUG] response_result:", result)
    response.raise_for_status()
    
    ttft = (
        first_token_time - start
        if first_token_time
        else None #total_latency
    )

    decode_time = (
        last_token_time - first_token_time
        if first_token_time and last_token_time
        else None
    )

    decode_tps = (output_tokens - 1) / decode_time \
                     if (decode_time and decode_time > 0 and output_tokens > 1) \
                     else None

    avg_itl = (np.mean(itls) if len(itls) > 0 else np.nan)
    return {

        "latency_sec": total_latency,
        "ttft_sec": ttft,
        "tokens_per_sec": decode_tps,
        "avg_itl_sec": avg_itl,
        "tokens_generated": output_tokens,
        "prompt_duration": prompt_duration

    }

async def send_request(context_length=DEFAULT_CTX_LEN, max_new_tokens=128,client=None):

    # prompt = create_prompts(
    #     batch_size=1,
    #     context_length=context_length
    # )
    print("Running: send_request for context length=", context_length)
    prompt_start_time = time.perf_counter()
    prompts = create_prompts(batch_size=1,context_length=context_length)
    prompt_end_time = time.perf_counter()
    prompt_duration = prompt_end_time - prompt_start_time
    print("Prompt duration:", prompt_duration)

    return await BACKEND_ASYNC_FN(
        prompt=prompts,
        prompt_duration=prompt_duration,
        max_tokens=max_new_tokens,
        client=client
    )

async def request_sweep():

    rows = []

    print("\nRunning Request Sweep\n")
    async with httpx.AsyncClient(
    timeout=900,
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        ),
    ) as client:
        for num_requests in REQUEST_COUNTS:

            print(f"Requests={num_requests}")

            start = time.time()

            tasks = [
                send_request(client=client)
                for _ in range(num_requests)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end = time.time()

            # filter errors safely
            results = [r for r in results if not isinstance(r, Exception)]

            latencies = [r["latency_sec"] for r in results]
            ttfts = [r["ttft_sec"] for r in results]
            avg_itl = [r["avg_itl_sec"] for r in results]
            tps_values = [r["tokens_per_sec"] for r in results]
            tokens_generated = [r["tokens_generated"] for r in results]
            prompt_durations = [r["prompt_duration"] for r in results]

            rows.append({
                "requests": num_requests,
                "batch_size": 1,
                "context_length": DEFAULT_CTX_LEN,
                "avg_latency_sec": np.mean(latencies),
                "p50_latency_sec": np.quantile(latencies, 0.5),
                "p95_latency_sec": np.quantile(latencies, 0.95),
                "p99_latency_sec": np.quantile(latencies, 0.99),
                "avg_ttft_sec": np.mean(ttfts),
                "avg_itl_sec": np.mean(avg_itl),
                "p99_itl_sec": np.quantile(avg_itl, 0.99),
                "avg_tps": np.mean(tps_values),
                "aggr_tps": np.sum(tps_values),
                "avg_tokens_generated": np.mean(tokens_generated),
                "avg_cpu_time_sec(prompt-builder)": np.mean(prompt_durations),
                "total_runtime_sec": end - start
            })

    df = pd.DataFrame(rows)

    print(df)

    USE_VLLM = 'vllm_'
    if not args.backend=='vllm':
        USE_VLLM = ''
    
    if ENABLED_CHUNKED_PREFILL=="--enable-chunked-prefill":
        USE_VLLM += 'chunkEN_'

    df.to_csv(
        f"{args.output_csv}/{USE_VLLM}request_sweep.csv",
        index=False
    )

    print(
        f"\nSaved: {args.output_csv}/{USE_VLLM}request_sweep.csv"
    )

async def mixed_context_sweep():

    print("\nRunning Mixed Context Sweep\n")

    rows = []

    contexts = [1024, 2048, 4096, 8192, 16384]
    MAX_MIXED_TOKENS=32

    random.shuffle(contexts)

    start = time.time()
    async with httpx.AsyncClient(
        timeout=900,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    ) as client:

        tasks = [send_request(client=client, context_length=ctx, max_new_tokens=MAX_MIXED_TOKENS)
            for ctx in contexts]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    end = time.time()

    # Print any failures
    for r in results:
        if isinstance(r, Exception):
            print("Error:", r)

    # Keep only successful requests
    results = [
        r for r in results
        if isinstance(r, dict)
    ]

    if len(results) == 0:
        print("No successful requests.")
        return

    latencies = [r["latency_sec"] for r in results]
    ttfts = [r["ttft_sec"] for r in results]
    avg_itl = [r["avg_itl_sec"] for r in results]
    tps_values = [r["tokens_per_sec"] for r in results]
    tokens_generated = [r["tokens_generated"] for r in results]
    prompt_durations = [r["prompt_duration"] for r in results]

    rows.append({
        "contexts":",".join(map(str, contexts)),
        "num_requests":len(results),
        "avg_latency_sec": np.mean(latencies),
        "p50_latency_sec": np.quantile(latencies, 0.5),
        "p95_latency_sec": np.quantile(latencies, 0.95),
        "p99_latency_sec": np.quantile(latencies, 0.99),
        "avg_ttft_sec": np.mean(ttfts),
        "avg_itl_sec": np.mean(avg_itl),
        "p99_itl_sec": np.quantile(avg_itl, 0.99),
        "avg_tps": np.mean(tps_values),
        "aggr_tps": np.sum(tps_values),
        "avg_tokens_generated": np.mean(tokens_generated),
        "avg_cpu_time_sec(prompt-builder)": np.mean(prompt_durations),
        "total_runtime_sec": end - start
    })

    df = pd.DataFrame(rows)

    print(df)
    USE_VLLM = 'vllm_'
    if not args.backend=='vllm':
        USE_VLLM = ''
    
    if ENABLED_CHUNKED_PREFILL=="--enable-chunked-prefill":
        USE_VLLM += 'chunkEN_'

    df.to_csv(
        f"{args.output_csv}/{USE_VLLM}mixed_context_sweep.csv",
        index=False
    )

    print(
        f"\nSaved: {args.output_csv}/{USE_VLLM}mixed_context_sweep.csv"
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    global BACKEND_FN, BACKEND_ASYNC_FN
    BACKEND_FN, BACKEND_ASYNC_FN = None, None

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=["transformer", "vllm"],
        default="transformer"
    )    

    parser.add_argument(
        "--dataset-choice",
        choices=["dummy", "longbench"],
        default="dummy")
        
    parser.add_argument(
        "--context-sweep",
        action="store_true"
    )

    parser.add_argument(
        "--batch-sweep",
        action="store_true"
    )

    parser.add_argument(
        "--request-sweep",
        action="store_true"
    )

    parser.add_argument(
        "--mixed-context-sweep",
        action="store_true"
    )

    parser.add_argument(
        "--output-csv",
        default="./results",
        type=str
    )

    args = parser.parse_args()

    if args.backend=='vllm':
        BACKEND_FN = run_vllm_request
    
    else:
        BACKEND_FN = run_transformer_request

    if args.context_sweep:
        context_sweep()

    elif args.batch_sweep:
        if not args.backend=='vllm':
            batch_sweep()
        else:
            print("batch-sweep not supported for vllm")

    elif args.request_sweep or args.mixed_context_sweep:
        if args.backend=='vllm':
            BACKEND_ASYNC_FN = run_vllm_request_async
        else:
            BACKEND_ASYNC_FN = run_request_async
        print("args.backend=", args.backend)
        print("BACKEND_ASYNC_FN:", BACKEND_ASYNC_FN.__name__)
        if args.mixed_context_sweep:
            asyncio.run(mixed_context_sweep())
        else:
            asyncio.run(request_sweep())

    else:
        print(
            """
Choose one:

python benchmark.py --context-sweep
python benchmark.py --batch-sweep
python benchmark.py --request-sweep
python benchmark.py --mixed-context-sweep
"""
        )