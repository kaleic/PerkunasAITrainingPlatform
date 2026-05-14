from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class Sample:
    latency_s: float
    tokens: int


async def run_one(client: httpx.AsyncClient, model: str, prompt: str, stream: bool) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "stream": stream,
    }
    start = time.perf_counter()
    tokens = 0
    if stream:
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    tokens += 1
    else:
        response = await client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        tokens = payload["usage"]["completion_tokens"]
    return Sample(latency_s=time.perf_counter() - start, tokens=tokens)


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", default="dev-token")
    parser.add_argument("--model", default="dev/kv-echo-chat")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt", default="Summarize the KV memory policy in one paragraph.")
    args = parser.parse_args()

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.token}"},
        timeout=60.0,
        limits=limits,
    ) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded() -> Sample:
            async with semaphore:
                return await run_one(client, args.model, args.prompt, args.stream)

        started = time.perf_counter()
        samples = await asyncio.gather(*(guarded() for _ in range(args.requests)))
        elapsed = time.perf_counter() - started

    latencies = [sample.latency_s for sample in samples]
    tokens = sum(sample.tokens for sample in samples)
    print(
        {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "elapsed_s": round(elapsed, 3),
            "rps": round(args.requests / elapsed, 3),
            "tokens_per_s": round(tokens / elapsed, 3),
            "p50_latency_s": round(statistics.median(latencies), 4),
            "p95_latency_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 4),
        }
    )


if __name__ == "__main__":
    asyncio.run(main_async())
