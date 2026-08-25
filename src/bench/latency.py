#!/usr/bin/env python3
"""Measure TTFT / E2E latency against an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "Fetch weather for a latitude/longitude pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]
SAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": "You are an expert in composing functions for a FinTech agent.",
    },
    {
        "role": "user",
        "content": "Fetch the current weather at latitude 45.4215 and longitude -75.6972.",
    },
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100) * (len(ordered) - 1)))))
    return ordered[idx]


async def one_request(client: httpx.AsyncClient, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    output_chars = 0
    async with client.stream("POST", url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if not piece and delta.get("tool_calls"):
                piece = json.dumps(delta["tool_calls"])
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()
            output_chars += len(piece or "")
    ended = time.perf_counter()
    return {
        "ttft_ms": ((first_token_at or ended) - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "output_chars": output_chars,
        "ok": True,
    }


async def run_concurrency(cfg: dict[str, Any], concurrency: int) -> dict[str, Any]:
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    n = int(cfg["requests_per_concurrency"])
    payload = {
        "model": cfg["model"],
        "messages": SAMPLE_MESSAGES,
        "tools": SAMPLE_TOOLS,
        "temperature": float(cfg.get("temperature", 0.0)),
        "max_tokens": int(cfg.get("max_tokens", 128)),
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    timeout = float(cfg.get("timeout_s", 120))
    results: list[dict[str, Any]] = []
    errors = 0
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {cfg.get('api_key', 'dummy')}"}) as client:
        sem = asyncio.Semaphore(concurrency)

        async def wrapped() -> None:
            nonlocal errors
            async with sem:
                try:
                    results.append(await one_request(client, url, payload, timeout))
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    results.append({"ok": False, "error": str(exc), "ttft_ms": None, "e2e_ms": None})

        await asyncio.gather(*[wrapped() for _ in range(n)])
    wall = time.perf_counter() - wall_start
    ok = [row for row in results if row.get("ok")]
    ttfts = [row["ttft_ms"] for row in ok]
    e2es = [row["e2e_ms"] for row in ok]
    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": errors,
        "throughput_rps": (len(ok) / wall) if wall else 0.0,
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
            "mean": statistics.fmean(ttfts) if ttfts else None,
        },
        "e2e_ms": {
            "p50": percentile(e2es, 50),
            "p95": percentile(e2es, 95),
            "p99": percentile(e2es, 99),
            "mean": statistics.fmean(e2es) if e2es else None,
        },
    }


async def async_main(cfg: dict[str, Any]) -> dict[str, Any]:
    report = {"endpoint": cfg["base_url"], "model": cfg["model"], "runs": []}
    for concurrency in cfg["concurrencies"]:
        print(f"Running concurrency={concurrency}", flush=True)
        report["runs"].append(await run_concurrency(cfg, int(concurrency)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bench.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = asyncio.run(async_main(cfg))
    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
