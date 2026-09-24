#!/usr/bin/env python3
"""Measure the real CPU ONNX serving queue; excludes HTTP and model loading."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from switchyard.batching import DynamicBatcher
from switchyard.models import ModelRegistry, bootstrap


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        f"p{p}": round(float(np.percentile(values, p)), 4) if values else 0.0
        for p in (50, 95, 99)
    }


async def run_case(registry, args, batch_size, concurrency, repeat):
    batcher = DynamicBatcher(
        registry,
        max_batch_size=batch_size,
        max_wait_ms=args.max_wait_ms,
        queue_capacity=args.queue_capacity,
    )
    await batcher.start()
    samples = registry.sample_data()
    version = args.version or registry.versions[0]
    durations, queues, inference, actual_batch = [], [], [], []
    errors = Counter()

    async def request(index, measured):
        start = time.perf_counter()
        try:
            result = await batcher.predict(
                version,
                samples[index % len(samples)]["pixels"],
                timeout_ms=args.timeout_ms,
            )
            if measured:
                queues.append(result.queue_ms)
                inference.append(result.inference_ms)
                actual_batch.append(result.batch_size)
        except Exception as exc:
            if measured:
                errors[type(exc).__name__] += 1
            else:
                raise
        finally:
            if measured:
                durations.append((time.perf_counter() - start) * 1000)

    async def phase(count, measured):
        async def worker(worker_id):
            for index in range(worker_id, count, concurrency):
                await request(index, measured)

        await asyncio.gather(*(worker(i) for i in range(min(count, concurrency))))

    try:
        await phase(args.warmup, False)
        start = time.perf_counter()
        await phase(args.samples, True)
        elapsed = time.perf_counter() - start
        successes = len(actual_batch)
        return {
            "version": version,
            "batch_size_limit": batch_size,
            "concurrency": concurrency,
            "repeat": repeat,
            "attempted": args.samples,
            "successful": successes,
            "errors": dict(errors),
            "elapsed_seconds": round(elapsed, 6),
            "successful_rps": round(successes / elapsed, 3),
            "attempted_rps": round(args.samples / elapsed, 3),
            "end_to_end_ms": percentiles(durations),
            "queue_ms": percentiles(queues),
            "inference_ms": percentiles(inference),
            "mean_observed_batch_size": round(float(np.mean(actual_batch)), 3)
            if actual_batch
            else 0,
            "batch_size_distribution": dict(sorted(Counter(actual_batch).items())),
        }
    finally:
        await batcher.close()


async def main(args):
    bootstrap(args.model_dir)
    registry = ModelRegistry(args.model_dir)
    results = []
    for batch_size in args.batch_sizes:
        for concurrency in args.concurrency:
            for repeat in range(1, args.repeats + 1):
                result = await run_case(registry, args, batch_size, concurrency, repeat)
                results.append(result)
                print(
                    f"batch={batch_size:2} concurrency={concurrency:2} run={repeat} "
                    f"success={result['successful']}/{result['attempted']} "
                    f"rps={result['successful_rps']:.1f} "
                    f"p95={result['end_to_end_ms']['p95']:.2f}ms",
                    flush=True,
                )
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "method": "closed-loop fixed concurrency; real ONNX CPU inference; no HTTP; per-case warmup excluded",
        "hardware": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "runtime": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "onnxruntime", "scikit-learn")
            },
        },
        "configuration": {
            "samples_per_run": args.samples,
            "warmup_per_run": args.warmup,
            "repeats": args.repeats,
            "max_wait_ms": args.max_wait_ms,
            "queue_capacity_per_version": args.queue_capacity,
            "timeout_ms": args.timeout_ms,
            "seed": 42,
        },
        "model": next(
            m for m in registry.list_models() if m["version"] == results[0]["version"]
        ),
        "results": results,
        "limitations": [
            "Single machine, process and model; no network or HTTP serialization.",
            "Small CPU workload; this is not a GPU/LLM benchmark.",
            "Closed-loop clients reduce offered load when latency rises; no open-loop saturation claim.",
            "Percentiles include failed requests; queue/inference metrics include successes only.",
            "Repeated sample vectors are intentional; inference results are not cached.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    if any(r["successful"] != r["attempted"] for r in results):
        raise SystemExit(
            "Benchmark recorded request failures; inspect errors before drawing conclusions."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/latest.json")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--batch-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[1, 16]
    )
    parser.add_argument(
        "--concurrency",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 8, 32],
    )
    parser.add_argument("--max-wait-ms", type=float, default=8)
    parser.add_argument("--queue-capacity", type=int, default=128)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--version")
    args = parser.parse_args()
    if (
        min(
            args.samples,
            args.warmup,
            args.repeats,
            *args.batch_sizes,
            *args.concurrency,
        )
        < 1
    ):
        parser.error(
            "Sample counts, repeats, batch sizes and concurrency must be positive"
        )
    asyncio.run(main(args))
