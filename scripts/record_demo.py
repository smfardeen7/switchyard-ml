#!/usr/bin/env python3
"""Capture an actual local HTTP canary/fault/rollback experiment for the static console."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def main(args):
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=10,
        headers={"Authorization": f"Bearer {args.api_key}"},
    ) as client:

        async def get(path):
            response = await client.get(path)
            response.raise_for_status()
            return response.json()

        async def post(path, body):
            response = await client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        health = await get("/health")
        if not health.get("ready") or not health.get("demo_mode"):
            raise SystemExit("Recording requires a ready server in local demo mode.")
        policy = await get("/v1/policy")
        if policy["canary"] is not None:
            raise SystemExit(
                "An existing canary is active. Finish it before recording an isolated experiment."
            )
        models = (await get("/v1/models"))["models"]
        samples = (await get("/v1/samples"))["samples"]
        champion = next(m for m in models if m["version"] == policy["champion"])
        candidates = [
            m
            for m in models
            if m["version"] != policy["champion"]
            and m["accuracy"] >= champion["accuracy"] - 0.03
        ]
        if not candidates:
            raise SystemExit("No candidate passes the quality gate.")
        candidate = candidates[0]["version"]
        frames, requests = [], []

        async def frame(label):
            frames.append(
                {
                    "label": label,
                    "policy": await get("/v1/policy"),
                    "metrics": await get("/v1/metrics"),
                    "events": (await get("/v1/events"))["events"],
                }
            )

        async def traffic(count, phase, concurrency=8):
            async def worker(worker_id):
                for index in range(worker_id, count, concurrency):
                    begin = time.perf_counter()
                    response = await client.post(
                        "/v1/predict",
                        json={
                            "pixels": samples[index % len(samples)]["pixels"],
                            "routing_key": f"record-{phase}-{index}",
                            "timeout_ms": 2000,
                        },
                    )
                    if response.status_code not in (200, 429, 503, 504):
                        response.raise_for_status()
                    payload = response.json()
                    requests.append(
                        {
                            "phase": phase,
                            "status": response.status_code,
                            "client_latency_ms": round(
                                (time.perf_counter() - begin) * 1000, 3
                            ),
                            "version": payload.get("version"),
                        }
                    )
                    if args.interval_ms:
                        await asyncio.sleep(args.interval_ms / 1000)

            await asyncio.gather(*(worker(i) for i in range(concurrency)))

        await traffic(80, "baseline")
        await frame("Baseline")
        await post(
            "/v1/rollouts",
            {
                "candidate": candidate,
                "weight": 0.5,
                "expected_revision": policy["revision"],
                "guardrails": {
                    "max_error_rate": 0.10,
                    "max_p95_ms": 150,
                    "min_samples": 20,
                    "window_size": 100,
                },
            },
        )
        await traffic(48, "healthy-canary")
        await frame("Healthy canary")
        await post("/v1/chaos", {"enabled": True, "error_rate": 1.0, "latency_ms": 40})
        # A small first wave captures actual observed faults before rollback, when possible.
        await traffic(4, "fault-onset", concurrency=1)
        await frame("Canary fault")
        for wave in range(8):
            if (await get("/v1/policy"))["status"] == "rolled_back":
                break
            await traffic(40, f"fault-{wave}")
        rolled_back = await get("/v1/policy")
        events = (await get("/v1/events"))["events"]
        if rolled_back["status"] != "rolled_back" or not any(
            e["type"] == "automatic_rollback"
            and e["revision"] == rolled_back["revision"]
            for e in events
        ):
            raise SystemExit(
                "Expected an automatic rollback backed by an audit event; no recording published."
            )
        await frame("Automatic rollback")
        await post("/v1/chaos", {"enabled": False, "error_rate": 0, "latency_ms": 0})
        await traffic(160, "recovery")
        await frame("Recovered")
        recovery = [r for r in requests if r["phase"] == "recovery"]
        if any(
            r["status"] != 200 or r["version"] != champion["version"] for r in recovery
        ):
            raise SystemExit(
                "Recovery did not serve all requests successfully from the original champion."
            )
        report = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "Real local HTTP requests to FastAPI and ONNX Runtime CPU; injected candidate faults; recorded snapshots, not a live backend.",
            "models": models,
            "samples": samples,
            "frames": frames,
            "experiment": {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "requests": len(requests),
                "status_counts": dict(Counter(str(r["status"]) for r in requests)),
                "recovery_successes": len(recovery),
                "recovery_attempts": len(recovery),
                "rollback_revision": rolled_back["revision"],
                "client_interval_ms": args.interval_ms,
                "notes": "60-second server metrics are rolling, so Recovered includes earlier fault observations. Recovery request checks are phase-specific.",
                "observations": requests,
            },
        }
        if args.benchmark.exists():
            report["benchmarks"] = json.loads(args.benchmark.read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Captured {len(frames)} frames, {len(requests)} actual requests, automatic rollback revision "
            f"{rolled_back['revision']}, {len(recovery)}/{len(recovery)} recovery successes to {args.output}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--api-key", default=os.getenv("SWITCHYARD_API_KEY", "local-development-only")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "frontend/public/demo.json"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "benchmarks/latest.json"
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=100,
        help="Pause per client after each response; produces an inspectable multi-second timeline",
    )
    args = parser.parse_args()
    if args.interval_ms < 0:
        parser.error("--interval-ms must be nonnegative")
    asyncio.run(main(args))
