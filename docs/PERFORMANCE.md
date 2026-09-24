# Performance evidence

The workload is a real ONNX Runtime **CPU** classifier over 64-feature handwritten digits. Its purpose is to expose serving behavior; it is too small to represent GPU/LLM inference. Batching can add delay and reduce performance at low concurrency. A configured batch size is a limit, not evidence that full batches actually occurred.

## Reproduce

```sh
.venv/bin/python scripts/benchmark.py
```

The default experiment measures batch limits 1 and 16 at concurrency 1, 8 and 32, with two runs per case, 64 excluded warmup requests and 512 measured requests per run. Requests use the real `DynamicBatcher` and ONNX sessions. Every run reports successful and attempted throughput, actual batch-size distribution, end-to-end p50/p95/p99, queue wait, inference time and exception counts. Model creation, loading, warmup, HTTP and serialization are excluded.

Results are written to `benchmarks/latest.json` with timestamp, OS/architecture, logical CPU count, Python/library versions, model checksum and settings. Do not compare numbers across machines without accounting for differences in hardware, concurrency, queue wait, model/runtime and workload.

The load generator is closed-loop: fixed workers issue another request after the preceding response. It measures attainable throughput under this workload, not an independent open-loop arrival-rate SLA. Queue overload, deadline, cancellation and shutdown behavior belong to targeted runtime tests. Queue and inference percentiles include successful requests; total latency includes failures. Error counts are always disclosed.

## Captured CPU results — September 24, 2026

The checked-in [raw result](../benchmarks/latest.json) contains **6,144 successful measured requests, zero errors**, and 768 additional warmup requests. Host: macOS 26.5.2 arm64, 10 logical CPUs; Python 3.12.14, ONNX Runtime 1.30.0, NumPy 2.5.3. Model: `digits-logreg-v1`, one CPU inference thread per ONNX session. The two-run ranges below are observations, not confidence intervals.

| Batch limit | Concurrency | Successful requests/s | End-to-end p95 (ms) | Mean observed batch size |
|---:|---:|---:|---:|---:|
| 1 | 1 | 4,450–4,518 | 0.24–0.24 | 1 |
| 1 | 8 | 5,819–5,832 | 1.54–1.58 | 1 |
| 1 | 32 | 5,646–5,712 | 5.84–5.95 | 1 |
| 16 | 1 | 94–102 | 10.99–11.28 | 1 |
| 16 | 8 | 850–853 | 9.91–9.93 | 8 |
| 16 | 32 | 8,483–9,323 | 3.68–4.85 | 16 |

At concurrency 1 and 8, the 8 ms coalescing allowance dominates this tiny model's execution time. At concurrency 32, full batches form immediately, and batch limit 16 improves measured throughput and p95 in these runs. This illustrates why batching must be tuned against the actual arrival pattern and latency budget rather than enabled as a blanket optimization. Background host activity was not isolated; repeat on the target deployment before choosing settings.

## Rollback recording

```sh
.venv/bin/python scripts/record_demo.py
```

This separate experiment uses actual HTTP requests, starts a quality-gated 50% canary, injects candidate-only errors/delay, waits for the real controller to persist an automatic rollback, and verifies that 160 recovery requests succeed on the original champion. It captures baseline, healthy canary, fault, rollback and recovery snapshots in `frontend/public/demo.json`. Request counts and client latencies are observed; no dashboard metrics are fabricated.

The checked-in recording was captured against a fresh local API instance: **332 requests, 326 HTTP 200 responses, six injected HTTP 503 responses, zero overload rejections, and 160/160 successful recovery requests**. The rollback occurred at revision 2 after four candidate errors among 35 candidate observations (11.4%, exceeding the 10% threshold); two already-dispatched candidate requests also failed. Candidate p95 at that decision was 40.533 ms, below the 150 ms latency guardrail, so the error-rate guardrail triggered it. Each client pauses 100 ms after a response to make the timeline inspectable; this recording is not a capacity benchmark.

The server's 60-second rolling metric window may still include faults in the recovered snapshot. The recorder separately checks the recovery phase rather than pretending the window instantly clears. Fault injection demonstrates controller behavior under a controlled failure, not a naturally occurring model regression.

## Interpretation

No speedup, production capacity, cost savings or reliability claim follows from the existence of batching or a successful local rollback. Use the recorded results and their limits. A CPU reference implementation demonstrates lifecycle, concurrency, observability and release-control engineering; it does not demonstrate GPU kernels, tensor parallelism, KV caching or distributed inference.
