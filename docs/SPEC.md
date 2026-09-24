# Switchyard: ML serving control plane

## Intent

Switchyard is a public reference project for **ML infrastructure and model serving**. This project demonstrates deployable systems behavior and measurable release safety. It must provide interview evidence through real models, tests, traces, benchmarks and a usable console. It does not claim to be an industrial replacement for Triton or prove GPU/LLM serving expertise.

## Product

A small ONNX CPU reference workload classifies 8×8 handwritten digits. Two independently trained versions have immutable artifact checksums, reproducible training splits and measured holdout quality. A FastAPI serving endpoint routes requests between champion and canary, with bounded asynchronous batching, deadlines and backpressure. A persisted release controller supports quality-gated canaries, explicit promotion, manual rollback and automatic rollback from observed canary error/p95 guardrails.

The console visualizes version metadata, live request metrics, traffic allocation and audit history. The primary demonstration starts a canary, sends real requests, injects canary-only faults in local demo mode and observes automatic rollback and recovery. All writes require a bearer token; release changes also require an optimistic revision. Production mode requires an explicitly supplied strong token and disables fault injection. It is a single-instance reference control plane, not a distributed consensus system.

## Runtime

Python 3.12, FastAPI, ONNX Runtime, numpy, scikit-learn only for training, sklearn-onnx conversion, SQLite control state, Prometheus metrics. Per-version queues are bounded. Cancellation/deadline expiry must not leak futures or execute cancelled work unnecessarily. Shutdown drains or explicitly fails queued work. Models are not uploaded through the API; verified local ONNX artifacts are the only executable model format.

## Scope and invariants

- No LLM API keys or paid external services are required.
- Digit inputs contain exactly 64 finite values between 0 and 16.
- Artifacts are checksum-verified before ONNX session creation and warmed before readiness.
- Canary candidate differs from champion and cannot be more than 3 percentage points worse on the stored holdout metric.
- Default rollout weight 25%, error-rate limit 10%, p95 limit 150 ms, minimum 20 canary observations, 100-observation rolling window.
- Stale observations from a previous rollout cannot roll back a new release.
- Restart safely rolls back an in-progress release because its observation window is process-local.
- Automatic rollback is a real persisted state change with reason and supporting metrics in the audit trail.
- Queue time, model execution time and end-to-end latency are measured separately.
- Metrics label cardinality is bounded by registered model versions.
- Demonstration faults are disabled in production mode and affect only the current canary.
- Benchmark results disclose sample count, warmup, concurrency, runtime/hardware and whether HTTP is included. No claimed speedup without measurement.

## Delivery

New public GitHub repository `smfardeen7/switchyard-ml`, with CI, Docker Compose (API, console, Prometheus), Kubernetes example manifests, tests and truthful resume/interview guidance. A GitHub Pages console allows inspection of a **recorded real experiment**, prominently labeled as a recording. Live controls run against the local API; the public static page does not impersonate a running backend. Container smoke tests run in CI if the local Docker daemon remains unavailable.

## Ownership

Runtime worker owns `switchyard/models.py`, `switchyard/batching.py`, and their tests. Frontend worker owns `frontend/`. Root owns API, release state, metrics, schema/contracts and integration. Benchmark/deployment worker owns scripts, deploy configuration, CI and its integration checks, after contracts are stable. Workers must not edit one another's files.
