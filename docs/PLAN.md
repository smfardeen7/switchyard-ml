# Switchyard implementation plan

**Goal:** A public, tested ML serving reference platform that demonstrates operational skills for ML infrastructure roles.
**Spec:** SPEC.md. **Shared API:** CONTRACT.md. **Architecture:** one backend process with bounded per-model batching queues, SQLite release state and audit history, Prometheus metrics and a React console. Parallel workers own disjoint modules; root integrates and verifies.

## Review focus

- Expired/cancelled requests must release pending capacity and not corrupt later batches.
- In-flight observations from an old rollout must not mutate a newer policy.
- Restart must not silently resume an unobserved canary.
- Queue overload/deadline errors must be observable and bounded, not hangs.
- Public replay must identify recorded measurements and cannot claim a live serving backend.

## Tasks

- [ ] Runtime: write failed model-integrity/shape tests; train/export deterministic ONNX versions; verify load/warmup and prediction parity; test corrupted files. Write concurrency/deadline/overload/shutdown tests before DynamicBatcher and verify them against actual inference.
- [ ] Control plane: test SQLite optimistic revisions, quality gate, rollback and restart recovery; implement state/audit. Test metrics window/percentiles/exporter and stale-rollout isolation. Implement validated HTTP API, authentication, canary-only faults and guardrails.
- [ ] Console: build accessible responsive React/TypeScript UI with model metadata, traffic allocation, percentiles, queue depth, audit timeline and sample inference; implement live mode and explicit recorded mode using CONTRACT.md.
- [ ] Experiments: real engine benchmark at batch sizes1/16 and concurrency1/8/32, with warmup/denominators/runtime metadata; real HTTP degraded-canary/rollback/recovery scenario; generate recording from captured API state.
- [ ] Operations: Docker/Compose/Prometheus and Kubernetes reference manifests, CI tests, container smoke, static Pages deployment. No unsupported scaling/throughput claims.
- [ ] Independent review, all tests and browser workflow; fix actual defects; record measured results and limitations; publish repo and static console, verify CI and public page; add truthful resume bullets and interview walkthrough.
