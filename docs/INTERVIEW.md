# Explain and demonstrate Switchyard

## Suggested portfolio description

**Switchyard ML — ONNX serving and canary release control**

- Built a FastAPI/ONNX Runtime serving platform with bounded per-model batching, request deadlines, versioned routing, Prometheus metrics and a React operations console.
- Implemented quality-gated canary releases and SQLite-backed automatic rollback from observed candidate error/latency thresholds; added reproducible load experiments and tests for cancellation, overload, stale observations and restart recovery.

Use these statements only after you have run the system and can explain the design. Describe the CPU reference workload and single-instance scope accurately. Do not present this as employment experience, GPU/LLM serving expertise or a deployment serving real customers. Development dates are September 2026; no historical commits are fabricated.

## A six-minute walkthrough

1. **Problem:** a model with acceptable offline quality can still fail operationally after release.
2. **Show lineage:** identify the seeded train/test split, holdout accuracy, ONNX parity check and artifact digest. Explain why verifying the digest before session construction matters.
3. **Show the release:** route a fraction of traffic to the second warmed version. Explain deterministic cohort assignment and optimistic policy revisions.
4. **Show failure:** inject candidate-only errors, send actual requests and inspect automatic rollback after the sample minimum. Then demonstrate a successful champion-only recovery burst.
5. **Explain the hard race:** an old candidate request finishes after rollback and a new release starts. Its revision/rollout ID prevents it from changing the new policy.
6. **Discuss evidence:** compare batch sizes at the same concurrency. Explain when waiting for a batch harms latency and why this workload says nothing about GPU throughput.

The public page is a recorded experiment that reviewers can inspect without installing anything. For an interview, run the live version so you can change traffic weights, thresholds and failures yourself.

## Questions you should be ready to answer

- Why is the model call offloaded from the event loop? What happens if native inference hangs?
- What is bounded: queued requests, in-flight batches, thread count, metrics retention?
- How do cancellation and a full queue differ? Why can a cancelled model call still consume CPU?
- Why are latency measurements different at the HTTP client, API, queue and runtime layers?
- Why does a restart roll back the candidate instead of resuming from the persisted policy?
- Why is a minimum sample count necessary? What does it fail to guarantee statistically?
- How would you replace the single-instance controller with a highly available design?
- How would you add GPU scheduling or a Triton adapter while preserving deadline and rollout semantics?
- Why is a trusted checksum manifest not sufficient to establish an untrusted model's provenance?
- What would you measure before recommending larger batches or changing serving hardware?

## Why these features match ML infrastructure work

Three official employer descriptions reviewed September 24, 2026 informed the scope. They are examples of requested skills, not promises of open positions, eligibility or hiring outcomes:

- [Quora — ML Platform, New Grad](https://jobs.ashbyhq.com/quora/452afc2e-0c79-41f8-8201-1aab7df775db) emphasizes serving reliability, distributed systems, latency/throughput/cost and profiling. Its indexed graduation-year criteria are specific; check current eligibility independently.
- [Yobi — Inference / Serving](https://jobs.ashbyhq.com/yobi/11855546-e216-4518-bfae-bb020d3a7baa?embed=js) names model packaging/versioning, rollout/rollback, batching, lineage and observability in an experienced role.
- [White Circle — MLOps Engineer](https://jobs.ashbyhq.com/whitecircle/2954648f-cc68-42f8-bd93-26b06dd8f1a1) describes release gates, canaries, rollback and serving telemetry for experienced GPU infrastructure work.

A useful next learning step is to implement a second serving adapter and compare it with the existing benchmark, then explain which operational assumptions changed. The repository already contains enough material for a concrete discussion of reliability tradeoffs; broad claims without evidence weaken that discussion.
