# Design and failure boundaries

## Data path

`POST /v1/predict` validates exactly 64 finite pixel values (0–16), authenticates a bearer token and snapshots the current release revision. The routing fraction is the first eight bytes of SHA-256 of `rollout_id:routing_key`, divided by 2^64. Clients can reuse a routing key for sticky assignment; absent keys use a fresh UUID. Client-chosen keys are for a trusted experiment workload, not an adversary-resistant public A/B allocator.

Each verified version has a bounded asynchronous queue and one worker. The worker groups compatible requests until the configured maximum batch size or maximum wait, then invokes its CPU ONNX session in a thread. Queue capacity is per version, excluding the batch already executing. Full queues return HTTP429; unavailable workers return503. The API deadline includes injected delay, batching wait and inference; expiry returns504. Native inference already in progress cannot be preempted by cancelling a Python future, so the worker finishes that batch and discards cancelled responses. This bounds work by worker count and batch size rather than spawning unbounded inference jobs.

Artifact loading verifies every size and SHA-256 digest before constructing sessions. Session input/output shapes and probabilities are checked, then warmed. The manifest is a **trusted local build artifact**, not a cryptographic signature or proof of model quality from an untrusted publisher. The service provides no model-upload or pickle-loading endpoint.

## Release controller

SQLite stores a single JSON policy plus append-only events. Mutations use `BEGIN IMMEDIATE` and an expected revision. A stale console receives409 and must refresh. A release uses a registered candidate different from the champion with measured accuracy no more than three percentage points lower on the common holdout split.

The controller observes **candidate requests** for the current rollout ID and revision only. It retains the most recent configured observation count. Once the minimum count is met, error rate or p95 above its threshold triggers a transaction restoring 100% champion traffic. Champion failures do not count toward candidate guardrails. Failed requests, including deadline and overload failures, are candidate errors when routed to that candidate. Manual promotion uses the same sample minimum and health checks. Disabling faults does not erase observations; start a fresh rollout after rollback for a fresh observation window.

Rollback changes admission for future requests; it cannot retract already admitted work. Old results retain their served version/revision for attribution. They cannot roll back a new release. A restart discards in-memory observations and explicitly rolls back a running canary before readiness, preserving that reason in SQLite.

The observation window is count-based, not a statistical significance test or accuracy monitor on production labels. Fixed p95 thresholds are demonstrative and should be calibrated against workload-specific SLOs. Model holdout accuracy does not prove performance under drift.

## Concurrency and persistence

Run exactly one API process/worker and one deployment replica. Queues, metrics, fault settings and guardrail observations are process-local. SQLite provides transactional persistence, not distributed consensus. Multiple API workers with independent windows would invalidate rollout decisions. Kubernetes is an operational example with a single replica and persistent volume, not a high-availability topology.

A scaled design would externalize versioned policy/watch distribution, aggregate revision-scoped observations across serving replicas, fence controller leadership, and define fail-open/fail-closed semantics if the policy service became unavailable. Serving replicas should receive immutable verified model manifests and report readiness before traffic assignment. These mechanisms are deliberately outside this reference implementation.

## Measurements

End-to-end latency starts after validation/authentication at the API handler; it includes routing, fault delay, batching and model work. Client HTTP measurements additionally include transport and serialization. Queue time runs from enqueue to batch execution start. Inference time is the wall time of the shared batch call, attributed to each successful member; summing it across requests would overcount CPU time.

Prometheus histograms/counters are lifetime process metrics. The console retains at most 10,000 latency observations. Separate per-second counters cover the current and 59 previous seconds, so throughput, error rate and request counts remain accurate when the latency sample buffer fills. The metrics response reports `retention.latency_samples_truncated` when percentiles cover only the retained subset. Lifetime totals remain separate. Throughput divides window request counts by the smaller of 60 seconds and process age. Use Prometheus counter rates for sustained monitoring. Percentiles use NumPy interpolation; queue/inference values in per-version summaries are successful-request medians. Empty metrics are zero, not latency measurements.

The benchmark directly exercises the runtime and batcher, excluding HTTP. It reports this distinction, runtime/hardware metadata, warmup, sample count and concurrency. A small CPU classifier is useful for testing release safety, but queue overhead may outweigh batching savings. No GPU, large-language-model, cost-saving or production throughput claims follow from this experiment.

## Deployment boundary

Local demo mode deliberately includes a public development token and fault controls. Production mode requires an explicit key of at least 32 characters and rejects fault injection. Metadata, health and metrics remain readable; place any internet-facing deployment behind TLS, network access controls and your own identity/rate-limiting layer. The example bearer key is not a multi-user authorization design. Secrets are supplied through environment variables or a Kubernetes Secret and never recorded in browser storage or public replay files.
