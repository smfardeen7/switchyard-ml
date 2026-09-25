from switchyard.telemetry import Telemetry, Observation, GuardrailMonitor


def obs(
    revision=1,
    rollout="r1",
    version="candidate",
    error=False,
    latency=10,
    at=100,
    queue=2,
    inference=1,
):
    return Observation(
        at,
        "2026-09-24T00:00:00+00:00",
        version,
        revision,
        rollout,
        latency,
        queue,
        inference,
        error,
        False,
    )


def policy():
    return dict(
        revision=1,
        rollout_id="r1",
        canary="candidate",
        status="running",
        guardrails=dict(
            min_samples=3, window_size=5, max_error_rate=0.2, max_p95_ms=100
        ),
    )


def test_guardrail_waits_for_sample_count_and_rejects_stale_observations():
    g = GuardrailMonitor()
    p = policy()
    assert g.observe(p, obs(revision=0, error=True)) is None
    assert g.observe(p, obs(version="champion", error=True)) is None
    assert g.observe(p, obs(error=True)) is None
    assert g.observe(p, obs()) is None
    assert g.observe(p, obs()) == "Candidate error rate 33.3% exceeds 20.0%"
    assert not g.eligible(p)
    p["revision"] = 2
    p["rollout_id"] = "r2"
    assert not g.eligible(p)


def test_latency_guardrail_and_healthy_promotion():
    g = GuardrailMonitor()
    p = policy()
    for _ in range(3):
        g.observe(p, obs())
    assert g.eligible(p)
    assert "p95" in g.observe(p, obs(latency=250))


def test_metrics_window_totals_and_prometheus():
    t = Telemetry(clock=lambda: 100)
    t.record(obs(at=20))
    t.record(obs(at=80, error=True))
    t.record(obs(at=90, latency=30))
    snap = t.snapshot({})
    assert snap["totals"]["requests"] == 3
    assert snap["aggregate"]["error_rate"] == 0.5
    assert snap["aggregate"]["p95_ms"] == 29
    assert snap["by_version"][0]["requests"] == 2
    assert sum(x["requests"] for x in snap["timeseries"]) == 2
    assert b"switchyard_requests_total" in t.prometheus()
    assert len(t.observations) == 2


def test_console_rate_and_errors_are_not_capped_by_latency_sample_buffer():
    t = Telemetry(clock=lambda: 100)
    t.started = 0
    for i in range(11000):
        t.record(obs(at=99, error=(i % 2 == 0)))
    snap = t.snapshot({})
    assert snap["aggregate"]["throughput_rps"] == round(11000 / 60, 3)
    assert snap["by_version"][0]["requests"] == 11000
    assert snap["by_version"][0]["errors"] == 5500
    assert snap["retention"]["latency_samples_truncated"] is True
    assert snap["retention"]["window_requests"] == 11000


def test_stage_histograms_resolve_submillisecond_timings():
    # ONNX inference for this workload takes about 0.1 ms. Prometheus' default
    # buckets start at 5 ms, which would put every observation in one bucket.
    t = Telemetry()
    t.record(obs(queue=0.3, inference=0.2))
    t.record(obs(queue=3, inference=2))
    for name in (
        "switchyard_queue_duration_seconds",
        "switchyard_inference_duration_seconds",
    ):
        finite = [
            sample.value
            for metric in t.registry.collect()
            if metric.name == name
            for sample in metric.samples
            if sample.name == f"{name}_bucket" and sample.labels["le"] != "+Inf"
        ]
        assert 1 in finite, f"{name} cannot separate 0.2-0.3 ms from 2-3 ms"

