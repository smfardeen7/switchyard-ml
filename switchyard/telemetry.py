"""Bounded request telemetry and rollout-scoped guardrails."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import time
import numpy as np
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)


@dataclass(frozen=True)
class Observation:
    at: float
    wall_at: str
    version: str
    revision: int
    rollout_id: str | None
    latency_ms: float
    queue_ms: float
    inference_ms: float
    error: bool
    rejection: bool


def percentile(rows, field, percentile_):
    return (
        round(float(np.percentile([getattr(r, field) for r in rows], percentile_)), 3)
        if rows
        else 0.0
    )


class GuardrailMonitor:
    def __init__(self):
        self.key = None
        self.rows = deque(maxlen=100)

    def _current(self, p):
        key = (p["rollout_id"], p["revision"])
        if key != self.key:
            self.key = key
            self.rows = deque(maxlen=p["guardrails"]["window_size"])

    def violation(self, p):
        g = p["guardrails"]
        if len(self.rows) < g["min_samples"]:
            return None
        error_rate = sum(r.error for r in self.rows) / len(self.rows)
        p95 = percentile(self.rows, "latency_ms", 95)
        if error_rate > g["max_error_rate"]:
            return f"Candidate error rate {error_rate:.1%} exceeds {g['max_error_rate']:.1%}"
        if p95 > g["max_p95_ms"]:
            return f"Candidate p95 {p95:.1f} ms exceeds {g['max_p95_ms']:.1f} ms"
        return None

    def observe(self, p, row):
        self._current(p)
        if (
            p["status"] != "running"
            or row.version != p["canary"]
            or (row.rollout_id, row.revision) != self.key
        ):
            return None
        self.rows.append(row)
        return self.violation(p)

    def eligible(self, p):
        self._current(p)
        return (
            p["status"] == "running"
            and len(self.rows) >= p["guardrails"]["min_samples"]
            and not self.violation(p)
        )


class Telemetry:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.observations = deque(maxlen=10000)
        self.totals = dict(requests=0, errors=0, rejections=0)
        self.buckets = {}
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "switchyard_requests",
            "Inference requests by served version and outcome",
            ["version", "outcome"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "switchyard_request_duration_seconds",
            "End-to-end inference request latency",
            ["version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        # Queue waits and CPU inference are often sub-millisecond; the client's
        # default buckets start at 5 ms and would collapse them into one bucket.
        stage_buckets = (
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1,
            2,
            5,
        )
        self.queue_latency = Histogram(
            "switchyard_queue_duration_seconds",
            "Queue wait for successful requests",
            ["version"],
            buckets=stage_buckets,
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "switchyard_inference_duration_seconds",
            "Batch execution time for successful requests",
            ["version"],
            buckets=stage_buckets,
            registry=self.registry,
        )
        self.depth = Gauge(
            "switchyard_queue_depth",
            "Pending queued requests across versions",
            registry=self.registry,
        )
        self.inflight = Gauge(
            "switchyard_inflight",
            "Requests executing across model workers",
            registry=self.registry,
        )
        self.revisions = Gauge(
            "switchyard_policy_revision",
            "Current persisted release revision",
            registry=self.registry,
        )

    def record(self, row):
        self.observations.append(row)
        second = int(row.at)
        for expired in [s for s in self.buckets if s < second - 59]:
            del self.buckets[expired]
        bucket = self.buckets.setdefault(second, dict(at=row.wall_at, versions={}))
        counts = bucket["versions"].setdefault(row.version, dict(requests=0, errors=0))
        counts["requests"] += 1
        counts["errors"] += int(row.error)
        self.totals["requests"] += 1
        self.totals["errors"] += int(row.error)
        self.totals["rejections"] += int(row.rejection)
        self.requests.labels(
            row.version,
            "rejected" if row.rejection else "error" if row.error else "success",
        ).inc()
        self.latency.labels(row.version).observe(row.latency_ms / 1000)
        if not row.error:
            self.queue_latency.labels(row.version).observe(row.queue_ms / 1000)
            self.inference_latency.labels(row.version).observe(row.inference_ms / 1000)

    def update_gauges(self, queue, revision):
        self.depth.set(queue.get("queue_depth", 0))
        self.inflight.set(queue.get("inflight", 0))
        self.revisions.set(revision)

    def prometheus(self):
        return generate_latest(self.registry)

    def snapshot(self, queue):
        now = self.clock()
        cutoff = int(now) - 59
        while self.observations and self.observations[0].at < cutoff:
            self.observations.popleft()
        for expired in [s for s in self.buckets if s < cutoff]:
            del self.buckets[expired]
        rows = list(self.observations)
        # Group in one pass; rescanning every retained row for each version and
        # each second costs ~600k comparisons per poll at full retention.
        rows_by_version, rows_by_second = {}, {}
        for r in rows:
            rows_by_version.setdefault(r.version, []).append(r)
            rows_by_second.setdefault(int(r.at), []).append(r)
        counts = {}
        for bucket in self.buckets.values():
            for version, value in bucket["versions"].items():
                target = counts.setdefault(version, dict(requests=0, errors=0))
                target["requests"] += value["requests"]
                target["errors"] += value["errors"]
        requests = sum(c["requests"] for c in counts.values())
        errors = sum(c["errors"] for c in counts.values())
        aggregate = dict(
            p50_ms=percentile(rows, "latency_ms", 50),
            p95_ms=percentile(rows, "latency_ms", 95),
            p99_ms=percentile(rows, "latency_ms", 99),
            throughput_rps=round(requests / max(0.001, min(60, now - self.started)), 3),
            error_rate=errors / requests if requests else 0,
        )
        by_version = []
        for version, count in sorted(counts.items()):
            vr = rows_by_version.get(version, [])
            success = [r for r in vr if not r.error]
            by_version.append(
                dict(
                    version=version,
                    **count,
                    error_rate=count["errors"] / count["requests"],
                    p50_ms=percentile(vr, "latency_ms", 50),
                    p95_ms=percentile(vr, "latency_ms", 95),
                    queue_ms=percentile(success, "queue_ms", 50),
                    inference_ms=percentile(success, "inference_ms", 50),
                )
            )
        series = []
        for second, bucket in sorted(self.buckets.items()):
            br = rows_by_second.get(second, [])
            series.append(
                dict(
                    at=bucket["at"],
                    requests=sum(v["requests"] for v in bucket["versions"].values()),
                    errors=sum(v["errors"] for v in bucket["versions"].values()),
                    p95_ms=percentile(br, "latency_ms", 95),
                )
            )
        retention = dict(
            window_requests=requests,
            latency_samples=len(rows),
            latency_sample_capacity=10000,
            latency_samples_truncated=len(rows) < requests,
        )
        return dict(
            window_seconds=60,
            totals=self.totals.copy(),
            aggregate=aggregate,
            by_version=by_version,
            timeseries=series,
            queue=queue,
            retention=retention,
        )
