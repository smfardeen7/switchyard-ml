import asyncio
from dataclasses import dataclass
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from switchyard.api import create_app, Settings
from switchyard.control import route_version


class FakeRegistry:
    versions = ["digits-logreg-v1", "digits-rf-v2"]

    def list_models(self):
        return [dict(version=v, accuracy=0.96) for v in self.versions]

    def sample_data(self):
        return [dict(id="zero", label=0, pixels=[0] * 64)]


@dataclass
class Result:
    probabilities: list
    queue_ms: float = 1
    inference_ms: float = 1
    batch_size: int = 1


class FakeBatcher:
    delay = 0

    async def start(self):
        pass

    async def close(self):
        pass

    async def predict(self, version, features, timeout_ms=2000):
        await asyncio.sleep(self.delay)
        return Result([1.0] + [0.0] * 9)

    def stats(self):
        return dict(
            queue_depth=0,
            inflight=0,
            batches=0,
            max_batch_size=16,
            queue_capacity_per_version=128,
        )


def client(tmp_path, *, production=False):
    settings = Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        env="production" if production else "demo",
        api_key="x" * 32 if production else "local-development-only",
    )
    app = create_app(settings, FakeRegistry(), FakeBatcher())
    return TestClient(app)


AUTH = {"Authorization": "Bearer local-development-only"}


def test_auth_validation_prediction_and_prometheus(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/ready").status_code == 200
        assert c.post("/v1/predict", json={"pixels": [0] * 64}).status_code == 401
        assert (
            c.post("/v1/predict", headers=AUTH, json={"pixels": [0] * 63}).status_code
            == 422
        )
        assert (
            c.post("/v1/predict", headers=AUTH, json={"pixels": [17] * 64}).status_code
            == 422
        )
        r = c.post("/v1/predict", headers=AUTH, json={"pixels": [0] * 64})
        assert r.status_code == 200 and r.json()["prediction"] == 0
        assert c.get("/v1/metrics").json()["totals"]["requests"] == 1
        assert "switchyard_request_duration_seconds" in c.get("/metrics").text


def start(c, min_samples=3):
    p = c.get("/v1/policy").json()
    return c.post(
        "/v1/rollouts",
        headers=AUTH,
        json=dict(
            candidate="digits-rf-v2",
            weight=0.5,
            expected_revision=p["revision"],
            guardrails=dict(
                min_samples=min_samples,
                window_size=10,
                max_error_rate=0.1,
                max_p95_ms=1000,
            ),
        ),
    )


def keys(p, count):
    return [str(i) for i in range(1000) if route_version(p, str(i)) == p["canary"]][
        :count
    ]


def test_canary_fault_automatically_rolls_back_and_recovers(tmp_path):
    with client(tmp_path) as c:
        p = start(c).json()
        assert (
            c.post(
                "/v1/rollouts/promote", headers=AUTH, json={"expected_revision": 1}
            ).status_code
            == 409
        )
        assert (
            c.post(
                "/v1/chaos",
                headers=AUTH,
                json=dict(enabled=True, error_rate=1, latency_ms=0),
            ).status_code
            == 200
        )
        for key in keys(p, 3):
            assert (
                c.post(
                    "/v1/predict",
                    headers=AUTH,
                    json=dict(pixels=[0] * 64, routing_key=key),
                ).status_code
                == 503
            )
        policy = c.get("/v1/policy").json()
        assert policy["status"] == "rolled_back" and policy["revision"] == 2
        assert c.get("/v1/chaos").json()["enabled"] is False
        event = c.get("/v1/events").json()["events"][0]
        assert event["type"] == "automatic_rollback"
        assert event["details"]["evidence"]["samples"] == 3
        assert event["details"]["evidence"]["error_rate"] == 1
        assert (
            c.post("/v1/predict", headers=AUTH, json=dict(pixels=[0] * 64)).status_code
            == 200
        )


def test_healthy_promotion_and_stale_revision(tmp_path):
    with client(tmp_path) as c:
        p = start(c).json()
        for key in keys(p, 3):
            assert (
                c.post(
                    "/v1/predict",
                    headers=AUTH,
                    json=dict(pixels=[0] * 64, routing_key=key),
                ).status_code
                == 200
            )
        assert (
            c.post(
                "/v1/rollouts/promote", headers=AUTH, json={"expected_revision": 0}
            ).status_code
            == 409
        )
        promoted = c.post(
            "/v1/rollouts/promote", headers=AUTH, json={"expected_revision": 1}
        )
        assert (
            promoted.status_code == 200
            and promoted.json()["champion"] == "digits-rf-v2"
        )


def test_deadline_covers_fault_delay_and_counts_error(tmp_path):
    with client(tmp_path) as c:
        p = start(c).json()
        c.post(
            "/v1/chaos",
            headers=AUTH,
            json=dict(enabled=True, error_rate=0, latency_ms=200),
        )
        r = c.post(
            "/v1/predict",
            headers=AUTH,
            json=dict(pixels=[0] * 64, routing_key=keys(p, 1)[0], timeout_ms=50),
        )
        assert r.status_code == 504
        assert c.get("/v1/metrics").json()["totals"]["errors"] == 1


def test_production_requires_key_and_disables_faults(tmp_path):
    with pytest.raises(ValueError, match="32"):
        create_app(Settings(data_dir=tmp_path, env="production", api_key="weak"))
    with client(tmp_path, production=True) as c:
        r = c.post(
            "/v1/chaos",
            headers={"Authorization": "Bearer " + "x" * 32},
            json=dict(enabled=True, error_rate=1, latency_ms=0),
        )
        assert r.status_code == 403
        assert not c.get("/health").json()["demo_mode"]


def test_database_closes_even_when_batcher_cleanup_fails(tmp_path):
    import sqlite3

    class BrokenCleanup(FakeBatcher):
        async def close(self):
            raise RuntimeError("cleanup failure")

    app = create_app(Settings(data_dir=tmp_path), FakeRegistry(), BrokenCleanup())
    with pytest.raises(RuntimeError, match="cleanup failure"):
        with TestClient(app) as c:
            assert c.get("/ready").status_code == 200
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        app.state.store.db.execute("SELECT 1")


def test_nonfinite_json_values_return_422_not_serialization_500(tmp_path):
    with client(tmp_path) as c:
        for token in ["1e400", "NaN", "Infinity", "-Infinity"]:
            body = '{"pixels":[' + ",".join([token] * 64) + "]}"
            response = c.post(
                "/v1/predict",
                headers={**AUTH, "Content-Type": "application/json"},
                content=body,
            )
            assert response.status_code == 422
            assert isinstance(response.json()["detail"], str)
        body = '{"candidate":"digits-rf-v2","weight":1e400,"expected_revision":0}'
        assert (
            c.post(
                "/v1/rollouts",
                headers={**AUTH, "Content-Type": "application/json"},
                content=body,
            ).status_code
            == 422
        )


def test_inflight_failure_from_old_canary_cannot_rollback_new_release(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from switchyard.batching import UnavailableError

    entered, release = threading.Event(), threading.Event()

    class ControlledBatcher(FakeBatcher):
        async def predict(self, version, features, timeout_ms=2000):
            if version == "digits-rf-v2":
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
                raise UnavailableError("old candidate failed")
            return Result([1.0] + [0.0] * 9)

    app = create_app(Settings(data_dir=tmp_path), FakeRegistry(), ControlledBatcher())
    with TestClient(app) as c, ThreadPoolExecutor(max_workers=1) as pool:
        old = start(c, min_samples=1).json()
        pending = pool.submit(
            c.post,
            "/v1/predict",
            headers=AUTH,
            json=dict(pixels=[0] * 64, routing_key=keys(old, 1)[0]),
        )
        try:
            assert entered.wait(2)
            assert (
                c.post(
                    "/v1/rollouts/rollback",
                    headers=AUTH,
                    json={"expected_revision": old["revision"]},
                ).status_code
                == 200
            )
            new = start(c, min_samples=1).json()
            release.set()
            assert pending.result(timeout=2).status_code == 503
            current = c.get("/v1/policy").json()
            assert current["status"] == "running"
            assert current["rollout_id"] == new["rollout_id"]
            assert current["revision"] == new["revision"]
        finally:
            release.set()
