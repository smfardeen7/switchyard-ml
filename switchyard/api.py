"""Single-worker ASGI application: release control, guarded routing and telemetry."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Annotated
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .control import PolicyStore, Conflict, InvalidRelease, now_iso, route_version
from .telemetry import Telemetry, Observation, GuardrailMonitor, percentile

log = logging.getLogger("switchyard")


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SWITCHYARD_DATA_DIR", "var"))
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SWITCHYARD_MODEL_DIR", "models"))
    )
    env: str = field(default_factory=lambda: os.getenv("SWITCHYARD_ENV", "demo"))
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "SWITCHYARD_API_KEY", "local-development-only"
        )
    )
    max_batch_size: int = field(
        default_factory=lambda: int(os.getenv("SWITCHYARD_MAX_BATCH_SIZE", "16"))
    )
    max_wait_ms: float = field(
        default_factory=lambda: float(os.getenv("SWITCHYARD_MAX_WAIT_MS", "8"))
    )
    queue_capacity: int = field(
        default_factory=lambda: int(os.getenv("SWITCHYARD_QUEUE_CAPACITY", "128"))
    )


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Pixel = Annotated[float, Field(ge=0, le=16)]


class PredictBody(StrictBody):
    pixels: list[Pixel] = Field(min_length=64, max_length=64)
    routing_key: str | None = Field(default=None, max_length=256)
    timeout_ms: int = Field(default=2000, ge=50, le=5000)


class Guardrails(StrictBody):
    max_error_rate: float = Field(default=0.10, ge=0, le=1)
    max_p95_ms: float = Field(default=150, gt=0, le=30000)
    min_samples: int = Field(default=20, ge=1, le=10000)
    window_size: int = Field(default=100, ge=1, le=10000)

    @model_validator(mode="after")
    def enough_window(self):
        if self.min_samples > self.window_size:
            raise ValueError("window_size must cover min_samples")
        return self


class RevisionBody(StrictBody):
    expected_revision: int = Field(ge=0)


class RolloutBody(RevisionBody):
    candidate: str = Field(min_length=1, max_length=128)
    weight: float = Field(ge=0.05, le=0.5)
    guardrails: Guardrails = Field(default_factory=Guardrails)


class RollbackBody(RevisionBody):
    reason: str = Field(
        default="Operator requested rollback", min_length=1, max_length=500
    )


class ChaosBody(StrictBody):
    enabled: bool
    error_rate: float = Field(default=0, ge=0, le=1)
    latency_ms: float = Field(default=0, ge=0, le=2000)


def create_app(settings=None, registry=None, batcher=None):
    cfg = settings or Settings()
    if cfg.env not in ("demo", "production"):
        raise ValueError("SWITCHYARD_ENV must be demo or production")
    if not cfg.api_key or (
        cfg.env == "production"
        and (len(cfg.api_key) < 32 or cfg.api_key == "local-development-only")
    ):
        raise ValueError(
            "Production requires an explicit SWITCHYARD_API_KEY with at least 32 characters"
        )
    if not (
        1 <= cfg.max_batch_size <= 256
        and 0 <= cfg.max_wait_ms <= 1000
        and 1 <= cfg.queue_capacity <= 10000
    ):
        raise ValueError("Invalid batch size, wait or queue capacity")

    @asynccontextmanager
    async def lifespan(app):
        nonlocal registry, batcher
        from .models import ModelRegistry, bootstrap
        from .batching import DynamicBatcher

        if registry is None:
            if not (cfg.model_dir / "registry.json").exists():
                await asyncio.to_thread(bootstrap, cfg.model_dir)
            registry = await asyncio.to_thread(ModelRegistry, cfg.model_dir)
        store = PolicyStore(cfg.data_dir / "state.db", registry.list_models())
        app.state.store = store
        try:
            if batcher is None:
                batcher = DynamicBatcher(
                    registry, cfg.max_batch_size, cfg.max_wait_ms, cfg.queue_capacity
                )
            app.state.batcher = batcher
            app.state.registry = registry
            app.state.telemetry = Telemetry()
            app.state.monitor = GuardrailMonitor()
            app.state.chaos = ChaosBody(enabled=False).model_dump()
            app.state.chaos_revision = None
            await batcher.start()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if batcher is not None:
                    await batcher.close()
            finally:
                store.close()

    app = FastAPI(
        title="Switchyard ML",
        version="1.0.0",
        description="Single-instance model serving and release control reference.",
        lifespan=lifespan,
    )
    app.state.ready = False

    async def authorized(authorization: Annotated[str | None, Header()] = None):
        expected = f"Bearer {cfg.api_key}"
        if not authorization or not secrets.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise HTTPException(
                401,
                "Valid bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def disable_chaos():
        app.state.chaos = ChaosBody(enabled=False).model_dump()
        app.state.chaos_revision = None

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        # FastAPI's default error echoes raw inputs; overflowed JSON numbers
        # cannot themselves be encoded in a standards-compliant JSON response.
        errors = exc.errors()
        detail = "; ".join(
            ".".join(map(str, e["loc"])) + ": " + e["msg"] for e in errors[:3]
        )
        if len(errors) > 3:
            detail += f"; and {len(errors) - 3} more validation errors"
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request: " + detail}
        )

    @app.exception_handler(Conflict)
    async def conflict_handler(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidRelease)
    async def invalid_handler(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health():
        return dict(status="ok", ready=app.state.ready, demo_mode=cfg.env == "demo")

    @app.get("/ready")
    async def ready():
        if not app.state.ready:
            raise HTTPException(503, "Models are not ready")
        return dict(ready=True)

    @app.get("/v1/models")
    async def models():
        return dict(models=app.state.registry.list_models())

    @app.get("/v1/samples")
    async def samples():
        return dict(samples=app.state.registry.sample_data())

    @app.get("/v1/policy")
    async def policy():
        return app.state.store.get()

    @app.get("/v1/events")
    async def events():
        return dict(events=app.state.store.events())

    @app.get("/v1/metrics")
    async def metrics():
        return app.state.telemetry.snapshot(app.state.batcher.stats())

    @app.get("/metrics")
    async def prometheus():
        app.state.telemetry.update_gauges(
            app.state.batcher.stats(), app.state.store.get()["revision"]
        )
        return Response(
            app.state.telemetry.prometheus(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/v1/chaos")
    async def get_chaos():
        return app.state.chaos

    @app.post("/v1/chaos", dependencies=[Depends(authorized)])
    async def chaos(body: ChaosBody):
        if cfg.env != "demo":
            raise HTTPException(403, "Fault injection is disabled in production mode")
        p = app.state.store.get()
        if body.enabled and p["status"] != "running":
            raise Conflict("Start a canary before enabling fault injection")
        app.state.chaos = body.model_dump()
        app.state.chaos_revision = p["revision"] if body.enabled else None
        app.state.store.record_event(
            "chaos_changed",
            "Candidate fault injection enabled"
            if body.enabled
            else "Candidate fault injection disabled",
            body.model_dump(),
        )
        return app.state.chaos

    @app.post("/v1/rollouts", dependencies=[Depends(authorized)])
    async def start(body: RolloutBody):
        p = app.state.store.start(
            body.candidate,
            body.weight,
            body.expected_revision,
            body.guardrails.model_dump(),
        )
        disable_chaos()
        return p

    @app.post("/v1/rollouts/promote", dependencies=[Depends(authorized)])
    async def promote(body: RevisionBody):
        p = app.state.store.get()
        if p["revision"] != body.expected_revision:
            raise Conflict("Policy changed; refresh and retry")
        if not app.state.monitor.eligible(p):
            raise Conflict(
                "Promotion requires a healthy running canary with minimum observed candidate samples"
            )
        result = app.state.store.promote(body.expected_revision)
        disable_chaos()
        return result

    @app.post("/v1/rollouts/rollback", dependencies=[Depends(authorized)])
    async def rollback(body: RollbackBody):
        p = app.state.store.rollback(body.expected_revision, body.reason)
        disable_chaos()
        return p

    @app.post("/v1/predict", dependencies=[Depends(authorized)])
    async def predict(body: PredictBody):
        from .batching import QueueFullError, DeadlineExceededError, UnavailableError

        started = time.monotonic()
        request_id = str(uuid.uuid4())
        p = app.state.store.get()
        version = route_version(
            p, body.routing_key if body.routing_key is not None else request_id
        )
        fault = dict(app.state.chaos)
        inject = (
            fault["enabled"]
            and p["status"] == "running"
            and version == p["canary"]
            and app.state.chaos_revision == p["revision"]
        )
        error, rejected, result = True, False, None

        async def execute():
            if inject:
                await asyncio.sleep(fault["latency_ms"] / 1000)
                chance = (
                    int.from_bytes(
                        hashlib.sha256(request_id.encode()).digest()[:8], "big"
                    )
                    / 2**64
                )
                if chance < fault["error_rate"]:
                    raise HTTPException(503, "Demo: injected candidate failure")
            return await app.state.batcher.predict(
                version, body.pixels, timeout_ms=body.timeout_ms
            )

        try:
            result = await asyncio.wait_for(execute(), timeout=body.timeout_ms / 1000)
            error = False
            return dict(
                request_id=request_id,
                version=version,
                prediction=max(
                    range(len(result.probabilities)),
                    key=result.probabilities.__getitem__,
                ),
                probabilities=result.probabilities,
                latency_ms=round((time.monotonic() - started) * 1000, 3),
                queue_ms=result.queue_ms,
                inference_ms=result.inference_ms,
                batch_size=result.batch_size,
                policy_revision=p["revision"],
            )
        except QueueFullError:
            rejected = True
            raise HTTPException(
                429,
                "Model queue is full; retry with backoff",
                headers={"Retry-After": "1"},
            )
        except (DeadlineExceededError, asyncio.TimeoutError):
            raise HTTPException(504, "Inference request deadline exceeded")
        except UnavailableError:
            raise HTTPException(503, "Model worker is unavailable")
        except HTTPException:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Inference failed for version %s", version)
            raise HTTPException(503, "Model inference failed")
        finally:
            finished = time.monotonic()
            observation = Observation(
                finished,
                now_iso(),
                version,
                p["revision"],
                p["rollout_id"],
                (finished - started) * 1000,
                result.queue_ms if result else 0,
                result.inference_ms if result else 0,
                error,
                rejected,
            )
            app.state.telemetry.record(observation)
            current = app.state.store.get()
            reason = app.state.monitor.observe(current, observation)
            if reason:
                try:
                    rows = app.state.monitor.rows
                    evidence = dict(
                        samples=len(rows),
                        errors=sum(row.error for row in rows),
                        error_rate=sum(row.error for row in rows) / len(rows),
                        p95_ms=percentile(rows, "latency_ms", 95),
                        guardrails=current["guardrails"],
                    )
                    app.state.store.rollback(
                        current["revision"], reason, automatic=True, evidence=evidence
                    )
                    disable_chaos()
                except Conflict:
                    pass  # A stale observation must never undo a later release.

    return app


app = create_app()
