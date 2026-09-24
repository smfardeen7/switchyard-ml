"""Bounded per-model dynamic batching with deadlines and graceful shutdown.

The caller's deadline includes queueing and model execution. Cancelling a caller
immediately removes queued work. Already-running ONNX calls cannot be preempted;
they finish in a thread, and abandoned results are discarded. close() rejects new
and queued work, then joins those active calls without cancelling their threads.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
import math
from numbers import Real
from time import perf_counter
import threading

import numpy as np


class QueueFullError(RuntimeError):
    """This model's pending queue has reached its configured bound."""


class DeadlineExceededError(RuntimeError):
    """The request did not complete before its total deadline."""


class UnavailableError(RuntimeError):
    """The batcher is not serving or its model execution failed."""


@dataclass(frozen=True)
class BatchResult:
    probabilities: list[float]
    queue_ms: float
    inference_ms: float
    batch_size: int


@dataclass(eq=False)
class _Request:
    features: np.ndarray
    future: asyncio.Future
    queued_at: float
    deadline: float
    queued_clock: float
    deadline_clock: float
    abandoned: threading.Event = field(default_factory=threading.Event)


class DynamicBatcher:
    def __init__(self, registry, max_batch_size=16, max_wait_ms=8, queue_capacity=128):
        for name, value in (
            ("max_batch_size", max_batch_size),
            ("queue_capacity", queue_capacity),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(max_wait_ms, Real)
            or not math.isfinite(max_wait_ms)
            or max_wait_ms < 0
        ):
            raise ValueError("max_wait_ms must be a finite non-negative number")
        self.registry = registry
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue_capacity = queue_capacity
        versions = list(registry.versions)
        if not versions or len(set(versions)) != len(versions):
            raise ValueError("Registry must expose unique model versions")
        self._queues = {version: deque() for version in versions}
        self._events = {version: asyncio.Event() for version in versions}
        self._active = {version: [] for version in versions}
        self._workers = []
        self._state = "new"
        self._batches = 0
        self._loop = None
        self._closing_task = None

    async def start(self):
        if self._state == "running":
            self._check_loop()
            return
        if self._state != "new":
            raise UnavailableError("A closed batcher cannot be restarted")
        self._loop = asyncio.get_running_loop()
        self._state = "running"
        self._workers = [
            asyncio.create_task(self._worker(version), name=f"batcher:{version}")
            for version in self._queues
        ]

    def _check_loop(self):
        if asyncio.get_running_loop() is not self._loop:
            raise UnavailableError("Use the batcher on the event loop that started it")

    async def close(self):
        if self._state == "closed":
            return
        if self._state == "new":
            self._state = "closed"
            return
        self._check_loop()
        if self._closing_task is None:
            self._state = "closing"
            for version, queue in self._queues.items():
                while queue:
                    self._fail(
                        queue.popleft(), UnavailableError("Batcher is shutting down")
                    )
                self._events[version].set()
            self._closing_task = asyncio.create_task(self._join_workers())
        # Cancellation of a lifespan caller must not orphan an inference thread.
        await asyncio.shield(self._closing_task)

    async def _join_workers(self):
        try:
            await asyncio.gather(*self._workers)
        finally:
            self._state = "closed"

    @staticmethod
    def _fail(request, error):
        if not request.future.done():
            request.future.set_exception(error)

    async def predict(self, version, features, timeout_ms=2000) -> BatchResult:
        if self._state != "running":
            raise UnavailableError("Batcher is not accepting requests")
        self._check_loop()
        if version not in self._queues:
            raise KeyError(version)
        if (
            not isinstance(timeout_ms, Real)
            or isinstance(timeout_ms, bool)
            or not math.isfinite(timeout_ms)
            or timeout_ms <= 0
        ):
            raise ValueError("timeout_ms must be a finite positive number")
        try:
            row = np.array(features, dtype=np.float32, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("features must contain 64 numeric pixels") from exc
        if (
            row.shape != (64,)
            or not np.isfinite(row).all()
            or np.any(row < 0)
            or np.any(row > 16)
        ):
            raise ValueError("features must contain 64 finite pixels between 0 and 16")
        queue = self._queues[version]
        if len(queue) >= self.queue_capacity:
            raise QueueFullError(f"Pending queue is full for {version}")
        now = self._loop.time()
        clock = perf_counter()
        request = _Request(
            row,
            self._loop.create_future(),
            now,
            now + timeout_ms / 1000,
            clock,
            clock + timeout_ms / 1000,
        )
        queue.append(request)
        self._events[version].set()
        try:
            return await asyncio.wait_for(
                asyncio.shield(request.future), timeout=timeout_ms / 1000
            )
        except TimeoutError as exc:
            raise DeadlineExceededError(
                "Request exceeded its total inference deadline"
            ) from exc
        finally:
            request.abandoned.set()
            if not request.future.done():
                request.future.cancel()
            elif not request.future.cancelled():
                # Consume a racing worker error even if wait_for timed out first.
                request.future.exception()
            # Deques allow immediate capacity recovery for cancelled/expired jobs.
            # _Request uses identity equality so ndarray values are never compared.
            try:
                queue.remove(request)
            except ValueError:
                pass
            self._events[version].set()

    def stats(self) -> dict:
        return {
            "queue_depth": sum(len(queue) for queue in self._queues.values()),
            "inflight": sum(len(batch) for batch in self._active.values()),
            "batches": self._batches,
            "max_batch_size": self.max_batch_size,
            "queue_capacity_per_version": self.queue_capacity,
        }

    def _drop_abandoned(self, batch):
        now = self._loop.time()
        live = []
        for request in batch:
            if request.future.done():
                continue
            if request.deadline <= now:
                self._fail(
                    request,
                    DeadlineExceededError("Request expired before model execution"),
                )
            else:
                live.append(request)
        batch[:] = live

    async def _collect(self, version):
        queue, event, batch = (
            self._queues[version],
            self._events[version],
            self._active[version],
        )
        cutoff = None
        while self._state == "running":
            self._drop_abandoned(batch)
            while queue and len(batch) < self.max_batch_size:
                request = queue.popleft()
                if request.future.done():
                    continue
                if request.deadline <= self._loop.time():
                    self._fail(
                        request,
                        DeadlineExceededError("Request expired before model execution"),
                    )
                    continue
                batch.append(request)
                if cutoff is None:
                    cutoff = request.queued_at + self.max_wait_ms / 1000
            if not batch or len(batch) == self.max_batch_size:
                return
            remaining = cutoff - self._loop.time()
            if remaining <= 0:
                return
            # Wake at an individual deadline as well as the coalescing deadline,
            # so abandoned work is retired promptly without running the model.
            remaining = min(
                remaining,
                min(request.deadline for request in batch) - self._loop.time(),
            )
            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=max(0, remaining))
            except TimeoutError:
                pass

    @staticmethod
    def _execute(registry, version, batch):
        # Recheck in the executor: scheduling can itself outlast a deadline or
        # caller cancellation. Events are safe to inspect across thread boundaries.
        executing = [
            request
            for request in batch
            if not request.abandoned.is_set()
            and perf_counter() < request.deadline_clock
        ]
        if not executing:
            return [], np.empty((0, 10), dtype=np.float32), 0.0, perf_counter()
        features = np.stack([request.features for request in executing])
        started = perf_counter()
        probabilities = registry.predict(version, features)
        return executing, probabilities, (perf_counter() - started) * 1000, started

    async def _worker(self, version):
        queue, event, active = (
            self._queues[version],
            self._events[version],
            self._active[version],
        )
        try:
            while self._state == "running":
                if not queue:
                    event.clear()
                    await event.wait()
                    if self._state != "running":
                        break
                await self._collect(version)
                self._drop_abandoned(active)
                if self._state != "running":
                    break
                if not active:
                    continue
                counted = False
                try:
                    (
                        executing,
                        probabilities,
                        inference_ms,
                        started,
                    ) = await asyncio.to_thread(
                        self._execute, self.registry, version, list(active)
                    )
                    if executing:
                        self._batches += 1
                        counted = True
                    probabilities = np.asarray(probabilities)
                    if (
                        probabilities.shape != (len(executing), 10)
                        or not np.isfinite(probabilities).all()
                    ):
                        raise ValueError("Model returned malformed probabilities")
                    completed = self._loop.time()
                    for request in active:
                        if request not in executing:
                            self._fail(
                                request,
                                DeadlineExceededError(
                                    "Request expired before its executor thread started"
                                ),
                            )
                    for request, scores in zip(executing, probabilities):
                        if request.future.done():
                            continue
                        if completed >= request.deadline:
                            self._fail(
                                request,
                                DeadlineExceededError(
                                    "Model execution exceeded the request deadline"
                                ),
                            )
                        else:
                            request.future.set_result(
                                BatchResult(
                                    scores.tolist(),
                                    (started - request.queued_clock) * 1000,
                                    inference_ms,
                                    len(executing),
                                )
                            )
                except Exception:
                    if not counted:
                        self._batches += 1
                    for request in active:
                        self._fail(
                            request,
                            UnavailableError(f"Model execution failed for {version}"),
                        )
                finally:
                    active.clear()
        finally:
            # Handles normal close and protects callers if a worker is cancelled.
            for request in active:
                self._fail(request, UnavailableError("Batch worker stopped"))
            active.clear()
            while queue:
                self._fail(queue.popleft(), UnavailableError("Batch worker stopped"))
