import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from switchyard.batching import (
    DeadlineExceededError,
    DynamicBatcher,
    QueueFullError,
    UnavailableError,
)


class RecordingRegistry:
    """Deterministic compute stand-in, with optional blocking/failure to test scheduling."""

    versions = ["a", "b"]

    def __init__(self, block=False, fail=False):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.fail = fail
        self.thread_ids = []

    def predict(self, version, rows):
        self.calls.append((version, rows.copy()))
        self.thread_ids.append(threading.get_ident())
        self.entered.set()
        if self.block and not self.release.wait(3):
            raise RuntimeError("Test did not release inference")
        if self.fail:
            raise RuntimeError("inference failed")
        output = np.zeros((len(rows), 10), dtype=np.float32)
        output[np.arange(len(rows)), rows[:, 0].astype(int)] = 1
        return output


def features(label=0):
    row = np.zeros(64, dtype=np.float32)
    row[0] = label
    return row


async def wait_until(predicate, timeout=1):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_requests_coalesce_per_version_off_the_event_loop():
    registry = RecordingRegistry()
    batcher = DynamicBatcher(registry, max_batch_size=4, max_wait_ms=30)
    await batcher.start()
    try:
        results = await asyncio.gather(
            *(batcher.predict("a", features(i)) for i in range(8)),
            batcher.predict("b", features(9)),
        )
        assert [int(np.argmax(result.probabilities)) for result in results] == [
            *range(8),
            9,
        ]
        assert max(len(rows) for _, rows in registry.calls) == 4
        assert all(len(rows) <= 4 for _, rows in registry.calls)
        assert all(
            thread_id != threading.get_ident() for thread_id in registry.thread_ids
        )
        assert all(
            result.queue_ms >= 0 and result.inference_ms >= 0 for result in results
        )
        assert batcher.stats()["queue_depth"] == 0
        assert batcher.stats()["inflight"] == 0
    finally:
        await batcher.close()


@pytest.mark.asyncio
async def test_queue_capacity_and_cancelled_work_release_backpressure():
    registry = RecordingRegistry(block=True)
    batcher = DynamicBatcher(
        registry, max_batch_size=1, max_wait_ms=0, queue_capacity=1
    )
    await batcher.start()
    first = asyncio.create_task(batcher.predict("a", features(1)))
    await wait_until(registry.entered.is_set)
    second = asyncio.create_task(batcher.predict("a", features(2)))
    await wait_until(lambda: batcher.stats()["queue_depth"] == 1)
    with pytest.raises(QueueFullError):
        await batcher.predict("a", features(3))
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert batcher.stats()["queue_depth"] == 0
    replacement = asyncio.create_task(batcher.predict("a", features(4)))
    registry.release.set()
    await asyncio.gather(first, replacement)
    await batcher.close()
    assert [int(rows[0, 0]) for _, rows in registry.calls] == [1, 4]


@pytest.mark.asyncio
async def test_deadline_expiry_does_not_execute_queued_request():
    registry = RecordingRegistry(block=True)
    batcher = DynamicBatcher(registry, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    first = asyncio.create_task(batcher.predict("a", features(1)))
    await wait_until(registry.entered.is_set)
    with pytest.raises(DeadlineExceededError):
        await batcher.predict("a", features(2), timeout_ms=15)
    assert batcher.stats()["queue_depth"] == 0
    registry.release.set()
    await first
    await batcher.close()
    assert len(registry.calls) == 1


@pytest.mark.asyncio
async def test_deadline_during_collection_skips_expired_work():
    registry = RecordingRegistry()
    batcher = DynamicBatcher(registry, max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    with pytest.raises(DeadlineExceededError):
        await batcher.predict("a", features(1), timeout_ms=10)
    await asyncio.sleep(0.06)
    assert registry.calls == []
    assert batcher.stats()["inflight"] == 0
    await batcher.close()


@pytest.mark.asyncio
async def test_caller_cancellation_during_inference_does_not_break_next_batch():
    registry = RecordingRegistry(block=True)
    batcher = DynamicBatcher(registry, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    first = asyncio.create_task(batcher.predict("a", features(1)))
    await wait_until(registry.entered.is_set)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    next_request = asyncio.create_task(batcher.predict("a", features(2)))
    registry.release.set()
    result = await next_request
    assert result.probabilities[2] == 1
    await batcher.close()
    assert batcher.stats()["inflight"] == 0


@pytest.mark.asyncio
async def test_shutdown_fails_queued_requests_and_joins_running_inference():
    registry = RecordingRegistry(block=True)
    batcher = DynamicBatcher(registry, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    first = asyncio.create_task(batcher.predict("a", features(1)))
    await wait_until(registry.entered.is_set)
    second = asyncio.create_task(batcher.predict("a", features(2)))
    await wait_until(lambda: batcher.stats()["queue_depth"] == 1)
    closing = asyncio.create_task(batcher.close())
    with pytest.raises(UnavailableError):
        await second
    with pytest.raises(UnavailableError):
        await batcher.predict("b", features())
    registry.release.set()
    await first
    await closing
    assert batcher.stats()["queue_depth"] == batcher.stats()["inflight"] == 0
    assert len(registry.calls) == 1
    await batcher.close()


@pytest.mark.asyncio
async def test_inference_failure_completes_futures_and_worker_recovers():
    registry = RecordingRegistry(fail=True)
    batcher = DynamicBatcher(registry, max_wait_ms=0)
    await batcher.start()
    with pytest.raises(UnavailableError):
        await batcher.predict("a", features())
    registry.fail = False
    result = await batcher.predict("a", features(3))
    assert result.probabilities[3] == 1
    await batcher.close()


@pytest.mark.asyncio
async def test_unstarted_unknown_and_invalid_inputs_fail_immediately():
    batcher = DynamicBatcher(RecordingRegistry())
    with pytest.raises(UnavailableError):
        await batcher.predict("a", features())
    await batcher.start()
    try:
        with pytest.raises(KeyError):
            await batcher.predict("unknown", features())
        with pytest.raises(ValueError):
            await batcher.predict("a", [0] * 63)
        with pytest.raises(ValueError):
            await batcher.predict("a", features(), timeout_ms=0)
    finally:
        await batcher.close()


@pytest.mark.asyncio
async def test_queue_time_includes_wait_for_the_executor_thread():
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    occupied, release = threading.Event(), threading.Event()

    def occupy():
        occupied.set()
        release.wait(2)

    blocker = asyncio.create_task(asyncio.to_thread(occupy))
    await wait_until(occupied.is_set)
    batcher = DynamicBatcher(RecordingRegistry(), max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    pending = asyncio.create_task(batcher.predict("a", features(4)))
    await wait_until(lambda: batcher.stats()["inflight"] == 1)
    await asyncio.sleep(0.05)
    release.set()
    result = await pending
    await blocker
    await batcher.close()
    executor.shutdown(wait=True)
    assert result.queue_ms >= 30


@pytest.mark.asyncio
async def test_cancellation_before_executor_starts_skips_model_execution():
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    occupied, release = threading.Event(), threading.Event()

    def occupy():
        occupied.set()
        release.wait(2)

    blocker = asyncio.create_task(asyncio.to_thread(occupy))
    await wait_until(occupied.is_set)
    registry = RecordingRegistry()
    batcher = DynamicBatcher(registry, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    pending = asyncio.create_task(batcher.predict("a", features(4)))
    await wait_until(lambda: batcher.stats()["inflight"] == 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()
    await blocker
    await batcher.close()
    executor.shutdown(wait=True)
    assert registry.calls == []


@pytest.mark.asyncio
async def test_deadline_during_execution_discards_late_result_and_recovers():
    registry = RecordingRegistry(block=True)
    batcher = DynamicBatcher(registry, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    expired = asyncio.create_task(batcher.predict("a", features(1), timeout_ms=20))
    await wait_until(registry.entered.is_set)
    with pytest.raises(DeadlineExceededError):
        await expired
    next_request = asyncio.create_task(batcher.predict("a", features(2)))
    registry.release.set()
    result = await next_request
    assert result.probabilities[2] == 1
    await batcher.close()
    assert batcher.stats()["inflight"] == batcher.stats()["queue_depth"] == 0
