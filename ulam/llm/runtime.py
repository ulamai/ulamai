from __future__ import annotations

import queue
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_runtime_controls(
    fn: Callable[[], T],
    *,
    timeout_s: float | None,
    heartbeat_s: float | None,
    label: str = "[llm]",
) -> T:
    timeout_enabled = timeout_s is not None and timeout_s > 0
    heartbeat_enabled = heartbeat_s is not None and heartbeat_s > 0
    if not timeout_enabled and not heartbeat_enabled:
        return fn()

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result = fn()
            result_queue.put((True, result))
        except Exception as exc:  # pragma: no cover - depends on external backends
            result_queue.put((False, exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    start = time.time()
    last_beat = start
    while True:
        try:
            ok, payload = result_queue.get(timeout=1.0)
            if ok:
                return payload  # type: ignore[return-value]
            raise payload  # type: ignore[misc]
        except queue.Empty:
            now = time.time()
            if heartbeat_enabled and now - last_beat >= float(heartbeat_s):
                elapsed = int(now - start)
                print(f"{label} still running ({elapsed}s)...", flush=True)
                last_beat = now
            if timeout_enabled and now - start >= float(timeout_s):
                raise RuntimeError(f"LLM request timed out after {float(timeout_s):.0f}s")
