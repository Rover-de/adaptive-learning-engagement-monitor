"""Latency instrumentation for the capture pipeline.

Mean latency hides the behaviour that actually matters. A pipeline that averages
8 ms but stalls to 90 ms on one frame in fifty will drop frames, and dropped
frames bias the frequency estimate. Percentiles are therefore tracked directly,
over a bounded rolling window so the numbers describe current behaviour rather
than the whole session.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Any, Deque, Iterator

import numpy as np

_PERCENTILES = (50.0, 95.0, 99.0)


class LatencyTracker:
    """Thread-safe rolling percentile tracker for one pipeline stage."""

    def __init__(self, name: str, maxlen: int = 2000) -> None:
        self.name = name
        self._samples: Deque[float] = deque(maxlen=maxlen)
        self._lock = Lock()

    def record(self, seconds: float) -> None:
        with self._lock:
            self._samples.append(seconds)

    @contextmanager
    def measure(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(time.perf_counter() - start)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = np.asarray(self._samples, dtype=np.float64)

        if samples.size == 0:
            return {"stage": self.name, "count": 0}

        ms = samples * 1000.0
        p50, p95, p99 = np.percentile(ms, _PERCENTILES)
        return {
            "stage": self.name,
            "count": int(samples.size),
            "mean_ms": round(float(ms.mean()), 3),
            "p50_ms": round(float(p50), 3),
            "p95_ms": round(float(p95), 3),
            "p99_ms": round(float(p99), 3),
            "max_ms": round(float(ms.max()), 3),
        }


class PipelineMetrics:
    """The set of stages timed in the capture loop."""

    def __init__(self) -> None:
        self.frame_read = LatencyTracker("frame_read")
        self.face_detect = LatencyTracker("face_detect")
        self.eye_detect = LatencyTracker("eye_detect")
        self.estimate = LatencyTracker("hr_hrv_estimate", maxlen=600)
        self.end_to_end = LatencyTracker("end_to_end_frame")
        self._frames = 0
        self._started_at = time.perf_counter()
        self._lock = Lock()

    def count_frame(self) -> None:
        with self._lock:
            self._frames += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            frames = self._frames
            elapsed = max(time.perf_counter() - self._started_at, 1e-9)

        return {
            "frames_processed": frames,
            "throughput_fps": round(frames / elapsed, 2),
            "uptime_seconds": round(elapsed, 1),
            "stages": [
                tracker.summary()
                for tracker in (
                    self.frame_read,
                    self.face_detect,
                    self.eye_detect,
                    self.estimate,
                    self.end_to_end,
                )
            ],
        }
