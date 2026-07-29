"""Blink event detection from a per-frame eye-visibility flag.

The upstream signal is binary and noisy: a Haar cascade drops the eye region on
individual frames for reasons unrelated to blinking (motion blur, head yaw,
lighting). Counting every dropout as a blink therefore massively overstates the
rate. The detector below treats a blink as an *event with a duration* and gates
candidates on three criteria, which is the cheapest way to remove the dominant
false-positive mode without adding a landmark model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import BlinkConfig


@dataclass(frozen=True)
class BlinkState:
    """Blink statistics as of the most recent update."""

    blink_rate_per_min: float = 0.0
    total_blinks: int = 0
    # Candidates rejected for being shorter than a physiological blink:
    # almost always single-frame detector dropouts.
    rejected_too_short: int = 0
    # Candidates rejected for lasting too long: treated as tracking loss.
    rejected_too_long: int = 0
    # Cumulative time with no eye region available (s). A large value relative
    # to session length means the rate below is not trustworthy.
    tracking_loss_seconds: float = 0.0

    @property
    def rejection_ratio(self) -> float:
        """Share of closure candidates that were discarded."""
        candidates = self.total_blinks + self.rejected_too_short + self.rejected_too_long
        if candidates == 0:
            return 0.0
        return (self.rejected_too_short + self.rejected_too_long) / candidates


class BlinkDetector:
    """Duration-gated blink detector over a binary eye-visibility stream."""

    def __init__(self, config: Optional[BlinkConfig] = None) -> None:
        self.config = config or BlinkConfig()
        self._closure_start: Optional[float] = None
        self._last_blink_at: Optional[float] = None
        self._events: Deque[float] = deque()
        self._total = 0
        self._rejected_short = 0
        self._rejected_long = 0
        self._tracking_loss = 0.0

    def reset(self) -> None:
        self.__init__(self.config)  # noqa: PLC2801 - deliberate full reset

    def update(self, eyes_detected: bool, timestamp: float) -> BlinkState:
        cfg = self.config

        if not eyes_detected:
            if self._closure_start is None:
                self._closure_start = timestamp
        elif self._closure_start is not None:
            duration = timestamp - self._closure_start
            self._closure_start = None
            self._tracking_loss += duration

            if duration < cfg.min_closure_seconds:
                self._rejected_short += 1
            elif duration > cfg.max_closure_seconds:
                self._rejected_long += 1
            elif (
                self._last_blink_at is not None
                and timestamp - self._last_blink_at < cfg.refractory_seconds
            ):
                self._rejected_short += 1
            else:
                self._events.append(timestamp)
                self._last_blink_at = timestamp
                self._total += 1

        while self._events and timestamp - self._events[0] > cfg.rate_window_seconds:
            self._events.popleft()

        return BlinkState(
            blink_rate_per_min=float(len(self._events)),
            total_blinks=self._total,
            rejected_too_short=self._rejected_short,
            rejected_too_long=self._rejected_long,
            tracking_loss_seconds=self._tracking_loss,
        )
