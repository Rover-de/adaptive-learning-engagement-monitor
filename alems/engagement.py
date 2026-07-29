"""Engagement regime classification from HRV and blink-rate features.

Two things are worth noting about the design.

First, the thresholds are demonstration values. They are not calibrated against
labelled human data, so the regime label carries no clinical or psychometric
authority. What the module does provide is a clean, gated interface between
noisy features and a downstream consumer.

Second, a naive threshold rule applied to a continuously updating feature
chatters: when RMSSD sits near a boundary the label flips on almost every
evaluation, and any consumer that acts on transitions (pausing a video, firing a
prompt) fires constantly. The classifier therefore separates the instantaneous
classification from the published one and requires N consecutive agreeing
observations before the published label moves. This is hysteresis in the same
sense used to suppress turnover in a threshold-crossing trading rule: it trades
a few evaluations of lag for a large reduction in transition count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from .blink import BlinkState
from .config import RegimeConfig
from .rppg import RppgEstimate


class Regime(str, Enum):
    """Published engagement state. String-valued so it serialises directly."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ENGAGED = "ENGAGED"
    MILD_FATIGUE = "MILD_FATIGUE"
    HIGH_FATIGUE = "HIGH_FATIGUE"


@dataclass(frozen=True)
class EngagementSnapshot:
    """Everything the API exposes for a single point in time."""

    regime: Regime = Regime.INSUFFICIENT_DATA
    hr_bpm: Optional[float] = None
    hr_peak_bpm: Optional[float] = None
    rmssd_ms: Optional[float] = None
    sqi: Optional[float] = None
    blink_rate_per_min: float = 0.0
    effective_fps: Optional[float] = None
    # Raw (pre-hysteresis) classification, exposed for diagnostics.
    raw_regime: Regime = Regime.INSUFFICIENT_DATA
    # Number of published regime changes so far; the analogue of turnover.
    regime_flips: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime"] = self.regime.value
        payload["raw_regime"] = self.raw_regime.value
        return payload


class EngagementClassifier:
    """Maps (rPPG estimate, blink state) to a hysteresis-filtered regime."""

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()
        self._published = Regime.INSUFFICIENT_DATA
        self._pending: Optional[Regime] = None
        self._pending_count = 0
        self._flips = 0

    def _classify_raw(self, estimate: RppgEstimate, blink: BlinkState) -> Regime:
        cfg = self.config

        # Reject rather than guess: an unreliable spectrum must not produce a
        # confident-looking label.
        if estimate.rmssd_ms is None or not estimate.is_valid:
            return Regime.INSUFFICIENT_DATA

        rmssd = estimate.rmssd_ms
        blink_rate = blink.blink_rate_per_min

        if rmssd < cfg.rmssd_low_ms and blink_rate > cfg.blink_high_per_min:
            return Regime.HIGH_FATIGUE
        if rmssd < cfg.rmssd_mild_ms or blink_rate > cfg.blink_mild_per_min:
            return Regime.MILD_FATIGUE
        return Regime.ENGAGED

    def update(
        self, estimate: RppgEstimate, blink: BlinkState, min_sqi: float
    ) -> EngagementSnapshot:
        raw = self._classify_raw(estimate, blink)

        # Low spectral quality invalidates the features regardless of the rule.
        if estimate.sqi is not None and estimate.sqi < min_sqi:
            raw = Regime.INSUFFICIENT_DATA

        if raw == self._published:
            self._pending = None
            self._pending_count = 0
        else:
            if raw == self._pending:
                self._pending_count += 1
            else:
                self._pending = raw
                self._pending_count = 1

            if self._pending_count >= self.config.confirmations_to_flip:
                self._published = raw
                self._pending = None
                self._pending_count = 0
                self._flips += 1

        return EngagementSnapshot(
            regime=self._published,
            hr_bpm=estimate.hr_psd_bpm,
            hr_peak_bpm=estimate.hr_peak_bpm,
            rmssd_ms=estimate.rmssd_ms,
            sqi=estimate.sqi,
            blink_rate_per_min=blink.blink_rate_per_min,
            effective_fps=estimate.effective_fps,
            raw_regime=raw,
            regime_flips=self._flips,
        )
