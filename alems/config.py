"""Tunable parameters for the engagement monitoring pipeline.

Every constant that affects a numerical result lives here so that benchmark
runs are reproducible from a single object. Units are stated explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    """Parameters of the rPPG signal-processing chain."""

    # Nominal camera frame rate (Hz). Used only to size buffers; the estimator
    # resamples on real timestamps, so jitter does not bias the result.
    nominal_fps: float = 30.0

    # Length of the rolling window (s). Frequency resolution of the periodogram
    # is 1/window_seconds, i.e. 0.05 Hz = 3.0 bpm at 20 s.
    window_seconds: float = 20.0

    # Minimum amount of data before any estimate is published (s).
    min_window_seconds: float = 8.0

    # Plausible heart-rate range (bpm). Doubles as the passband of the
    # bandpass filter and as the outlier gate on RR intervals.
    hr_min_bpm: float = 42.0
    hr_max_bpm: float = 180.0

    # Butterworth bandpass order. Applied forward-backward (zero phase).
    filter_order: int = 3

    # How individual beats are located for HRV: "phase" (instantaneous-phase
    # crossings of the analytic signal) or "peak" (interpolated local maxima).
    # "phase" is the default on measured evidence; see the benchmark output.
    beat_timing_method: str = "phase"

    # Maximum accepted absolute change between consecutive RR intervals (s).
    # Guards RMSSD against a single mis-detected peak.
    max_rr_jump_seconds: float = 0.25

    # Minimum number of surviving RR intervals required for HR / RMSSD.
    min_rr_for_hr: int = 3
    min_rr_for_rmssd: int = 4

    # Minimum spectral signal-quality index (fraction of total in-band power
    # concentrated around the dominant peak) for an estimate to be trusted.
    # Calibrated on synthetic data; see benchmarks/validate_hr_accuracy.py.
    min_sqi: float = 0.30

    @property
    def band_hz(self) -> tuple[float, float]:
        return self.hr_min_bpm / 60.0, self.hr_max_bpm / 60.0

    @property
    def rr_bounds_seconds(self) -> tuple[float, float]:
        return 60.0 / self.hr_max_bpm, 60.0 / self.hr_min_bpm

    @property
    def max_samples(self) -> int:
        return int(self.window_seconds * self.nominal_fps)


@dataclass(frozen=True)
class BlinkConfig:
    """Parameters of the blink event detector."""

    # Rolling window over which blink rate is reported (s).
    rate_window_seconds: float = 60.0

    # A closure must persist at least this long to count as a blink (s).
    # Rejects single-frame cascade dropouts, the dominant false-positive mode.
    min_closure_seconds: float = 0.06

    # A closure longer than this is treated as a tracking loss, not a blink (s).
    max_closure_seconds: float = 0.60

    # Refractory period after a confirmed blink (s).
    refractory_seconds: float = 0.15


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds of the engagement-state classifier.

    These are demonstration defaults, not calibrated on labelled human data.
    Treat them as an interface, not as a validated finding.
    """

    # RMSSD below this is treated as strongly suppressed vagal tone (ms).
    rmssd_low_ms: float = 20.0
    rmssd_mild_ms: float = 35.0

    # Blink rate above this is treated as elevated (blinks/min).
    blink_high_per_min: float = 25.0
    blink_mild_per_min: float = 18.0

    # Consecutive confirmations required before the published regime flips.
    # Acts as a hysteresis band against threshold-boundary chattering.
    confirmations_to_flip: int = 3


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5050
    # Upper bound on MJPEG preview frame rate (Hz).
    stream_fps: float = 15.0
    # JPEG quality for the preview stream (0-100).
    stream_jpeg_quality: int = 70


@dataclass(frozen=True)
class AppConfig:
    signal: SignalConfig = SignalConfig()
    blink: BlinkConfig = BlinkConfig()
    regime: RegimeConfig = RegimeConfig()
    server: ServerConfig = ServerConfig()

    # Interval between HR / RMSSD recomputations (s). Decoupled from the frame
    # rate so that estimator cost does not scale with capture rate.
    estimate_interval_seconds: float = 1.0

    # Fraction of the face box height used as the forehead ROI.
    roi_height_fraction: float = 0.30
    # Horizontal inset applied to the ROI to drop hair and background edges.
    roi_width_inset_fraction: float = 0.15
