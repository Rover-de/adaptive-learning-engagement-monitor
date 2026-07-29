"""Remote photoplethysmography (rPPG) estimation from a scalar intensity stream.

The estimator is deliberately decoupled from the camera: it consumes
``(value, timestamp)`` pairs and knows nothing about faces or frames. That makes
the whole numerical chain testable against synthetic signals with known ground
truth, which is what benchmarks/validate_hr_accuracy.py exploits.

Processing chain
----------------
1.  Resample the irregularly-timed stream onto a uniform grid. Webcam frame
    delivery jitters; interpolating on real timestamps keeps the frequency axis
    unbiased instead of assuming a constant frame period.
2.  Remove a linear trend (illumination drift, slow head motion).
3.  Zero-phase Butterworth bandpass over the plausible heart-rate band. Applied
    forward-backward so no group delay is introduced into peak timings.
4.  Estimate the dominant frequency from a zero-padded periodogram.
5.  Re-filter narrowly around that frequency and only then locate individual
    beats. Searching for peaks in the wide passband is what makes naive
    implementations produce unusable HRV: broadband noise and the waveform's own
    second harmonic both generate spurious peaks, and every spurious peak splits
    one interval into two.
6.  Report a spectral quality index so downstream consumers can reject
    estimates instead of acting on noise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from scipy.signal import butter, find_peaks, get_window, hilbert, periodogram, sosfiltfilt

from .config import SignalConfig

# Half-width of the band around the spectral peak used for the quality index (Hz).
_SQI_HALF_WIDTH_HZ = 0.15

# Zero-padding factor for the periodogram. Does not improve true resolution
# (which is 1 / window_seconds); it only refines localisation of the peak.
_ZERO_PAD_FACTOR = 8

# Half-width of the narrow band placed around the estimated pulse frequency
# before beat localisation (Hz). Wide enough to pass genuine beat-to-beat
# variation -- at 72 bpm it admits periods from 0.63 s to 1.25 s -- while
# rejecting the second harmonic and most broadband noise.
_BEAT_BAND_HALF_WIDTH_HZ = 0.40

# Peak prominence required, as a multiple of the narrowband signal's standard
# deviation.
_PEAK_PROMINENCE_RATIO = 0.4

# Seconds trimmed from each end of the narrowband signal before phase-based beat
# timing. The Hilbert transform is not well behaved at the boundaries of a finite
# record, and a distorted first or last cycle is charged straight to RMSSD.
_PHASE_EDGE_TRIM_SECONDS = 1.0


@dataclass(frozen=True)
class RppgEstimate:
    """Output of one estimator evaluation.

    ``hr_psd_bpm`` is the frequency-domain estimate and is the one that should
    be preferred: it integrates the whole window instead of relying on
    individual peak timings. ``hr_peak_bpm`` is retained because RMSSD is only
    definable on beat-to-beat intervals.
    """

    hr_psd_bpm: Optional[float] = None
    hr_peak_bpm: Optional[float] = None
    rmssd_ms: Optional[float] = None
    sqi: Optional[float] = None
    n_beats: int = 0
    effective_fps: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.hr_psd_bpm is not None


class RppgEstimator:
    """Rolling-window HR / HRV estimator over a scalar rPPG stream."""

    def __init__(self, config: Optional[SignalConfig] = None) -> None:
        self.config = config or SignalConfig()
        maxlen = self.config.max_samples
        self._values: Deque[float] = deque(maxlen=maxlen)
        self._timestamps: Deque[float] = deque(maxlen=maxlen)

    def reset(self) -> None:
        self._values.clear()
        self._timestamps.clear()

    def add_sample(self, value: float, timestamp: float) -> None:
        self._values.append(float(value))
        self._timestamps.append(float(timestamp))

    @property
    def n_samples(self) -> int:
        return len(self._values)

    @property
    def span_seconds(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    def _resample_uniform(self) -> Optional[tuple[np.ndarray, float]]:
        """Interpolate the buffer onto a uniform grid.

        Returns the resampled series and the effective sample rate, or ``None``
        if the buffer does not yet cover ``min_window_seconds``.
        """
        n = len(self._values)
        if n < 2:
            return None

        t = np.asarray(self._timestamps, dtype=np.float64)
        x = np.asarray(self._values, dtype=np.float64)

        span = t[-1] - t[0]
        if span < self.config.min_window_seconds:
            return None

        # Guard against non-monotonic timestamps from a stalled capture thread.
        keep = np.concatenate(([True], np.diff(t) > 0.0))
        t, x = t[keep], x[keep]
        if t.size < 2:
            return None

        fs = (t.size - 1) / (t[-1] - t[0])
        grid = np.linspace(t[0], t[-1], t.size)
        return np.interp(grid, t, x), float(fs)

    @staticmethod
    def _detrend_linear(x: np.ndarray) -> np.ndarray:
        """Remove the least-squares linear trend in closed form.

        Written out rather than delegated to ``scipy.signal.detrend`` because
        that routine goes through ``lstsq``, which on some BLAS builds raises
        spurious floating-point warnings, and because the closed form is cheaper
        in a loop that runs once per second.
        """
        n = x.size
        if n < 3:
            return x - x.mean()

        idx = np.arange(n, dtype=np.float64)
        idx_centred = idx - idx.mean()
        denom = float(idx_centred @ idx_centred)
        if denom <= 0.0:
            return x - x.mean()

        slope = float(idx_centred @ (x - x.mean())) / denom
        return x - x.mean() - slope * idx_centred

    def _bandpass(
        self, x: np.ndarray, fs: float, lo_hz: float, hi_hz: float, order: int
    ) -> Optional[np.ndarray]:
        """Zero-phase Butterworth bandpass via second-order sections.

        SOS rather than transfer-function coefficients: the narrow band used for
        beat localisation is numerically fragile in ``b, a`` form.
        """
        nyquist = fs / 2.0
        # Frame rate too low to represent the top of the requested passband.
        hi_hz = min(hi_hz, nyquist * 0.95)
        lo_hz = max(lo_hz, 1e-6)
        if lo_hz >= hi_hz:
            return None

        # Forward-backward filtering needs enough samples to pad the edges.
        if x.size <= 6 * order + 1:
            return None

        sos = butter(order, [lo_hz / nyquist, hi_hz / nyquist], btype="band", output="sos")
        return sosfiltfilt(sos, x)

    def _spectral_hr(self, x: np.ndarray, fs: float) -> tuple[Optional[float], Optional[float]]:
        """Dominant in-band frequency (bpm) and the spectral quality index."""
        window = get_window("hann", x.size)
        freqs, power = periodogram(
            x, fs=fs, window=window, nfft=_ZERO_PAD_FACTOR * x.size, scaling="density"
        )

        lo_hz, hi_hz = self.config.band_hz
        in_band = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not np.any(in_band):
            return None, None

        band_power = power[in_band]
        band_freqs = freqs[in_band]
        total = float(band_power.sum())
        if total <= 0.0 or not np.isfinite(total):
            return None, None

        peak_hz = float(band_freqs[int(np.argmax(band_power))])

        # Quality index: how concentrated in-band power is around the peak.
        # A pure tone approaches 1.0; band-limited noise sits near
        # (2 * half_width) / bandwidth, i.e. ~0.13 for this passband.
        near_peak = np.abs(band_freqs - peak_hz) <= _SQI_HALF_WIDTH_HZ
        sqi = float(band_power[near_peak].sum() / total)

        return peak_hz * 60.0, sqi

    @staticmethod
    def _refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Sub-sample peak positions by fitting a parabola to each triplet.

        At 30 fps the sample grid quantises every beat time to 33 ms. Two
        independent quantisation errors enter each interval difference, injecting
        roughly 14 ms of pure artefact into RMSSD -- comparable to the quantity
        being measured. Interpolating the peak of the filtered waveform recovers
        most of that resolution at negligible cost.
        """
        interior = peaks[(peaks > 0) & (peaks < x.size - 1)].astype(np.int64)
        if interior.size == 0:
            return peaks.astype(np.float64)

        left, centre, right = x[interior - 1], x[interior], x[interior + 1]
        curvature = left - 2.0 * centre + right

        offset = np.zeros(interior.size, dtype=np.float64)
        valid = np.abs(curvature) > 1e-12
        offset[valid] = 0.5 * (left[valid] - right[valid]) / curvature[valid]
        # A well-formed peak sits within half a sample of its discrete maximum;
        # anything further indicates a bad fit, so fall back to the grid.
        offset = np.clip(offset, -0.5, 0.5)

        return interior.astype(np.float64) + offset

    def _beat_times_by_peak(
        self, narrow: np.ndarray, fs: float, f0_hz: float
    ) -> np.ndarray:
        """Beat times (samples) from interpolated local maxima."""
        std = float(narrow.std())
        if std <= 1e-9:
            return np.empty(0)

        # Expect one beat per 1/f0 seconds; require peaks to be at least 70% of
        # that apart, which tolerates fast beats without admitting harmonics.
        min_separation = max(1, int(round(0.7 * fs / f0_hz)))
        peaks, _ = find_peaks(
            narrow, distance=min_separation, prominence=_PEAK_PROMINENCE_RATIO * std
        )
        if peaks.size < 2:
            return np.empty(0)
        return self._refine_peaks(narrow, peaks)

    @staticmethod
    def _beat_times_by_phase(narrow: np.ndarray, fs: float) -> np.ndarray:
        """Beat times (samples) from instantaneous-phase crossings.

        A local maximum is defined by three samples, so its position inherits
        whatever noise sits on those three samples. The instantaneous phase of
        the analytic signal is instead determined by the whole cycle, which makes
        crossing times markedly more stable at low SNR -- the regime rPPG
        actually operates in. This is the estimator used for RMSSD; the benchmark
        quantifies the margin over peak picking.
        """
        trim = int(round(_PHASE_EDGE_TRIM_SECONDS * fs))
        core = narrow[trim:-trim] if narrow.size > 2 * trim + 8 else narrow
        if core.size < 8:
            return np.empty(0)

        phase = np.unwrap(np.angle(hilbert(core)))
        phase = phase - phase[0]
        if phase[-1] <= 0:
            return np.empty(0)

        # np.interp needs a monotone abscissa; noise can locally reverse phase.
        # Enforcing monotonicity biases individual crossings slightly but is far
        # less damaging than dropping the record.
        phase = np.maximum.accumulate(phase)

        n_cycles = int(np.floor(phase[-1] / (2.0 * np.pi)))
        if n_cycles < 2:
            return np.empty(0)

        targets = 2.0 * np.pi * np.arange(1, n_cycles + 1)
        positions = np.interp(targets, phase, np.arange(core.size, dtype=np.float64))
        return positions + trim if core is not narrow else positions

    def _beat_intervals(
        self, x: np.ndarray, fs: float, hr_bpm: float, method: str = "phase"
    ) -> np.ndarray:
        """Beat-to-beat intervals (s), located in a band centred on ``hr_bpm``."""
        f0_hz = hr_bpm / 60.0
        band_lo, band_hi = self.config.band_hz
        narrow = self._bandpass(
            x,
            fs,
            lo_hz=max(f0_hz - _BEAT_BAND_HALF_WIDTH_HZ, band_lo),
            hi_hz=min(f0_hz + _BEAT_BAND_HALF_WIDTH_HZ, band_hi),
            order=2,
        )
        if narrow is None:
            return np.empty(0)

        if method == "phase":
            beat_samples = self._beat_times_by_phase(narrow, fs)
        elif method == "peak":
            beat_samples = self._beat_times_by_peak(narrow, fs, f0_hz)
        else:
            raise ValueError(f"unknown beat timing method: {method!r}")

        if beat_samples.size < 2:
            return np.empty(0)

        rr = np.diff(beat_samples) / fs
        lo_s, hi_s = self.config.rr_bounds_seconds
        rr = rr[(rr >= lo_s) & (rr <= hi_s)]
        if rr.size < 2:
            return rr

        # Drop intervals far from the window median: a dropped beat doubles an
        # interval and a spurious peak halves one, and either would dominate a
        # squared statistic.
        median_rr = float(np.median(rr))
        return rr[np.abs(rr - median_rr) <= self.config.max_rr_jump_seconds]

    def _rmssd_ms(self, rr: np.ndarray) -> Optional[float]:
        """RMSSD over successive interval differences.

        This is the root mean square of first differences of the RR series --
        structurally the same quantity as realised volatility computed from a
        return series, and it inherits the same sensitivity to outliers. A
        single spurious peak splits one interval into two and inflates the
        statistic, so differences that exceed ``max_rr_jump_seconds`` are
        dropped rather than winsorised.
        """
        if rr.size < self.config.min_rr_for_rmssd:
            return None

        diffs = np.diff(rr)
        accepted = diffs[np.abs(diffs) <= self.config.max_rr_jump_seconds]
        if accepted.size < 1:
            return None
        return float(np.sqrt(np.mean(accepted**2)) * 1000.0)

    def estimate(self, beat_timing_method: Optional[str] = None) -> RppgEstimate:
        """Evaluate the current window. Cheap enough to call at 1 Hz.

        ``beat_timing_method`` overrides the configured beat-localisation method
        and exists so that the benchmark can compare the two on identical inputs.
        """
        method = beat_timing_method or self.config.beat_timing_method
        resampled = self._resample_uniform()
        if resampled is None:
            return RppgEstimate()

        x_raw, fs = resampled
        lo_hz, hi_hz = self.config.band_hz
        x = self._bandpass(
            self._detrend_linear(x_raw),
            fs,
            lo_hz=lo_hz,
            hi_hz=hi_hz,
            order=self.config.filter_order,
        )
        if x is None:
            return RppgEstimate(effective_fps=fs)

        hr_psd, sqi = self._spectral_hr(x, fs)
        if hr_psd is None:
            return RppgEstimate(sqi=sqi, effective_fps=fs)

        rr = self._beat_intervals(x, fs, hr_psd, method=method)

        hr_peak: Optional[float] = None
        if rr.size >= self.config.min_rr_for_hr:
            mean_rr = float(rr.mean())
            if mean_rr > 1e-6:
                hr_peak = 60.0 / mean_rr

        return RppgEstimate(
            hr_psd_bpm=hr_psd,
            hr_peak_bpm=hr_peak,
            rmssd_ms=self._rmssd_ms(rr),
            sqi=sqi,
            n_beats=int(rr.size) + 1 if rr.size else 0,
            effective_fps=fs,
        )
