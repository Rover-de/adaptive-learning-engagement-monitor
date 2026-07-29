"""Synthetic rPPG generator with exactly known ground truth.

There is no way to validate a heart-rate estimator against webcam footage
without reference hardware, so the accuracy claims in this repository are made
against synthesised signals whose true heart rate and true RMSSD are known by
construction. The generator reproduces the four degradations that actually
matter for camera-based rPPG:

* **Sensor and quantisation noise** at a controlled signal-to-noise ratio. Real
  rPPG pulsatile amplitude is a fraction of a percent of the DC level, so the
  operating regime is genuinely low-SNR.
* **Low-frequency baseline drift** from illumination change and slow head
  motion. Deliberately placed below the passband, so it tests the filter.
* **Frame timestamp jitter**, because webcam delivery is not isochronous.
* **Dropped frames**, because detection fails intermittently.

Note the direction of the bias this introduces: synthetic evaluation is
optimistic. It contains no motion artefact correlated with the heart-rate band,
which is the failure mode that degrades real rPPG most. The numbers should be
read as an upper bound on field accuracy, not an estimate of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Relative amplitude of the second harmonic. Real PPG waveforms are not
# sinusoidal; the harmonic makes peak detection a non-trivial task.
_HARMONIC_RATIO = 0.35

# Typical webcam green-channel DC level and pulsatile amplitude, in digital
# numbers. The ratio (~0.4%) is representative of forehead rPPG.
_DC_LEVEL = 120.0
_PULSE_AMPLITUDE = 0.5


@dataclass(frozen=True)
class SyntheticSeries:
    """A synthetic rPPG observation window and its ground truth."""

    timestamps: np.ndarray
    values: np.ndarray
    true_hr_bpm: float
    true_rmssd_ms: float
    snr_db: float

    def __len__(self) -> int:
        return int(self.values.size)


def synthesize_rppg(
    duration_seconds: float = 20.0,
    fps: float = 30.0,
    hr_bpm: float = 72.0,
    rmssd_ms: float = 35.0,
    snr_db: float = 0.0,
    drift_amplitude: float = 2.0,
    drift_hz: float = 0.05,
    jitter_seconds: float = 0.004,
    dropout_rate: float = 0.02,
    seed: int | None = None,
) -> SyntheticSeries:
    """Generate one window of synthetic rPPG.

    Parameters
    ----------
    snr_db
        Ratio of pulsatile power to additive noise power, in decibels. Noise
        variance is solved for exactly from the realised pulse variance, so the
        requested SNR holds for the generated series rather than in expectation.
    rmssd_ms
        Target RMSSD of the beat-to-beat interval series. Intervals are
        generated as an i.i.d. perturbation around the mean, for which
        ``RMSSD = sqrt(2) * sigma``; sigma is set accordingly.
    dropout_rate
        Fraction of frames removed at random, simulating detection failure.
    """
    rng = np.random.default_rng(seed)

    if hr_bpm <= 0 or duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration, fps and hr_bpm must be positive")

    # --- Beat-to-beat interval series ------------------------------------
    mean_rr = 60.0 / hr_bpm
    sigma = (rmssd_ms / 1000.0) / np.sqrt(2.0)
    n_beats = int(np.ceil(duration_seconds / mean_rr)) + 4
    rr = mean_rr + rng.normal(0.0, sigma, size=n_beats)
    # Keep intervals physiological even at aggressive RMSSD settings.
    rr = np.clip(rr, 0.34, 1.42)
    beat_times = np.concatenate(([0.0], np.cumsum(rr)))

    true_hr = 60.0 / float(rr.mean())
    true_rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)

    # --- Sampling grid with jitter and dropouts --------------------------
    n_frames = int(round(duration_seconds * fps))
    t_ideal = np.arange(n_frames) / fps
    t = t_ideal + rng.normal(0.0, jitter_seconds, size=n_frames)
    t = np.maximum.accumulate(t)

    if dropout_rate > 0.0:
        keep = rng.random(n_frames) >= dropout_rate
        keep[0] = keep[-1] = True
        t = t[keep]

    # --- Pulsatile component ---------------------------------------------
    # Phase advances by exactly one cycle per generated beat interval, so the
    # instantaneous period of the waveform matches the RR series by design.
    phase = np.interp(t, beat_times, np.arange(beat_times.size))
    pulse = np.sin(2.0 * np.pi * phase) + _HARMONIC_RATIO * np.sin(4.0 * np.pi * phase)
    pulse *= _PULSE_AMPLITUDE

    # --- Additive noise at the requested SNR ------------------------------
    pulse_power = float(np.var(pulse))
    noise_power = pulse_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=t.size)

    # --- Out-of-band baseline drift ---------------------------------------
    drift = drift_amplitude * np.sin(2.0 * np.pi * drift_hz * t + rng.uniform(0, 2 * np.pi))
    drift += drift_amplitude * 0.5 * (t / max(t[-1], 1e-9))

    values = _DC_LEVEL + pulse + noise + drift

    return SyntheticSeries(
        timestamps=t,
        values=values,
        true_hr_bpm=true_hr,
        true_rmssd_ms=true_rmssd,
        snr_db=snr_db,
    )


def synthesize_blink_stream(
    duration_seconds: float = 120.0,
    fps: float = 30.0,
    blink_rate_per_min: float = 17.0,
    blink_duration_seconds: float = 0.15,
    dropout_rate: float = 0.01,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate an eye-visibility stream with a known number of true blinks.

    ``dropout_rate`` injects isolated single-frame detector failures -- the
    false-positive source the duration gate in :mod:`alems.blink` exists to
    reject. Returns ``(timestamps, eyes_detected, n_true_blinks)``.
    """
    rng = np.random.default_rng(seed)

    n_frames = int(round(duration_seconds * fps))
    t = np.arange(n_frames) / fps
    eyes = np.ones(n_frames, dtype=bool)

    n_blinks = int(round(blink_rate_per_min * duration_seconds / 60.0))
    # Place blinks on a jittered regular grid to avoid overlap.
    slots = np.linspace(1.0, duration_seconds - 1.0, n_blinks)
    slots = slots + rng.uniform(-0.4, 0.4, size=n_blinks)

    for start in slots:
        i0 = int(round(start * fps))
        i1 = int(round((start + blink_duration_seconds) * fps))
        eyes[i0:i1] = False

    # Single-frame dropouts, never adjacent to a real blink.
    for i in np.flatnonzero(rng.random(n_frames) < dropout_rate):
        if 1 <= i < n_frames - 1 and eyes[i - 1] and eyes[i + 1]:
            eyes[i] = False

    return t, eyes, n_blinks
