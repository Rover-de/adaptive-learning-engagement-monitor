"""Accuracy validation of the rPPG estimator against synthetic ground truth.

What this measures
------------------
The estimator is fed synthetic windows whose true heart rate and true RMSSD are
known by construction (see :mod:`alems.synthetic`), across a grid of
signal-to-noise ratios. For each SNR level we report error dispersion, not just
a central estimate, plus the hit rate inside the estimator's own frequency
resolution of 3.0 bpm at a 20 s window.

Three comparisons are the point of the exercise:

1.  **Frequency domain versus time domain.** The spectral estimate integrates
    the entire window; any time-domain estimate relies on individual beat
    timings and degrades faster as noise rises. The sweep quantifies the gap.
2.  **Phase crossings versus peak picking.** Both locate individual beats, and
    RMSSD is only definable once they are located. Each trial is evaluated
    under both methods on the *same* buffer, so the comparison is paired and
    the difference is not confounded by trial-to-trial noise. This is what
    justifies the configured default.
3.  **The precision/coverage frontier of the quality gate.** The spectral
    quality index is not an output, it is a filter. Raising the threshold
    lowers error on surviving observations and discards more of them. That is
    the same trade a signal filter makes against sample count, and the right
    operating point is a choice, not a constant -- so the frontier is reported
    rather than a single number.

Run with:  python -m benchmarks.validate_hr_accuracy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from alems.config import SignalConfig
from alems.rppg import RppgEstimator
from alems.synthetic import synthesize_rppg

# Master seed. Every derived trial seed is a deterministic function of it, so the
# whole table reproduces bit-for-bit.
SEED = 20240301

SNR_GRID_DB: tuple[float, ...] = (-20.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0)
TRIALS_PER_LEVEL = 200

HR_RANGE_BPM = (50.0, 110.0)
RMSSD_RANGE_MS = (15.0, 60.0)

SQI_GRID = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)

# Beat-localisation methods compared head to head on identical inputs.
BEAT_METHODS: tuple[str, ...] = ("phase", "peak")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Trial:
    snr_db: float
    true_hr_bpm: float
    true_rmssd_ms: float
    sqi: Optional[float]
    # Frequency-domain heart rate. Independent of beat localisation.
    hr_psd_bpm: Optional[float]
    # Time-domain heart rate and RMSSD, keyed by beat-localisation method.
    hr_by_method: dict[str, Optional[float]]
    rmssd_by_method: dict[str, Optional[float]]


def _tolerance_bpm(config: SignalConfig) -> float:
    """Frequency resolution of the window, expressed in bpm.

    A periodogram over T seconds resolves 1/T Hz. Zero-padding refines peak
    localisation but cannot beat this, so it is the natural error tolerance.
    """
    return 60.0 / config.window_seconds


def run_sweep(
    config: SignalConfig,
    snr_grid: Sequence[float] = SNR_GRID_DB,
    trials_per_level: int = TRIALS_PER_LEVEL,
) -> list[Trial]:
    rng = np.random.default_rng(SEED)
    results: list[Trial] = []

    for snr_db in snr_grid:
        for _ in range(trials_per_level):
            hr = float(rng.uniform(*HR_RANGE_BPM))
            rmssd = float(rng.uniform(*RMSSD_RANGE_MS))
            trial_seed = int(rng.integers(0, 2**32 - 1))

            series = synthesize_rppg(
                duration_seconds=config.window_seconds,
                fps=config.nominal_fps,
                hr_bpm=hr,
                rmssd_ms=rmssd,
                snr_db=snr_db,
                seed=trial_seed,
            )

            estimator = RppgEstimator(config)
            for value, ts in zip(series.values, series.timestamps):
                estimator.add_sample(value, ts)

            # One buffer, both methods: a paired comparison. The estimator does
            # not mutate state on evaluation, so the spectral part is identical
            # across the two calls by construction.
            estimates = {m: estimator.estimate(beat_timing_method=m) for m in BEAT_METHODS}
            reference = estimates[BEAT_METHODS[0]]

            results.append(
                Trial(
                    snr_db=snr_db,
                    true_hr_bpm=series.true_hr_bpm,
                    true_rmssd_ms=series.true_rmssd_ms,
                    sqi=reference.sqi,
                    hr_psd_bpm=reference.hr_psd_bpm,
                    hr_by_method={m: e.hr_peak_bpm for m, e in estimates.items()},
                    rmssd_by_method={m: e.rmssd_ms for m, e in estimates.items()},
                )
            )

    return results


def _error_stats(errors: np.ndarray, tolerance: float, n_total: int) -> dict[str, Any]:
    if errors.size == 0:
        return {"n": 0, "coverage": 0.0}
    return {
        "n": int(errors.size),
        "coverage": round(errors.size / n_total, 4),
        "mae_bpm": round(float(np.mean(np.abs(errors))), 3),
        "rmse_bpm": round(float(np.sqrt(np.mean(errors**2))), 3),
        "bias_bpm": round(float(np.mean(errors)), 3),
        "p95_abs_err_bpm": round(float(np.percentile(np.abs(errors), 95)), 3),
        f"within_{tolerance:.1f}bpm": round(
            float(np.mean(np.abs(errors) <= tolerance)), 4
        ),
    }


def summarise_by_snr(trials: list[Trial], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for snr in sorted({t.snr_db for t in trials}):
        level = [t for t in trials if t.snr_db == snr]
        n_total = len(level)

        psd_err = np.array(
            [t.hr_psd_bpm - t.true_hr_bpm for t in level if t.hr_psd_bpm is not None]
        )
        sqi = np.array([t.sqi for t in level if t.sqi is not None])

        row: dict[str, Any] = {
            "snr_db": snr,
            "n_trials": n_total,
            "mean_sqi": round(float(sqi.mean()), 4) if sqi.size else None,
            "spectral": _error_stats(psd_err, tolerance, n_total),
        }
        for method in BEAT_METHODS:
            err = np.array(
                [
                    t.hr_by_method[method] - t.true_hr_bpm
                    for t in level
                    if t.hr_by_method.get(method) is not None
                ]
            )
            row[f"time_domain_{method}"] = _error_stats(err, tolerance, n_total)

        rows.append(row)

    return rows


def summarise_sqi_frontier(
    trials: list[Trial], tolerance: float, sqi_grid: Sequence[float] = SQI_GRID
) -> list[dict[str, Any]]:
    """Error and retained sample count as a function of the gate threshold."""
    usable = [t for t in trials if t.hr_psd_bpm is not None and t.sqi is not None]
    n_total = len(usable)
    rows: list[dict[str, Any]] = []

    for threshold in sqi_grid:
        kept = [t for t in usable if t.sqi >= threshold]
        errors = np.array([t.hr_psd_bpm - t.true_hr_bpm for t in kept])
        stats = _error_stats(errors, tolerance, n_total)
        stats["min_sqi"] = threshold
        rows.append(stats)

    return rows


def summarise_rmssd(trials: list[Trial], min_sqi: float) -> dict[str, Any]:
    """RMSSD accuracy on gated observations, per beat-localisation method.

    RMSSD is far harder than heart rate: it depends on individual beat timings
    rather than on an integrated spectrum, so a single mis-placed beat moves it
    materially. It is reported separately and honestly.
    """
    gated = [t for t in trials if t.sqi is not None and t.sqi >= min_sqi]
    out: dict[str, Any] = {"min_sqi": min_sqi, "n_gated": len(gated)}

    for method in BEAT_METHODS:
        kept = [t for t in gated if t.rmssd_by_method.get(method) is not None]
        if not kept:
            out[method] = {"n": 0}
            continue

        est = np.array([t.rmssd_by_method[method] for t in kept])
        true = np.array([t.true_rmssd_ms for t in kept])
        err = est - true

        out[method] = {
            "n": len(kept),
            "coverage_of_all_trials": round(len(kept) / len(trials), 4),
            "mae_ms": round(float(np.mean(np.abs(err))), 3),
            "rmse_ms": round(float(np.sqrt(np.mean(err**2))), 3),
            "bias_ms": round(float(np.mean(err)), 3),
            "pearson_r": round(float(np.corrcoef(est, true)[0, 1]), 4),
        }

    return out


def _fmt(value: Any, spec: str = ".2f") -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return format(value, spec)


def print_report(
    by_snr: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    rmssd: dict[str, Any],
    tolerance: float,
) -> None:
    hit_key = f"within_{tolerance:.1f}bpm"

    def hit(stats: dict[str, Any]) -> str:
        value = stats.get(hit_key)
        return "--" if value is None else format(value * 100, ".1f")

    print(f"\nHeart-rate accuracy by SNR  (tolerance = {tolerance:.1f} bpm)")
    print(
        f"{'SNR dB':>7} | {'mean SQI':>8} | {'PSD MAE':>8} {'PSD RMSE':>9} "
        f"{'PSD bias':>9} {'PSD hit':>8} | {'phase MAE':>10} {'phase hit':>10} | "
        f"{'peak MAE':>9} {'peak hit':>9}"
    )
    print("-" * 112)
    for row in by_snr:
        s = row["spectral"]
        ph = row["time_domain_phase"]
        pk = row["time_domain_peak"]
        print(
            f"{row['snr_db']:>7.0f} | {_fmt(row['mean_sqi'], '.3f'):>8} | "
            f"{_fmt(s.get('mae_bpm')):>8} {_fmt(s.get('rmse_bpm')):>9} "
            f"{_fmt(s.get('bias_bpm'), '+.2f'):>9} {hit(s):>7}% | "
            f"{_fmt(ph.get('mae_bpm')):>10} {hit(ph):>9}% | "
            f"{_fmt(pk.get('mae_bpm')):>9} {hit(pk):>8}%"
        )

    print("\nQuality-gate frontier  (spectral HR, pooled across all SNR levels)")
    print(
        f"{'min SQI':>8} | {'retained':>9} | {'MAE bpm':>8} {'RMSE bpm':>9} "
        f"{'p95 |err|':>10} {'hit rate':>9}"
    )
    print("-" * 66)
    for row in frontier:
        print(
            f"{row['min_sqi']:>8.2f} | {row['coverage'] * 100:>8.1f}% | "
            f"{_fmt(row.get('mae_bpm')):>8} {_fmt(row.get('rmse_bpm')):>9} "
            f"{_fmt(row.get('p95_abs_err_bpm')):>10} {hit(row):>8}%"
        )

    print(f"\nRMSSD accuracy on gated observations  (SQI >= {rmssd['min_sqi']:.2f})")
    print(
        f"{'method':>8} | {'n':>5} {'coverage':>9} | {'MAE ms':>7} {'RMSE ms':>8} "
        f"{'bias ms':>8} {'r':>7}"
    )
    print("-" * 60)
    for method in BEAT_METHODS:
        stats = rmssd.get(method, {})
        if not stats.get("n"):
            print(f"{method:>8} |    no gated observations")
            continue
        print(
            f"{method:>8} | {stats['n']:>5} "
            f"{stats['coverage_of_all_trials'] * 100:>8.1f}% | "
            f"{stats['mae_ms']:>7.1f} {stats['rmse_ms']:>8.1f} "
            f"{stats['bias_ms']:>+8.1f} {stats['pearson_r']:>7.3f}"
        )


def main() -> int:
    config = SignalConfig()
    tolerance = _tolerance_bpm(config)

    print("Adaptive Learning Engagement Monitoring System -- accuracy validation")
    print(
        f"window = {config.window_seconds:.0f} s @ {config.nominal_fps:.0f} fps  "
        f"passband = {config.band_hz[0]:.2f}-{config.band_hz[1]:.2f} Hz  "
        f"({config.hr_min_bpm:.0f}-{config.hr_max_bpm:.0f} bpm)"
    )
    print(f"trials = {len(SNR_GRID_DB) * TRIALS_PER_LEVEL}  seed = {SEED}")

    trials = run_sweep(config)
    by_snr = summarise_by_snr(trials, tolerance)
    frontier = summarise_sqi_frontier(trials, tolerance)
    rmssd = summarise_rmssd(trials, config.min_sqi)

    print_report(by_snr, frontier, rmssd, tolerance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "trials_per_level": TRIALS_PER_LEVEL,
        "tolerance_bpm": tolerance,
        "beat_methods": list(BEAT_METHODS),
        "config": {
            "window_seconds": config.window_seconds,
            "nominal_fps": config.nominal_fps,
            "hr_min_bpm": config.hr_min_bpm,
            "hr_max_bpm": config.hr_max_bpm,
            "filter_order": config.filter_order,
            "beat_timing_method": config.beat_timing_method,
            "min_sqi": config.min_sqi,
        },
        "by_snr": by_snr,
        "sqi_frontier": frontier,
        "rmssd": rmssd,
    }
    out = RESULTS_DIR / "hr_accuracy.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    display = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print(f"\nwrote {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
