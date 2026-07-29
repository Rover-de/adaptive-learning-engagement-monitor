"""Render the quality-gate frontier as a figure.

Reads the committed output of :mod:`benchmarks.validate_hr_accuracy` and plots
tail error against retained sample count as the gate threshold varies.

A frontier is the honest way to present a filter whose operating point is a
choice: there is no single "accuracy" number, only a curve, and where to sit on
it depends on what the consumer needs. The elbow is the whole story, and it is
the one result in the benchmark that a table genuinely obscures.

The SNR sweep is deliberately not plotted. Its interesting feature is the
phase-versus-peak ranking, and those differences run to hundredths of a bpm —
invisible at any sensible figure scale, and already legible in the table.

Reads only the JSON, never re-runs the sweep, so the figure cannot drift from
the committed numbers.

Run with:  python -m benchmarks.plot_results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_JSON = RESULTS_DIR / "hr_accuracy.json"

DPI = 130

SPECTRAL_COLOUR = "#1f4e8c"
ACCENT = "#c1121f"
GRID_COLOUR = "#d6dbe3"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4a5568",
            "axes.labelcolor": "#1a202c",
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.grid": True,
            "grid.color": GRID_COLOUR,
            "grid.linewidth": 0.7,
            "text.color": "#1a202c",
            "xtick.color": "#4a5568",
            "ytick.color": "#4a5568",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "font.family": "sans-serif",
        }
    )


def plot_frontier(payload: dict[str, Any], out: Path) -> None:
    """Retained coverage against tail error, as the gate threshold varies."""
    rows = payload["sqi_frontier"]
    default_sqi = payload["config"]["min_sqi"]

    # Thresholds below the noise floor of the metric map to an identical point.
    # Plotting them separately would imply distinct operating points that do
    # not exist, so co-located thresholds share one marker and one label.
    merged: list[dict[str, Any]] = []
    for row in rows:
        key = (round(row["coverage"], 6), round(row["p95_abs_err_bpm"], 6))
        if merged and merged[-1]["key"] == key:
            merged[-1]["thresholds"].append(row["min_sqi"])
        else:
            merged.append({"key": key, "thresholds": [row["min_sqi"]], "row": row})

    coverage = [m["row"]["coverage"] * 100.0 for m in merged]
    p95 = [m["row"]["p95_abs_err_bpm"] for m in merged]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    ax.plot(coverage, p95, "-", color=SPECTRAL_COLOUR, linewidth=1.8, zorder=2)
    ax.plot(
        coverage, p95, "o", color="white", markeredgecolor=SPECTRAL_COLOUR,
        markeredgewidth=1.8, markersize=8, zorder=3,
    )

    for m, x, y in zip(merged, coverage, p95):
        thresholds = m["thresholds"]
        label = (
            f"{thresholds[0]:.2f}"
            if len(thresholds) == 1
            else f"{min(thresholds):.2f}–{max(thresholds):.2f}"
        )
        is_default = any(abs(t - default_sqi) < 1e-9 for t in thresholds)

        if is_default:
            ax.plot(
                x, y, "o", color=ACCENT, markersize=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2,
            )
            ax.annotate(
                f"SQI ≥ {label}\nconfigured default",
                xy=(x, y),
                xytext=(14, 26),
                textcoords="offset points",
                fontsize=10,
                fontweight="600",
                color=ACCENT,
                ha="left",
                arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.1),
            )
        else:
            # On the steep upper arm the curve runs up and to the right, so a
            # label placed underneath lands on the line itself. Those go left.
            on_steep_arm = x >= 90.0
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(-10, 3) if on_steep_arm else (0, -17),
                textcoords="offset points",
                fontsize=9,
                color="#4a5568",
                ha="right" if on_steep_arm else "center",
            )

    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 25])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(1.0, 40.0)
    ax.set_xlim(48, 106)

    ax.set_xlabel("Observations retained by the gate  (%)")
    ax.set_ylabel("95th-percentile absolute error  (bpm, log scale)")
    ax.set_title("Quality-gate frontier: tail error bought with coverage")

    # The frontier is only meaningful relative to what the window can resolve.
    tolerance = payload["tolerance_bpm"]
    ax.axhline(tolerance, color="#718096", linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate(
        f"frequency resolution of a {payload['config']['window_seconds']:.0f} s "
        f"window  ({tolerance:.1f} bpm)",
        xy=(49.5, tolerance),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color="#718096",
    )

    ax.annotate(
        "raising the threshold\n← discards more, errs less",
        xy=(0.04, 0.20),
        xycoords="axes fraction",
        fontsize=9.5,
        color="#4a5568",
        style="italic",
    )

    n_total = payload["trials_per_level"] * len(payload["by_snr"])
    fig.text(
        0.5, 0.005,
        f"{n_total} synthetic trials, seed {payload['seed']} · labels are the "
        f"minimum spectral quality index required to publish an estimate",
        ha="center", fontsize=8.5, color="#718096",
    )

    fig.tight_layout(rect=(0, 0.028, 1, 1))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> int:
    if not RESULTS_JSON.exists():
        print(f"missing {RESULTS_JSON}; run python -m benchmarks.validate_hr_accuracy first")
        return 1

    payload = json.loads(RESULTS_JSON.read_text())
    _style()

    out = RESULTS_DIR / "sqi_frontier.png"
    plot_frontier(payload, out)
    print(f"wrote {out.name}  ({out.stat().st_size / 1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
