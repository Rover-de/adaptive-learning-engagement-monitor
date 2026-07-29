# Adaptive Learning Engagement Monitoring System

**English** · [简体中文](README.zh-CN.md)

<p align="center">
  <a href="https://rover-de.github.io/adaptive-learning-engagement-monitor/"><strong>Open results page</strong></a>
  · key numbers and the quality-gate frontier, no install required
</p>

Estimating heart rate and beat-to-beat variability from a commodity webcam, and
turning them into a **quality-gated** engagement signal that a downstream
consumer can act on.

The physiological application is the vehicle. The engineering problem is the one
that generalises: extract a periodic signal buried in noise, quantify how wrong
the estimate is *before* publishing it, and refuse to emit a confident-looking
number when the evidence does not support one.

<p align="center"><em>webcam → signal processing → quality gate → engagement regime → JSON API</em></p>

---

## Why this might interest a quantitative reader

The domain is remote photoplethysmography; the machinery is signal estimation
under a low signal-to-noise ratio, which is the same problem shape as extracting
a weak predictive signal from market data.

| Component here | The equivalent problem |
| :---: | :---: |
| Spectral quality index used as a **gate**, not an output | Trading only when signal confidence clears a threshold; the operating point sits on a precision/coverage frontier |
| Regime label requires 3 consecutive confirmations | Hysteresis to suppress turnover at a threshold boundary |
| RMSSD = RMS of first differences of the interval series | Structurally identical to realised volatility from a return series, and inherits the same outlier sensitivity |
| Latency tracked as p50/p95/p99, never as a mean | Tail latency is what causes dropped frames; dropped frames bias the frequency estimate |
| Every constant lives in one frozen config object, one master seed | Bit-for-bit reproducible results |

The headline finding below is a **negative** one, reported rather than buried.

---

## Measured results

All numbers come from synthetic signals whose true heart rate and true RMSSD are
known by construction, so error is measured against ground truth rather than
against another estimator. **This validates the numerical chain, not the
physiology** — there is no pulse-oximeter reference and no labelled human data
in this repository. See [Limitations](#limitations).

Reproduce the whole table, bit for bit:

```bash
python -m benchmarks.validate_hr_accuracy   # tables and results JSON
python -m benchmarks.plot_results           # figure; needs requirements-dev.txt
```

The plotting step reads the committed JSON and never re-runs the sweep, so the
figure cannot drift from the numbers beside it.

1800 trials · seed `20240301` · 20 s window @ 30 fps · passband 0.70–3.00 Hz
(42–180 bpm). Committed output: [`benchmarks/results/hr_accuracy.json`](benchmarks/results/hr_accuracy.json).

The error tolerance used throughout is **3.0 bpm**, which is not a preference —
a periodogram over a 20 s window resolves 1/20 Hz, and that is 3.0 bpm. Claiming
accuracy finer than the window's own frequency resolution would be meaningless.

### 1. Heart-rate accuracy across the noise grid

Spectral estimate, 200 trials per level:

| SNR (dB) | mean SQI | MAE (bpm) | RMSE (bpm) | bias (bpm) | within 3 bpm |
| :---: | :---: | :---: | :---: | :---: | :---: |
| −20 | 0.261 | 19.13 | 28.40 | +11.58 | 34.0% |
| −15 | 0.302 | 9.24 | 19.88 | +4.96 | 72.5% |
| −12 | 0.379 | 1.40 | 4.93 | +0.57 | 95.5% |
| −9 | 0.510 | 0.55 | 0.85 | +0.06 | 99.5% |
| −6 | 0.636 | 0.45 | 0.59 | −0.01 | 100.0% |
| −3 | 0.743 | 0.44 | 0.60 | −0.03 | 100.0% |
| 0 | 0.821 | 0.40 | 0.58 | +0.01 | 100.0% |
| +3 | 0.867 | 0.44 | 0.63 | −0.03 | 100.0% |
| +6 | 0.899 | 0.41 | 0.59 | −0.01 | 100.0% |

Two things worth reading off this table. The transition is sharp rather than
gradual: between −15 dB and −12 dB the hit rate moves from 72.5% to 95.5%, so
performance is governed by whether the pulse peak dominates its neighbourhood at
all, not by noise level in any smooth sense. And the bias is strictly positive
where the estimator fails (+11.58 bpm at −20 dB) — when the true peak is lost,
the argmax lands elsewhere in a passband that extends to 180 bpm, so failures
are asymmetric and skew high. A mean absolute error alone would hide that.

### 2. The quality gate is the actual contribution

The spectral quality index measures how concentrated in-band power is around the
dominant peak. It is not published as a feature; it decides whether the estimate
is published at all. Raising the threshold buys precision and pays for it in
coverage:

<p align="center">
  <img src="benchmarks/results/sqi_frontier.png" alt="Quality-gate frontier" width="70%" />
</p>

The shape is the result. There is an elbow, and past it the curve flattens hard:
further tightening buys almost no precision and costs coverage steadily. Pooled
over all 1800 trials:

| min SQI | retained | MAE (bpm) | RMSE (bpm) | p95 abs err | within 3 bpm |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 100.0% | 3.61 | 11.68 | 25.20 | 89.1% |
| 0.10 | 100.0% | 3.61 | 11.68 | 25.20 | 89.1% |
| 0.20 | 96.8% | 2.91 | 10.17 | 18.31 | 91.1% |
| **0.30** | **83.5%** | **1.28** | **5.96** | **1.83** | **97.1%** |
| 0.40 | 70.8% | 0.56 | 2.34 | 1.42 | 99.7% |
| 0.50 | 62.9% | 0.49 | 1.62 | 1.36 | 99.8% |
| 0.60 | 53.7% | 0.43 | 0.61 | 1.35 | 100.0% |

The configured default of 0.30 discards 16.5% of observations and in exchange
cuts the 95th-percentile absolute error from **25.20 bpm to 1.83 bpm** — a
13.8× reduction in the tail, for a sixth of the sample. That is the single most
valuable line in this repository.

Note also that the p95 error collapses much faster than the RMSE, which is still
5.96 bpm at the same threshold. The gate is efficient at removing *typical* bad
estimates and much weaker against the rare catastrophic one, so the residual
risk after gating is concentrated in a thin tail. A consumer that cares about
worst case should gate at 0.40 or higher and accept 70.8% coverage.

Raising the threshold from 0.00 to 0.10 changes nothing at all: band-limited
noise already scores about 0.13 by construction, so no realisable observation
lives below 0.10. The frontier's flat segment is a property of the metric, not a
measurement artefact.

### 3. Beat localisation: phase crossings beat peak picking where it matters

RMSSD requires locating individual beats, and a local maximum is defined by only
three samples — so its position inherits whatever noise sits on those three
samples. The alternative is to track the instantaneous phase of the analytic
signal, which is determined by the whole cycle. Both methods are evaluated on
the *same* buffer in every trial, making the comparison paired.

Time-domain heart rate, mean absolute error in bpm:

| SNR (dB) | phase | peak | verdict |
| :---: | :---: | :---: | :---: |
| −20 | 18.58 | 19.02 | phase |
| −15 | 9.84 | 10.04 | phase |
| −12 | 2.10 | 2.22 | phase |
| −9 | 0.74 | 0.79 | phase |
| −6 | 0.39 | 0.38 | peak, marginally |
| 0 | 0.26 | 0.23 | peak, marginally |
| +6 | 0.24 | 0.20 | peak, marginally |

The crossover is the point: peak picking wins slightly on clean signals, phase
tracking wins on noisy ones. Since webcam rPPG operates at low SNR, `phase` is
the configured default — and the benchmark, not intuition, is what settles it.
On gated RMSSD the margin is unambiguous (see below).

### 4. Negative result: RMSSD carries no usable cross-sectional information

On gated observations (SQI ≥ 0.30, n = 1503):

| method | MAE (ms) | RMSE (ms) | bias (ms) | Pearson r vs truth |
| :---: | :---: | :---: | :---: | :---: |
| phase | 28.3 | 37.2 | +14.5 | **0.046** |
| peak | 31.3 | 40.9 | +19.5 | 0.033 |

Phase tracking dominates peak picking on every column, which justifies the
default. But the column that matters is the last one: **r ≈ 0.05 against known
ground truth is indistinguishable from no relationship.** The estimator recovers
heart rate to well under 2 bpm on the same gated observations and simultaneously
fails to recover variability at all.

This is not a tuning problem, and the reason is structural. Heart rate is a
property of the integrated spectrum, so averaging over a 20 s window helps it.
RMSSD is a function of differences between adjacent intervals, so window
averaging cannot help — and at 30 fps each beat time is quantised to 33 ms,
which enters every interval difference twice. Sub-sample parabolic interpolation
of peak positions is implemented to claw some of that back, and it is not
enough. The positive bias of +14.5 ms is consistent with residual timing noise
inflating a root-mean-square statistic, exactly as it would inflate a realised
volatility estimate computed from noisy prices.

The honest conclusion: **the fatigue thresholds in `RegimeConfig` rest on an
input this pipeline cannot currently measure.** They remain in the code as a
documented interface, explicitly not as a validated finding. Fixing this needs a
higher frame rate or a genuinely better beat-timing estimator, not different
thresholds.

### 5. Latency

The estimator is decoupled from the capture rate and re-evaluated at 1 Hz, so
its cost does not scale with frame rate. Over a full 586-sample window
(Apple silicon, Python 3.9, n = 300 evaluations):

| p50 | p95 | p99 | max |
| :---: | :---: | :---: | :---: |
| 1.41 ms | 1.68 ms | 1.91 ms | 2.27 ms |

Per-stage percentiles for the live pipeline — frame read, face detect, eye
detect, estimate, end-to-end — are exposed at `/metrics` and rendered on the
dashboard. Percentiles rather than means, because a pipeline averaging 8 ms that
stalls to 90 ms on one frame in fifty drops frames, and dropped frames bias the
frequency estimate.

---

## Quick start

Requires Python 3.9+, a webcam, and camera permission for your terminal.

```bash
git clone https://github.com/Rover-de/adaptive-learning-engagement-monitor.git
cd adaptive-learning-engagement-monitor

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py
```

Then open <http://127.0.0.1:5050/>. Heart rate appears after the buffer reaches
`min_window_seconds` (8 s) and stabilises over about 20 s.

No camera? The full benchmark suite runs headless and needs neither a webcam nor
the Flask server:

```bash
python -m benchmarks.validate_hr_accuracy
```

Options:

```bash
python main.py --camera 1              # select a different capture device
python main.py --port 8080             # bind elsewhere
python main.py --window-seconds 30     # longer window: finer resolution, more lag
```

The window length is the central trade-off. Frequency resolution is
`60 / window_seconds` bpm, so 30 s resolves 2.0 bpm instead of 3.0 bpm, at the
cost of responding to change more slowly.

---

## API

| Endpoint | Returns |
| :---: | :---: |
| `GET /` | Monitoring dashboard |
| `GET /status` | Current engagement snapshot (JSON) |
| `GET /metrics` | Per-stage latency percentiles and throughput (JSON) |
| `GET /video_feed` | Annotated MJPEG preview |
| `GET /static/lesson_player.html` | Optional demo: adapts video playback to the published regime |

```jsonc
// GET /status
{
  "regime": "ENGAGED",            // published label, after hysteresis
  "raw_regime": "MILD_FATIGUE",   // instantaneous classification, for diagnostics
  "regime_flips": 3,              // published transitions so far — the turnover analogue
  "hr_bpm": 72.4,                 // spectral estimate; prefer this one
  "hr_peak_bpm": 71.9,            // time-domain estimate, from beat intervals
  "rmssd_ms": 34.1,               // see the negative result above before trusting this
  "sqi": 0.71,                    // below RegimeConfig's gate, regime becomes INSUFFICIENT_DATA
  "blink_rate_per_min": 14.0,
  "effective_fps": 29.7,
  "face_locked": true
}
```

Every numeric field is nullable. A `null` means the estimator declined to
publish, which is a meaningful answer and not an error condition.

---

## Design decisions worth defending

**Resample on real timestamps.** Webcam frame delivery jitters. Assuming a
constant frame period puts a bias directly on the frequency axis, so the buffer
is interpolated onto a uniform grid using measured arrival times.

**Zero-phase filtering.** The Butterworth bandpass is applied forward and
backward. A causal filter's group delay would shift beat timings, and beat
timings are exactly what RMSSD is computed from.

**Locate beats in a narrow band, not the passband.** Searching for peaks across
0.70–3.00 Hz is what makes naive implementations produce unusable HRV: broadband
noise and the waveform's own second harmonic both generate spurious peaks, and
every spurious peak splits one interval into two. Beats are located only after
re-filtering ±0.40 Hz around the already-estimated pulse frequency.

**Drop outlier intervals rather than winsorise them.** A dropped beat doubles an
interval and a spurious peak halves one. Either would dominate a squared
statistic, and neither represents physiology, so they are excluded rather than
shrunk toward the median.

**One writer.** The capture thread owns the camera and all estimator state.
The web layer reads immutable snapshots under a lock and never touches an
estimator object.

**Rate-limit the MJPEG stream and skip unchanged frames.** The obvious
implementation — loop, re-encode whatever is in the buffer — spins a core on
hundreds of redundant JPEG encodes per second and starves the capture thread,
which shows up directly as frame loss in the estimator.

---

## Repository layout

```
alems/
  config.py       every constant that affects a number, in frozen dataclasses
  rppg.py         the signal chain: resample, detrend, filter, spectrum, beats
  blink.py        duration-gated blink detection over a binary visibility stream
  engagement.py   feature → regime classification with hysteresis
  metrics.py      rolling latency percentiles
  pipeline.py     capture thread; the only writer of estimator state
  server.py       read-only Flask API
  synthetic.py    ground-truth signal generator used by the benchmarks
benchmarks/
  validate_hr_accuracy.py   the tables above
  plot_results.py           renders the frontier figure from the JSON
  results/hr_accuracy.json  committed output
  results/sqi_frontier.png  committed figure
static/
  dashboard.html            monitoring console
  lesson_player.html        optional adaptive-playback demo
main.py
requirements.txt            runtime dependencies
requirements-dev.txt        adds matplotlib, for the figure only
docs/                       static results page (GitHub Pages)
```

The estimator consumes `(value, timestamp)` pairs and knows nothing about faces,
frames or cameras. That decoupling is what makes the whole numerical chain
testable against synthetic ground truth.

---

## Limitations

Stated plainly, because a benchmark that only reports its wins is not a
benchmark.

- **Validation is synthetic.** Ground truth comes from a generator, so the
  numbers above bound the estimator's numerical behaviour, not the accuracy of
  camera-based physiology. There is no pulse-oximeter comparison here.
- **RMSSD is not usable.** r ≈ 0.05 against known truth. Documented above in
  full; the fatigue rule that depends on it is unvalidated by construction.
- **Regime thresholds are demonstration values.** Not fitted to labelled human
  data. No clinical, diagnostic or psychometric authority whatsoever.
- **Haar cascades are a weak front end.** Blink detection infers eye closure
  from cascade dropout. Duration gating and a refractory period remove the
  dominant false-positive mode, but landmark-based eye-aspect-ratio would be
  strictly better.
- **Single subject, frontal, reasonably still.** The largest detected face wins.
  Head motion and illumination change both degrade the ROI signal, and the
  quality gate will correctly refuse to publish under either.
- **No pull-based backpressure.** A slow consumer of `/video_feed` is handled by
  frame skipping, not by flow control.

