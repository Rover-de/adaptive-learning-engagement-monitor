"""Adaptive Learning Engagement Monitoring System.

A real-time pipeline that turns a webcam stream into a gated engagement signal:
forehead ROI intensity -> bandpass filter -> spectral heart-rate estimate and
beat-to-beat HRV -> quality-gated regime classification -> low-latency feed.
"""

from .blink import BlinkDetector, BlinkState
from .config import (
    AppConfig,
    BlinkConfig,
    RegimeConfig,
    ServerConfig,
    SignalConfig,
)
from .engagement import EngagementClassifier, EngagementSnapshot, Regime
from .metrics import LatencyTracker, PipelineMetrics
from .pipeline import CapturePipeline, SharedState
from .rppg import RppgEstimate, RppgEstimator
from .server import create_app

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "BlinkConfig",
    "BlinkDetector",
    "BlinkState",
    "CapturePipeline",
    "EngagementClassifier",
    "EngagementSnapshot",
    "LatencyTracker",
    "PipelineMetrics",
    "Regime",
    "RegimeConfig",
    "RppgEstimate",
    "RppgEstimator",
    "ServerConfig",
    "SharedState",
    "SignalConfig",
    "create_app",
    "__version__",
]
