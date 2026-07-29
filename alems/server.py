"""Flask read-only API over the shared pipeline state.

Three endpoints:

``GET /status``      current engagement snapshot (the feed a consumer acts on)
``GET /metrics``     per-stage latency percentiles and throughput
``GET /video_feed``  annotated MJPEG preview

The preview stream is rate-limited and emits only frames it has not already
sent. The obvious implementation -- loop, re-encode whatever is in the buffer --
spins a core at hundreds of redundant JPEG encodes per second and starves the
capture thread, which shows up directly as frame loss in the estimator.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
from flask import Flask, Response, jsonify, send_from_directory

from .config import AppConfig
from .metrics import PipelineMetrics
from .pipeline import SharedState


def create_app(
    state: SharedState,
    metrics: Optional[PipelineMetrics] = None,
    config: Optional[AppConfig] = None,
) -> Flask:
    cfg = config or AppConfig()
    app = Flask(__name__, static_folder="../static", static_url_path="/static")

    @app.get("/")
    def index() -> Response:
        return send_from_directory(app.static_folder, "dashboard.html")

    @app.get("/status")
    def status() -> Response:
        return jsonify(state.snapshot())

    @app.get("/metrics")
    def latency_metrics() -> Response:
        if metrics is None:
            return jsonify({"error": "metrics not attached"}), 503
        return jsonify(metrics.summary())

    @app.get("/video_feed")
    def video_feed() -> Response:
        min_period = 1.0 / max(cfg.server.stream_fps, 1.0)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.server.stream_jpeg_quality]

        def generate() -> Iterator[bytes]:
            last_seq = -1
            while True:
                started = time.perf_counter()
                frame, seq = state.frame()

                if frame is None or seq == last_seq:
                    time.sleep(min_period / 4.0)
                    continue

                last_seq = seq
                ok, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                remaining = min_period - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app
