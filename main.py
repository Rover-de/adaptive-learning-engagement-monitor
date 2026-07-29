"""Entry point: start the capture thread and serve the read-only API."""

from __future__ import annotations

import argparse
import sys

from waitress import serve

from alems import AppConfig, CapturePipeline, ServerConfig, SharedState, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Learning Engagement Monitoring System"
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=5050, help="bind port")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="rolling analysis window; longer is more accurate but less responsive",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    signal = base.signal
    if args.window_seconds is not None:
        signal = type(signal)(**{**vars(signal), "window_seconds": args.window_seconds})
    return AppConfig(
        signal=signal,
        blink=base.blink,
        regime=base.regime,
        server=ServerConfig(host=args.host, port=args.port),
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)

    state = SharedState()
    pipeline = CapturePipeline(state, config=config, camera_index=args.camera)

    try:
        pipeline.start_thread()
    except RuntimeError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1

    app = create_app(state, metrics=pipeline.metrics, config=config)
    url = f"http://{config.server.host}:{config.server.port}"
    print(f"dashboard  {url}/")
    print(f"status     {url}/status")
    print(f"metrics    {url}/metrics")
    print("Ctrl+C to stop.")

    try:
        serve(app, host=config.server.host, port=config.server.port, _quiet=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
