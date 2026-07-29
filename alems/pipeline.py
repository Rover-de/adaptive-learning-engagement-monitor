"""Camera capture loop: frames in, engagement snapshots out.

Threading model: one capture thread owns the camera and all estimator state and
is the only writer. The web server reads through :class:`SharedState`, which
hands out immutable copies under a lock. No estimator object is ever touched
from a request thread.
"""

from __future__ import annotations

import time
from threading import Event, Lock, Thread
from typing import Any, Optional

import cv2
import numpy as np

from .blink import BlinkDetector, BlinkState
from .config import AppConfig
from .engagement import EngagementClassifier, EngagementSnapshot
from .metrics import PipelineMetrics
from .rppg import RppgEstimate, RppgEstimator


class SharedState:
    """Single synchronisation point between the capture thread and the server."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = EngagementSnapshot()
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._face_locked = False

    def publish(
        self, snapshot: EngagementSnapshot, frame: Optional[np.ndarray], face_locked: bool
    ) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._face_locked = face_locked
            if frame is not None:
                self._frame = frame
                self._frame_seq += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = self._snapshot.to_dict()
            payload["face_locked"] = self._face_locked
        return payload

    def frame(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._frame_seq


class CapturePipeline:
    """Owns the camera, the estimators and the annotated preview frame."""

    def __init__(
        self,
        state: SharedState,
        config: Optional[AppConfig] = None,
        camera_index: int = 0,
    ) -> None:
        self.config = config or AppConfig()
        self.state = state
        self.camera_index = camera_index
        self.metrics = PipelineMetrics()

        self._rppg = RppgEstimator(self.config.signal)
        self._blink = BlinkDetector(self.config.blink)
        self._classifier = EngagementClassifier(self.config.regime)
        self._stop = Event()
        self._cap: Optional[cv2.VideoCapture] = None

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def open(self) -> None:
        """Acquire the camera and validate models.

        Called from the main thread so that a missing camera surfaces as a clean
        startup failure instead of an exception lost inside a worker thread.
        """
        if self._face_cascade.empty() or self._eye_cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascades; check the install.")

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Check that no other application holds it and that camera "
                "permission is granted to your terminal."
            )
        self._cap = cap

    def stop(self) -> None:
        self._stop.set()

    def _forehead_roi(self, frame: np.ndarray, face: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = face
        inset = int(w * self.config.roi_width_inset_fraction)
        left, right = x + inset, x + w - inset
        top, bottom = y, y + int(h * self.config.roi_height_fraction)

        left, top = max(left, 0), max(top, 0)
        right = min(right, frame.shape[1])
        bottom = min(bottom, frame.shape[0])
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    def _annotate(
        self,
        frame: np.ndarray,
        snapshot: EngagementSnapshot,
        blink: BlinkState,
        roi_box: Optional[tuple[int, int, int, int]],
        eye_boxes: list[tuple[int, int, int, int]],
    ) -> None:
        if roi_box is not None:
            left, top, right, bottom = roi_box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        for ex, ey, ew, eh in eye_boxes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 128, 0), 2)

        hr = f"{snapshot.hr_bpm:5.1f}" if snapshot.hr_bpm is not None else "  --"
        rmssd = f"{snapshot.rmssd_ms:6.1f}" if snapshot.rmssd_ms is not None else "    --"
        sqi = f"{snapshot.sqi:.2f}" if snapshot.sqi is not None else "--"

        lines = [
            f"HR {hr} bpm   RMSSD {rmssd} ms   SQI {sqi}",
            f"Blink {blink.blink_rate_per_min:.0f}/min   Regime {snapshot.regime.value}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (10, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def run(self) -> None:
        if self._cap is None:
            self.open()
        cap = self._cap
        assert cap is not None

        last_estimate_at = 0.0
        estimate = RppgEstimate()
        blink = BlinkState()
        snapshot = EngagementSnapshot()

        try:
            while not self._stop.is_set():
                with self.metrics.end_to_end.measure():
                    with self.metrics.frame_read.measure():
                        ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.01)
                        continue

                    now = time.time()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    with self.metrics.face_detect.measure():
                        faces = self._face_cascade.detectMultiScale(
                            gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
                        )

                    roi_box: Optional[tuple[int, int, int, int]] = None
                    eye_boxes: list[tuple[int, int, int, int]] = []
                    face_locked = len(faces) > 0

                    if face_locked:
                        # Largest detection: the subject, not a background face.
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                        roi = self._forehead_roi(frame, (x, y, w, h))
                        if roi is not None and roi.size > 0:
                            # Green channel carries the strongest haemoglobin
                            # absorption contrast of the three RGB channels.
                            self._rppg.add_sample(float(np.mean(roi[:, :, 1])), now)
                            inset = int(w * self.config.roi_width_inset_fraction)
                            roi_box = (
                                x + inset,
                                y,
                                x + w - inset,
                                y + int(h * self.config.roi_height_fraction),
                            )

                        with self.metrics.eye_detect.measure():
                            eyes = self._eye_cascade.detectMultiScale(
                                gray[y : y + h, x : x + w],
                                scaleFactor=1.1,
                                minNeighbors=3,
                                minSize=(20, 20),
                            )
                        eye_boxes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
                        blink = self._blink.update(len(eyes) > 0, now)
                    else:
                        blink = self._blink.update(False, now)

                    if now - last_estimate_at >= self.config.estimate_interval_seconds:
                        last_estimate_at = now
                        with self.metrics.estimate.measure():
                            estimate = self._rppg.estimate()
                        snapshot = self._classifier.update(
                            estimate, blink, self.config.signal.min_sqi
                        )

                    self._annotate(frame, snapshot, blink, roi_box, eye_boxes)
                    self.state.publish(snapshot, frame, face_locked)
                    self.metrics.count_frame()
        finally:
            cap.release()
            self._cap = None

    def start_thread(self) -> Thread:
        """Open the camera here, then hand the loop to a daemon thread."""
        if self._cap is None:
            self.open()
        thread = Thread(target=self.run, name="capture", daemon=True)
        thread.start()
        return thread
