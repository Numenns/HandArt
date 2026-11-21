from dataclasses import dataclass
import sys
from threading import Lock
from time import monotonic

import cv2
import numpy as np
import torch
from .gestures import GestureFilter, Recognition
from .model import GestureModel
from .tracking import HandTracker, TrackedHand
from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    hand: TrackedHand | None
    gesture: Recognition
    captured_at: float


def open_camera(index: int):
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return capture
        capture.release()
    raise RuntimeError(f"No se pudo abrir la cámara {index}. Revisa el índice, los permisos y si otra aplicación la está usando.")


class CameraWorker(QThread):
    failed = Signal(str)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._lock = Lock()
        self._latest: CameraFrame | None = None

    def take_latest(self) -> CameraFrame | None:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def run(self):
        try:
            cv2.setNumThreads(1)
            torch.set_num_threads(1)
            model = GestureModel()
            if self.isInterruptionRequested():
                return
            gesture_filter = GestureFilter()
            with HandTracker() as tracker:
                if self.isInterruptionRequested():
                    return
                capture = open_camera(self.index)
                try:
                    self._read_frames(capture, tracker, model, gesture_filter)
                finally:
                    capture.release()
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))

    def _read_frames(self, capture, tracker, model, gesture_filter):
        while not self.isInterruptionRequested():
            ok, bgr = capture.read()
            captured_at = monotonic()
            if self.isInterruptionRequested():
                return
            if not ok or bgr is None:
                raise RuntimeError("Se perdió la captura. Comprueba la conexión e inicia la cámara de nuevo.")
            rgb = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
            hand = tracker.detect(rgb, captured_at)

            scores = model.predict(rgb) if hand is not None else {}
            gesture = gesture_filter.update(scores, hand.world_points if hand else None, captured_at)
            with self._lock:
                self._latest = CameraFrame(rgb, hand, gesture, captured_at)
