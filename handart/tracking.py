from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .model import ASSETS


INDEX_TIP = 8
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass(frozen=True)
class TrackedHand:
    image_points: tuple[tuple[float, float], ...]
    world_points: np.ndarray

    @property
    def tip(self) -> tuple[float, float]:
        return self.image_points[INDEX_TIP]


class HandTracker:
    def __init__(self, path: Path = ASSETS / "modelos" / "hand_landmarker.task"):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp = -1
        self._start = monotonic()

    def detect(self, rgb: np.ndarray, captured_at: float) -> TrackedHand | None:
        self._timestamp = max(self._timestamp + 1, int((captured_at - self._start) * 1000))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, self._timestamp)
        if not result.hand_landmarks or not result.hand_world_landmarks:
            return None
        points = tuple((p.x, p.y) for p in result.hand_landmarks[0])
        world = np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]])
        if len(points) != 21 or world.shape != (21, 3):
            return None
        if not np.isfinite(points).all() or not np.isfinite(world).all():
            return None
        return TrackedHand(points, world)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._landmarker.close()
