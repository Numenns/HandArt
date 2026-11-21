from dataclasses import dataclass
from math import acos, copysign, degrees, exp, isfinite

import numpy as np


DRAW = "Indice"
COLOR = "2 dedos"
ERASE = "Palma"
RESIZE = "Pinza"
INK_GESTURES = (DRAW, ERASE)

MAX_FRAME_GAP = 0.25
ENTER_SCORE = 0.65
HOLD_SCORE = 0.50
ENTER_MARGIN = 0.15
HOLD_MARGIN = 0.08
MIN_FRAMES = 3
MIN_DURATION = 0.09
PINCH_ENTER = 0.32
PINCH_RELEASE = 0.42
EXTENDED_REACH = 1.42
FOLDED_REACH = 1.33
RELEASE_GRACE = {DRAW: 0.10}


def angle(a, joint, b) -> float:
    left, right = a - joint, b - joint
    lengths = np.linalg.norm(left) * np.linalg.norm(right)
    if lengths <= 1e-10:
        return float("nan")
    return degrees(acos(float(np.clip(np.dot(left, right) / lengths, -1, 1))))


def extended(points, mcp: int) -> bool | None:

    pip = angle(points[mcp], points[mcp + 1], points[mcp + 2])
    base_length = np.linalg.norm(points[mcp] - points[0])
    if base_length <= 1e-6 or not np.isfinite(pip):
        return None
    reach = np.linalg.norm(points[mcp + 3] - points[0]) / base_length
    if pip >= 135 and reach >= EXTENDED_REACH:
        return True
    if pip <= 120 or reach <= FOLDED_REACH:
        return False
    return None


def pinch_aperture(points: np.ndarray) -> float | None:
    palm_length = np.linalg.norm(points[9] - points[0])
    if palm_length <= 1e-6:
        return None
    return float(np.linalg.norm(points[4] - points[8]) / palm_length)


def matches_pose(label: str, points: np.ndarray, holding: bool) -> bool:
    gap = pinch_aperture(points)
    if gap is None:
        return False
    if label == RESIZE:
        return gap <= (PINCH_RELEASE if holding else PINCH_ENTER)

    if gap <= PINCH_RELEASE:
        return False

    index, middle, ring, pinky = (extended(points, mcp) for mcp in (5, 9, 13, 17))
    if label == DRAW:
        return index is True and middle is False and ring is False and pinky is False
    if label == COLOR:
        return index is True and middle is True and ring is False and pinky is False
    return label == ERASE and all((index, middle, ring, pinky))


def geometric_label(points: np.ndarray, active: str | None = None) -> str | None:


    for label in (RESIZE, DRAW, COLOR, ERASE):
        if matches_pose(label, points, holding=label == active):
            return label
    return None


@dataclass(frozen=True)
class Recognition:
    raw: str | None = None
    confidence: float | None = None
    active: str | None = None
    reason: str = "Sin mano"
    activation_id: int = 0
    aperture: float | None = None


class GestureFilter:
    def __init__(self, rescue: bool = True):


        self.rescue = rescue
        self._last_good = None
        self._activation_id = 0
        self._active_id = 0
        self._color_activation = self._color_released_at = None
        self._reset()

    def _reset(self):
        self._active = self._candidate = None
        self._count = 0
        self._start = self._last_frame = None

    def update(self, scores: dict[str, float], points: np.ndarray | None,
               captured_at: float) -> Recognition:
        if self._color_activation is not None:
            self._check_color_release(points, captured_at)
        if not scores or points is None:
            self._reset()
            return Recognition()

        if self._last_frame is not None and captured_at - self._last_frame > MAX_FRAME_GAP:
            self._reset()
        self._last_frame = captured_at
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        label, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        holding = label == self._active
        min_score = HOLD_SCORE if holding else ENTER_SCORE
        min_margin = HOLD_MARGIN if holding else ENTER_MARGIN
        rescued = False
        if score < min_score or score - second < min_margin:
            reason = "Confianza insuficiente"
        elif not matches_pose(label, points, holding):
            reason = "Postura sin confirmar"
        else:
            reason = None
        if reason and self.rescue:

            geo = geometric_label(points, self._active)
            if geo is not None:
                label, score = geo, scores.get(geo, 0.0)
                holding, reason, rescued = label == self._active, None, True
        if reason is None:
            if not holding:

                self._active = None
                if label != self._candidate:
                    self._candidate, self._start, self._count = label, captured_at, 0
                self._count += 1
                if self._count < MIN_FRAMES or captured_at - self._start < MIN_DURATION:
                    return Recognition(label, score, reason="Estabilizando gesto")
                self._active = label
                self._active_id = self._begin_activation(label)
            aperture = pinch_aperture(points) if label == RESIZE else None
            self._last_good = captured_at
            return Recognition(label, score, label,
                               "Rescate geométrico" if rescued else "Confirmado",
                               self._active_id, aperture)

        grace = RELEASE_GRACE.get(self._active, 0.0)
        if grace and self._last_good is not None and captured_at - self._last_good <= grace:
            return Recognition(self._active, None, self._active, "Gracia de salida",
                               self._active_id)
        self._active = self._candidate = None
        return Recognition(label, score, reason=reason)

    def _begin_activation(self, label: str) -> int:
        if label == COLOR and self._color_activation is not None:
            return self._color_activation
        self._activation_id += 1
        if label == COLOR:
            self._color_activation = self._activation_id
        return self._activation_id

    def _check_color_release(self, points: np.ndarray | None, captured_at: float):

        if points is not None and matches_pose(COLOR, points, holding=True):
            self._color_released_at = None
        elif self._color_released_at is None:
            self._color_released_at = captured_at
        elif captured_at - self._color_released_at >= MIN_DURATION:
            self._color_activation = self._color_released_at = None


class PinchSizeController:


    SMOOTHING_SECONDS = 0.10
    APERTURE_DEADBAND = 0.012
    PIXELS_PER_PALM_LENGTH = 70
    MAX_PIXELS_PER_SECOND = 12.0

    def __init__(self, limits: tuple[int, int]):
        self.limits = limits
        self.reset()

    def reset(self):
        self._activation_id = None
        self._timestamp = None
        self._filtered = self._anchor = None
        self._value = self._target = None

    def update(self, aperture: float | None, captured_at: float,
               activation_id: int, width: int) -> int:
        if aperture is None or not isfinite(aperture) or aperture < 0:
            self.reset()
            return width
        gap = captured_at - self._timestamp if self._timestamp is not None else None
        if (self._activation_id != activation_id or gap is None or gap > MAX_FRAME_GAP
                or round(self._value) != width):
            self._activation_id = activation_id
            self._timestamp = captured_at
            self._filtered = self._anchor = aperture
            self._value = self._target = float(width)
            return width
        if gap <= 0:
            return width

        self._timestamp = captured_at

        dt = min(gap, 0.10)
        alpha = 1 - exp(-dt / self.SMOOTHING_SECONDS)
        self._filtered += alpha * (aperture - self._filtered)
        delta = self._filtered - self._anchor
        movement = copysign(max(0.0, abs(delta) - self.APERTURE_DEADBAND), delta)
        self._anchor += movement
        low, high = self.limits

        self._target = min(high, max(low, self._target + movement * self.PIXELS_PER_PALM_LENGTH))
        step = self.MAX_PIXELS_PER_SECOND * dt
        self._value += min(step, max(-step, self._target - self._value))
        return round(self._value)
