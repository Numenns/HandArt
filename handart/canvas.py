from collections import deque
from math import exp, hypot

from PySide6.QtCore import QIODevice, QPointF, QRectF, QSaveFile, Qt, Signal
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .gestures import DRAW, ERASE, INK_GESTURES, MAX_FRAME_GAP


WIDTH, HEIGHT = 1200, 900
EDGE_MARGIN = 0.08
WIDTH_LIMITS = (2, 96)
CURSOR_TIME_CONSTANT = 0.045
MAX_CURSOR_JUMP = 0.18


class Canvas(QWidget):
    history_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAccessibleName("Lienzo HandArt")
        self.color = QColor("#17253a")
        self.tool = DRAW
        self._width_px = 9
        self._paper = self._blank("white")
        self._ink = self._blank(Qt.GlobalColor.transparent)
        self._history = deque(maxlen=20)
        self.reset_pointer()

    @staticmethod
    def _blank(color) -> QImage:
        image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(color))
        return image

    def _remember(self):
        self._history.append((QImage(self._paper), QImage(self._ink)))
        self.history_changed.emit(True)

    def undo(self):
        if self._history:
            self._paper, self._ink = self._history.pop()
            self.history_changed.emit(bool(self._history))
            self.reset_pointer()

    def clear(self):
        self._remember()
        self._paper = self._blank("white")
        self._ink = self._blank(Qt.GlobalColor.transparent)
        self.reset_pointer()

    def open_image(self, path: str):
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(reader.errorString())
        self._remember()
        self._paper = self._blank("white")
        self._ink = self._blank(Qt.GlobalColor.transparent)
        rect = self._fit_rect(WIDTH, HEIGHT, image.width(), image.height())
        with QPainter(self._paper) as painter:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, image)
        self.reset_pointer()

    def save_png(self, path: str):
        flattened = self._paper.copy()
        with QPainter(flattened) as painter:
            painter.drawImage(0, 0, self._ink)
        output = QSaveFile(path)
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            raise ValueError(output.errorString())
        if not flattened.save(output, "PNG"):
            output.cancelWriting()
            raise ValueError("No se pudo guardar el PNG en la ubicación seleccionada.")
        if not output.commit():
            raise ValueError(output.errorString())

    def lift_tool(self):
        self._last_stroke = None

    @property
    def width_px(self) -> int:
        return self._width_px

    @width_px.setter
    def width_px(self, width: int):
        lowest, highest = WIDTH_LIMITS
        self._width_px = min(highest, max(lowest, int(width)))
        self.update()

    def reset_pointer(self):
        self._raw_point = self._cursor = self._last_frame = None
        self.lift_tool()
        self.update()

    @staticmethod
    def _map_coordinate(coordinate: float) -> float:
        return min(1.0, max(0.0, (coordinate - EDGE_MARGIN) / (1 - 2 * EDGE_MARGIN)))

    def track(self, tip: tuple[float, float] | None, gesture: str | None, stroke_id: int,
              captured_at: float):
        if tip is None:
            self.reset_pointer()
            return

        position = tuple(map(self._map_coordinate, tip))
        dt = captured_at - self._last_frame if self._last_frame is not None else 0.0
        if self._raw_point is not None and (
            dt > MAX_FRAME_GAP
            or hypot(*(p - q for p, q in zip(position, self._raw_point))) > MAX_CURSOR_JUMP
        ):
            self.reset_pointer()

        previous = self._cursor
        fraction = 1 - exp(-max(0.0, dt) / CURSOR_TIME_CONSTANT)
        self._cursor = position
        if previous is not None:
            self._cursor = tuple(old + fraction * (new - old)
                                 for old, new in zip(previous, position))
        self._raw_point = position
        self._last_frame = captured_at
        if gesture in INK_GESTURES:
            self.tool = gesture
            new_stroke = stroke_id != self._last_stroke
            if new_stroke:
                self._remember()
            end = self._pixels(self._cursor)
            start = end if new_stroke else self._pixels(previous)
            self._paint_segment(start, end, erase=gesture == ERASE)
            self._last_stroke = stroke_id
        else:
            self.lift_tool()
        self.update()

    @staticmethod
    def _pixels(point: tuple[float, float]) -> QPointF:
        return QPointF(point[0] * (WIDTH - 1), point[1] * (HEIGHT - 1))

    def _paint_segment(self, start: QPointF, end: QPointF, erase: bool):
        with QPainter(self._ink) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            width = self.width_px
            pen = QPen(self.color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            if erase:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            if start == end:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self.color)
                painter.drawEllipse(end, width / 2, width / 2)
            else:
                painter.drawLine(start, end)

    @staticmethod
    def _fit_rect(width, height, image_width, image_height):
        scale = min(width / image_width, height / image_height)
        draw_width, draw_height = image_width * scale, image_height * scale
        return QRectF((width - draw_width) / 2, (height - draw_height) / 2,
                      draw_width, draw_height)

    def paintEvent(self, event):
        rect = self._fit_rect(self.width(), self.height(), WIDTH, HEIGHT)
        with QPainter(self) as painter:
            painter.fillRect(self.rect(), QColor("#0d121b"))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, self._paper)
            painter.drawImage(rect, self._ink)
            if self._cursor is not None:
                point = self._pixels(self._cursor)
                scale = rect.width() / WIDTH
                center = rect.topLeft() + point * scale
                radius = max(4, self.width_px * scale * 0.5)
                painter.setClipRect(rect)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor("#ffffff"), 3))
                painter.drawEllipse(center, radius, radius)
                painter.setPen(QPen(QColor("#087f72"), 1.5))
                painter.drawEllipse(center, radius, radius)
