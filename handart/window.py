from math import ceil, gcd
from time import monotonic

from .camera import CameraFrame, CameraWorker
from .canvas import Canvas, EDGE_MARGIN, HEIGHT, WIDTH, WIDTH_LIMITS
from .gestures import COLOR, DRAW, ERASE, MAX_FRAME_GAP, RESIZE, PinchSizeController, Recognition
from .tracking import CONNECTIONS, INDEX_TIP
from .gesture_sidebar import GestureSidebar

from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget,
)


ACTIONS = {DRAW: "Dibujar", ERASE: "Borrar", COLOR: "Cambiar color", RESIZE: "Ajustar grosor"}
GESTURE_NAMES = {DRAW: "Índice"}
TOOL_NAMES = {DRAW: "Pincel", ERASE: "Borrador"}
SWATCHES = (
    ("Tinta", "#17253a"), ("Turquesa", "#088c7b"), ("Azul", "#3468d9"),
    ("Coral", "#e05864"), ("Dorado", "#da930d"),
)

STYLE = """
QMainWindow, QWidget#root, QWidget#sidebar, QScrollArea#sidebarScroll { background: #0d121b; }
QWidget { color: #edf2f8; font-family: "Segoe UI"; font-size: 13px; }
QFrame#panel { background: #151d2a; border: 1px solid #273548; border-radius: 12px; }
QLabel#title { font-size: 27px; font-weight: 700; color: #e9fcf8; }
QLabel#section { font-size: 15px; font-weight: 600; }
QLabel[muted="true"], QCheckBox { color: #aebed0; }
QLabel#action { font-size: 17px; font-weight: 600; color: #8be8d5; }
QPushButton { background: #202c3d; border: 1px solid #35475e;
    border-radius: 6px; padding: 5px 9px; min-height: 21px; }
QPushButton:hover { background: #2e4055; }
QPushButton:focus { border-color: #77dcca; }
QPushButton:disabled { color: #8391a0; background: #1b2533; border-color: #283548; }
QPushButton#primary { background: #8be8d5; color: #102c27; font-weight: 700; }
QPushButton#primary:hover { background: #b1f5e7; }
QPushButton#primary:disabled { background: #36534f; color: #b9cfcb; }
QPushButton#swatch { padding: 0; min-width: 22px; min-height: 22px; border-radius: 11px; }
QSlider::groove:horizontal { height: 4px; background: #35475e; border-radius: 2px; }
QSlider::handle:horizontal { background: #8be8d5; width: 13px; margin: -5px 0; border-radius: 6px; }
"""


class CameraPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(155)
        self.setAccessibleName("Vista previa de la cámara")
        self._image = QImage()
        self._points = ()
        self._overlay = True
        self._placeholder = "Cámara detenida"

    def set_frame(self, frame):
        height, width = frame.rgb.shape[:2]
        self._image = QImage(frame.rgb.data, width, height, frame.rgb.strides[0],
                             QImage.Format.Format_RGB888).copy()
        self._points = frame.hand.image_points if frame.hand else ()
        self.update()

    def clear(self, text):
        self._image = QImage()
        self._points = ()
        self._placeholder = text
        self.update()

    @Slot(bool)
    def set_overlay(self, enabled):
        self._overlay = enabled
        self.update()

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.fillRect(self.rect(), QColor("#0d121b"))
            if self._image.isNull():
                painter.setPen(QColor("#aebed0"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
                return
            size = self._image.size()
            size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            rect = QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                          size.width(), size.height())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, self._image)
            if self._overlay:
                self._paint_landmarks(painter, rect)

    def _paint_landmarks(self, painter, rect):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor("#a4fff0"), 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect.adjusted(rect.width() * EDGE_MARGIN, rect.height() * EDGE_MARGIN,
                                       -rect.width() * EDGE_MARGIN, -rect.height() * EDGE_MARGIN))
        points = [QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
                  for x, y in self._points]
        if points:
            painter.setPen(QPen(QColor("#a4fff0"), 1.5))
            for a, b in CONNECTIONS:
                painter.drawLine(points[a], points[b])
            painter.setBrush(QColor("#a4fff0"))
            for point in points:
                painter.drawEllipse(point, 2, 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(points[INDEX_TIP], 5, 5)


class DrawingWorkspace(QWidget):


    SPACING = 16

    def __init__(self, canvas: Canvas, gesture_sidebar: GestureSidebar):
        super().__init__()
        self.canvas = canvas
        self.gesture_sidebar = gesture_sidebar
        canvas.setParent(self)
        gesture_sidebar.setParent(self)
        divisor = gcd(WIDTH, HEIGHT)
        self._aspect_width = WIDTH // divisor
        self._aspect_height = HEIGHT // divisor
        sidebar_width = gesture_sidebar.width()
        minimum_height = max(canvas.minimumHeight(), gesture_sidebar.minimumSizeHint().height())
        units = max(ceil(minimum_height / self._aspect_height),
                    ceil(canvas.minimumWidth() / self._aspect_width))
        self.setMinimumSize(sidebar_width + self.SPACING + units * self._aspect_width,
                            units * self._aspect_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sidebar_width = self.gesture_sidebar.width()
        available_width = self.width() - sidebar_width - self.SPACING

        units = min(available_width // self._aspect_width, self.height() // self._aspect_height)
        paper_width = units * self._aspect_width
        paper_height = units * self._aspect_height
        left = (self.width() - sidebar_width - self.SPACING - paper_width) // 2
        top = (self.height() - paper_height) // 2
        self.gesture_sidebar.setGeometry(left, top, sidebar_width, paper_height)
        self.canvas.setGeometry(left + sidebar_width + self.SPACING, top, paper_width, paper_height)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandArt")
        self.resize(1160, 600)
        self._worker = None
        self._camera_state = "Detenida"
        self._closing = False
        self._editing = False
        self._require_release = False
        self._last_frame_at = None
        self._last_color_activation = None
        self._pinch_size = PinchSizeController(WIDTH_LIMITS)

        self.canvas = Canvas()
        self.preview = CameraPreview()
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(13)
        layout.addLayout(self._toolbar())

        content = QHBoxLayout()
        content.setSpacing(16)
        self.gesture_sidebar = GestureSidebar()
        self.drawing_workspace = DrawingWorkspace(self.canvas, self.gesture_sidebar)
        content.addWidget(self.drawing_workspace, stretch=1)
        sidebar_scroll = QScrollArea(objectName="sidebarScroll")
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setFixedWidth(316)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(self._sidebar())
        content.addWidget(sidebar_scroll)
        layout.addLayout(content, stretch=1)
        self._set_camera_state("Detenida")

        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._poll_camera)
        self._timer.start()

    def _button(self, text, handler, shortcut=None):
        button = QPushButton(text)
        button.clicked.connect(handler)
        if shortcut:
            button.setShortcut(shortcut)
        return button

    @staticmethod
    def _label(text, *, name=None, muted=False):
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        label.setProperty("muted", muted)
        return label

    def _toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._label("HandArt", name="title"))
        row.addStretch()
        row.addWidget(self._button("Abrir", self._open, "Ctrl+O"))
        self._undo_button = self._button("Deshacer", self._undo, "Ctrl+Z")
        self._undo_button.setEnabled(False)
        self.canvas.history_changed.connect(self._undo_button.setEnabled)
        row.addWidget(self._undo_button)
        row.addWidget(self._button("Limpiar", self._clear))
        save = self._button("Guardar PNG", self._save, "Ctrl+S")
        save.setObjectName("primary")
        row.addWidget(save)
        return row

    def _section(self, title):
        panel = QFrame(objectName="panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(13, 9, 13, 9)
        layout.setSpacing(5)
        layout.addWidget(self._label(title, name="section"))
        return panel, layout

    def _sidebar(self):
        sidebar = QWidget(objectName="sidebar")
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self._camera_panel())


        column.addWidget(self._tools_panel())
        column.addStretch()
        return sidebar

    def _camera_panel(self):
        panel, layout = self._section("Cámara")
        layout.addWidget(self.preview)
        self._status = self._label("", muted=True)
        layout.addWidget(self._status)
        settings = QHBoxLayout()
        settings.addStretch()
        overlay = QCheckBox("Landmarks")
        overlay.setChecked(True)
        overlay.toggled.connect(self.preview.set_overlay)
        settings.addWidget(overlay)
        layout.addLayout(settings)

        controls = QHBoxLayout()
        controls.setSpacing(5)
        self._start = self._button("Iniciar", self._start_camera)
        self._start.setObjectName("primary")
        self._pause = self._button("Pausar", lambda: self._stop_camera(paused=True))
        self._stop = self._button("Detener", self._stop_camera)
        for button in (self._start, self._pause, self._stop):
            controls.addWidget(button)
        layout.addLayout(controls)
        return panel

    def _tools_panel(self):
        panel, layout = self._section("Colores")
        row = QHBoxLayout()
        for name, color in SWATCHES:
            swatch = self._button("", lambda checked=False, value=color: self._set_color(QColor(value)))
            swatch.setObjectName("swatch")
            swatch.setAccessibleName(name)
            swatch.setToolTip(name)
            swatch.setStyleSheet(f"background: {color};")
            row.addWidget(swatch)
        layout.addLayout(row)
        row = QHBoxLayout()
        self._tool_label = self._label("", muted=True)
        row.addWidget(self._tool_label)
        self._width_slider = QSlider(Qt.Orientation.Horizontal)
        self._width_slider.valueChanged.connect(self._set_manual_width)
        row.addWidget(self._width_slider, stretch=1)
        self._width_label = QLabel()
        row.addWidget(self._width_label)
        self._sync_width()
        layout.addLayout(row)
        return panel

    def _set_color(self, color):
        self.canvas.lift_tool()
        self.canvas.color = color

    def _advance_color(self):
        colors = [color for _, color in SWATCHES]
        current = self.canvas.color.name()
        next_index = colors.index(current) + 1 if current in colors else 0
        self._set_color(QColor(colors[next_index % len(colors)]))

    def _set_manual_width(self, width):
        self._pinch_size.reset()
        self.canvas.lift_tool()
        self.canvas.width_px = width
        self._width_label.setText(f"{self.canvas.width_px} px")

    def _sync_width(self):
        name = TOOL_NAMES[self.canvas.tool]
        self._tool_label.setText(name)
        self._width_label.setText(f"{self.canvas.width_px} px")
        with QSignalBlocker(self._width_slider):
            self._width_slider.setAccessibleName("Grosor compartido del pincel y borrador")
            self._width_slider.setRange(*WIDTH_LIMITS)
            self._width_slider.setValue(self.canvas.width_px)

    def _resize_with_pinch(self, recognition: Recognition, captured_at: float):
        width = self._pinch_size.update(
            recognition.aperture, captured_at, recognition.activation_id, self.canvas.width_px
        )
        if width != self.canvas.width_px:
            self.canvas.width_px = width
            self._sync_width()

    def _set_camera_state(self, state):
        self._camera_state = state
        if state != "En directo":
            self.gesture_sidebar.set_active(None)
        self._status.setText(state)
        self._start.setText("Reanudar" if state == "Pausada" else "Iniciar")
        self._start.setEnabled(self._worker is None and not self._closing)
        self._pause.setEnabled(state == "En directo")
        self._stop.setEnabled(state in ("Iniciando…", "En directo", "Pausada"))

    @Slot()
    def _start_camera(self):
        if self._worker is not None:
            return
        worker = CameraWorker(0, self)
        worker.failed.connect(self._camera_failed)
        worker.finished.connect(self._camera_finished)
        self._worker = worker
        self.canvas.reset_pointer()
        self._last_frame_at = None
        self._require_release = False
        self._last_color_activation = None
        self._pinch_size.reset()
        self._set_camera_state("Iniciando…")
        self.preview.clear("Iniciando cámara…")

        worker.start()

    @Slot()
    def _stop_camera(self, *, paused=False):
        if self._worker is not None:
            self._worker.requestInterruption()
        self._set_camera_state("Pausada" if paused else "Detenida")
        self.canvas.reset_pointer()
        self._last_frame_at = None
        self._pinch_size.reset()
        if not paused:
            self.preview.clear("Cámara detenida")


    @Slot(str)
    def _camera_failed(self, error):
        if self._camera_state in ("Iniciando…", "En directo"):
            self._stop_camera()
            self.preview.clear("Cámara no disponible")


    @Slot()
    def _camera_finished(self):
        worker = self._worker

        worker.wait()
        self._worker = None
        worker.deleteLater()
        self._set_camera_state(self._camera_state)
        if self._closing:
            self.close()

    @Slot()
    def _poll_camera(self):
        if self._worker is None or self._camera_state not in ("Iniciando…", "En directo"):
            return
        frame = self._worker.take_latest()
        now = monotonic()
        if frame is not None:
            if self._camera_state == "Iniciando…":
                self._set_camera_state("En directo")
            self.preview.set_frame(frame)
            if now - frame.captured_at <= MAX_FRAME_GAP:
                self._last_frame_at = frame.captured_at

                if not self._editing:
                    self._apply_gesture(frame)
                return
        if self._last_frame_at is not None and now - self._last_frame_at > MAX_FRAME_GAP:
            self._last_frame_at = None
            self.gesture_sidebar.set_active(None)
            self._pinch_size.reset()
            self.canvas.reset_pointer()


    def _apply_gesture(self, frame: CameraFrame):
        result = frame.gesture
        label = result.active
        if label is None:
            self._require_release = False
        if self._require_release:

            label = None

        self.gesture_sidebar.set_active(label if frame.hand is not None else None)
        if label == RESIZE:
            self._resize_with_pinch(result, frame.captured_at)
        else:
            self._pinch_size.reset()
        if label == COLOR and result.activation_id != self._last_color_activation:
            self._advance_color()
            self._last_color_activation = result.activation_id

        previous_tool = self.canvas.tool
        self.canvas.track(frame.hand.tip if frame.hand else None,
                          label, result.activation_id, frame.captured_at)
        if self.canvas.tool != previous_tool:
            self._sync_width()


    def _edit_canvas(self, operation):
        self.gesture_sidebar.set_active(None)
        self.canvas.reset_pointer()
        self._editing = self._require_release = True
        try:
            operation()
        except ValueError as exc:
            QMessageBox.warning(self, "HandArt", str(exc))
        finally:
            self._editing = False

    @Slot()
    def _undo(self):
        self._edit_canvas(self.canvas.undo)

    @Slot()
    def _clear(self):
        def clear():
            result = QMessageBox.question(self, "Limpiar lienzo", "¿Limpiar todo el lienzo?")
            if result == QMessageBox.StandardButton.Yes:
                self.canvas.clear()
        self._edit_canvas(clear)

    @Slot()
    def _open(self):
        def load():
            path, _ = QFileDialog.getOpenFileName(
                self, "Abrir imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.webp)"
            )
            if path:
                self.canvas.open_image(path)
        self._edit_canvas(load)

    @Slot()
    def _save(self):
        def save():
            dialog = QFileDialog(self, "Guardar dibujo")
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setNameFilter("PNG (*.png)")
            dialog.setDefaultSuffix("png")
            dialog.selectFile("HandArt.png")
            if dialog.exec():
                self.canvas.save_png(dialog.selectedFiles()[0])
        self._edit_canvas(save)

    def closeEvent(self, event):
        if self._worker is not None:
            self._closing = True
            self._stop_camera()
            event.ignore()
        else:
            super().closeEvent(event)


def run() -> int:
    app = QApplication([])
    app.setApplicationName("HandArt")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
