from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from .gestures import COLOR, DRAW, ERASE, RESIZE


ICONS = Path(__file__).resolve().parent.parent / "assets" / "iconos"
GESTURES = (
    (DRAW, "indice.svg", "Índice", "Dibujar"),
    (ERASE, "palma.svg", "Palma", "Borrar"),
    (COLOR, "2dedos.svg", "2 dedos", "Cambiar color"),
    (RESIZE, "pinza.svg", "Pinza", "Ajustar grosor"),
)
STYLE = """
QFrame#gestureSidebar { background: #151d2a; border: 1px solid #273548; border-radius: 12px; }
QFrame#gestureCell { background: #e1e8ee; border: 2px solid #e1e8ee; border-radius: 9px; }
QFrame#gestureCell[active="true"] { background: #8be8d5; border-color: #37bba0; }
QFrame#gestureCell QLabel { background: transparent; border: none; color: #203448; }
QFrame#gestureCell QLabel#gestureName { font-weight: 700; }
QFrame#gestureCell QSvgWidget { background: transparent; border: none; }
"""


class GestureSidebar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gestureSidebar")
        self.setAccessibleName("Gestos de la mano")
        self.setFixedWidth(120)
        self.setStyleSheet(STYLE)
        self.active_gesture = None
        self.cells = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(8)
        for gesture, filename, name, action in GESTURES:
            cell = QFrame(objectName="gestureCell")
            cell.setProperty("active", False)
            cell.setAccessibleName(f"{name} · {action}")
            cell.setToolTip(f"{name} · {action}")
            layout = QVBoxLayout(cell)
            layout.setContentsMargins(4, 6, 4, 6)
            layout.setSpacing(3)
            icon = QSvgWidget(str(ICONS / filename))
            if not icon.renderer().isValid():
                raise ValueError(f"No se pudo cargar el SVG de {name}: {ICONS / filename}")
            icon.setFixedSize(46, 54)
            layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignHCenter)
            for text, object_name in ((name, "gestureName"), (action, "gestureAction")):
                label = QLabel(text, objectName=object_name)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
            self.cells[gesture] = cell
            column.addWidget(cell)
        column.addStretch()

    def set_active(self, gesture: str | None):
        gesture = gesture if gesture in self.cells else None
        if gesture == self.active_gesture:
            return
        self.active_gesture = gesture
        for label, cell in self.cells.items():
            active = label == gesture
            if cell.property("active") != active:
                cell.setProperty("active", active)
                cell.style().unpolish(cell)
                cell.style().polish(cell)
                cell.update()
