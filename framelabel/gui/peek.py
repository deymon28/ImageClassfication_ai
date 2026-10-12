"""A small photo that appears beside the frame list when the mouse rests on a row: look at a frame (with its boxes) without
opening it, so a long list can be scanned. It follows the mouse, never takes the focus and closes on the first click,
scroll or when the mouse leaves."""
from __future__ import annotations

from collections import OrderedDict

import cv2
from PyQt6.QtCore import QEvent, QObject, QPoint, QRectF, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QListView

from .canvas import to_qimage
from .theme import qc

DELAY_MS = 350      # the mouse has to rest this long on a row
WIDTH = 380         # photo width in logical pixels
CACHE = 40          # photos kept


class FramePeek(QObject):
    """Shows the frame under the mouse in `view` (rows are mapped to frames by `frame_at(index)`)."""

    def __init__(self, view: QListView, frame_at, grid, parent=None):
        super().__init__(parent)
        self.view, self.frame_at, self.grid = view, frame_at, grid
        self.enabled = True
        self.pix: OrderedDict[tuple, QPixmap] = OrderedDict()
        self.shown_stem: str | None = None
        self.pos = QPoint()
        self.label = QLabel(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.label.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.setStyleSheet("QLabel { background: #0b0c0f; border: 1px solid #3a3f4b; padding: 0; }")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DELAY_MS)
        self.timer.timeout.connect(self.show_now)
        view.viewport().installEventFilter(self)
        view.verticalScrollBar().valueChanged.connect(self.hide)

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self.hide()

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        t = e.type()
        if t == QEvent.Type.MouseMove:
            self.pos = e.globalPosition().toPoint()
            row = self.view.indexAt(e.position().toPoint())
            fi = self.frame_at(row) if row.isValid() else None
            if fi is None:
                self.hide()
            elif fi.stem != self.shown_stem:
                self.hide()
                self.timer.start()
            elif self.label.isVisible():
                self.place(row)
        elif t in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel, QEvent.Type.Hide,
                   QEvent.Type.KeyPress):
            self.hide()
        return False

    def hide(self, *_args) -> None:
        self.timer.stop()
        self.shown_stem = None
        self.label.hide()

    def photo(self, fi) -> QPixmap | None:
        """The frame at the preview size with its boxes and a caption (cached per frame state)."""
        key = (fi.stem, fi.status, round(self.grid.conf, 3), id(fi))
        hit = self.pix.get(key)
        if hit is not None:
            self.pix.move_to_end(key)
            return hit
        img = cv2.imread(str(fi.path), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            return None
        dpr = self.label.devicePixelRatioF() or 1.0
        w = int(WIDTH * dpr)
        h = max(1, int(img.shape[0] * w / img.shape[1]))
        pm = QPixmap.fromImage(to_qimage(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))
        pm.setDevicePixelRatio(dpr)
        lw, lh = w / dpr, h / dpr
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.grid.draw_overlays(p, fi, lambda x1, y1, x2, y2: QRectF(x1 * lw, y1 * lh, (x2 - x1) * lw, (y2 - y1) * lh), 1.2)
        tag = {"pos": "labeled", "neg": "empty", "skip": "skipped", "todo": "marked"}.get(fi.status or "", "new")
        f = QFont(self.label.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.fillRect(QRectF(0, lh - 20, lw, 20), QColor(0, 0, 0, 170))
        p.setPen(QPen(QColor(255, 255, 255, 235)))
        p.drawText(QRectF(8, lh - 20, lw - 16, 20), Qt.AlignmentFlag.AlignVCenter, f"{fi.stem}   {tag}"
                   + (f"   model {fi.max_conf:.2f}" if fi.max_conf > 0 else ""))
        p.end()
        self.pix[key] = pm
        while len(self.pix) > CACHE:
            self.pix.popitem(last=False)
        return pm

    def show_now(self) -> None:
        if not self.enabled:
            return
        local = self.view.viewport().mapFromGlobal(self.pos)
        row = self.view.indexAt(local)
        fi = self.frame_at(row) if row.isValid() else None
        if fi is None or not self.view.viewport().rect().contains(local):
            return
        pm = self.photo(fi)
        if pm is None:
            return
        self.shown_stem = fi.stem
        self.label.setPixmap(pm)
        self.label.adjustSize()
        self.place(row)
        self.label.show()

    def place(self, row) -> None:
        """Beside the list (to its right) at the height of the row, kept on the screen."""
        r = self.view.visualRect(row)
        anchor = self.view.viewport().mapToGlobal(QPoint(self.view.viewport().width() + 10, r.center().y()))
        screen = QApplication.screenAt(anchor) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(anchor.x(), area.right() - self.label.width())
        y = min(max(anchor.y() - self.label.height() // 2, area.top()), area.bottom() - self.label.height())
        self.label.move(x, y)
