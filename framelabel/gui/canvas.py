"""The box editor canvas: shows one frame with its boxes; zoom at the cursor, pan, draw / move / resize / delete boxes,
model proposals (dashed, with confidence), weak proposals (faint, click to accept), crosshair and loupe."""
from __future__ import annotations

import copy

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from framelabel.core.boxes import Box, box_at, corner_at, edge_at
from framelabel.core.history import History
from framelabel.core.view import View

from .theme import qc


LOUPE_SIDE = 200          # loupe size on screen, px
LOUPE_MIN, LOUPE_MAX = 1.5, 10.0
LOUPE_MAX_SCALE = 40.0    # at most this many screen px per image px


def loupe_window(ix: float, iy: float, view_scale: float, side: float, zoom: float,
                 max_scale: float = LOUPE_MAX_SCALE) -> tuple[float, float, float, float, float]:
    """Image window (x1, y1, x2, y2) shown in a loupe of `side` screen px centred on image point (ix, iy), and its
    scale in screen px per image px: `zoom` times the view's current scale, so the loupe always magnifies what is on
    screen by the same factor and follows the view zoom smoothly (capped at max_scale)."""
    k = min(max(view_scale * zoom, 1e-6), max_scale)
    half = side / 2 / k
    return ix - half, iy - half, ix + half, iy + half, k


def to_qimage(img: np.ndarray) -> QImage:
    """BGR (or gray) array -> QImage in a 32-bit format, which Qt paints without converting on every repaint
    (BGRA memory order is Format_RGB32 on little-endian machines)."""
    bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA if img.ndim == 2 else cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    return QImage(bgra.data, w, h, 4 * w, QImage.Format.Format_RGB32).copy()


class Canvas(QWidget):
    changed = pyqtSignal()               # boxes edited
    selection_changed = pyqtSignal(int)
    hover_info = pyqtSignal(str)
    sam_requested = pyqtSignal(object)   # Box (rough box) or tuple (x, y) point
    loupe_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 200)
        self.img: np.ndarray | None = None
        self.qimg: QImage | None = None
        self._small: tuple[float, QImage] | None = None
        self.view = View(1, 1)
        self.boxes: list[Box] = []
        self.weak: list[Box] = []             # proposals below the shown confidence (faint)
        self.history = History()
        self.dirty = False
        self.sel = -1
        self.hover = -1
        self.cls = 0
        self.class_names: list[str] = ["object"]
        self.colors: list[QColor] = [QColor("#ff2828")]
        self.show_boxes = True
        self.show_weak = False
        self.loupe = True
        self.loupe_zoom = 3.0  # loupe magnification relative to the current view
        self.sam_on_draw = False
        self.mouse = QPointF(-1, -1)
        self.drag: dict | None = None
        self.pan_from: QPointF | None = None
        self.message = ""

    # ---------------- data
    def set_classes(self, names: list[str], colors: list[str]) -> None:
        self.class_names, self.colors = list(names), [QColor(c) for c in colors]
        self.cls = min(self.cls, len(names) - 1)
        self.update()

    def set_frame(self, img: np.ndarray, boxes: list[Box], weak: list[Box] | None = None, keep_view: bool = False,
                  qimg: QImage | None = None) -> None:
        # `img` is replaced, never modified in place: the SAM thread may still be reading the previous array
        same_size = self.img is not None and self.img.shape[:2] == img.shape[:2]
        self.img, self.qimg, self._small = img, qimg if qimg is not None else to_qimage(img), None
        if not (keep_view and same_size):
            self.view = View(img.shape[1], img.shape[0])
        self.view.fit(self.width(), self.height())
        self.boxes, self.weak = boxes, weak or []
        self.history.clear()
        self.sel, self.hover, self.drag, self.dirty = -1, -1, None, False
        self.update()

    @property
    def iw(self) -> int:
        return self.img.shape[1] if self.img is not None else 1

    @property
    def ih(self) -> int:
        return self.img.shape[0] if self.img is not None else 1

    def snapshot(self) -> None:
        self.history.push(self.boxes)
        self.dirty = True

    def edited(self) -> None:
        self.dirty = True
        self.changed.emit()
        self.update()

    def replace_boxes(self, boxes: list[Box]) -> None:
        self.snapshot()
        self.boxes = boxes
        self.sel = -1
        self.edited()

    def undo(self) -> bool:
        prev = self.history.undo(self.boxes)
        if prev is None:
            return False
        self.boxes, self.sel = prev, -1
        self.edited()
        return True

    def redo(self) -> bool:
        nxt = self.history.redo(self.boxes)
        if nxt is None:
            return False
        self.boxes, self.sel = nxt, -1
        self.edited()
        return True

    def delete_selected(self) -> None:
        if 0 <= self.sel < len(self.boxes):
            self.snapshot()
            del self.boxes[self.sel]
            self.sel = -1
            self.selection_changed.emit(-1)
            self.edited()

    def set_class_of(self, idx: int, cls: int) -> None:
        if 0 <= idx < len(self.boxes) and (self.boxes[idx].cls != cls or self.boxes[idx].conf is not None):
            self.snapshot()
            self.boxes[idx].cls, self.boxes[idx].conf = cls, None
            self.edited()

    def accept_all(self) -> None:
        """Mark every proposal as reviewed (drops the confidence tag)."""
        if any(b.conf is not None for b in self.boxes):
            self.snapshot()
            for b in self.boxes:
                b.conf = None
            self.edited()

    def add_box(self, b: Box, select: bool = True) -> None:
        self.snapshot()
        self.boxes.append(b)
        if select:
            self.sel = len(self.boxes) - 1
            self.selection_changed.emit(self.sel)
        self.edited()

    def select(self, idx: int, center: bool = False) -> None:
        self.sel = idx if 0 <= idx < len(self.boxes) else -1
        if center and self.sel >= 0:
            b = self.boxes[self.sel].norm()
            self.view.zoom_to(b.x1, b.y1, b.x2, b.y2, margin=4.0)
        self.selection_changed.emit(self.sel)
        self.update()

    def target_box(self) -> int:
        """The selected box, else the one under the cursor."""
        if self.sel >= 0:
            return self.sel
        ix, iy = self.view.to_img(self.mouse.x(), self.mouse.y())
        return box_at(self.boxes, ix, iy)

    # ---------------- geometry helpers
    def img_pt(self, p: QPointF, clamp: bool = True) -> tuple[float, float]:
        x, y = self.view.to_img(p.x(), p.y())
        if clamp:
            x, y = min(max(x, 0.0), float(self.iw)), min(max(y, 0.0), float(self.ih))
        return x, y

    def on_image(self, p: QPointF) -> bool:
        x, y = self.view.to_img(p.x(), p.y())
        return self.img is not None and 0 <= x <= self.iw and 0 <= y <= self.ih

    def wpt(self, x: float, y: float) -> QPointF:
        return QPointF(*self.view.to_widget(x, y))

    def wrect(self, b: Box) -> QRectF:
        b = b.norm()
        return QRectF(self.wpt(b.x1, b.y1), self.wpt(b.x2, b.y2))

    # ---------------- painting
    def resizeEvent(self, e) -> None:
        self.view.fit(self.width(), self.height())
        super().resizeEvent(e)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), qc("bg0"))
        if self.img is None:
            f = QFont(self.font())
            f.setPointSizeF(11)
            p.setFont(f)
            p.setPen(qc("text2"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message or "Open a project or a folder")
            return
        v = self.view
        v.fit(self.width(), self.height())
        s = v.s
        if s < 1:  # area-averaged downscale (cached): QPainter's smoothing aliases thin details when shrinking
            if self._small is None or abs(self._small[0] - s) > 1e-6:
                size = (max(1, round(self.iw * s)), max(1, round(self.ih * s)))
                self._small = (s, to_qimage(cv2.resize(self.img, size, interpolation=cv2.INTER_AREA)))
            x0, y0 = v.to_widget(0, 0)
            p.drawImage(QRectF(x0, y0, self.iw * s, self.ih * s), self._small[1],
                        QRectF(0, 0, self._small[1].width(), self._small[1].height()))
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 3)
            x1, y1, x2, y2 = v.visible_rect()
            x1, y1, x2, y2 = int(x1), int(y1), min(self.iw, int(x2) + 1), min(self.ih, int(y2) + 1)
            p.drawImage(QRectF(self.wpt(x1, y1), self.wpt(x2, y2)), self.qimg, QRectF(x1, y1, x2 - x1, y2 - y1))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        font = QFont(self.font())
        font.setPointSizeF(9)
        p.setFont(font)
        if self.show_weak:
            for b in self.weak:
                c = QColor(self.color(b.cls))
                c.setAlpha(110)
                p.setPen(QPen(c, 1, Qt.PenStyle.DotLine))
                p.drawRect(self.wrect(b))
        boxes = list(self.boxes)
        if self.drag and self.drag["kind"] == "new":
            boxes.append(self.drag["box"])
        if self.show_boxes:
            for i, b in enumerate(boxes):
                r = self.wrect(b)
                color = self.color(b.cls)
                width = 3 if i == self.sel else 2 if i == self.hover else 1.5
                pen = QPen(color, width, Qt.PenStyle.DashLine if b.conf is not None else Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.drawRect(r)
                label = self.class_names[b.cls] if 0 <= b.cls < len(self.class_names) else str(b.cls)
                if b.conf is not None:
                    label += f" {b.conf:.2f}"
                tw = p.fontMetrics().horizontalAdvance(label) + 6
                th = p.fontMetrics().height()
                ty = r.top() - th if r.top() - th > 0 else r.bottom()
                bg = QColor(color)
                bg.setAlpha(170)
                p.fillRect(QRectF(r.left(), ty, tw, th), bg)
                p.setPen(QColor(0, 0, 0))
                p.drawText(QRectF(r.left() + 3, ty, tw, th), Qt.AlignmentFlag.AlignVCenter, label)
                if i == self.sel:
                    p.setPen(QPen(QColor(0, 255, 255), 1))
                    for cx, cy in b.norm().corners():
                        q = self.wpt(cx, cy)
                        p.drawRect(QRectF(q.x() - 4, q.y() - 4, 8, 8))
        if self.on_image(self.mouse) and self.pan_from is None:
            mx, my = self.mouse.x(), self.mouse.y()
            p.setPen(QPen(QColor(150, 150, 150, 160), 1))
            p.drawLine(QPointF(mx, 0), QPointF(mx, self.height()))
            p.drawLine(QPointF(0, my), QPointF(self.width(), my))
            if self.loupe:
                self.paint_loupe(p, mx, my)
        if self.message:
            p.setPen(QColor(255, 220, 120))
            p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 22), Qt.AlignmentFlag.AlignLeft, self.message)

    def paint_loupe(self, p: QPainter, mx: float, my: float) -> None:
        ix, iy = self.view.to_img(mx, my)
        side = LOUPE_SIDE
        x1, y1, x2, y2, k = loupe_window(ix, iy, self.view.s, side, self.loupe_zoom)
        ox = 10 if mx > self.width() / 2 else self.width() - side - 10
        target = QRectF(ox, 10, side, side)
        p.fillRect(target, QColor(0, 0, 0))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, k < 3)
        # the part of the window inside the image, so the loupe stays exact at the image borders
        cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(self.iw)), min(y2, float(self.ih))
        if cx2 > cx1 and cy2 > cy1:
            p.drawImage(QRectF(ox + (cx1 - x1) * k, 10 + (cy1 - y1) * k, (cx2 - cx1) * k, (cy2 - cy1) * k), self.qimg,
                        QRectF(cx1, cy1, cx2 - cx1, cy2 - cy1))
        p.save()
        p.setClipRect(target)
        for b in self.boxes:
            bb = b.norm()
            p.setPen(QPen(self.color(b.cls), 1))
            p.drawRect(QRectF(ox + (bb.x1 - x1) * k, 10 + (bb.y1 - y1) * k, bb.w * k, bb.h * k))
        cx, cy = ox + (ix - x1) * k, 10 + (iy - y1) * k
        p.setPen(QPen(QColor(0, 255, 255), 1))
        p.drawLine(QPointF(cx, 10), QPointF(cx, cy - 6))
        p.drawLine(QPointF(cx, cy + 6), QPointF(cx, 10 + side))
        p.drawLine(QPointF(ox, cy), QPointF(cx - 6, cy))
        p.drawLine(QPointF(cx + 6, cy), QPointF(ox + side, cy))
        p.drawText(QRectF(ox + 4, 10 + side - 18, side - 8, 16), Qt.AlignmentFlag.AlignRight,
                   f"{self.loupe_zoom:.1f}x  (Ctrl+wheel, +/-)")
        p.restore()
        p.setPen(QPen(QColor(0, 255, 255), 1))
        p.drawRect(target)

    def change_loupe_zoom(self, factor: float) -> None:
        self.loupe_zoom = min(max(self.loupe_zoom * factor, LOUPE_MIN), LOUPE_MAX)
        self.loupe_changed.emit(self.loupe_zoom)
        self.update()

    def color(self, cls: int) -> QColor:
        return self.colors[cls] if 0 <= cls < len(self.colors) else QColor(255, 255, 255)

    # ---------------- mouse
    def wheelEvent(self, e) -> None:
        if self.img is None:
            return
        up = e.angleDelta().y() > 0
        if self.loupe and e.modifiers() & Qt.KeyboardModifier.ControlModifier:  # loupe magnification
            self.change_loupe_zoom(1.25 if up else 0.8)
            return
        pos = e.position()
        self.view.zoom_at(pos.x(), pos.y(), 1.25 if up else 0.8)
        self.update()

    def mousePressEvent(self, e) -> None:
        if self.img is None:
            return
        pos = e.position()
        self.mouse = pos
        btn = e.button()
        if btn == Qt.MouseButton.MiddleButton:
            self.pan_from = pos
            return
        ix, iy = self.img_pt(pos)
        if btn == Qt.MouseButton.RightButton:
            if not self.drag and (i := box_at(self.boxes, ix, iy)) >= 0:
                self.snapshot()
                del self.boxes[i]
                self.sel = -1
                self.selection_changed.emit(-1)
                self.edited()
            return
        if btn != Qt.MouseButton.LeftButton or not self.on_image(pos):
            return
        if e.modifiers() & Qt.KeyboardModifier.AltModifier:  # SAM from a click
            self.sam_requested.emit((ix, iy))
            return
        tol = 7 / self.view.s
        if self.sel >= 0:
            b = self.boxes[self.sel].norm()
            corner = corner_at(b, ix, iy, tol)
            edge = "" if corner >= 0 else edge_at(b, ix, iy, tol)
            if corner >= 0 or edge:
                self.snapshot()
                self.boxes[self.sel] = b
                self.drag = {"kind": "resize", "corner": corner, "edge": edge, "moved": False}
                return
        i = box_at(self.boxes, ix, iy)
        if i >= 0:
            was_dirty = self.dirty
            self.sel = i
            self.selection_changed.emit(i)
            self.snapshot()
            b = self.boxes[i].norm()
            self.drag = {"kind": "move", "start": (ix, iy, b.x1, b.y1, b.x2, b.y2), "moved": False, "was_dirty": was_dirty}
            return
        if self.show_weak and (j := box_at(self.weak, ix, iy)) >= 0:  # accept a weak proposal
            w = self.weak.pop(j)
            self.add_box(Box(*w.xyxy(), w.cls, None))
            return
        self.sel = -1
        self.selection_changed.emit(-1)
        self.drag = {"kind": "new", "start": (ix, iy), "box": Box(ix, iy, ix, iy, self.cls)}
        self.update()

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        prev, self.mouse = self.mouse, pos
        if self.img is None:
            return
        if self.pan_from is not None:
            self.view.pan(pos.x() - prev.x(), pos.y() - prev.y())
            self.update()
            return
        ix, iy = self.img_pt(pos)
        d = self.drag
        if d is None:
            h = box_at(self.boxes, ix, iy)
            if h != self.hover:
                self.hover = h
            self.cursor_for(ix, iy)
            self.hover_info.emit(f"x {ix:.0f}  y {iy:.0f}")
            self.update()
            return
        if d["kind"] == "new":
            sx, sy = d["start"]
            d["box"] = Box(sx, sy, ix, iy, self.cls)
            b = d["box"].norm()
            self.hover_info.emit(f"new box {b.w:.0f} x {b.h:.0f} px")
        elif d["kind"] == "move":
            sx, sy, x1, y1, x2, y2 = d["start"]
            dx = min(max(ix - sx, -x1), self.iw - x2)
            dy = min(max(iy - sy, -y1), self.ih - y2)
            b = self.boxes[self.sel]
            b.x1, b.y1, b.x2, b.y2, b.conf = x1 + dx, y1 + dy, x2 + dx, y2 + dy, None
            d["moved"] = d["moved"] or abs(dx) + abs(dy) > 0
        else:
            b = self.boxes[self.sel]
            c, edge = d["corner"], d["edge"]
            if c in (0, 3) or edge == "l":
                b.x1 = ix
            if c in (1, 2) or edge == "r":
                b.x2 = ix
            if c in (0, 1) or edge == "t":
                b.y1 = iy
            if c in (2, 3) or edge == "b":
                b.y2 = iy
            b.conf = None
            d["moved"] = True
            nb = b.norm()
            self.hover_info.emit(f"box {nb.w:.0f} x {nb.h:.0f} px")
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan_from = None
            self.update()
            return
        d, self.drag = self.drag, None
        if d is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if d["kind"] == "new":
            b = d["box"].norm()
            if b.w >= 2 and b.h >= 2:
                self.add_box(b)
                if self.sam_on_draw:
                    self.sam_requested.emit(b)
            self.update()
            return
        self.boxes[self.sel] = self.boxes[self.sel].norm()
        if not d["moved"]:  # a plain click: selection only, not an edit
            self.history.drop_last()
            self.dirty = d.get("was_dirty", self.dirty)
            self.update()
            return
        self.edited()

    def leaveEvent(self, _e) -> None:
        self.mouse = QPointF(-1, -1)
        self.update()

    def cursor_for(self, ix: float, iy: float) -> None:
        shape = Qt.CursorShape.CrossCursor
        if self.sel >= 0:
            b = self.boxes[self.sel].norm()
            tol = 7 / self.view.s
            c = corner_at(b, ix, iy, tol)
            if c in (0, 2):
                shape = Qt.CursorShape.SizeFDiagCursor
            elif c in (1, 3):
                shape = Qt.CursorShape.SizeBDiagCursor
            else:
                edge = edge_at(b, ix, iy, tol)
                if edge in ("l", "r"):
                    shape = Qt.CursorShape.SizeHorCursor
                elif edge in ("t", "b"):
                    shape = Qt.CursorShape.SizeVerCursor
        if shape == Qt.CursorShape.CrossCursor and box_at(self.boxes, ix, iy) >= 0:
            shape = Qt.CursorShape.SizeAllCursor
        self.setCursor(shape)

    def boxes_copy(self) -> list[Box]:
        return copy.deepcopy(self.boxes)
