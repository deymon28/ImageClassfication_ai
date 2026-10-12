"""Grid triage: a page of frame tiles to confirm quickly. LMB marks 'has objects', MMB or Ctrl+LMB marks 'skip',
RMB opens a full-size preview (wheel zoom, drag pan, click toggles the mark). Committing the page saves unmarked
frames as empty, marked ones go to the edit queue, skip marks are saved as skipped."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from framelabel.core.queues import FrameInfo
from framelabel.core.view import View
from .canvas import to_qimage
from .theme import qc

GAP = 4
TILE_RADIUS = 4
STATUS_TAGS = {"neg": "empty", "pos": "labeled", "skip": "skipped", "todo": "marked"}
STATUS_TAG_COLORS = {"neg": "neutral", "pos": "ok", "skip": "skip", "todo": "warn"}
PEEK_HOLD_S = 0.3  # a right button held longer than this is a 'peek': the photo closes when the button is released
RESIZE_SETTLE_MS = 150  # while the window is being resized the tiles of the old size are stretched instead of reloaded


def grid_layout(aw: int, ah: int, cols: int, rows: int, aspect: tuple[int, int]) -> tuple[int, int, int, int]:
    """Largest aspect-correct tiles that fit the area, the grid centred -> (x0, y0, tile w, tile h)."""
    h0, w0 = aspect
    tw = (aw - GAP * (cols + 1)) // cols
    th = round(tw * h0 / w0)
    if th * rows + GAP * (rows + 1) > ah:
        th = (ah - GAP * (rows + 1)) // rows
        tw = round(th * w0 / h0)
    tw, th = max(tw, 16), max(th, 9)
    x0 = (aw - (cols * tw + (cols - 1) * GAP)) // 2
    y0 = (ah - (rows * th + (rows - 1) * GAP)) // 2
    return x0, y0, tw, th


class Grid(QWidget):
    marks_changed = pyqtSignal()
    page_changed = pyqtSignal()
    tile_ready = pyqtSignal()  # a tile page finished loading in the pool (emitted from the pool thread)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.items: list[FrameInfo] = []
        self.cols, self.rows = 3, 3
        self.page = 0
        self.marks: dict[str, str] = {}
        self.ui_threshold = 0.3
        self.show_weak = True
        self.show_ignored = False  # draw proposals hidden by ignore zones (the "zone" view)
        self.aspect = (9, 16)
        self.pool = ThreadPoolExecutor(2)
        self.cache: dict[tuple, Future] = {}
        self.last_tiles: tuple[int, list[QImage]] | None = None  # (page, tiles) shown last, for stretching on resize
        self.resizing = False
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.end_resize)
        self.tile_ready.connect(self.update)
        self.hover = -1
        self.preview: FrameInfo | None = None
        self.preview_img: np.ndarray | None = None
        self.preview_q: QImage | None = None
        self.preview_view: View | None = None
        self.press: QPointF | None = None
        self.peek_since: float | None = None  # when the right button opened the preview; None if it did not
        self.last: QPointF | None = None
        self.title = ""
        self.message = ""  # shown instead of the tiles when there are none (e.g. while a session loads)
        self.class_colors: list[QColor] = [QColor("#ff2828")]

    @property
    def per_page(self) -> int:
        return self.cols * self.rows

    @property
    def sticky(self) -> bool:
        """With one tile per page the tile is the whole photo: it is shown like the preview (zoom, pan) all the time
        instead of needing a right click to open it."""
        return self.per_page == 1

    @property
    def modal_preview(self) -> bool:
        """A preview that was opened on purpose (right click) and has to be closed before anything else."""
        return self.preview is not None and not self.sticky

    @property
    def n_pages(self) -> int:
        return max(1, -(-len(self.items) // self.per_page))

    def page_items(self, page: int | None = None) -> list[FrameInfo]:
        page = self.page if page is None else page
        return self.items[page * self.per_page:(page + 1) * self.per_page]

    def clear_cache(self) -> None:
        for f in self.cache.values():
            f.cancel()  # a no-op for loads already running
        self.cache.clear()
        self.last_tiles = None

    def shutdown(self) -> None:
        self.clear_cache()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def set_items(self, items: list[FrameInfo], page: int = 0, initial_mark=None) -> None:
        self.items = items
        self.clear_cache()
        if items:
            img = cv2.imread(str(items[0].path), cv2.IMREAD_REDUCED_COLOR_4)
            if img is not None:
                self.aspect = img.shape[:2]
        self.initial_mark = initial_mark or (lambda fi: None)
        self.set_page(page)

    def set_page(self, page: int) -> None:
        self.page = min(max(page, 0), self.n_pages - 1)
        self.marks = {fi.stem: m for fi in self.page_items() if (m := self.initial_mark(fi))}
        self.close_preview()
        if self.sticky and self.items:
            self.open_preview(self.page_items()[0])
        self.page_changed.emit()
        self.update()

    def toggle(self, stem: str, kind: str) -> None:
        if self.marks.get(stem) == kind:
            del self.marks[stem]
        else:
            self.marks[stem] = kind
        self.marks_changed.emit()
        self.update()

    # ---------------- tiles
    def _load(self, page: int, tw: int, th: int) -> list[QImage]:
        out = []
        for fi in self.page_items(page):
            img = cv2.imread(str(fi.path), cv2.IMREAD_REDUCED_COLOR_2 if tw <= self.aspect[1] * 2 else cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((th, tw, 3), np.uint8)
            out.append(to_qimage(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)))
        return out

    def _notify(self, _f) -> None:
        try:
            self.tile_ready.emit()
        except RuntimeError:  # the widget was deleted while a load was finishing
            pass

    def tiles(self, tw: int, th: int) -> list[QImage] | None:
        """The current page's tiles at this size, or None while they load (a repaint follows when they are ready)."""
        for key in list(self.cache):
            if abs(key[0] - self.page) > 2 or key[1:] != (tw, th):
                self.cache.pop(key).cancel()
        for p in (self.page, self.page + 1, self.page - 1):  # the current page first
            if 0 <= p < self.n_pages and (p, tw, th) not in self.cache:
                f = self.cache[(p, tw, th)] = self.pool.submit(self._load, p, tw, th)
                f.add_done_callback(self._notify)
        f = self.cache[(self.page, tw, th)]
        if not f.done() or f.cancelled():
            return None
        tiles = f.result()
        self.last_tiles = (self.page, tiles)
        return tiles

    def resizeEvent(self, e) -> None:
        self.resizing = True
        self.resize_timer.start(RESIZE_SETTLE_MS)
        super().resizeEvent(e)

    def end_resize(self) -> None:
        self.resizing = False
        self.update()

    def layout_now(self) -> tuple[int, int, int, int]:
        return grid_layout(self.width(), self.height(), self.cols, self.rows, self.aspect)

    def tile_at(self, x: float, y: float) -> int:
        x0, y0, tw, th = self.layout_now()
        c, r = int((x - x0) // (tw + GAP)), int((y - y0) // (th + GAP))
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return -1
        if (x - x0) - c * (tw + GAP) >= tw or (y - y0) - r * (th + GAP) >= th:
            return -1
        i = r * self.cols + c
        return i if i < len(self.page_items()) else -1

    # ---------------- painting
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), qc("bg0"))
        if self.preview is not None:
            self.paint_preview(p)
            return
        if not self.items:
            f = QFont(self.font())
            f.setPointSizeF(11)
            p.setFont(f)
            p.setPen(qc("text2"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message or "No frames in this view")
            return
        x0, y0, tw, th = self.layout_now()
        if self.resizing:  # stretch the last tiles of this page; new sizes are requested once the resize settles
            tiles = self.last_tiles[1] if self.last_tiles and self.last_tiles[0] == self.page else None
        else:
            tiles = self.tiles(tw, th)
        font = QFont(self.font())
        font.setPointSizeF(8)
        p.setFont(font)
        for i, fi in enumerate(self.page_items()):
            x, y = x0 + (i % self.cols) * (tw + GAP), y0 + (i // self.cols) * (th + GAP)
            rect = QRectF(x, y, tw, th)
            clip = QPainterPath()
            clip.addRoundedRect(rect, TILE_RADIUS, TILE_RADIUS)
            p.save()
            p.setClipPath(clip)
            if tiles is not None:
                p.drawImage(rect, tiles[i])
            else:
                p.fillRect(rect, qc("bg2"))
            if self.show_weak and fi.prop_xyxy is not None:
                for (bx1, by1, bx2, by2), c in zip(fi.prop_xyxy, fi.confs):
                    col = QColor(255, 200, 0, 90 + int(150 * min(c / 0.3, 1.0)))
                    p.setPen(QPen(col, 1, Qt.PenStyle.DotLine))
                    p.drawRect(QRectF(x + bx1 * tw, y + by1 * th, (bx2 - bx1) * tw, (by2 - by1) * th))
            if self.show_ignored and fi.ignored_xyxy is not None:
                p.setPen(QPen(QColor(0, 220, 255), 2, Qt.PenStyle.DotLine))
                for bx1, by1, bx2, by2 in fi.ignored_xyxy:
                    p.drawRect(QRectF(x + bx1 * tw, y + by1 * th, (bx2 - bx1) * tw, (by2 - by1) * th))
                self.pill(p, x + 6, y + 6, f"ignore zone {fi.ignored_conf:.2f}", QColor(0, 0, 0, 170), QColor(0, 220, 255))
            # caption: a dark fade at the bottom with the frame name and its state
            fade = QLinearGradient(0, y + th - 26, 0, y + th)
            fade.setColorAt(0, QColor(0, 0, 0, 0))
            fade.setColorAt(1, QColor(0, 0, 0, 180))
            p.fillRect(QRectF(x, y + th - 26, tw, 26), fade)
            p.restore()
            mark = self.marks.get(fi.stem)
            if mark == "people":
                p.setPen(QPen(qc("accent"), 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1.5, 1.5, -1.5, -1.5), TILE_RADIUS, TILE_RADIUS)
                self.pill(p, x + 6, y + 6, "HAS OBJECTS", qc("accent"), QColor("#ffffff"))
            elif mark == "skip":
                p.save()
                p.setClipPath(clip)
                p.fillRect(rect, QColor(10, 11, 14, 175))
                p.restore()
                p.setPen(QPen(qc("line2"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), TILE_RADIUS, TILE_RADIUS)
                self.pill(p, x + 6, y + 6, "SKIP", qc("bg3"), qc("text2"))
            elif i == self.hover:
                p.setPen(QPen(QColor(255, 255, 255, 150), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), TILE_RADIUS, TILE_RADIUS)
            tag = STATUS_TAGS.get(fi.status or "", "")
            right = f"model {fi.max_conf:.2f}" if fi.max_conf > 0 else ""
            if fi.ui < self.ui_threshold:
                right += "  UI?"
            cap = QRectF(x + 7, y + th - 19, tw - 14, 17)
            p.setPen(QColor(255, 255, 255, 235))
            p.drawText(cap, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, fi.stem)
            if tag:
                off = p.fontMetrics().horizontalAdvance(fi.stem) + 10
                p.setPen(qc(STATUS_TAG_COLORS[fi.status]))
                p.drawText(QRectF(cap.x() + off, cap.y(), cap.width() - off, cap.height()),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, tag)
            p.setPen(QColor(255, 255, 255, 170))
            p.drawText(cap, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, right.strip())

    def pill(self, p: QPainter, x: float, y: float, text: str, bg: QColor, fg: QColor, size: float = 8) -> None:
        """A rounded label ('HAS OBJECTS', 'SKIP') on a tile or on the photo."""
        p.save()
        f = QFont(self.font())
        f.setPointSizeF(size)
        f.setBold(True)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 14
        h = p.fontMetrics().height() + 5
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)
        p.setPen(fg)
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def badge(self, p: QPainter, x: float, y: float, text: str, color: QColor) -> None:
        """The caption of the full-size preview."""
        p.save()
        f = QFont(self.font())
        f.setPointSizeF(10)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 24
        h = p.fontMetrics().height() + 12
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 11, 14, 190))
        p.drawRoundedRect(QRectF(x, y, w, h), 8, 8)
        p.setPen(color)
        p.drawText(QRectF(x + 12, y, w, h), Qt.AlignmentFlag.AlignVCenter, text)
        p.restore()

    def paint_preview(self, p: QPainter) -> None:
        v = self.preview_view
        v.fit(self.width(), self.height())
        x1, y1, x2, y2 = v.visible_rect()
        x1, y1, x2, y2 = int(x1), int(y1), min(v.iw, int(x2) + 1), min(v.ih, int(y2) + 1)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, v.s < 3)
        p.drawImage(QRectF(QPointF(*v.to_widget(x1, y1)), QPointF(*v.to_widget(x2, y2))), self.preview_q,
                    QRectF(x1, y1, x2 - x1, y2 - y1))
        fi = self.preview
        if fi.ignored_xyxy is not None and self.show_ignored:
            p.setPen(QPen(QColor(0, 220, 255), 2, Qt.PenStyle.DotLine))
            for bx1, by1, bx2, by2 in fi.ignored_xyxy:
                p.drawRect(QRectF(QPointF(*v.to_widget(bx1 * v.iw, by1 * v.ih)), QPointF(*v.to_widget(bx2 * v.iw, by2 * v.ih))))
        if fi.prop_xyxy is not None and self.show_weak:
            for (bx1, by1, bx2, by2), c in zip(fi.prop_xyxy, fi.confs):
                p.setPen(QPen(QColor(255, 200, 0, 200), 1, Qt.PenStyle.DotLine))
                p.drawRect(QRectF(QPointF(*v.to_widget(bx1 * v.iw, by1 * v.ih)), QPointF(*v.to_widget(bx2 * v.iw, by2 * v.ih))))
        kind = self.marks.get(fi.stem, "")
        if kind == "people":
            p.setPen(QPen(qc("accent"), 5))
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        elif kind == "skip":
            p.fillRect(self.rect(), QColor(10, 11, 14, 130))
            p.setPen(QPen(qc("line2"), 5))
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        mark = {"people": "HAS OBJECTS", "skip": "SKIP"}.get(kind, "not marked")
        hint = ("click: has objects | Ctrl+click or MMB: skip | wheel: zoom | drag: pan | Z: fit" if self.sticky
                else "click: toggle 'has objects' | RMB / Esc: back")
        self.badge(p, 10, 10, f"{fi.stem}  [{mark}]  zoom {v.zoom:.1f}x   {hint}", qc("text"))

    # ---------------- preview
    def open_preview(self, fi: FrameInfo) -> None:
        img = cv2.imread(str(fi.path))
        if img is None:
            return
        self.preview, self.preview_img, self.preview_q = fi, img, to_qimage(img)
        self.preview_view = View(img.shape[1], img.shape[0])
        self.update()

    def close_preview(self) -> None:
        self.preview = self.preview_img = self.preview_q = self.preview_view = None
        self.press = self.peek_since = None
        self.update()

    # ---------------- mouse
    def mousePressEvent(self, e) -> None:
        pos = e.position()
        self.last = pos
        if self.preview is not None:
            if e.button() == Qt.MouseButton.LeftButton or e.button() == Qt.MouseButton.MiddleButton:
                self.press = pos
            elif e.button() == Qt.MouseButton.RightButton and not self.sticky:
                self.close_preview()
            return
        i = self.tile_at(pos.x(), pos.y())
        if i < 0:
            return
        fi = self.page_items()[i]
        ctrl = e.modifiers() & Qt.KeyboardModifier.ControlModifier
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and ctrl):
            self.toggle(fi.stem, "skip")
        elif e.button() == Qt.MouseButton.LeftButton:
            self.toggle(fi.stem, "people")
        elif e.button() == Qt.MouseButton.RightButton:
            self.open_preview(fi)
            self.peek_since = time.monotonic()  # held: closes on release; a quick click leaves it open (click again to close)

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        if self.preview is not None:
            if self.press is not None and self.last is not None:
                self.preview_view.pan(pos.x() - self.last.x(), pos.y() - self.last.y())
                self.update()
            self.last = pos
            return
        h = self.tile_at(pos.x(), pos.y())
        if h != self.hover:
            self.hover = h
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        pos = e.position()
        if e.button() == Qt.MouseButton.RightButton and self.peek_since is not None:
            held = time.monotonic() - self.peek_since
            self.peek_since = None
            if self.preview is not None and held >= PEEK_HOLD_S:
                self.close_preview()  # a peek: the button was held, so letting go closes the photo
            return
        if self.preview is not None and self.press is not None:
            moved = abs(pos.x() - self.press.x()) + abs(pos.y() - self.press.y())
            self.press = None
            if moved <= 4 and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                skip = e.button() == Qt.MouseButton.MiddleButton or bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.toggle(self.preview.stem, "skip" if skip else "people")
                if not self.sticky:
                    self.close_preview()

    def wheelEvent(self, e) -> None:
        if self.preview is not None:
            pos = e.position()
            self.preview_view.zoom_at(pos.x(), pos.y(), 1.25 if e.angleDelta().y() > 0 else 0.8)
            self.update()

    def leaveEvent(self, _e) -> None:
        self.hover = -1
        self.update()

    def stems_on_page(self) -> list[str]:
        return [fi.stem for fi in self.page_items()]

    def paths_on_page(self) -> list[Path]:
        return [fi.path for fi in self.page_items()]
