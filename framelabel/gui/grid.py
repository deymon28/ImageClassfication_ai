"""A page of frame tiles. LMB marks 'has objects', MMB or Ctrl+LMB marks 'skip', double-click opens the frame in the
editor, RMB peeks at the photo (wheel zoom, drag pan). Committing the page saves unmarked frames as empty, marked ones
go to the label stage, skip marks are saved as skipped; labeled frames are locked (they only open in the editor).

Tiles show the model's proposals (dotted when weak, dashed when the labels do not cover them) and, for labeled frames,
their boxes in the class colours, so the same grid serves triage, review and browsing."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from framelabel.core.queues import FrameInfo, unmatched_proposals
from framelabel.core.view import View
from framelabel.core.images import imread
from .canvas import to_qimage
from .theme import qc

GAP = 4
TILE_RADIUS = 4
STATUS_TAGS = {"neg": "empty", "pos": "labeled", "skip": "skipped", "todo": "marked"}
STATUS_TAG_COLORS = {"neg": "neutral", "pos": "ok", "skip": "skip", "todo": "warn"}
STORED_MARK = {"pos": "people", "todo": "people", "skip": "skip"}  # the mark a decision already stands for
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
    open_requested = pyqtSignal(str)  # double-click: open this frame (stem) in the editor
    locked = pyqtSignal(str)          # a click on a labeled frame (stem): it cannot be changed from the grid

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
        self.class_names: list[str] = []
        self.conf = 0.3            # the shown confidence: proposals from it up are drawn as proposals, below it as weak
        self.by_stem: dict[str, FrameInfo] = {}
        self.overlay_cache: dict[tuple, tuple] = {}
        self.aspect = (9, 16)
        self.pool = ThreadPoolExecutor(2)
        self.cache: dict[tuple, Future] = {}
        self.last_tiles: tuple[int, list[QImage]] | None = None  # (page, tiles) shown last, for stretching on resize
        self.resizing = False
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.end_resize)
        self.tile_ready.connect(self.update)
        self.hover = -1             # the tile under the mouse, or the one the arrow keys moved to
        self.kbd = False            # the keys moved it last (then it is drawn as a cursor and a new page starts on tile 0)
        self.return_mode = False    # the Checked view: a tick means 'return this frame to Review', labeled tiles can be ticked
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
        self.by_stem = {fi.stem: fi for fi in items}
        self.overlay_cache.clear()
        self.clear_cache()
        if items:
            img = imread(str(items[0].path), cv2.IMREAD_REDUCED_COLOR_4)
            if img is not None:
                self.aspect = img.shape[:2]
        self.initial_mark = initial_mark or (lambda fi: None)
        self.marks = {}  # a new list of frames starts without ticks
        self.set_page(page)

    def set_page(self, page: int) -> None:
        self.page = min(max(page, 0), self.n_pages - 1)
        for fi in self.page_items():  # ticks not saved yet survive paging; a page seen for the first time starts from its status
            if fi.stem not in self.marks and (m := self.initial_mark(fi)):
                self.marks[fi.stem] = m
        self.hover = (0 if self.kbd else self.hover) if self.page_items() else -1
        self.hover = min(self.hover, len(self.page_items()) - 1)
        self.close_preview()
        if self.sticky and self.items:
            self.open_preview(self.page_items()[0])
        self.page_changed.emit()
        self.update()

    def toggle(self, stem: str, kind: str) -> bool:
        """Tick / untick a mark. A labeled frame cannot be changed from the grid (a stray click must not cost its
        boxes): that says so through `locked` and changes nothing. Returns whether the mark changed."""
        fi = self.by_stem.get(stem)
        if self.return_mode:
            if kind != "people":
                return False
        elif fi is not None and fi.status == "pos":
            self.locked.emit(stem)
            return False
        if self.marks.get(stem) == kind:
            del self.marks[stem]
        else:
            self.marks[stem] = kind
        self.marks_changed.emit()
        self.update()
        return True

    def pending(self, fi: FrameInfo) -> str | None:
        """The mark that is not saved yet: what the tile shows as a change ('people', 'skip', 'empty'), None when the
        tile just shows the frame's decision."""
        mark, stored = self.marks.get(fi.stem), STORED_MARK.get(fi.status)
        if self.return_mode:
            return "people" if mark == "people" else None
        if mark == stored:
            return None
        return mark or "empty"

    def tooltip_for(self, fi: FrameInfo) -> str:
        """What a tile says when the mouse rests on it: the decision, the boxes, what the model saw."""
        lines = [f"{fi.stem}  -  {STATUS_TAGS.get(fi.status or '', 'new')}"]
        if fi.status == "pos" and fi.label_cls:
            names = {}
            for c in fi.label_cls:
                name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
                names[name] = names.get(name, 0) + 1
            lines.append("boxes: " + ", ".join(f"{n} x{k}" if k > 1 else n for n, k in names.items()))
        if fi.confs:
            lines.append(f"model: {len(fi.confs)} proposal{'s' if len(fi.confs) != 1 else ''}, best {fi.max_conf:.2f}")
        else:
            lines.append("model: no proposals" + (f" (hidden by an ignore zone: {fi.ignored_conf:.2f})" if fi.ignored_conf else ""))
        if fi.ui < self.ui_threshold:
            lines.append("looks like an interface screen (menu, map, inventory)")
        return "\n".join(lines)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip and self.preview is None:
            i = self.tile_at(e.pos().x(), e.pos().y())
            if i >= 0:
                QToolTip.showText(e.globalPos(), self.tooltip_for(self.page_items()[i]), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    def hovered_stem(self) -> str | None:
        """The frame under the mouse, else the first one of the page (what a key like O acts on)."""
        items = self.page_items()
        if 0 <= self.hover < len(items):
            return items[self.hover].stem
        return items[0].stem if items else None

    @property
    def people_text(self) -> str:
        return "RETURN" if self.return_mode else "HAS OBJECTS"

    def move_cursor(self, dx: int, dy: int) -> bool:
        """Move the arrow-key cursor by a tile (dx) or a row (dy) within the page; the first press only shows it on the
        first tile. Returns whether it moved."""
        n = len(self.page_items())
        if n == 0:
            return False
        if self.hover < 0:
            self.hover, self.kbd = 0, True
            self.update()
            return True
        self.kbd = True
        self.update()
        i, j = self.hover, self.hover + dx + dy * self.cols
        if dy > 0 and j >= n and i // self.cols < (n - 1) // self.cols:
            j = n - 1  # the row below is shorter: its last tile
        if not 0 <= j < n or (dx and j // self.cols != i // self.cols):
            return False
        self.hover = j
        self.update()
        return True

    def mark_at_cursor(self, kind: str, start: bool = True) -> bool:
        """Tick (or untick) the tile under the cursor / mouse like a click would. Without a tile under either, `start`
        begins at the first tile; False when nothing was ticked."""
        items = self.page_items()
        if not 0 <= self.hover < len(items):
            if not items or not start:
                return False
            self.hover, self.kbd = 0, True
        return self.toggle(items[self.hover].stem, kind)

    # ---------------- what is drawn on a frame
    def overlays(self, fi: FrameInfo) -> tuple[list, list, list, list]:
        """(labels, proposals, weak, hidden) of a frame as normalized boxes: labels (xyxy, class) of a labeled frame,
        the proposals at the shown confidence the labels do not cover, the weak ones (new frames only), and the ones
        an ignore zone hides."""
        key = (fi.stem, round(self.conf, 3))
        hit = self.overlay_cache.get(key)
        if hit is None or hit[0] is not fi:  # a saved frame is a new FrameInfo: its boxes are read again
            labels = ([(tuple(b), c) for b, c in zip(fi.label_xyxy, fi.label_cls)]
                      if fi.status == "pos" and fi.label_xyxy is not None else [])
            strong = [tuple(b) for b in unmatched_proposals(fi, self.conf)]
            undecided = fi.status in (None, "todo")
            weak = ([(tuple(b), c) for b, c in zip(fi.prop_xyxy, fi.confs) if c < self.conf]
                    if undecided and fi.prop_xyxy is not None else [])
            hidden = [tuple(b) for b in fi.ignored_xyxy] if fi.ignored_xyxy is not None else []
            hit = self.overlay_cache[key] = (fi, labels, strong, weak, hidden)
        return hit[1:]

    def draw_overlays(self, p: QPainter, fi: FrameInfo, rect_of, width: float = 1.0) -> None:
        """Paint the boxes of a frame; `rect_of(x1, y1, x2, y2)` maps normalized coordinates to the widget."""
        labels, strong, weak, hidden = self.overlays(fi)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.show_weak:
            for (x1, y1, x2, y2), c in weak:
                p.setPen(QPen(QColor(255, 200, 0, 90 + int(150 * min(c / 0.3, 1.0))), width, Qt.PenStyle.DotLine))
                p.drawRect(rect_of(x1, y1, x2, y2))
        p.setPen(QPen(qc("warn"), width * 1.4, Qt.PenStyle.DashLine))
        for x1, y1, x2, y2 in strong:
            p.drawRect(rect_of(x1, y1, x2, y2))
        for (x1, y1, x2, y2), cls in labels:
            col = self.class_colors[cls] if 0 <= cls < len(self.class_colors) else QColor("#ffffff")
            p.setPen(QPen(col, width * 1.6))
            p.drawRect(rect_of(x1, y1, x2, y2))
        if self.show_ignored and hidden:
            p.setPen(QPen(QColor(0, 220, 255), width * 2, Qt.PenStyle.DotLine))
            for x1, y1, x2, y2 in hidden:
                p.drawRect(rect_of(x1, y1, x2, y2))

    # ---------------- tiles
    def _load(self, page: int, tw: int, th: int) -> list[QImage]:
        out = []
        for fi in self.page_items(page):
            img = imread(str(fi.path), cv2.IMREAD_REDUCED_COLOR_2 if tw <= self.aspect[1] * 2 else cv2.IMREAD_COLOR)
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
            self.draw_overlays(p, fi, lambda x1, y1, x2, y2, x=x, y=y, tw=tw, th=th:
                               QRectF(x + x1 * tw, y + y1 * th, (x2 - x1) * tw, (y2 - y1) * th))
            if self.show_ignored and fi.ignored_xyxy is not None:
                self.pill(p, x + 6, y + 6, f"ignore zone {fi.ignored_conf:.2f}", QColor(0, 0, 0, 170), QColor(0, 220, 255))
            if fi.status == "skip" and self.pending(fi) is None:
                p.fillRect(rect, QColor(10, 11, 14, 120))  # a skipped frame is faded: it is out of the dataset
            # caption: a dark fade at the bottom with the frame name and its state
            fade = QLinearGradient(0, y + th - 26, 0, y + th)
            fade.setColorAt(0, QColor(0, 0, 0, 0))
            fade.setColorAt(1, QColor(0, 0, 0, 180))
            p.fillRect(QRectF(x, y + th - 26, tw, 26), fade)
            p.restore()
            change = self.pending(fi)
            if change == "people":
                p.setPen(QPen(qc("accent"), 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1.5, 1.5, -1.5, -1.5), TILE_RADIUS, TILE_RADIUS)
                self.pill(p, x + 6, y + 6, self.people_text, qc("accent"), QColor("#ffffff"))
            elif change == "skip":
                p.save()
                p.setClipPath(clip)
                p.fillRect(rect, QColor(10, 11, 14, 175))
                p.restore()
                p.setPen(QPen(qc("line2"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), TILE_RADIUS, TILE_RADIUS)
                self.pill(p, x + 6, y + 6, "SKIP", qc("bg3"), qc("text2"))
            elif change == "empty":
                self.pill(p, x + 6, y + 6, "EMPTY", qc("bg3"), qc("text2"))
            elif i == self.hover:
                p.setPen(QPen(QColor(255, 255, 255, 150), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), TILE_RADIUS, TILE_RADIUS)
            if self.kbd and i == self.hover:  # the cursor of the arrow keys, also over a tile that has a pending change
                p.setPen(QPen(QColor(255, 255, 255, 235), 2, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(5, 5, -5, -5), TILE_RADIUS, TILE_RADIUS)
            tag = STATUS_TAGS.get(fi.status or "", "")
            if fi.status == "pos" and fi.label_cls:
                tag += f" {len(fi.label_cls)}"
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
        self.draw_overlays(p, fi, lambda x1, y1, x2, y2: QRectF(QPointF(*v.to_widget(x1 * v.iw, y1 * v.ih)),
                                                               QPointF(*v.to_widget(x2 * v.iw, y2 * v.ih))), width=1.4)
        kind = self.pending(fi) or ""
        if kind == "people":
            p.setPen(QPen(qc("accent"), 5))
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        elif kind == "skip":
            p.fillRect(self.rect(), QColor(10, 11, 14, 130))
            p.setPen(QPen(qc("line2"), 5))
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        status = STATUS_TAGS.get(fi.status or "", "new")
        mark = {"people": self.people_text, "skip": "SKIP", "empty": "EMPTY"}.get(kind, status)
        if self.return_mode:
            hint = ("click: tick to return | double-click: editor | wheel: zoom | drag: pan | Z: fit" if self.sticky
                    else "click: tick to return | RMB / Esc: back")
        else:
            hint = ("click: has objects | Ctrl+click or MMB: skip | double-click: editor | wheel: zoom | drag: pan | Z: fit"
                    if self.sticky else "click: toggle 'has objects' | RMB / Esc: back")
        self.badge(p, 10, 10, f"{fi.stem}  [{mark}]  zoom {v.zoom:.1f}x   {hint}", qc("text"))

    # ---------------- preview
    def open_preview(self, fi: FrameInfo) -> None:
        img = imread(str(fi.path))
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

    def mouseDoubleClickEvent(self, e) -> None:
        """A plain double-click opens the frame in the editor. The first click of the pair already ticked the tile, so it is
        ticked back (the same kind of mark, which Ctrl and the middle button change); only the plain click opens."""
        if e.button() == Qt.MouseButton.RightButton:
            self.mousePressEvent(e)  # the second press of a quick double right-click closes the photo the first one opened
            return
        left = e.button() == Qt.MouseButton.LeftButton
        skip = e.button() == Qt.MouseButton.MiddleButton or (left and bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
        if not (left or skip):
            return
        if self.preview is not None and not self.sticky:
            self.mousePressEvent(e)
            return
        fi = self.preview
        if fi is None:
            i = self.tile_at(e.position().x(), e.position().y())
            if i < 0:
                return
            fi = self.page_items()[i]
        if fi.status != "pos":
            self.toggle(fi.stem, "skip" if skip else "people")
        if not skip:
            self.open_requested.emit(fi.stem)

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        if self.preview is not None:
            if self.press is not None and self.last is not None:
                self.preview_view.pan(pos.x() - self.last.x(), pos.y() - self.last.y())
                self.update()
            self.last = pos
            return
        h = self.tile_at(pos.x(), pos.y())
        if h != self.hover or self.kbd:
            self.hover = h
            self.kbd = False
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
        if not self.kbd:  # the arrow-key cursor stays when the mouse leaves
            self.hover = -1
            self.update()

    def stems_on_page(self) -> list[str]:
        return [fi.stem for fi in self.page_items()]

    def paths_on_page(self) -> list[Path]:
        return [fi.path for fi in self.page_items()]
