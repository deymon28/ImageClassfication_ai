"""Reusable widgets of the main window: the workflow / library list with counts and progress, the grid / editor
switch, the context bar that always says where you are, the hint card, and the toast that confirms what an action did."""
from __future__ import annotations

import html

from PyQt6.QtCore import QRectF, QSize, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QStackedWidget, QStyledItemDelegate, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from framelabel.core.queues import SORTS
from framelabel.core.views import GROUPS, ViewDef
from .icons import dot_icon, mode_icon
from .theme import T, qc

GRID_SIZES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (4, 3), (4, 4), (5, 4))
PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 1   # 0..1 for a stage that has a progress bar
LIBRARY_DOTS = {"all": "text3", "unreviewed": "text", "pos": "ok", "neg": "neutral", "skip": "skip"}  # theme tokens


def grid_size_text(cols: int, rows: int, short: bool = False) -> str:
    if short:
        return f"{cols} × {rows}" + ("  photo" if cols * rows == 1 else "")
    note = "  (one photo per page)" if cols * rows == 1 else f"  ({cols * rows} per page)"
    return f"{cols} × {rows}{note}"


class StageDelegate(QStyledItemDelegate):
    """Draws a thin progress line under the name of a stage (how much of it is done)."""

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        frac = index.data(PROGRESS_ROLE)
        if frac is None or index.column() != 0:
            return
        r = option.rect
        track = QRectF(r.left() + 6, r.bottom() - 3.5, r.width() - 8, 2.5)
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qc("bg3"))
        painter.drawRoundedRect(track, 1.2, 1.2)
        if frac > 0:
            painter.setBrush(qc("ok") if frac >= 1 else qc("accent"))
            painter.drawRoundedRect(QRectF(track.left(), track.top(), max(track.width() * frac, 2.5), track.height()), 1.2, 1.2)
        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        if index.data(PROGRESS_ROLE) is not None:
            s.setHeight(s.height() + 6)
        return s


class ViewTree(QTreeWidget):
    """The workflow (three stages, each with its lenses) and the library, every row with its frame count. The
    highlighted row is where you are; only the current stage shows its lenses."""

    view_chosen = pyqtSignal(str)  # a click on a view (not set by code)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAllColumnsShowFocus(True)
        self.setItemDelegate(StageDelegate(self))
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, self.header().ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, self.header().ResizeMode.ResizeToContents)
        self.items: dict[str, QTreeWidgetItem] = {}
        self.groups: dict[str, QTreeWidgetItem] = {}
        self.defs: dict[str, ViewDef] = {}
        self.itemClicked.connect(self.on_clicked)
        self.currentItemChanged.connect(self.on_current)
        self.quiet = False
        self.counts: dict[str, int] = {}
        self.next_key: str | None = None   # the stage to go to next: its count is highlighted

    def set_views(self, defs: list[ViewDef], class_colors: list[str] | None = None) -> None:
        self.quiet = True
        self.clear()
        self.items.clear()
        self.groups.clear()
        self.defs = {d.key: d for d in defs}
        head = QFont(self.font())
        head.setBold(True)
        head.setPointSizeF(max(head.pointSizeF() * 0.85, 7))
        head.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        stage_font = QFont(self.font())
        stage_font.setBold(True)
        lens_font = QFont(self.font())
        lens_font.setPointSizeF(max(lens_font.pointSizeF() * 0.95, 7))
        for gkey, gname in GROUPS:
            g = QTreeWidgetItem([gname.upper(), ""])
            g.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a heading: neither selectable nor a view
            g.setFont(0, head)
            g.setForeground(0, QBrush(qc("text3")))
            self.addTopLevelItem(g)
            self.groups[gkey] = g
        for d in defs:
            if d.parent is not None:
                continue
            it = self.make_item(self.groups[d.group], d)
            if d.is_stage:
                it.setText(0, f"{d.step}  {d.label}")
                it.setFont(0, stage_font)
                it.setData(0, PROGRESS_ROLE, 0.0 if d.key != "review" else None)
                it.setIcon(0, mode_icon(d.mode))
            elif d.key.startswith("class:"):
                i = int(d.key.split(":", 1)[1])
                it.setIcon(0, dot_icon(class_colors[i] if class_colors and i < len(class_colors) else "text2"))
            else:
                it.setIcon(0, dot_icon(LIBRARY_DOTS.get(d.key, "text2")))
        for d in defs:
            if d.parent is not None:
                it = self.make_item(self.items[d.parent], d)
                it.setFont(0, lens_font)
        for g in self.groups.values():
            g.setExpanded(True)
        for key, it in self.items.items():
            if self.defs[key].is_stage:
                it.setExpanded(False)
        self.quiet = False

    def make_item(self, parent: QTreeWidgetItem, d: ViewDef) -> QTreeWidgetItem:
        it = QTreeWidgetItem(parent, [d.label, ""])
        it.setData(0, Qt.ItemDataRole.UserRole, d.key)
        it.setToolTip(0, d.hint)
        it.setToolTip(1, d.hint)
        it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.items[d.key] = it
        return it

    def set_tooltips(self, tips: dict[str, str]) -> None:
        for key, text in tips.items():
            if key in self.items:
                self.items[key].setToolTip(0, text)
                self.items[key].setToolTip(1, text)

    def set_mode_icons(self, modes: dict[str, str]) -> None:
        """The icon of a stage shows the way it opens: a grid of tiles, or a box with handles (the editor)."""
        for key, mode in modes.items():
            it = self.items.get(key)
            if it is not None and self.defs[key].is_stage:
                it.setIcon(0, mode_icon(mode))

    def sizeHint(self):
        """Tall enough for every visible row, so the list needs no scrolling (the frame list below takes the rest)."""
        total = 8

        def visit(item: QTreeWidgetItem) -> None:
            nonlocal total
            total += max(self.sizeHintForIndex(self.indexFromItem(item)).height(), 20)
            if item.isExpanded():
                for i in range(item.childCount()):
                    visit(item.child(i))

        for g in self.groups.values():
            visit(g)
        return QSize(220, total)

    def set_counts(self, counts: dict[str, int], progress: dict[str, tuple[int, int]] | None = None,
                   next_key: str | None = None) -> None:
        self.counts = dict(counts)
        self.next_key = next_key
        for key, it in self.items.items():
            n = counts.get(key)
            stage = self.defs[key].is_stage
            it.setText(1, "" if n is None else ("✓" if n == 0 and stage else f"{n:,}".replace(",", " ")))
            if stage and key in (progress or {}):
                left, total = progress[key]
                it.setData(0, PROGRESS_ROLE, (1 - left / total) if total else 1.0)
        self.paint_text()
        self.viewport().update()

    def paint_text(self) -> None:
        """Muted counts, empty views dimmed, a finished stage green, the next stage to work on highlighted, the selected
        row white (an explicit colour beats the style sheet)."""
        cur = self.currentItem()
        for key, it in self.items.items():
            n = self.counts.get(key)
            d = self.defs[key]
            for col in (0, 1):
                if it is cur:
                    color = QColor("#ffffff")
                elif col == 1 and d.is_stage and n == 0:
                    color = qc("ok")
                elif col == 1 and key == self.next_key:
                    color = qc("accent_hi")
                elif n == 0:
                    color = qc("text3")
                elif col == 0:
                    color = qc("text") if (d.is_stage or d.parent is None) else qc("text2")
                else:
                    color = qc("text2")
                it.setForeground(col, QBrush(color))

    def select(self, key: str) -> None:
        it = self.items.get(key)
        if it is None:
            return
        self.quiet = True
        self.setCurrentItem(it)
        stage = self.defs[key].stage
        for k, other in self.items.items():  # only the current stage shows its lenses
            if self.defs[k].is_stage:
                other.setExpanded(k == stage)
        if it.parent() is not None:
            it.parent().setExpanded(True)
        self.scrollToItem(it)
        self.quiet = False
        self.paint_text()

    def current_key(self) -> str | None:
        it = self.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def keys(self) -> list[str]:
        return list(self.items)

    def on_clicked(self, it: QTreeWidgetItem, _col: int) -> None:
        key = it.data(0, Qt.ItemDataRole.UserRole)
        if key and not self.quiet:
            self.view_chosen.emit(key)

    def on_current(self, it, _prev) -> None:
        self.paint_text()
        key = it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None
        if key and not self.quiet:  # arrow keys in the list
            self.view_chosen.emit(key)


class SwitchBox(QWidget):
    """Like a QStackedWidget, but only the current page takes space (a stack is as big as its biggest page)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(0)
        self.pages: list[QWidget] = []
        self.index = -1

    def addWidget(self, w: QWidget) -> int:
        self.box.addWidget(w, 0)
        w.setVisible(False)
        self.pages.append(w)
        if self.index < 0:
            self.setCurrentIndex(0)
        return len(self.pages) - 1

    def widget(self, i: int) -> QWidget:
        return self.pages[i]

    def currentIndex(self) -> int:
        return self.index

    def setCurrentIndex(self, i: int) -> None:
        self.index = i
        for k, w in enumerate(self.pages):
            w.setVisible(k == i)


class ModeSwitch(QWidget):
    """Grid | Editor as a two-segment switch: both options are visible and the active one is highlighted. The icons are
    the ones the workflow list shows next to each stage, so a stage's icon tells which segment it opens in."""

    mode_chosen = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segTrack")  # styled in theme.py: a track with one coloured segment
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        self.tips = {"grid": "Grid: many frames at once, click tiles", "edit": "Editor: one frame, draw and fix boxes"}
        for mode, text in (("grid", "Grid"), ("edit", "Editor")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setIcon(mode_icon(mode))
            b.setIconSize(QSize(14, 14))
            b.setToolTip(self.tips[mode])
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keys stay with the canvas / grid
            b.clicked.connect(lambda _=False, m=mode: self.mode_chosen.emit(m))
            self.group.addButton(b)
            self.buttons[mode] = b
            lay.addWidget(b, 1)
        self.group.setExclusive(True)

    def set_mode(self, mode: str) -> None:
        self.buttons[mode].setChecked(True)

    def set_hint(self, key: str) -> None:
        for mode, b in self.buttons.items():
            b.setToolTip(self.tips[mode] + (f"   [{key} switches]" if key else ""))


class ContextBar(QFrame):
    """One thin row above the work area: where you are (stage and lens, position), what the grid page will do, the
    order (Label stage), the tiles per page and the Grid | Editor switch. The description of the view is the tooltip of
    its title; the keys of the mode are in the status bar."""

    grid_size_chosen = pyqtSignal(int, int)
    sort_chosen = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("contextBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 5, 8, 5)
        lay.setSpacing(10)
        self.title = QLabel()
        self.title.setProperty("role", "title")
        self.lens = QLabel()
        self.lens.setProperty("role", "lens")
        self.where = QLabel()
        self.where.setTextFormat(Qt.TextFormat.RichText)
        self.where.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # clips instead of widening
        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.sort = QComboBox()
        for key, text in SORTS:
            self.sort.addItem(text, key)
        self.sort.setToolTip("The order of the frames of this stage")
        self.sort.activated.connect(lambda i: self.sort_chosen.emit(self.sort.itemData(i)))
        self.tiles = QComboBox()
        for c, r in GRID_SIZES:
            self.tiles.addItem(grid_size_text(c, r, short=True), f"{c}x{r}")
        self.tiles.setToolTip("Tiles per page")
        self.tiles.activated.connect(lambda i: self.grid_size_chosen.emit(*map(int, self.tiles.itemData(i).split("x"))))
        self.mode_switch = ModeSwitch()
        for w, stretch in ((self.title, 0), (self.lens, 0), (self.where, 1), (self.stats, 0), (self.sort, 0),
                           (self.tiles, 0), (self.mode_switch, 0)):
            lay.addWidget(w, stretch)

    def set_grid_size(self, cols: int, rows: int) -> None:
        i = self.tiles.findData(f"{cols}x{rows}")
        if i >= 0:
            self.tiles.setCurrentIndex(i)

    def set_sort(self, key: str) -> None:
        i = self.sort.findData(key)
        if i >= 0:
            self.sort.setCurrentIndex(i)

    def show_tiles(self, on: bool) -> None:
        self.tiles.setVisible(on)
        self.stats.setVisible(on)

    def show_sort(self, on: bool) -> None:
        self.sort.setVisible(on)

    def set_text(self, title: str, where: str, hint: str, stats: str = "", lens: str = "") -> None:
        self.title.setText(title)
        self.title.setToolTip(hint)
        self.lens.setText(f"›  {lens}" if lens else "")
        self.lens.setVisible(bool(lens))
        parts = [html.escape(p) for p in where.split("  |  ")]
        sep = f"<span style='color:{T['text3']}'>&nbsp;&middot;&nbsp;</span>"
        self.where.setText(f"<span style='color:{T['text2']}'>" + sep.join(parts) + "</span>")
        self.stats.setText(stats)


class CoachCard(QLabel):
    """Under the list of views: what the current view is for and what to do in it, in a sentence or two."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("coach")
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)

    def show_hint(self, title: str, mode: str, hint: str) -> None:
        self.setText(f"<b style='color:{T['text']}'>{html.escape(title)}</b>"
                     f"<span style='color:{T['text3']}'> &nbsp;&middot;&nbsp; {html.escape(mode)}</span><br>{html.escape(hint)}")


class Toast(QFrame):
    """A short message over the work area ('Saved as empty'). It fades after a moment and ignores the mouse, unless it
    carries an action ('Undo'), which stays longer."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("toast")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(12)
        self.label = QLabel()
        self.button = QPushButton()
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.clicked.connect(self.run_action)
        lay.addWidget(self.label)
        lay.addWidget(self.button)
        self.action = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.hide()

    def text(self) -> str:
        return self.label.text()

    def show_message(self, text: str, ms: int = 1800, action: tuple | None = None) -> None:
        """`action` = (button text, callable)."""
        self.label.setText(text)
        self.action = action[1] if action else None
        self.button.setVisible(action is not None)
        if action:
            self.button.setText(action[0])
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, action is None)
        self.adjustSize()
        self.place()
        self.show()
        self.raise_()
        self.timer.start(max(ms, 6000) if action else ms)

    def run_action(self) -> None:
        action, self.action = self.action, None
        self.hide()
        if action:
            action()

    def place(self) -> None:
        p = self.parentWidget()
        self.move(max((p.width() - self.width()) // 2, 0), max(p.height() - self.height() - 48, 0))


class Central(QWidget):
    """Context bar above the grid / editor stack, with the toast floating over the stack."""

    def __init__(self, bar: ContextBar, stack: QStackedWidget, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(stack, 1)
        self.toast = Toast(self)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.toast.isVisible():
            self.toast.place()


def section_label(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setProperty("role", "section")
    f = QFont(lab.font())
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
    lab.setFont(f)
    return lab


def note_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "note")
    lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    return lab
