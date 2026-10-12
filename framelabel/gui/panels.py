"""Small reusable widgets of the main window: the view list with counts, the grid / editor switch, the context bar
that always says where you are, and the toast that confirms what an action did."""
from __future__ import annotations

from PyQt6.QtCore import QSize, QTimer, Qt, pyqtSignal
import html

from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from framelabel.core.views import GROUPS, ViewDef
from .theme import T, qc

GRID_SIZES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (4, 3), (4, 4), (5, 4))
DIM = qc("text3")


def grid_size_text(cols: int, rows: int) -> str:
    note = "  (one photo per page)" if cols * rows == 1 else f"  ({cols * rows} per page)"
    return f"{cols} × {rows}{note}"


class ViewTree(QTreeWidget):
    """The views of the session in groups, each with its frame count. The highlighted row is where you are."""

    view_chosen = pyqtSignal(str)  # a click on a view (not set by code)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.setIndentation(10)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAllColumnsShowFocus(True)
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

    def set_views(self, defs: list[ViewDef], collapsed: set[str] = frozenset()) -> None:
        self.quiet = True
        self.clear()
        self.items.clear()
        self.groups.clear()
        self.defs = {d.key: d for d in defs}
        head = QFont(self.font())
        head.setBold(True)
        head.setPointSizeF(max(head.pointSizeF() * 0.85, 7))
        head.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        for gkey, gname in GROUPS:
            members = [d for d in defs if d.group == gkey]
            if not members:
                continue
            g = QTreeWidgetItem([gname.upper(), ""])
            g.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a heading: neither selectable nor a view
            g.setFont(0, head)
            g.setForeground(0, QBrush(DIM))
            self.addTopLevelItem(g)
            self.groups[gkey] = g
            for d in members:
                it = QTreeWidgetItem(g, [d.label, ""])
                it.setData(0, Qt.ItemDataRole.UserRole, d.key)
                it.setToolTip(0, d.hint)
                it.setToolTip(1, d.hint)
                it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.items[d.key] = it
            g.setExpanded(gkey not in collapsed)
        self.quiet = False

    def sizeHint(self):
        """Tall enough for every row, so the list needs no scrolling while the groups are open (the frame list below
        takes the rest of the panel)."""
        rows = sum(1 + (g.childCount() if g.isExpanded() else 0) for g in self.groups.values())
        h = self.sizeHintForRow(0) if self.groups else 24
        return QSize(220, max(h, 20) * rows + 8)

    def collapsed_groups(self) -> set[str]:
        return {k for k, g in self.groups.items() if not g.isExpanded()}

    def set_counts(self, counts: dict[str, int]) -> None:
        self.counts = dict(counts)
        for key, it in self.items.items():
            n = counts.get(key)
            it.setText(1, "" if n is None else f"{n:,}".replace(",", " "))
        self.paint_text()

    def paint_text(self) -> None:
        """Counts muted, empty views dimmed, the selected row white (an explicit colour beats the style sheet)."""
        cur = self.currentItem()
        for key, it in self.items.items():
            n = self.counts.get(key)
            for col in (0, 1):
                if it is cur:
                    it.setForeground(col, QBrush(QColor("#ffffff")))
                elif n == 0:
                    it.setForeground(col, QBrush(qc("text3")))
                else:
                    it.setForeground(col, QBrush(qc("text") if col == 0 else qc("text2")))

    def select(self, key: str) -> None:
        it = self.items.get(key)
        if it is None:
            return
        self.quiet = True
        self.setCurrentItem(it)
        it.parent().setExpanded(True)  # the current view is never hidden inside a collapsed group
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
    """Grid | Editor as a two-segment switch: both options are visible and the active one is highlighted."""

    mode_chosen = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segTrack")  # styled in theme.py: a track with one coloured segment
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(3)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        for mode, text, tip in (("grid", "Grid", "Many frames at once: triage by clicking tiles"),
                                ("edit", "Editor", "One frame at a time: draw and fix boxes")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
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
        for b in self.buttons.values():
            b.setToolTip(b.toolTip().split("  [")[0] + (f"  [{key}]" if key else ""))


class ContextBar(QFrame):
    """Always visible above the work area: which view, which mode, where in it, and what you can do here."""

    grid_size_chosen = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("contextBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(3)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setProperty("role", "title")
        self.where = QLabel()
        self.where.setTextFormat(Qt.TextFormat.RichText)
        self.tiles_label = QLabel("Tiles")
        self.tiles_label.setProperty("role", "muted")
        self.tiles = QComboBox()
        for c, r in GRID_SIZES:
            self.tiles.addItem(grid_size_text(c, r), f"{c}x{r}")
        self.tiles.setToolTip("How many frames the grid shows per page")
        self.tiles.activated.connect(lambda i: self.grid_size_chosen.emit(*map(int, self.tiles.itemData(i).split("x"))))
        top.addWidget(self.title)
        top.addSpacing(14)
        top.addWidget(self.where, 1)
        top.addWidget(self.tiles_label)
        top.addWidget(self.tiles)
        lay.addLayout(top)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setProperty("role", "muted")
        lay.addWidget(self.hint)

    def set_grid_size(self, cols: int, rows: int) -> None:
        i = self.tiles.findData(f"{cols}x{rows}")
        if i >= 0:
            self.tiles.setCurrentIndex(i)

    def show_tiles(self, on: bool) -> None:
        self.tiles.setVisible(on)
        self.tiles_label.setVisible(on)

    def set_text(self, title: str, where: str, hint: str) -> None:
        self.title.setText(title)
        head, *rest = [html.escape(p) for p in where.split("  |  ")]
        sep = f" <span style='color:{T['text3']}'>&nbsp;/&nbsp;</span> "
        self.where.setText(f"<b style='color:{T['accent_hi']}'>{head}</b>" + (sep + sep.join(rest) if rest else ""))
        self.hint.setText(hint)


class Toast(QLabel):
    """A short confirmation over the work area ('Saved as empty'); it fades after a moment and ignores the mouse."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setObjectName("toast")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, text: str, ms: int = 1800) -> None:
        self.setText(text)
        self.adjustSize()
        self.place()
        self.show()
        self.raise_()
        self.timer.start(ms)

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
