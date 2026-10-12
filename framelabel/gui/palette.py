"""The command palette (Ctrl+K): type a few letters to find a view, an action or a frame, and run it with Enter. It lists
the same things the menus and the key map have, with their shortcuts, so it also teaches the keys."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate,
                             QVBoxLayout)

from .theme import qc

MAX_RESULTS = 12
WORD_BREAKS = " >-:/·_(.,"


@dataclass
class Entry:
    title: str
    detail: str = ""        # muted text on the right: a count, a hint
    shortcut: str = ""      # shown on the right instead of the detail when there is one
    run: Callable[[], None] | None = None
    kind: str = ""          # "view", "action" or "frame": only used to order equal matches


def fuzzy_score(query: str, text: str) -> int | None:
    """None when the letters of the query do not appear in order in the text, else a score (higher = better): a
    prefix or a whole-word hit beats letters scattered over the text."""
    q, t = query.lower().replace(" ", ""), text.lower()
    if not q:
        return 0
    score, at, prev = 0, 0, -2
    for ch in q:
        j = t.find(ch, at)
        if j < 0:
            return None
        score += 10
        if j == prev + 1:
            score += 14                   # consecutive letters
        if j == 0 or t[j - 1] in WORD_BREAKS:
            score += 12                   # the start of a word
        score -= min(j - at, 12)          # a gap costs a little
        prev, at = j, j + 1
    if t.startswith(q):
        score += 50
    elif q in t:
        score += 30
    return score - len(t) // 10


def rank(entries: list[Entry], query: str, limit: int = MAX_RESULTS) -> list[Entry]:
    """The entries matching the query, best first; without a query the first `limit` in their given order."""
    if not query.strip():
        return entries[:limit]
    scored = []
    for i, e in enumerate(entries):
        s = fuzzy_score(query, e.title)
        if s is not None:
            scored.append((-s, i, e))
    return [e for *_k, e in sorted(scored, key=lambda x: (x[0], x[1]))[:limit]]


class _Delegate(QStyledItemDelegate):
    def paint(self, p, option, index) -> None:
        e: Entry = index.data(Qt.ItemDataRole.UserRole)
        r = QRectF(option.rect)
        p.save()
        p.setRenderHint(p.RenderHint.Antialiasing)
        if option.state & QStyle.StateFlag.State_Selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(108, 140, 255, 60))
            p.drawRoundedRect(r.adjusted(2, 1, -2, -1), 5, 5)
        f = QFont(option.font)
        p.setFont(f)
        right = e.shortcut or e.detail
        rw = p.fontMetrics().horizontalAdvance(right) + 8 if right else 0
        p.setPen(qc("text"))
        p.drawText(r.adjusted(12, 0, -rw - 12, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   p.fontMetrics().elidedText(e.title, Qt.TextElideMode.ElideRight, int(r.width() - rw - 24)))
        if right:
            p.setPen(qc("text2") if e.shortcut else qc("text3"))
            p.drawText(r.adjusted(0, 0, -12, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, right)
        p.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 30)


class CommandPalette(QDialog):
    """`entries` are the fixed choices (views, actions); `more(query)` adds choices that depend on what was typed
    (frames). The entry that was chosen is in `chosen` after exec(): the caller runs it once the palette is gone."""

    def __init__(self, entries: list[Entry], more: Callable[[str], list[Entry]] | None = None, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("palette")
        self.entries, self.more = entries, more
        self.chosen: Entry | None = None
        self.resize(580, 60)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Go to a view or frame, or run an action...")
        self.edit.textChanged.connect(self.refill)
        self.edit.installEventFilter(self)
        self.list = QListWidget()
        self.list.setItemDelegate(_Delegate(self.list))
        self.list.itemClicked.connect(lambda _i: self.accept_current())
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hint = QLabel("↑↓ choose   Enter run   Esc close")
        self.hint.setProperty("role", "note")
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        lay.addWidget(self.hint)
        self.refill("")

    def showEvent(self, e) -> None:
        super().showEvent(e)
        p = self.parentWidget()
        if p is not None:
            top = p.mapToGlobal(p.rect().topLeft())
            self.move(top.x() + (p.width() - self.width()) // 2, top.y() + 70)
        self.edit.setFocus()

    def refill(self, query: str) -> None:
        found = rank(self.entries + (self.more(query) if self.more and query.strip() else []), query)
        self.list.clear()
        for e in found:
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, e)
            it.setSizeHint(QSize(100, 30))
            self.list.addItem(it)
        if found:
            self.list.setCurrentRow(0)
        self.list.setVisible(bool(found))
        self.hint.setText("↑↓ choose   Enter run   Esc close" if found else "Nothing matches")
        self.resize(self.width(), 62 + (min(len(found), MAX_RESULTS) * 30 + 8 if found else 0) + 20)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.edit and ev.type() == QEvent.Type.KeyPress:
            key = ev.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                n = self.list.count()
                if n:
                    self.list.setCurrentRow((self.list.currentRow() + (1 if key == Qt.Key.Key_Down else -1)) % n)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.accept_current()
                return True
        return super().eventFilter(obj, ev)

    def accept_current(self) -> None:
        it = self.list.currentItem()
        if it is not None:
            self.chosen = it.data(Qt.ItemDataRole.UserRole)
            self.accept()
