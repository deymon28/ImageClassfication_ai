"""The left panel's frame list as a model/view pair: rows are built on demand, so switching queues of thousands of
frames does not create thousands of widgets."""
from __future__ import annotations

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt
from PyQt6.QtGui import QColor

from framelabel.core.queues import FrameInfo

STATUS_COLORS = {"pos": "#4ade80", "neg": "#8b93a1", "skip": "#e0a458", "todo": "#fbbf24", None: "#e8eaed"}


def frame_label(fi: FrameInfo) -> str:
    tag = {"pos": "labeled", "neg": "empty", "skip": "skipped", "todo": "marked"}.get(fi.status or "", "new")
    extra = f"  {fi.max_conf:.2f}" if fi.max_conf else ""
    return f"{fi.stem}  [{tag}]{extra}"


class FrameListModel(QAbstractListModel):
    """Shows `queue` (the same list object the window edits in place); call row_changed(i) after replacing item i."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue: list[FrameInfo] = []
        self.notes: dict[str, str] = {}   # stem -> a short note shown after the name (what Review found)
        self.colors = {k: QColor(v) for k, v in STATUS_COLORS.items()}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.queue)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.queue):
            return None
        fi = self.queue[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            note = self.notes.get(fi.stem)
            return frame_label(fi) + (f"  {note}" if note else "")
        if role == Qt.ItemDataRole.ForegroundRole:
            return self.colors.get(fi.status, self.colors[None])
        return None

    def set_queue(self, queue: list[FrameInfo]) -> None:
        self.beginResetModel()
        self.queue = queue
        self.endResetModel()

    def row_changed(self, i: int) -> None:
        if 0 <= i < len(self.queue):
            ix = self.index(i)
            self.dataChanged.emit(ix, ix)
