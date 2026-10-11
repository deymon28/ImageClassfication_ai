"""Undo/redo of a frame's box list (snapshots; box lists are small)."""
from __future__ import annotations

import copy

from .boxes import Box


class History:
    def __init__(self, limit: int = 200):
        self.limit = limit
        self.undo_stack: list[list[Box]] = []
        self.redo_stack: list[list[Box]] = []

    def push(self, boxes: list[Box]) -> None:
        """Record the state before a change."""
        self.undo_stack.append(copy.deepcopy(boxes))
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def drop_last(self) -> list[Box] | None:
        """Forget the last snapshot (a click that turned out not to be an edit) and return it."""
        return self.undo_stack.pop() if self.undo_stack else None

    def undo(self, current: list[Box]) -> list[Box] | None:
        if not self.undo_stack:
            return None
        self.redo_stack.append(copy.deepcopy(current))
        return self.undo_stack.pop()

    def redo(self, current: list[Box]) -> list[Box] | None:
        if not self.redo_stack:
            return None
        self.undo_stack.append(copy.deepcopy(current))
        return self.redo_stack.pop()

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)
