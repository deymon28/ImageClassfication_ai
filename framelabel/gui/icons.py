"""Small vector icons drawn in code (no image files): the two ways of looking (a grid of tiles, a box with handles) and
colour dots for statuses and classes. Each icon has a muted look and a bright one (selected / checked)."""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from .theme import T, qc

SIZE = 16
DPR = 2.0


def _pixmap(draw, color: QColor) -> QPixmap:
    pm = QPixmap(int(SIZE * DPR), int(SIZE * DPR))
    pm.setDevicePixelRatio(DPR)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw(p, color)
    p.end()
    return pm


def _draw_grid(p: QPainter, c: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    for x in (2.0, 9.0):
        for y in (2.0, 9.0):
            p.drawRoundedRect(QRectF(x, y, 5, 5), 1.2, 1.2)


def _draw_editor(p: QPainter, c: QColor) -> None:
    """A box with a handle on every corner: what the editor is about."""
    pen = QPen(c, 1.3)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.5, 3.5, 9, 9))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    for x, y in ((3.5, 3.5), (12.5, 3.5), (3.5, 12.5), (12.5, 12.5)):
        p.drawRect(QRectF(x - 1.6, y - 1.6, 3.2, 3.2))


def _draw_dot(p: QPainter, c: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QPointF(8, 8), 3.6, 3.6)


def _icon(draw, normal: QColor, bright: QColor) -> QIcon:
    ic = QIcon()
    dim, lit = _pixmap(draw, normal), _pixmap(draw, bright)
    ic.addPixmap(dim, QIcon.Mode.Normal, QIcon.State.Off)
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Selected, QIcon.Mode.Active):
        ic.addPixmap(lit, mode, QIcon.State.On)
    for mode in (QIcon.Mode.Selected, QIcon.Mode.Active):
        ic.addPixmap(lit, mode, QIcon.State.Off)
    return ic


@lru_cache(maxsize=None)
def mode_icon(mode: str) -> QIcon:
    """'grid' or 'edit': muted normally, white when selected in a list or checked on a button."""
    return _icon(_draw_grid if mode == "grid" else _draw_editor, qc("text2"), QColor("#ffffff"))


@lru_cache(maxsize=None)
def dot_icon(color: str) -> QIcon:
    """A round dot of a colour ('#rrggbb' or a theme token such as 'ok')."""
    c = qc(color) if color in T else QColor(color)  # a theme token, else any Qt colour ('#rrggbb', 'red', ...)
    return _icon(_draw_dot, c, c)
