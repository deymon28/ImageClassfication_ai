"""Boxes in image pixels and the YOLO txt format (rows 'cls cx cy w h [conf]', normalized)."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0
    conf: float | None = None  # set only for unreviewed model proposals

    def norm(self) -> Box:
        return Box(min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2), self.cls, self.conf)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def w(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def h(self) -> float:
        return abs(self.y2 - self.y1)

    def area(self) -> float:
        return self.w * self.h

    def corners(self) -> list[tuple[float, float]]:
        """Top-left, top-right, bottom-right, bottom-left."""
        return [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]

    def xyxy(self) -> tuple[float, float, float, float]:
        b = self.norm()
        return b.x1, b.y1, b.x2, b.y2

    def copy(self, **changes) -> Box:
        return replace(self, **changes)


def grow_box(b: Box, px: float, w: float, h: float) -> Box:
    """The box with every side moved `px` pixels out (positive) or in (negative), kept inside the w x h image and at
    least 2 px wide and high: the correction for a fitter that is a pixel or two off on every side."""
    n = b.norm()
    x1, y1, x2, y2 = n.x1 - px, n.y1 - px, n.x2 + px, n.y2 + px
    if x2 - x1 < 2:
        x1, x2 = (n.x1 + n.x2) / 2 - 1, (n.x1 + n.x2) / 2 + 1
    if y2 - y1 < 2:
        y1, y2 = (n.y1 + n.y2) / 2 - 1, (n.y1 + n.y2) / 2 + 1
    return replace(n, x1=max(0.0, x1), y1=max(0.0, y1), x2=min(float(w), x2), y2=min(float(h), y2))


def in_zones(cx: float, cy: float, zones) -> bool:
    """Is the normalized point inside any normalized (x1, y1, x2, y2) zone."""
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in zones)


def parse_rows(text: str) -> list[tuple[int, float, float, float, float, float | None]]:
    """YOLO rows -> (cls, cx, cy, w, h, conf|None). Polygon rows (more than 6 values) become their bounding box. Rows
    that are not numbers (a header, a truncated line) are skipped: one bad row must not make a session unreadable."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            vals = [float(v) for v in p]
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        cls, vals = int(vals[0]), vals[1:]
        if len(vals) in (4, 5):
            conf = vals[4] if len(vals) == 5 else None
            out.append((cls, vals[0], vals[1], vals[2], vals[3], conf))
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            out.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, None))
    return out


def read_rows(path: Path) -> list[tuple[int, float, float, float, float, float | None]]:
    """parse_rows of a label file; bytes that are not UTF-8 only cost their own row."""
    return parse_rows(Path(path).read_text(encoding="utf-8", errors="replace"))


def read_yolo(path: Path, w: int, h: int, min_conf: float = 0.0, ignore_zones=()) -> list[Box]:
    """Label file -> pixel boxes. Proposal rows (with conf) below min_conf or centred in an ignore zone are dropped;
    reviewed labels are returned as written."""
    if not path.exists():
        return []
    boxes = []
    for cls, cx, cy, bw, bh, conf in read_rows(path):
        if conf is not None and (conf < min_conf or in_zones(cx, cy, ignore_zones)):
            continue
        boxes.append(Box((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, cls, conf))
    return boxes


def yolo_lines(boxes: list[Box], w: int, h: int, with_conf: bool = False) -> list[str]:
    """Pixel boxes -> normalized YOLO rows, clipped to the image; degenerate boxes are dropped."""
    out = []
    for b in boxes:
        b = b.norm()
        x1, x2 = max(0.0, b.x1), min(float(w), b.x2)
        y1, y2 = max(0.0, b.y1), min(float(h), b.y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        row = f"{b.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        if with_conf and b.conf is not None:
            row += f" {b.conf:.3f}"
        out.append(row)
    return out


def write_yolo(path: Path, boxes: list[Box], w: int, h: int, with_conf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(yolo_lines(boxes, w, h, with_conf)), encoding="utf-8")
    tmp.replace(path)


def box_at(boxes: list[Box], x: float, y: float) -> int:
    """Index of the smallest box containing the point (small boxes inside big ones stay clickable), -1 if none."""
    hits = [(b.area(), i) for i, b in enumerate(boxes) if b.norm().contains(x, y)]
    return min(hits)[1] if hits else -1


def corner_at(box: Box, x: float, y: float, tol: float) -> int:
    for i, (cx, cy) in enumerate(box.corners()):
        if abs(cx - x) <= tol and abs(cy - y) <= tol:
            return i
    return -1


def edge_at(box: Box, x: float, y: float, tol: float) -> str:
    """Which edge of the box ('l', 'r', 't', 'b') the point is on, '' if none."""
    b = box.norm()
    inside_y, inside_x = b.y1 - tol <= y <= b.y2 + tol, b.x1 - tol <= x <= b.x2 + tol
    if inside_y and abs(x - b.x1) <= tol:
        return "l"
    if inside_y and abs(x - b.x2) <= tol:
        return "r"
    if inside_x and abs(y - b.y1) <= tol:
        return "t"
    if inside_x and abs(y - b.y2) <= tol:
        return "b"
    return ""


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N, 4 xyxy) with every box in b (M, 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    tl, br = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def to_array(boxes: list[Box]) -> np.ndarray:
    return np.array([b.xyxy() for b in boxes], float).reshape(-1, 4)
