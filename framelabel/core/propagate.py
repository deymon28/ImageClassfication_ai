"""Carry boxes from one frame to the next: neighbouring frames are near-duplicates, so the previous frame's boxes are a
good start. Each box is tracked by template matching its patch in a search window of the new frame, then snapped to
an overlapping model proposal on the new frame when there is one (the model's box is fitted to this frame)."""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box, iou_matrix, to_array

MIN_SCORE = 0.5  # template match score below which the box keeps its old position


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def track_box(prev: np.ndarray, cur: np.ndarray, box: Box, search: float = 1.0, max_side: int = 96) -> tuple[Box, float]:
    """Find the box's patch from `prev` in `cur` within `search` box sizes around it -> (moved box, match score)."""
    h, w = prev.shape[:2]
    b = box.norm()
    bw, bh = max(b.w, 2.0), max(b.h, 2.0)
    # the template is the box plus a small margin (context helps for thin boxes), downscaled to at most max_side
    pad = 0.15
    tx1, ty1 = int(max(0, b.x1 - pad * bw)), int(max(0, b.y1 - pad * bh))
    tx2, ty2 = int(min(w, b.x2 + pad * bw + 1)), int(min(h, b.y2 + pad * bh + 1))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return box.copy(), 0.0
    sx1, sy1 = int(max(0, tx1 - search * bw)), int(max(0, ty1 - search * bh))
    sx2, sy2 = int(min(w, tx2 + search * bw)), int(min(h, ty2 + search * bh))
    k = min(1.0, max_side / max(tx2 - tx1, ty2 - ty1))
    tmpl = _gray(prev)[ty1:ty2, tx1:tx2]
    area = _gray(cur)[sy1:sy2, sx1:sx2]
    if k < 1.0:
        tmpl = cv2.resize(tmpl, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        area = cv2.resize(area, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if area.shape[0] < tmpl.shape[0] or area.shape[1] < tmpl.shape[1] or tmpl.std() < 1e-3:
        return box.copy(), 0.0
    res = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    dx, dy = sx1 + mx / k - tx1, sy1 + my / k - ty1
    if score < MIN_SCORE:
        return box.copy(), float(score)
    return Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, box.cls, None), float(score)


def snap_to_proposals(boxes: list[Box], proposals: list[Box], min_iou: float = 0.3) -> list[Box]:
    """Replace each box's geometry with the best-overlapping proposal (one proposal per box, class kept)."""
    if not boxes or not proposals:
        return [b.copy() for b in boxes]
    ious = iou_matrix(to_array(boxes), to_array(proposals))
    out, used = [], set()
    for i in np.argsort(-ious.max(1)):
        j = int(np.argmax(np.where([k in used for k in range(len(proposals))], -1, ious[i])))
        if ious[i, j] >= min_iou and j not in used:
            used.add(j)
            p = proposals[j].norm()
            out.append((i, Box(p.x1, p.y1, p.x2, p.y2, boxes[i].cls, None)))
        else:
            out.append((i, boxes[i].copy(conf=None)))
    return [b for _, b in sorted(out, key=lambda x: x[0])]


def propagate(prev_img: np.ndarray, cur_img: np.ndarray, boxes: list[Box], proposals: list[Box] | None = None) -> list[Box]:
    """Boxes of the previous frame moved onto the current frame (tracked, then snapped to proposals)."""
    moved = [track_box(prev_img, cur_img, b)[0] for b in boxes]
    return snap_to_proposals(moved, proposals or [])
