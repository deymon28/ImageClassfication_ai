"""SAM-assisted box fitting: a rough box or a click on an object -> a tight box around SAM's mask.

Small (far) objects are a few pixels in SAM's 1024 px input when the whole frame is used, so SAM runs on a crop
around the prompt (about 3x the box, at least 192 px) and the result is mapped back to frame pixels.
GPU work: the caller checks that the game is closed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from framelabel.core.boxes import Box


def crop_window(x1: float, y1: float, x2: float, y2: float, w: int, h: int, scale: float = 3.0,
                min_side: int = 192) -> tuple[int, int, int, int]:
    """Square-ish window around a box, clamped to the image."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, scale * max(x2 - x1, y2 - y1))
    wx1, wy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    wx2, wy2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return wx1, wy1, wx2, wy2


def mask_box(mask: np.ndarray, near: tuple[float, float] | None = None) -> tuple[float, float, float, float] | None:
    """Bounding box of the mask's connected component that contains (or is nearest to) `near`, else the largest."""
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    comps = list(range(1, n))
    if near is not None:
        px, py = int(near[0]), int(near[1])
        if 0 <= py < lab.shape[0] and 0 <= px < lab.shape[1] and lab[py, px] > 0:
            best = lab[py, px]
        else:
            best = min(comps, key=lambda i: (cent[i][0] - near[0]) ** 2 + (cent[i][1] - near[1]) ** 2)
    else:
        best = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw, bh = stats[best, 0], stats[best, 1], stats[best, 2], stats[best, 3]
    return float(x), float(y), float(x + bw), float(y + bh)


class BoxFitter:
    """Loads SAM once (lazy); fit() refines a box or a click."""

    def __init__(self, weights: Path, device: int | str = 0, half: bool = True, max_vram_gb: float = 3.0):
        self.weights, self.device, self.half, self.max_vram_gb = Path(weights), device, half, max_vram_gb
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from ultralytics import SAM

        from .guard import cap_vram

        if torch.cuda.is_available() and self.device != "cpu":
            cap_vram(self.max_vram_gb)
        else:
            self.device, self.half = "cpu", False
        self.model = SAM(str(self.weights))

    def _segment(self, crop: np.ndarray, bbox=None, point=None) -> np.ndarray | None:
        kw = {}
        if bbox is not None:
            kw["bboxes"] = [list(map(float, bbox))]
        if point is not None:
            kw["points"] = [list(map(float, point))]
            kw["labels"] = [1]
        res = self.model.predict(crop, device=self.device, half=self.half, verbose=False, **kw)
        if not res or res[0].masks is None or not len(res[0].masks.data):
            return None
        masks = res[0].masks.data.cpu().numpy()
        return masks[0]

    def fit(self, img: np.ndarray, box: Box | None = None, point: tuple[float, float] | None = None) -> Box | None:
        """Tight box from SAM for a rough box (preferred) or a point; None when SAM finds nothing."""
        self.load()
        h, w = img.shape[:2]
        if box is not None:
            b = box.norm()
            wx1, wy1, wx2, wy2 = crop_window(b.x1, b.y1, b.x2, b.y2, w, h)
            crop = img[wy1:wy2, wx1:wx2]
            mask = self._segment(crop, bbox=(b.x1 - wx1, b.y1 - wy1, b.x2 - wx1, b.y2 - wy1))
            near = ((b.x1 + b.x2) / 2 - wx1, (b.y1 + b.y2) / 2 - wy1)
            cls = box.cls
        elif point is not None:
            px, py = point
            wx1, wy1, wx2, wy2 = crop_window(px - 40, py - 80, px + 40, py + 80, w, h, scale=2.0, min_side=256)
            crop = img[wy1:wy2, wx1:wx2]
            mask = self._segment(crop, point=(px - wx1, py - wy1))
            near, cls = (px - wx1, py - wy1), 0
        else:
            return None
        if mask is None:
            return None
        if mask.shape[:2] != crop.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        mb = mask_box(mask, near)
        if mb is None:
            return None
        x1, y1, x2, y2 = mb
        return Box(x1 + wx1, y1 + wy1, x2 + wx1, y2 + wy1, cls, None)
