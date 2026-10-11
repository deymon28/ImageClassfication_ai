"""Image <-> widget mapping for one image shown in an area. Zoom 1 = whole image fitted and centred; zooming keeps
the image point under the cursor in place; panning is clamped so the image never leaves empty space on a side it
could fill. Coordinates are continuous (pixel i spans [i, i+1))."""
from __future__ import annotations


class View:
    MAX_ZOOM = 32.0

    def __init__(self, iw: int, ih: int):
        self.iw, self.ih = max(iw, 1), max(ih, 1)
        self.aw, self.ah = self.iw, self.ih
        self.zoom, self.cx, self.cy = 1.0, self.iw / 2, self.ih / 2  # (cx, cy) = image point at the area centre

    @property
    def s(self) -> float:
        return min(self.aw / self.iw, self.ah / self.ih) * self.zoom

    def fit(self, aw: int, ah: int) -> None:
        self.aw, self.ah = max(aw, 1), max(ah, 1)
        self.clamp()

    def clamp(self) -> None:
        s = self.s
        half_w, half_h = self.aw / 2 / s, self.ah / 2 / s
        self.cx = self.iw / 2 if self.iw <= 2 * half_w else min(max(self.cx, half_w), self.iw - half_w)
        self.cy = self.ih / 2 if self.ih <= 2 * half_h else min(max(self.cy, half_h), self.ih - half_h)

    def to_widget(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.cx) * self.s + self.aw / 2, (y - self.cy) * self.s + self.ah / 2

    def to_img(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.aw / 2) / self.s + self.cx, (y - self.ah / 2) / self.s + self.cy

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        px, py = self.to_img(x, y)
        self.zoom = min(max(self.zoom * factor, 1.0), self.MAX_ZOOM)
        self.cx, self.cy = px - (x - self.aw / 2) / self.s, py - (y - self.ah / 2) / self.s
        self.clamp()

    def zoom_to(self, x1: float, y1: float, x2: float, y2: float, margin: float = 3.0) -> None:
        """Zoom so that the image rectangle (with `margin` times its size around it) fills the area."""
        w, h = max(x2 - x1, 1.0) * margin, max(y2 - y1, 1.0) * margin
        base = min(self.aw / self.iw, self.ah / self.ih)
        self.zoom = min(max(min(self.aw / w, self.ah / h) / base, 1.0), self.MAX_ZOOM)
        self.cx, self.cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.clamp()

    def pan(self, dx: float, dy: float) -> None:
        self.cx -= dx / self.s
        self.cy -= dy / self.s
        self.clamp()

    def reset(self) -> None:
        self.zoom, self.cx, self.cy = 1.0, self.iw / 2, self.ih / 2
        self.clamp()

    def visible_rect(self) -> tuple[float, float, float, float]:
        """Image rectangle (x1, y1, x2, y2) currently visible, clipped to the image."""
        x1, y1 = self.to_img(0, 0)
        x2, y2 = self.to_img(self.aw, self.ah)
        return max(0.0, x1), max(0.0, y1), min(float(self.iw), x2), min(float(self.ih), y2)
